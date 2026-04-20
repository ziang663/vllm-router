use super::{
    get_healthy_worker_indices, BackendObservedLoad, CacheAwareConfig, LoadBalancingPolicy,
    RequestHeaders,
};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::normalize_model_key;
use crate::tree::Tree;
use dashmap::DashMap;
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread;
use std::time::Duration;
use tracing::{debug, info};

/// Cache-aware routing policy that never queues onto a busy worker.
///
/// Selection is restricted to idle workers only. When every healthy worker is
/// already running or waiting on backend work, selection fails so the caller can
/// return 503 immediately.
#[derive(Debug)]
pub struct CacheAwareNoQueuePolicy {
    config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    observed: RwLock<HashMap<String, BackendObservedLoad>>,
    eviction_handle: Option<thread::JoinHandle<()>>,
}

impl CacheAwareNoQueuePolicy {
    pub fn new() -> Self {
        Self::with_config(CacheAwareConfig::default())
    }

    pub fn with_config(config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());

        let eviction_handle = if config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = config.max_tree_size;
            let interval = config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));
                for entry in trees_clone.iter() {
                    let model_id = entry.key();
                    let tree = entry.value();
                    tree.evict_tenant_by_size(max_tree_size);
                    debug!(
                        "Cache eviction completed for model {}, max_size: {}",
                        model_id, max_tree_size
                    );
                }
            }))
        } else {
            None
        };

        Self {
            config,
            trees,
            observed: RwLock::new(HashMap::new()),
            eviction_handle,
        }
    }

    fn worker_is_idle(&self, worker: &dyn Worker) -> bool {
        if let Ok(observed) = self.observed.read() {
            if let Some(state) = observed.get(worker.url()) {
                return state.running <= 0 && state.waiting <= 0;
            }
        }

        worker.load() == 0
    }

    pub fn remove_worker_by_url(&self, url: &str) {
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    fn select_worker_min_load(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        candidate_indices: &[usize],
        model_id: &str,
        max_load: usize,
        min_load: usize,
    ) -> Option<usize> {
        if tracing::enabled!(tracing::Level::DEBUG) {
            let worker_loads: Vec<(&str, usize)> =
                workers.iter().map(|w| (w.url(), w.load())).collect();
            debug!(
                "Load balancing triggered | max: {} | min: {} | workers: {:?}",
                max_load, min_load, worker_loads
            );
        }

        RouterMetrics::record_load_balancing_event();
        RouterMetrics::set_load_range(max_load, min_load);

        let min_load_idx = candidate_indices
            .iter()
            .min_by_key(|&&idx| workers[idx].load())
            .copied()?;

        if let Some(text) = request_text {
            let tree = self.trees.get(model_id).map(|entry| entry.value().clone());
            if let Some(tree) = tree {
                tree.insert(text, workers[min_load_idx].url());
            }
        }

        workers[min_load_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[min_load_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[min_load_idx].url());

        Some(min_load_idx)
    }
}

impl LoadBalancingPolicy for CacheAwareNoQueuePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let idle_indices: Vec<usize> = healthy_indices
            .iter()
            .copied()
            .filter(|&idx| self.worker_is_idle(workers[idx].as_ref()))
            .collect();
        if idle_indices.is_empty() {
            debug!("No idle workers available for cache_aware_no_queue");
            return None;
        }

        let model_id = normalize_model_key(workers[idle_indices[0]].model_id());
        let (min_load, max_load) =
            idle_indices
                .iter()
                .fold((usize::MAX, 0usize), |(min, max), &idx| {
                    let load = workers[idx].load();
                    (min.min(load), max.max(load))
                });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };

        let is_imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        if is_imbalanced {
            return self.select_worker_min_load(
                workers,
                request_text,
                &idle_indices,
                model_id,
                max_load,
                min_load,
            );
        }

        let text = request_text.unwrap_or("");
        let tree = self.trees.get(model_id).map(|entry| entry.value().clone());

        let Some(tree) = tree else {
            let mut rng = rand::rng();
            let random_idx = rng.random_range(0..idle_indices.len());
            let selected_idx = idle_indices[random_idx];
            workers[selected_idx].increment_processed();
            RouterMetrics::record_processed_request(workers[selected_idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());
            return Some(selected_idx);
        };

        let result = tree.prefix_match_with_counts(text);
        let match_rate = if result.input_char_count == 0 {
            0.0
        } else {
            result.matched_char_count as f32 / result.input_char_count as f32
        };

        let selected_idx = if match_rate > self.config.cache_threshold {
            let tenant_url: &str = &result.tenant;
            idle_indices
                .iter()
                .copied()
                .find(|&idx| workers[idx].url() == tenant_url)
        } else {
            idle_indices
                .iter()
                .min_by_key(|&&idx| workers[idx].load())
                .copied()
        };

        if let Some(idx) = selected_idx {
            tree.insert(text, workers[idx].url());
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
            return Some(idx);
        }

        if match_rate > self.config.cache_threshold {
            tree.remove_tenant(&result.tenant);
        }

        if let Some(idx) = idle_indices.first().copied() {
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
            Some(idx)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "cache_aware_no_queue"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn update_backend_observations(&self, observations: &HashMap<String, BackendObservedLoad>) {
        if let Ok(mut observed) = self.observed.write() {
            *observed = observations.clone();
        }
    }

    fn select_worker_pair_with_headers(
        &self,
        prefill_workers: &[Arc<dyn Worker>],
        decode_workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, usize)> {
        let prefill_idx =
            self.select_worker_with_headers(prefill_workers, request_text, headers)?;

        let healthy_decode = get_healthy_worker_indices(decode_workers);
        let idle_decode: Vec<usize> = healthy_decode
            .iter()
            .copied()
            .filter(|&idx| self.worker_is_idle(decode_workers[idx].as_ref()))
            .collect();
        let decode_idx = idle_decode
            .iter()
            .min_by_key(|&&idx| decode_workers[idx].load())
            .copied()?;

        Some((prefill_idx, decode_idx))
    }

    fn requires_initialization(&self) -> bool {
        true
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        info!(
            "Initializing workers for cache-aware-no-queue policy: {}",
            workers
                .iter()
                .map(|w| w.url())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut model_workers: HashMap<String, Vec<&Arc<dyn Worker>>> = HashMap::new();
        for worker in workers {
            let tree_key = normalize_model_key(worker.model_id());
            model_workers
                .entry(tree_key.to_string())
                .or_default()
                .push(worker);
        }

        for (tree_key, model_workers) in model_workers {
            let tree = self
                .trees
                .entry(tree_key)
                .or_insert_with(|| Arc::new(Tree::new()))
                .clone();
            for worker in model_workers {
                tree.insert("", worker.url());
            }
        }
    }
}

impl Default for CacheAwareNoQueuePolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CacheAwareNoQueuePolicy {
    fn drop(&mut self) {
        if let Some(handle) = self.eviction_handle.take() {
            drop(handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_cache_aware_no_queue_skips_busy_worker_and_returns_none_when_all_busy() {
        let config = CacheAwareConfig {
            eviction_interval_secs: 0,
            ..Default::default()
        };
        let policy = CacheAwareNoQueuePolicy::with_config(config);
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        policy.init_workers(&workers);

        let first = policy
            .select_worker(&workers, Some("same prompt"))
            .expect("first request should select a worker");
        workers[first].increment_load();

        let second = policy
            .select_worker(&workers, Some("same prompt"))
            .expect("busy worker should be skipped");
        assert_ne!(first, second);

        workers[second].increment_load();
        assert!(
            policy.select_worker(&workers, Some("same prompt")).is_none(),
            "all busy workers should return no selection"
        );
    }
}
