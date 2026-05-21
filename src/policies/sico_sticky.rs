//! Sico-style sticky session load balancing policy.
//!
//! This policy ports the core behavior of Sico's `StickySessionLB` into the
//! native router policy interface:
//! - router-local load imbalance bypasses stickiness and picks the least-loaded worker
//! - later requests for the same session stay sticky
//! - new or missing sessions fall back to cache-aware prefix routing
//! - migration only happens when the mapped worker is sufficiently worse than
//!   the best worker and a global cooldown has elapsed
//! - a small local waiting bump reduces stampedes between load refreshes

use super::{
    get_healthy_worker_indices, normalize_model_key, BackendObservedLoad, CacheAwareConfig,
    LoadBalancingPolicy, RequestHeaders,
};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::tree::Tree;
use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

const STICKY_WAIT_WEIGHT: i64 = 4;
const STICKY_SLACK: i64 = 80;
const STICKY_MIGRATE_COOLDOWN: Duration = Duration::from_secs(10 * 60);
const STICKY_CLIENT_COUNT: i64 = 1;

#[derive(Debug, Default)]
struct StickyState {
    start_idx: usize,
    session_map: HashMap<String, String>,
    local_counts: HashMap<String, (i64, i64)>,
    last_tokens: HashMap<String, i64>,
    last_migration_at: Option<Instant>,
}

#[derive(Debug)]
pub struct SicoStickyPolicy {
    state: Mutex<StickyState>,
    cache_config: CacheAwareConfig,
    trees: Arc<DashMap<String, Arc<Tree>>>,
    cached_loads: RwLock<HashMap<String, isize>>,
    cached_load_generation: AtomicI64,
    observed: RwLock<HashMap<String, BackendObservedLoad>>,
    eviction_handle: Option<thread::JoinHandle<()>>,
}

impl SicoStickyPolicy {
    pub fn new() -> Self {
        Self::with_cache_config(CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 120,
            max_tree_size: 1 << 26,
        })
    }

    pub fn with_cache_config(cache_config: CacheAwareConfig) -> Self {
        let trees = Arc::new(DashMap::<String, Arc<Tree>>::new());
        let eviction_handle = if cache_config.eviction_interval_secs > 0 {
            let trees_clone = Arc::clone(&trees);
            let max_tree_size = cache_config.max_tree_size;
            let interval = cache_config.eviction_interval_secs;

            Some(thread::spawn(move || loop {
                thread::sleep(Duration::from_secs(interval));
                for entry in trees_clone.iter() {
                    entry.value().evict_tenant_by_size(max_tree_size);
                }
            }))
        } else {
            None
        };

        Self {
            state: Mutex::new(StickyState::default()),
            cache_config,
            trees,
            cached_loads: RwLock::new(HashMap::new()),
            cached_load_generation: AtomicI64::new(0),
            observed: RwLock::new(HashMap::new()),
            eviction_handle,
        }
    }

    fn score(waiting: i64, running: i64) -> i64 {
        waiting.saturating_mul(STICKY_WAIT_WEIGHT) + running
    }

    fn get_or_init_tree(
        &self,
        model_id: &str,
        candidates: &[(usize, Arc<dyn Worker>)],
    ) -> Arc<Tree> {
        let tree = self
            .trees
            .entry(model_id.to_string())
            .or_insert_with(|| Arc::new(Tree::new()))
            .clone();

        for (_, worker) in candidates {
            tree.insert("", worker.url());
        }

        tree
    }

    fn local_load_bounds(workers: &[Arc<dyn Worker>]) -> (usize, usize) {
        let (min_load, max_load) =
            workers
                .iter()
                .fold((usize::MAX, 0usize), |(min, max), worker| {
                    let load = worker.load();
                    (min.min(load), max.max(load))
                });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };
        (min_load, max_load)
    }

    fn is_router_load_imbalanced(&self, min_load: usize, max_load: usize) -> bool {
        max_load.saturating_sub(min_load) > self.cache_config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.cache_config.balance_rel_threshold)
    }

    fn pick_min_router_load(candidates: &[(usize, Arc<dyn Worker>)]) -> Option<usize> {
        candidates
            .iter()
            .enumerate()
            .min_by_key(|(_, (_, worker))| worker.load())
            .map(|(idx, _)| idx)
    }

    fn pick_by_cache_aware(
        &self,
        candidates: &[(usize, Arc<dyn Worker>)],
        request_text: Option<&str>,
    ) -> Option<usize> {
        let model_id = normalize_model_key(candidates[0].1.model_id());
        let tree = self.get_or_init_tree(model_id, candidates);
        let text = request_text.unwrap_or("");
        let result = tree.prefix_match_with_counts(text);
        let match_rate = if result.input_char_count == 0 {
            0.0
        } else {
            result.matched_char_count as f32 / result.input_char_count as f32
        };

        let selected_idx = if match_rate > self.cache_config.cache_threshold {
            let tenant_url: &str = &result.tenant;
            candidates
                .iter()
                .position(|(_, worker)| worker.url() == tenant_url && worker.is_healthy())
        } else {
            Self::pick_min_router_load(candidates)
        };

        if let Some(idx) = selected_idx {
            tree.insert(text, candidates[idx].1.url());
            return Some(idx);
        }

        if match_rate > self.cache_config.cache_threshold {
            tree.remove_tenant(&result.tenant);
        }

        Some(0)
    }

    fn bump_local_waiting(state: &mut StickyState, engine_id: &str) {
        let entry = state
            .local_counts
            .entry(engine_id.to_string())
            .or_insert((0, 0));
        entry.0 += STICKY_CLIENT_COUNT;
    }

    pub fn remove_worker_by_url(&self, url: &str) {
        for tree_ref in self.trees.iter() {
            tree_ref.value().remove_tenant(url);
        }
    }

    fn extract_sticky_key(
        &self,
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<String> {
        if let Some(hdrs) = headers {
            for header_name in ConsistentHeaderKeys::HEADER_NAMES {
                if let Some(value) = hdrs.get(*header_name) {
                    if !value.is_empty() {
                        return Some(format!("header:{}:{}", header_name, value));
                    }
                }
            }
        }

        let text = request_text.unwrap_or("");
        if text.is_empty() {
            return None;
        }

        if let Some(session_id) =
            ConsistentHeaderKeys::extract_nested_field_value(text, "session_params", "session_id")
        {
            return Some(format!("session:{}", session_id));
        }
        if let Some(session_id) = ConsistentHeaderKeys::extract_field_value(text, "session_id") {
            return Some(format!("session:{}", session_id));
        }

        None
    }

    fn current_waiting(&self, worker: &dyn Worker) -> i64 {
        if let Ok(observed) = self.observed.read() {
            if let Some(state) = observed.get(worker.url()) {
                return state.waiting;
            }
        }
        if let Ok(loads) = self.cached_loads.read() {
            if let Some(load) = loads.get(worker.url()) {
                return *load as i64;
            }
        }
        worker.load() as i64
    }

    fn current_running(&self, worker: &dyn Worker) -> i64 {
        if let Ok(observed) = self.observed.read() {
            if let Some(state) = observed.get(worker.url()) {
                return state.running;
            }
        }
        0
    }

    fn current_scrape_token(&self, worker: &dyn Worker) -> i64 {
        if let Ok(observed) = self.observed.read() {
            if let Some(state) = observed.get(worker.url()) {
                return state.scrape_token;
            }
        }
        if let Ok(loads) = self.cached_loads.read() {
            if loads.contains_key(worker.url()) {
                return self.cached_load_generation.load(Ordering::Relaxed);
            }
        }
        0
    }

    fn argmin_rr(scores: &[i64], start_idx: usize) -> usize {
        let mut best_idx = 0usize;
        let mut best_val = i64::MAX;

        for offset in 0..scores.len() {
            let idx = (start_idx + offset) % scores.len();
            let val = scores[idx];
            if val < best_val {
                best_val = val;
                best_idx = idx;
            }
        }

        best_idx
    }
}

impl LoadBalancingPolicy for SicoStickyPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);
        if healthy_indices.is_empty() {
            return None;
        }

        let mut candidates: Vec<(usize, Arc<dyn Worker>)> = healthy_indices
            .into_iter()
            .map(|idx| (idx, workers[idx].clone()))
            .collect();
        candidates.sort_by(|(_, lhs), (_, rhs)| lhs.url().cmp(rhs.url()));

        let engine_ids: Vec<String> = candidates
            .iter()
            .map(|(_, worker)| worker.url().to_string())
            .collect();
        let observed_waiting: Vec<i64> = candidates
            .iter()
            .map(|(_, worker)| self.current_waiting(worker.as_ref()))
            .collect();
        let observed_running: Vec<i64> = candidates
            .iter()
            .map(|(_, worker)| self.current_running(worker.as_ref()))
            .collect();
        let scrape_tokens: Vec<i64> = candidates
            .iter()
            .map(|(_, worker)| self.current_scrape_token(worker.as_ref()))
            .collect();
        let session_id = self.extract_sticky_key(request_text, headers);

        let mut state = self.state.lock().unwrap();

        let active_engines: std::collections::HashSet<&str> =
            engine_ids.iter().map(|s| s.as_str()).collect();
        state
            .local_counts
            .retain(|engine, _| active_engines.contains(engine.as_str()));
        state
            .last_tokens
            .retain(|engine, _| active_engines.contains(engine.as_str()));
        state
            .session_map
            .retain(|_, engine| active_engines.contains(engine.as_str()));

        for ((engine_id, waiting), (running, token)) in engine_ids
            .iter()
            .zip(observed_waiting.iter())
            .zip(observed_running.iter().zip(scrape_tokens.iter()))
        {
            let token = *token;
            match state.last_tokens.get(engine_id) {
                Some(last_token) if *last_token == token => {
                    let entry = state
                        .local_counts
                        .entry(engine_id.clone())
                        .or_insert((*waiting, *running));
                    entry.0 = entry.0.max(*waiting);
                    entry.1 = *running;
                }
                _ => {
                    state
                        .local_counts
                        .insert(engine_id.clone(), (*waiting, *running));
                }
            }
            state.last_tokens.insert(engine_id.clone(), token);
        }

        if !engine_ids.is_empty() {
            state.start_idx %= engine_ids.len();
        } else {
            state.start_idx = 0;
        }

        let pick_by_observed = |state: &StickyState| -> usize {
            let scores: Vec<i64> = observed_waiting
                .iter()
                .zip(observed_running.iter())
                .map(|(waiting, running)| Self::score(*waiting, *running))
                .collect();
            Self::argmin_rr(&scores, state.start_idx)
        };

        let (min_load, max_load) = Self::local_load_bounds(workers);
        let (chosen_sorted_idx, migrated) = if self.is_router_load_imbalanced(min_load, max_load) {
            let chosen_idx = Self::pick_min_router_load(&candidates)?;
            if let Some(text) = request_text {
                let model_id = normalize_model_key(candidates[chosen_idx].1.model_id());
                let tree = self.get_or_init_tree(model_id, &candidates);
                tree.insert(text, candidates[chosen_idx].1.url());
            }
            Self::bump_local_waiting(&mut state, &engine_ids[chosen_idx]);
            (chosen_idx, false)
        } else {
            match session_id {
                None => {
                    let chosen_idx = self.pick_by_cache_aware(&candidates, request_text)?;
                    Self::bump_local_waiting(&mut state, &engine_ids[chosen_idx]);
                    (chosen_idx, false)
                }
                Some(session_id) => {
                    if let Some(mapped_engine) = state.session_map.get(&session_id).cloned() {
                        if let Some(mapped_idx) = engine_ids
                            .iter()
                            .position(|engine| engine == &mapped_engine)
                        {
                            let best_idx = pick_by_observed(&state);
                            let mut chosen_idx = mapped_idx;
                            let mut migrated = false;

                            if best_idx != mapped_idx {
                                let mapped_score = Self::score(
                                    observed_waiting[mapped_idx],
                                    observed_running[mapped_idx],
                                );
                                let best_score = Self::score(
                                    observed_waiting[best_idx],
                                    observed_running[best_idx],
                                );
                                let migration_allowed = state
                                    .last_migration_at
                                    .map(|ts| ts.elapsed() >= STICKY_MIGRATE_COOLDOWN)
                                    .unwrap_or(true);

                                if mapped_score - best_score >= STICKY_SLACK && migration_allowed {
                                    chosen_idx = best_idx;
                                    state
                                        .session_map
                                        .insert(session_id.clone(), engine_ids[best_idx].clone());
                                    state.last_migration_at = Some(Instant::now());
                                    migrated = true;
                                }
                            }

                            let chosen_engine = &engine_ids[chosen_idx];
                            Self::bump_local_waiting(&mut state, chosen_engine);

                            (chosen_idx, migrated)
                        } else {
                            let chosen_idx = self.pick_by_cache_aware(&candidates, request_text)?;
                            Self::bump_local_waiting(&mut state, &engine_ids[chosen_idx]);
                            state
                                .session_map
                                .insert(session_id, engine_ids[chosen_idx].clone());
                            (chosen_idx, false)
                        }
                    } else {
                        let chosen_idx = self.pick_by_cache_aware(&candidates, request_text)?;
                        Self::bump_local_waiting(&mut state, &engine_ids[chosen_idx]);
                        state
                            .session_map
                            .insert(session_id, engine_ids[chosen_idx].clone());
                        (chosen_idx, false)
                    }
                }
            }
        };

        let (original_idx, worker) = &candidates[chosen_sorted_idx];
        worker.increment_processed();
        RouterMetrics::record_processed_request(worker.url());
        RouterMetrics::record_policy_decision(self.name(), worker.url());
        if migrated {
            tracing::info!("Sico sticky session migrated to {}", worker.url());
        }

        Some(*original_idx)
    }

    fn name(&self) -> &'static str {
        "sico_sticky"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads.clone();
            self.cached_load_generation.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn update_backend_observations(&self, observations: &HashMap<String, BackendObservedLoad>) {
        if let Ok(mut observed) = self.observed.write() {
            *observed = observations.clone();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn requires_initialization(&self) -> bool {
        true
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
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

impl Drop for SicoStickyPolicy {
    fn drop(&mut self) {
        let _ = self.eviction_handle.take();
    }
}

#[derive(Debug)]
struct ConsistentHeaderKeys;

impl ConsistentHeaderKeys {
    const HEADER_NAMES: &'static [&'static str] = &[
        "x-session-id",
        "x-user-id",
        "x-tenant-id",
        "x-correlation-id",
        "x-request-id",
        "x-trace-id",
    ];

    fn extract_nested_field_value(
        text: &str,
        parent_field: &str,
        child_field: &str,
    ) -> Option<String> {
        if let Some(parent_start) = Self::find_field_start(text, parent_field) {
            if let Some(obj_start) = text[parent_start..].find('{') {
                let obj_start_pos = parent_start + obj_start;
                if let Some(obj_content) = Self::extract_json_object(&text[obj_start_pos..]) {
                    return Self::extract_field_value(&obj_content, child_field);
                }
            }
        }
        None
    }

    fn find_field_start(text: &str, field_name: &str) -> Option<usize> {
        let patterns = [
            format!("\"{}\"", field_name),
            format!("'{}'", field_name),
            field_name.to_string(),
        ];
        patterns.into_iter().find_map(|pattern| text.find(&pattern))
    }

    fn extract_json_object(text: &str) -> Option<String> {
        let mut brace_count = 0i32;
        let mut in_string = false;
        let mut string_char = '\0';

        for (i, ch) in text.char_indices() {
            match ch {
                '"' | '\'' if !in_string => {
                    in_string = true;
                    string_char = ch;
                }
                ch if ch == string_char && in_string => {
                    in_string = false;
                }
                '{' if !in_string => brace_count += 1,
                '}' if !in_string => {
                    brace_count -= 1;
                    if brace_count == 0 {
                        return Some(text[..=i].to_string());
                    }
                }
                _ => {}
            }
        }

        None
    }

    fn extract_field_value(text: &str, field_name: &str) -> Option<String> {
        let field_patterns = [
            format!("\"{}\"", field_name),
            format!("'{}'", field_name),
            field_name.to_string(),
        ];

        for pattern in field_patterns {
            if let Some(field_pos) = text.find(&pattern) {
                let after_field = &text[field_pos + pattern.len()..];
                if let Some(colon_pos) = after_field.find(':') {
                    let value_part = after_field[colon_pos + 1..].trim_start();

                    if let Some(stripped) = value_part.strip_prefix('"') {
                        if let Some(end_quote) = stripped.find('"') {
                            return Self::non_empty_value(&stripped[..end_quote]);
                        }
                    } else if let Some(stripped) = value_part.strip_prefix('\'') {
                        if let Some(end_quote) = stripped.find('\'') {
                            return Self::non_empty_value(&stripped[..end_quote]);
                        }
                    } else {
                        let end_pos = value_part
                            .find(&[',', ' ', '}', ']', '\n', '\r', '\t'][..])
                            .unwrap_or(value_part.len());
                        if end_pos > 0 {
                            return Self::non_empty_value(&value_part[..end_pos]);
                        }
                    }
                }
            }
        }

        None
    }

    fn non_empty_value(value: &str) -> Option<String> {
        if value.trim().is_empty() {
            None
        } else {
            Some(value.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    fn policy() -> SicoStickyPolicy {
        SicoStickyPolicy::with_cache_config(CacheAwareConfig {
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 0,
            max_tree_size: 1 << 26,
        })
    }

    fn workers() -> Vec<Arc<dyn Worker>> {
        vec![
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w3:8000".to_string(),
                WorkerType::Regular,
            )),
        ]
    }

    fn add_load(worker: &Arc<dyn Worker>, count: usize) {
        for _ in 0..count {
            worker.increment_load();
        }
    }

    fn remove_load(worker: &Arc<dyn Worker>, count: usize) {
        for _ in 0..count {
            worker.decrement_load();
        }
    }

    fn headers(name: &str, value: &str) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(name.to_string(), value.to_string());
        headers
    }

    #[test]
    fn sticky_session_keeps_same_worker() {
        let policy = policy();
        let workers = workers();
        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "session-1".to_string());

        let first = policy
            .select_worker_with_headers(&workers, Some("{\"prompt\":\"a\"}"), Some(&headers))
            .unwrap();

        for _ in 0..10 {
            let next = policy
                .select_worker_with_headers(&workers, Some("{\"prompt\":\"b\"}"), Some(&headers))
                .unwrap();
            assert_eq!(next, first);
        }
    }

    #[test]
    fn requests_without_session_use_cache_aware_fallback() {
        let policy = policy();
        let workers = workers();

        let first = policy
            .select_worker_with_headers(&workers, Some("{}"), None)
            .unwrap();
        let second = policy
            .select_worker_with_headers(&workers, Some("{}"), None)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn openai_user_field_is_not_sticky_session() {
        let policy = policy();
        let workers = workers();
        let body = r#"{"model":"test","messages":[],"user":"test"}"#;

        assert_eq!(policy.extract_sticky_key(Some(body), None), None);

        let first = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        let second = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn empty_string_fields_are_not_sticky_sessions() {
        let policy = policy();
        let workers = workers();

        for body in [
            r#"{"user":""}"#,
            r#"{"user_id":""}"#,
            r#"{"session_id":""}"#,
            r#"{"session_params":{"session_id":""}}"#,
        ] {
            assert_eq!(policy.extract_sticky_key(Some(body), None), None);
        }

        let first = policy
            .select_worker_with_headers(&workers, Some(r#"{"user":""}"#), None)
            .unwrap();
        let second = policy
            .select_worker_with_headers(&workers, Some(r#"{"user":""}"#), None)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn explicit_body_session_id_remains_sticky() {
        let policy = policy();
        let workers = workers();
        let body = r#"{"session_id":"session-1","user":"test"}"#;

        assert_eq!(
            policy.extract_sticky_key(Some(body), None),
            Some("session:session-1".to_string())
        );

        let first = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        let second = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn unmatched_session_falls_back_to_cache_aware_and_then_sticks() {
        let policy = policy();
        let workers = workers();
        let text = r#"{"prompt":"shared-prefix-for-cache"}"#;

        // Make the cache owner non-trivial: with w1 and w3 busier, the seed
        // request chooses w2. Then make w2 busier than w1/w3 so a later choice
        // can only pick w2 because of the prefix cache hit, not min-load.
        add_load(&workers[1], 10);
        add_load(&workers[2], 10);
        let cached_worker = policy
            .select_worker_with_headers(&workers, Some(text), None)
            .unwrap();
        assert_eq!(workers[cached_worker].url(), "http://w2:8000");

        add_load(&workers[cached_worker], 20);

        let body = r#"{"prompt":"shared-prefix-for-cache","session_id":"new-session"}"#;
        let first_session_worker = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        assert_eq!(first_session_worker, cached_worker);

        let second_session_worker = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"session_id":"new-session","prompt":"different"}"#),
                None,
            )
            .unwrap();
        assert_eq!(second_session_worker, first_session_worker);
    }

    #[test]
    fn local_load_imbalance_bypasses_existing_sticky_session() {
        let policy = policy();
        let workers = workers();
        let body = r#"{"session_id":"session-1","prompt":"a"}"#;

        let sticky_worker = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        for _ in 0..65 {
            workers[sticky_worker].increment_load();
        }

        let chosen = policy
            .select_worker_with_headers(&workers, Some(body), None)
            .unwrap();
        assert_ne!(chosen, sticky_worker);
        assert_eq!(workers[chosen].load(), 0);
    }

    #[test]
    fn different_header_sessions_with_same_prompt_share_cache_owner() {
        let policy = policy();
        let workers = workers();
        let text = r#"{"prompt":"same-prefix-from-two-header-sessions"}"#;

        add_load(&workers[1], 10);
        add_load(&workers[2], 10);
        let session_a = headers("x-session-id", "session-a");
        let first = policy
            .select_worker_with_headers(&workers, Some(text), Some(&session_a))
            .unwrap();
        assert_eq!(workers[first].url(), "http://w2:8000");

        add_load(&workers[first], 20);
        let session_b = headers("x-session-id", "session-b");
        let second = policy
            .select_worker_with_headers(&workers, Some(text), Some(&session_b))
            .unwrap();
        assert_eq!(second, first);

        let second_sticky = policy
            .select_worker_with_headers(
                &workers,
                Some(r#"{"prompt":"different-text-after-session-b-is-bound"}"#),
                Some(&session_b),
            )
            .unwrap();
        assert_eq!(second_sticky, second);
    }

    #[test]
    fn different_body_sessions_share_cache_when_common_prompt_prefix_is_first() {
        let policy = policy();
        let workers = workers();
        let body_a = r#"{"prompt":"very-long-common-prefix-for-body-session-cache-aware-routing","session_id":"body-a"}"#;
        let body_b = r#"{"prompt":"very-long-common-prefix-for-body-session-cache-aware-routing","session_id":"body-b"}"#;

        add_load(&workers[1], 10);
        add_load(&workers[2], 10);
        let first = policy
            .select_worker_with_headers(&workers, Some(body_a), None)
            .unwrap();
        assert_eq!(workers[first].url(), "http://w2:8000");

        add_load(&workers[first], 20);
        let second = policy
            .select_worker_with_headers(&workers, Some(body_b), None)
            .unwrap();
        assert_eq!(second, first);
    }

    #[test]
    fn load_imbalance_takes_precedence_over_cache_hit_for_unkeyed_request() {
        let policy = policy();
        let workers = workers();
        let text = r#"{"prompt":"cache-hit-but-overloaded"}"#;

        let cached_worker = policy
            .select_worker_with_headers(&workers, Some(text), None)
            .unwrap();
        add_load(&workers[cached_worker], 65);

        let chosen = policy
            .select_worker_with_headers(&workers, Some(text), None)
            .unwrap();
        assert_ne!(chosen, cached_worker);
        assert_eq!(workers[chosen].load(), 0);
    }

    #[test]
    fn sticky_mapping_survives_temporary_router_load_imbalance() {
        let policy = policy();
        let workers = workers();
        let headers = headers("x-session-id", "stable-session");
        let body = r#"{"prompt":"stable"}"#;

        let sticky_worker = policy
            .select_worker_with_headers(&workers, Some(body), Some(&headers))
            .unwrap();
        add_load(&workers[sticky_worker], 65);

        let overloaded_choice = policy
            .select_worker_with_headers(&workers, Some(body), Some(&headers))
            .unwrap();
        assert_ne!(overloaded_choice, sticky_worker);

        remove_load(&workers[sticky_worker], 65);
        let restored_choice = policy
            .select_worker_with_headers(&workers, Some(body), Some(&headers))
            .unwrap();
        assert_eq!(restored_choice, sticky_worker);
    }

    #[test]
    fn backend_observed_load_can_still_migrate_existing_sticky_session_when_balanced() {
        let policy = policy();
        let workers = workers();
        let headers = headers("x-session-id", "migrating-session");
        let body = r#"{"prompt":"migrate"}"#;

        let sticky_worker = policy
            .select_worker_with_headers(&workers, Some(body), Some(&headers))
            .unwrap();
        let sticky_url = workers[sticky_worker].url().to_string();

        let mut observations = HashMap::new();
        for worker in &workers {
            let waiting = if worker.url() == sticky_url { 100 } else { 0 };
            observations.insert(
                worker.url().to_string(),
                BackendObservedLoad {
                    waiting,
                    running: 0,
                    scrape_token: 1,
                },
            );
        }
        policy.update_backend_observations(&observations);

        let migrated = policy
            .select_worker_with_headers(&workers, Some(body), Some(&headers))
            .unwrap();
        assert_ne!(migrated, sticky_worker);
        assert_eq!(workers[migrated].load(), 0);
    }
}
