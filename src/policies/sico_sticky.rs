//! Sico-style sticky session load balancing policy.
//!
//! This policy ports the core behavior of Sico's `StickySessionLB` into the
//! native router policy interface:
//! - first placement picks the least-loaded worker with round-robin tie break
//! - later requests for the same session stay sticky
//! - migration only happens when the mapped worker is sufficiently worse than
//!   the best worker and a global cooldown has elapsed
//! - a small local waiting bump reduces stampedes between load refreshes

use super::{
    get_healthy_worker_indices, BackendObservedLoad, LoadBalancingPolicy, RequestHeaders,
};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
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

#[derive(Debug, Default)]
pub struct SicoStickyPolicy {
    state: Mutex<StickyState>,
    cached_loads: RwLock<HashMap<String, isize>>,
    observed: RwLock<HashMap<String, BackendObservedLoad>>,
}

impl SicoStickyPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    fn score(waiting: i64, running: i64) -> i64 {
        waiting.saturating_mul(STICKY_WAIT_WEIGHT) + running
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
        if let Some(user) = ConsistentHeaderKeys::extract_field_value(text, "user") {
            return Some(format!("user:{}", user));
        }
        if let Some(session_id) = ConsistentHeaderKeys::extract_field_value(text, "session_id") {
            return Some(format!("session:{}", session_id));
        }
        if let Some(user_id) = ConsistentHeaderKeys::extract_field_value(text, "user_id") {
            return Some(format!("user:{}", user_id));
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
        self.current_waiting(worker)
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

        let pick_by_local = |state: &mut StickyState| -> usize {
            let scores: Vec<i64> = engine_ids
                .iter()
                .map(|engine_id| {
                    let (waiting, running) =
                        state.local_counts.get(engine_id).copied().unwrap_or((0, 0));
                    Self::score(waiting, running)
                })
                .collect();

            let chosen = Self::argmin_rr(&scores, state.start_idx);
            state.start_idx = (chosen + 1) % engine_ids.len();

            let chosen_engine = &engine_ids[chosen];
            let entry = state
                .local_counts
                .entry(chosen_engine.clone())
                .or_insert((0, 0));
            entry.0 += STICKY_CLIENT_COUNT;

            chosen
        };

        let pick_by_observed = |state: &StickyState| -> usize {
            let scores: Vec<i64> = observed_waiting
                .iter()
                .zip(observed_running.iter())
                .map(|(waiting, running)| Self::score(*waiting, *running))
                .collect();
            Self::argmin_rr(&scores, state.start_idx)
        };

        let (chosen_sorted_idx, migrated) = match session_id {
            None => (pick_by_local(&mut state), false),
            Some(session_id) => {
                if let Some(mapped_engine) = state.session_map.get(&session_id).cloned() {
                    if let Some(mapped_idx) = engine_ids.iter().position(|engine| engine == &mapped_engine)
                    {
                        let best_idx = pick_by_observed(&state);
                        let mut chosen_idx = mapped_idx;
                        let mut migrated = false;

                        if best_idx != mapped_idx {
                            let mapped_score =
                                Self::score(observed_waiting[mapped_idx], observed_running[mapped_idx]);
                            let best_score =
                                Self::score(observed_waiting[best_idx], observed_running[best_idx]);
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
                        let entry = state
                            .local_counts
                            .entry(chosen_engine.clone())
                            .or_insert((0, 0));
                        entry.0 += STICKY_CLIENT_COUNT;

                        (chosen_idx, migrated)
                    } else {
                        let chosen_idx = pick_by_local(&mut state);
                        state
                            .session_map
                            .insert(session_id, engine_ids[chosen_idx].clone());
                        (chosen_idx, false)
                    }
                } else {
                    let chosen_idx = pick_by_local(&mut state);
                    state
                        .session_map
                        .insert(session_id, engine_ids[chosen_idx].clone());
                    (chosen_idx, false)
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
                            return Some(stripped[..end_quote].to_string());
                        }
                    } else if let Some(stripped) = value_part.strip_prefix('\'') {
                        if let Some(end_quote) = stripped.find('\'') {
                            return Some(stripped[..end_quote].to_string());
                        }
                    } else {
                        let end_pos = value_part
                            .find(&[',', ' ', '}', ']', '\n', '\r', '\t'][..])
                            .unwrap_or(value_part.len());
                        if end_pos > 0 {
                            return Some(value_part[..end_pos].to_string());
                        }
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

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

    #[test]
    fn sticky_session_keeps_same_worker() {
        let policy = SicoStickyPolicy::new();
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
    fn requests_without_session_rotate_by_load() {
        let policy = SicoStickyPolicy::new();
        let workers = workers();

        let first = policy.select_worker_with_headers(&workers, Some("{}"), None).unwrap();
        let second = policy.select_worker_with_headers(&workers, Some("{}"), None).unwrap();
        assert_ne!(first, second);
    }
}
