use rhino_backend::{AsrBackend, PcmBuffer, WordToken};
use rhino_engine::{AgreementConfig, AgreementEngine, EngineEvent, WordHypothesis};
use rhino_protocol::AsrEvent;

const SAMPLE_RATE: usize = 16_000;

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Minimum new audio (seconds) between transcriptions.
    pub step_secs: f32,
    /// Maximum audio buffer before forced trim (seconds).
    pub max_buffer_secs: f32,
    /// Minimum audio before first transcription (seconds).
    pub min_chunk_secs: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            step_secs: 0.5,
            max_buffer_secs: 30.0,
            min_chunk_secs: 1.0,
        }
    }
}

/// Sync ASR pipeline wiring backend + agreement engine.
///
/// Manages a growing audio buffer, runs transcription at intervals,
/// and produces self-correcting ASR events via LocalAgreement-2.
///
/// VAD is **not** inside the pipeline — the session layer (Phase 4)
/// gates which audio reaches here.
pub struct AsrPipeline<B: AsrBackend> {
    backend: B,
    engine: AgreementEngine,
    config: PipelineConfig,
    audio_buf: PcmBuffer,
    /// Total samples trimmed from the start of the stream.
    trim_offset: usize,
    /// Buffer length at last transcription (to enforce step_secs).
    samples_at_last_transcribe: usize,
    /// Number of committed words already accounted for by buffer trimming.
    /// After trimming, the token window no longer contains these words, so
    /// `find_last_committed_end` must skip them.
    trimmed_committed_count: usize,
}

impl<B: AsrBackend> AsrPipeline<B> {
    pub fn new(backend: B, engine_config: AgreementConfig, config: PipelineConfig) -> Self {
        Self {
            backend,
            engine: AgreementEngine::new(engine_config),
            config,
            audio_buf: PcmBuffer::new(),
            trim_offset: 0,
            samples_at_last_transcribe: 0,
            trimmed_committed_count: 0,
        }
    }

    /// Append samples. Runs transcription if enough new audio since last step.
    pub fn push_audio(&mut self, samples: &[f32]) -> anyhow::Result<Vec<AsrEvent>> {
        self.audio_buf.0.extend_from_slice(samples);

        let min_samples = (self.config.min_chunk_secs * SAMPLE_RATE as f32) as usize;
        if self.audio_buf.len() < min_samples {
            return Ok(vec![]);
        }

        let step_samples = (self.config.step_secs * SAMPLE_RATE as f32) as usize;
        let new_since_last = self.audio_buf.len() - self.samples_at_last_transcribe;
        if new_since_last < step_samples {
            return Ok(vec![]);
        }

        // Cap buffer to prevent unbounded growth.
        let max_samples = (self.config.max_buffer_secs * SAMPLE_RATE as f32) as usize;
        if self.audio_buf.len() > max_samples {
            self.force_trim();
        }

        self.step()
    }

    /// Commit all pending words and reset. Call at utterance boundary (VAD SpeechEnd).
    ///
    /// On backend error during the final transcription step: logs a warning,
    /// still flushes the engine (preserving prior accumulated commits),
    /// emits `EndOfUtterance`, and clears state. Returns `Ok` with whatever
    /// events could be salvaged.
    pub fn flush_utterance(&mut self) -> anyhow::Result<Vec<AsrEvent>> {
        // Determine if an utterance was active: either audio remains in the buffer,
        // or audio was previously consumed and trimmed (trim_offset > 0).
        let had_utterance = !self.audio_buf.is_empty() || self.trim_offset > 0;
        if !had_utterance {
            return Ok(vec![]);
        }

        let mut events = Vec::new();

        // Final transcription of remaining audio (skip if buffer was fully trimmed).
        if !self.audio_buf.is_empty() {
            match self.step() {
                Ok(evs) => events.extend(evs),
                Err(e) => {
                    tracing::warn!("final transcription failed during flush: {e:#}");
                }
            }
        }

        // Flush engine — commits everything regardless of agreement.
        let flush_events = self.engine.flush();
        events.extend(flush_events.into_iter().map(engine_to_asr));
        events.push(AsrEvent::EndOfUtterance);

        // Clear state for next utterance.
        self.audio_buf.0.clear();
        self.trim_offset = 0;
        self.samples_at_last_transcribe = 0;
        self.trimmed_committed_count = 0;
        self.engine.reset();

        Ok(events)
    }

    /// Full reset for new session.
    pub fn reset(&mut self) {
        self.audio_buf.0.clear();
        self.trim_offset = 0;
        self.samples_at_last_transcribe = 0;
        self.trimmed_committed_count = 0;
        self.engine.reset();
        self.backend.reset();
    }

    /// Run one transcription step.
    fn step(&mut self) -> anyhow::Result<Vec<AsrEvent>> {
        let tokens = self.backend.transcribe(&self.audio_buf)?;
        self.samples_at_last_transcribe = self.audio_buf.len();

        let buf_duration = self.audio_buf.duration_secs();
        let hypotheses = tokens_to_hypotheses(&tokens);

        let engine_events = self.engine.push_hypothesis(hypotheses, buf_duration);
        let asr_events: Vec<AsrEvent> = engine_events.into_iter().map(engine_to_asr).collect();

        // Trim buffer up to last confirmed word's end timestamp.
        self.trim_to_confirmed(&tokens);

        Ok(asr_events)
    }

    /// Trim oldest audio up to the last confirmed word's end timestamp.
    ///
    /// Only matches committed words added since the last trim — earlier
    /// committed words are no longer in the token window after trimming.
    /// Clears engine hypothesis state after trimming because buffer-relative
    /// timestamps shift, invalidating the agreement counters.
    fn trim_to_confirmed(&mut self, tokens: &[WordToken]) {
        let committed = self.engine.committed_text();
        let new_committed = &committed[self.trimmed_committed_count..];
        if new_committed.is_empty() {
            return;
        }

        // Find the end timestamp of the last newly-committed word in the token list.
        if let Some(last_end) = find_last_committed_end(new_committed, tokens) {
            let trim_samples = (last_end * SAMPLE_RATE as f32) as usize;
            let trim_samples = trim_samples.min(self.audio_buf.len());
            if trim_samples > 0 {
                self.trim_offset += trim_samples;
                self.audio_buf.0.drain(..trim_samples);
                self.samples_at_last_transcribe =
                    self.samples_at_last_transcribe.saturating_sub(trim_samples);
                self.trimmed_committed_count = committed.len();
                // Timestamps shifted — agreement state is invalid.
                // Freeze committed words so they're not retracted by post-trim confirmations.
                self.engine.freeze_committed();
                self.engine.clear_hypothesis();
            }
        }
    }

    /// Emergency trim: drop oldest 25% of buffer when it exceeds max size.
    ///
    /// Safety valve for pathological cases where committed-word trimming
    /// cannot keep up (e.g., backend produces no committable words for 30+
    /// seconds of continuous speech). Resets the engine's positional tracking
    /// since timestamps are no longer aligned after the trim. Committed words
    /// are preserved — only agreement-in-progress is lost.
    fn force_trim(&mut self) {
        let trim = self.audio_buf.len() / 4;
        tracing::warn!(
            trimmed_samples = trim,
            buffer_samples = self.audio_buf.len(),
            "emergency buffer trim — agreement progress reset"
        );
        self.trim_offset += trim;
        self.audio_buf.0.drain(..trim);
        self.samples_at_last_transcribe = self.samples_at_last_transcribe.saturating_sub(trim);
        self.trimmed_committed_count = self.engine.committed_text().len();
        self.engine.freeze_committed();
        self.engine.clear_hypothesis();
    }
}

/// Find the end timestamp of the last committed word in the token stream.
///
/// Walks `committed` and `tokens` forward in lockstep: each committed word
/// is matched to the next token with the same text (case-insensitive).
/// Returns the `end` time of the last matched token, or `None` if no
/// committed word appears in the token list.
fn find_last_committed_end(committed: &[String], tokens: &[WordToken]) -> Option<f32> {
    let mut last_end = None;
    let mut token_idx = 0;

    for word in committed {
        while token_idx < tokens.len() {
            if tokens[token_idx].word.trim().eq_ignore_ascii_case(word) {
                last_end = Some(tokens[token_idx].end);
                token_idx += 1;
                break;
            }
            token_idx += 1;
        }
    }

    last_end
}

fn tokens_to_hypotheses(tokens: &[WordToken]) -> Vec<WordHypothesis> {
    tokens
        .iter()
        .map(|t| WordHypothesis {
            word: t.word.clone(),
            end_time: t.end,
        })
        .collect()
}

fn engine_to_asr(event: EngineEvent) -> AsrEvent {
    match event {
        EngineEvent::Commit(text) => AsrEvent::Commit { text },
        EngineEvent::Retract(n) => AsrEvent::Retract { count: n },
        EngineEvent::Interim(text) => AsrEvent::Interim { text },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhino_backend::mock::MockBackend;

    fn token(word: &str, start: f32, end: f32) -> WordToken {
        WordToken {
            word: word.to_string(),
            start,
            end,
        }
    }

    fn pipeline_with_mock(mock: MockBackend) -> AsrPipeline<MockBackend> {
        AsrPipeline::new(
            mock,
            AgreementConfig {
                min_agreement: 2,
                commit_lookahead_secs: 0.5,
            },
            PipelineConfig {
                step_secs: 0.0,       // transcribe on every push for testing
                max_buffer_secs: 30.0,
                min_chunk_secs: 0.0,   // no minimum for testing
            },
        )
    }

    /// Push 1 second of silence (zeros).
    fn push_one_sec(pipeline: &mut AsrPipeline<MockBackend>) -> anyhow::Result<Vec<AsrEvent>> {
        pipeline.push_audio(&[0.0; SAMPLE_RATE])
    }

    #[test]
    fn basic_transcription() {
        let mut mock = MockBackend::new();
        let words = vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)];
        mock.set_default_response(words);

        let mut pipeline = pipeline_with_mock(mock);

        // First push — agreement = 1 (first observation), only interim.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Interim { .. })));
        assert!(!events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));

        // Second push — agreement = 2 = k, commit (LA-2: two matching hypotheses).
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
    }

    #[test]
    fn step_interval_respected() {
        let mut mock = MockBackend::new();
        mock.queue_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = AsrPipeline::new(
            mock,
            AgreementConfig::default(),
            PipelineConfig {
                step_secs: 1.0,        // require 1s between transcriptions
                max_buffer_secs: 30.0,
                min_chunk_secs: 0.0,
            },
        );

        // Push 0.5s — not enough new audio since last transcribe.
        let events = pipeline.push_audio(&[0.0; 8000]).unwrap();
        assert!(events.is_empty());

        // Push another 0.5s — now have 1s total, triggers transcription.
        let events = pipeline.push_audio(&[0.0; 8000]).unwrap();
        assert!(!events.is_empty()); // Should have at least an Interim.
    }

    #[test]
    fn no_events_below_min_chunk() {
        let mut mock = MockBackend::new();
        mock.queue_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = AsrPipeline::new(
            mock,
            AgreementConfig::default(),
            PipelineConfig {
                step_secs: 0.0,
                max_buffer_secs: 30.0,
                min_chunk_secs: 1.0, // require 1s minimum
            },
        );

        // Push 0.5s — below minimum.
        let events = pipeline.push_audio(&[0.0; 8000]).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn buffer_trimming() {
        let mut mock = MockBackend::new();
        let words = vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)];
        mock.set_default_response(words);

        let mut pipeline = pipeline_with_mock(mock);

        // Two pushes → commit (LA-2). Buffer should be trimmed after commit.
        for _ in 0..2 {
            push_one_sec(&mut pipeline).unwrap();
        }

        // Buffer should have been trimmed (committed words' audio removed).
        assert!(pipeline.audio_buf.len() < 2 * SAMPLE_RATE);
    }

    #[test]
    fn flush_commits_pending() {
        let mut mock = MockBackend::new();
        // Only one response — not enough for agreement.
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = pipeline_with_mock(mock);

        // One push — only interim, no commit.
        push_one_sec(&mut pipeline).unwrap();

        // Flush forces commit of everything.
        let events = pipeline.flush_utterance().unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
    }

    #[test]
    fn reset_clears_state() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);

        push_one_sec(&mut pipeline).unwrap();
        assert!(!pipeline.audio_buf.is_empty());

        pipeline.reset();
        assert!(pipeline.audio_buf.is_empty());
        assert_eq!(pipeline.trim_offset, 0);
        assert_eq!(pipeline.samples_at_last_transcribe, 0);
    }

    #[test]
    fn flush_on_empty_buffer() {
        let mock = MockBackend::new();
        let mut pipeline = pipeline_with_mock(mock);

        let events = pipeline.flush_utterance().unwrap();
        assert!(events.is_empty());
    }

    // --- New tests: find_last_committed_end ---

    #[test]
    fn repeated_word_trimming() {
        // committed=["go"], tokens=["go","stop","go"] → should match first "go"
        let committed = vec!["go".to_string()];
        let tokens = vec![
            token("go", 0.0, 0.5),
            token("stop", 0.5, 0.8),
            token("go", 0.8, 1.0),
        ];
        let end = find_last_committed_end(&committed, &tokens);
        assert_eq!(end, Some(0.5)); // first "go", not the last
    }

    #[test]
    fn ordered_matching_with_duplicates() {
        let committed = vec!["hello".to_string(), "world".to_string()];
        let tokens = vec![
            token("hello", 0.0, 0.3),
            token("world", 0.3, 0.6),
            token("hello", 0.6, 0.9),
            token("world", 0.9, 1.2),
        ];
        let end = find_last_committed_end(&committed, &tokens);
        // Matches first "hello" (0.3) then first "world" (0.6)
        assert_eq!(end, Some(0.6));
    }

    #[test]
    fn committed_subset_at_start() {
        let committed = vec!["a".to_string(), "b".to_string()];
        let tokens = vec![
            token("a", 0.0, 0.2),
            token("b", 0.2, 0.4),
            token("c", 0.4, 0.6),
        ];
        assert_eq!(find_last_committed_end(&committed, &tokens), Some(0.4));
    }

    #[test]
    fn no_match_returns_none() {
        let committed = vec!["xyz".to_string()];
        let tokens = vec![token("abc", 0.0, 0.5)];
        assert_eq!(find_last_committed_end(&committed, &tokens), None);
    }

    // --- New tests: EndOfUtterance emission ---

    #[test]
    fn flush_emits_end_of_utterance() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);
        push_one_sec(&mut pipeline).unwrap();

        let events = pipeline.flush_utterance().unwrap();
        assert_eq!(events.last(), Some(&AsrEvent::EndOfUtterance));
    }

    #[test]
    fn flush_empty_no_end_of_utterance() {
        let mock = MockBackend::new();
        let mut pipeline = pipeline_with_mock(mock);

        let events = pipeline.flush_utterance().unwrap();
        assert!(events.is_empty()); // no utterance = no boundary event
    }

    // --- New tests: flush error handling ---

    #[test]
    fn flush_with_backend_error_still_returns_events() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = pipeline_with_mock(mock);

        // Build up engine state with one successful push.
        push_one_sec(&mut pipeline).unwrap();

        // Make next transcribe fail (the one inside flush_utterance's step()).
        pipeline.backend.set_fail_count(1);

        let events = pipeline.flush_utterance().unwrap();
        // Engine flush should still produce events from the prior hypothesis.
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
        assert_eq!(events.last(), Some(&AsrEvent::EndOfUtterance));
    }

    #[test]
    fn flush_with_backend_error_clears_state() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);
        push_one_sec(&mut pipeline).unwrap();

        pipeline.backend.set_fail_count(1);
        pipeline.flush_utterance().unwrap();

        assert!(pipeline.audio_buf.is_empty());
        assert_eq!(pipeline.trim_offset, 0);
        assert_eq!(pipeline.samples_at_last_transcribe, 0);
    }

    // --- New tests: force_trim ---

    #[test]
    fn force_trim_resets_engine_tracking() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = AsrPipeline::new(
            mock,
            AgreementConfig {
                min_agreement: 2,
                commit_lookahead_secs: 0.5,
            },
            PipelineConfig {
                step_secs: 0.0,
                max_buffer_secs: 0.5, // very small — triggers force_trim on second push
                min_chunk_secs: 0.0,
            },
        );

        // First push: 1s of audio, exceeds 0.5s max → force_trim fires.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(!events.is_empty()); // should still get interim events

        // Pipeline should still be functional after force trim.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    fn force_trim_no_retract_on_next_update() {
        let mut mock = MockBackend::new();
        // Word with small end timestamp — trim_to_confirmed won't reclaim much buffer,
        // so the buffer grows past max_buffer_secs on the third push.
        mock.set_default_response(vec![token("hello", 0.0, 0.3)]);

        let mut pipeline = AsrPipeline::new(
            mock,
            AgreementConfig {
                min_agreement: 2,
                commit_lookahead_secs: 0.5,
            },
            PipelineConfig {
                step_secs: 0.0,
                max_buffer_secs: 2.0,
                min_chunk_secs: 0.0,
            },
        );

        // Push 1: agreement = 1, interim only.
        let events1 = push_one_sec(&mut pipeline).unwrap();
        assert!(!events1.iter().any(|e| matches!(e, AsrEvent::Retract { .. })));

        // Push 2: agreement = 2 → commit "hello". Buffer trimmed slightly (to 0.3s).
        let events2 = push_one_sec(&mut pipeline).unwrap();
        assert!(
            events2.iter().any(|e| matches!(e, AsrEvent::Commit { .. })),
            "should commit after two matching hypotheses"
        );

        // Push 3: buffer exceeds max → force_trim → clear_hypothesis.
        // First post-trim hypothesis: agreement = 1, empty confirmed target.
        // Must NOT retract the committed "hello".
        let events3 = push_one_sec(&mut pipeline).unwrap();
        assert!(
            !events3.iter().any(|e| matches!(e, AsrEvent::Retract { .. })),
            "force_trim must not cause retraction on next update"
        );
    }

    // --- Sliding-window trim correctness ---

    /// Exercises multiple rounds of normal trimming. After the first commit
    /// and trim, the token window no longer contains already-trimmed words.
    /// Subsequent commits must still trigger trimming to keep the buffer bounded.
    #[test]
    fn multi_round_trimming_stays_bounded() {
        // Backend returns incrementally longer sequences to simulate real
        // whisper output over a growing-then-trimmed buffer.
        let mut mock = MockBackend::new();

        // Rounds 1-2: "hello world" → commit + trim after round 2.
        mock.queue_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);
        mock.queue_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        // After trim, buffer restarts. Rounds 3-4: "foo bar" from the new origin.
        mock.queue_response(vec![token("foo", 0.0, 0.5), token("bar", 0.5, 1.0)]);
        mock.queue_response(vec![token("foo", 0.0, 0.5), token("bar", 0.5, 1.0)]);

        // After second trim. Rounds 5-6: "baz" from the new origin.
        mock.queue_response(vec![token("baz", 0.0, 0.5)]);
        mock.queue_response(vec![token("baz", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);

        // Round 1: agreement=1, interim only. No trim.
        push_one_sec(&mut pipeline).unwrap();
        let buf_after_1 = pipeline.audio_buf.len();
        assert_eq!(buf_after_1, SAMPLE_RATE); // 1s

        // Round 2: agreement=2, commit "hello world". Trim to 1.0s.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
        let buf_after_2 = pipeline.audio_buf.len();
        assert!(
            buf_after_2 < 2 * SAMPLE_RATE,
            "buffer should have been trimmed after first commit: {buf_after_2}"
        );
        assert_eq!(pipeline.trimmed_committed_count, 2); // "hello", "world"

        // Round 3: fresh agreement after trim, agreement=1 for "foo bar".
        push_one_sec(&mut pipeline).unwrap();

        // Round 4: agreement=2, commit "foo bar". Second trim.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(
            events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })),
            "should commit 'foo bar' in second trim round: {events:?}"
        );
        let buf_after_4 = pipeline.audio_buf.len();
        assert!(
            buf_after_4 < 3 * SAMPLE_RATE,
            "buffer should have been trimmed after second commit: {buf_after_4}"
        );
        assert_eq!(pipeline.trimmed_committed_count, 4); // +2 for "foo", "bar"

        // Round 5-6: third commit cycle.
        push_one_sec(&mut pipeline).unwrap();
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(
            events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })),
            "should commit 'baz' in third trim round: {events:?}"
        );
        assert_eq!(pipeline.trimmed_committed_count, 5);

        // Full committed history preserved across all trims.
        assert_eq!(
            pipeline.engine.committed_text(),
            &["hello", "world", "foo", "bar", "baz"]
        );
    }

    /// Verifies exact event sequence across a commit-trim-commit cycle.
    #[test]
    fn exact_event_sequence_across_trims() {
        let mut mock = MockBackend::new();
        mock.queue_response(vec![token("a", 0.0, 0.5)]);
        mock.queue_response(vec![token("a", 0.0, 0.5)]);
        // Post-trim: new tokens from buffer origin.
        mock.queue_response(vec![token("b", 0.0, 0.5)]);
        mock.queue_response(vec![token("b", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);

        let e1 = push_one_sec(&mut pipeline).unwrap();
        assert_eq!(
            e1.iter()
                .filter(|e| matches!(e, AsrEvent::Interim { .. }))
                .count(),
            1,
            "round 1: exactly one interim"
        );

        let e2 = push_one_sec(&mut pipeline).unwrap();
        assert!(
            e2.iter().any(|e| *e == AsrEvent::Commit { text: "a".into() }),
            "round 2: commit 'a': {e2:?}"
        );

        let e3 = push_one_sec(&mut pipeline).unwrap();
        assert!(
            e3.iter().any(|e| matches!(e, AsrEvent::Interim { .. })),
            "round 3: interim for 'b': {e3:?}"
        );
        assert!(
            !e3.iter().any(|e| matches!(e, AsrEvent::Retract { .. })),
            "round 3: no retract: {e3:?}"
        );

        let e4 = push_one_sec(&mut pipeline).unwrap();
        assert!(
            e4.iter().any(|e| *e == AsrEvent::Commit { text: "b".into() }),
            "round 4: commit 'b': {e4:?}"
        );
        assert!(
            !e4.iter().any(|e| matches!(e, AsrEvent::Retract { .. })),
            "round 4: no retract: {e4:?}"
        );
    }

    /// After flush, trimmed_committed_count resets so the next utterance starts clean.
    #[test]
    fn flush_resets_trimmed_committed_count() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);

        // Commit + trim.
        push_one_sec(&mut pipeline).unwrap();
        push_one_sec(&mut pipeline).unwrap();
        assert!(pipeline.trimmed_committed_count > 0);

        pipeline.flush_utterance().unwrap();
        assert_eq!(pipeline.trimmed_committed_count, 0);
        assert_eq!(pipeline.trim_offset, 0);
    }
}
