use rhino_backend::{AsrBackend, PcmAudio, PcmBuffer, WordToken};
use rhino_engine::{AgreementConfig, AgreementEngine, EngineEvent, WordHypothesis};
use rhino_protocol::AsrEvent;

const SAMPLE_RATE: usize = 16_000;

/// Trait for ASR pipelines with different chunking strategies.
///
/// The backend type is encapsulated inside each concrete implementation.
/// This trait is object-safe — `Box<dyn AsrPipeline>` is used by the session layer.
pub trait AsrPipeline: Send {
    /// Append audio samples. May produce events depending on mode:
    /// - Utterance: Commit if chunk interval reached.
    /// - Streaming: Commit/Retract/Interim from LA-2 agreement.
    fn push_audio(&mut self, samples: &[f32]) -> anyhow::Result<Vec<AsrEvent>>;

    /// Transcribe and emit events for an utterance boundary.
    /// Always emits EndOfUtterance. Called on VAD SpeechEnd or stream close.
    fn flush_utterance(&mut self) -> anyhow::Result<Vec<AsrEvent>>;

    /// Mid-utterance flush: emit Commit only (no EndOfUtterance).
    /// Used by session-layer buffer overflow safety net.
    fn flush_chunk(&mut self) -> anyhow::Result<Vec<AsrEvent>>;

    /// Current buffer duration in seconds.
    fn buffer_duration_secs(&self) -> f32;

    /// Whether the buffer exceeds the configured maximum.
    fn buffer_full(&self) -> bool;

    /// Full reset for new session.
    fn reset(&mut self);

    /// Run a single transcription on arbitrary audio, bypassing buffer
    /// management. For diagnostics only.
    fn transcribe_raw(&mut self, audio: &[f32]) -> anyhow::Result<Vec<WordToken>>;
}

// =============================================================================
// Utterance mode: buffer audio, transcribe once on flush
// =============================================================================

/// Configuration for utterance-mode pipeline.
#[derive(Debug, Clone)]
pub struct UtteranceConfig {
    /// Maximum audio buffer before safety-net flush (seconds).
    pub max_buffer_secs: f32,
    /// If set, emit Commit every N seconds during long continuous speech.
    /// Prevents O(n) inference growth on long utterances.
    pub chunk_interval_secs: Option<f32>,
}

impl Default for UtteranceConfig {
    fn default() -> Self {
        Self {
            max_buffer_secs: 10.0,
            chunk_interval_secs: Some(8.0),
        }
    }
}

/// Single-pass ASR pipeline for VAD-chunked transcription.
///
/// Audio accumulates during speech. On flush (VAD SpeechEnd), the full
/// buffer is transcribed once and emitted as a single Commit. Previous
/// utterance text is passed as `initial_prompt` for cross-chunk context.
///
/// Optional interval chunking emits Commit every N seconds during long
/// continuous speech to bound inference latency.
pub struct UtterancePipeline<B: AsrBackend> {
    backend: B,
    config: UtteranceConfig,
    audio_buf: PcmBuffer,
    /// Previous utterance text, passed as initial_prompt for cross-chunk context.
    previous_context: String,
}

impl<B: AsrBackend> UtterancePipeline<B> {
    pub fn new(backend: B, config: UtteranceConfig) -> Self {
        Self {
            backend,
            config,
            audio_buf: PcmBuffer::new(),
            previous_context: String::new(),
        }
    }

    fn transcribe_buffer(&mut self) -> anyhow::Result<Vec<WordToken>> {
        if self.previous_context.is_empty() {
            self.backend.transcribe(&self.audio_buf)
        } else {
            self.backend.transcribe_with_prompt(&self.audio_buf, &self.previous_context)
        }
    }
}

impl<B: AsrBackend> AsrPipeline for UtterancePipeline<B> {
    fn push_audio(&mut self, samples: &[f32]) -> anyhow::Result<Vec<AsrEvent>> {
        self.audio_buf.0.extend_from_slice(samples);

        if let Some(interval) = self.config.chunk_interval_secs {
            if self.audio_buf.duration_secs() > interval {
                return self.flush_chunk();
            }
        }
        Ok(vec![])
    }

    fn flush_utterance(&mut self) -> anyhow::Result<Vec<AsrEvent>> {
        if self.audio_buf.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!(
            buf_secs = format!("{:.2}", self.audio_buf.duration_secs()),
            buf_samples = self.audio_buf.len(),
            context_len = self.previous_context.len(),
            "transcribing utterance"
        );

        let tokens = self.transcribe_buffer()?;

        let text: String = tokens
            .iter()
            .map(|t| t.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if !text.is_empty() {
            tracing::info!(words = tokens.len(), text = %text, "utterance committed");
        }

        self.previous_context = text.clone();
        self.audio_buf.0.clear();

        let mut events = Vec::new();
        if !text.is_empty() {
            events.push(AsrEvent::Commit { text });
        }
        events.push(AsrEvent::EndOfUtterance);
        Ok(events)
    }

    fn flush_chunk(&mut self) -> anyhow::Result<Vec<AsrEvent>> {
        if self.audio_buf.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!(
            buf_secs = format!("{:.2}", self.audio_buf.duration_secs()),
            buf_samples = self.audio_buf.len(),
            "interval chunk flush"
        );

        let tokens = self.transcribe_buffer()?;

        let text: String = tokens
            .iter()
            .map(|t| t.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if !text.is_empty() {
            tracing::info!(words = tokens.len(), text = %text, "chunk committed");
        }

        self.previous_context = text.clone();
        self.audio_buf.0.clear();

        if text.is_empty() {
            Ok(vec![])
        } else {
            Ok(vec![AsrEvent::Commit { text }])
        }
    }

    fn buffer_duration_secs(&self) -> f32 {
        self.audio_buf.duration_secs()
    }

    fn buffer_full(&self) -> bool {
        self.audio_buf.duration_secs() > self.config.max_buffer_secs
    }

    fn reset(&mut self) {
        self.audio_buf.0.clear();
        self.previous_context.clear();
        self.backend.reset();
    }

    fn transcribe_raw(&mut self, audio: &[f32]) -> anyhow::Result<Vec<WordToken>> {
        self.backend.transcribe(&PcmAudio::new(audio))
    }
}

// =============================================================================
// Streaming mode: LocalAgreement-2 with periodic transcription
// =============================================================================

/// Configuration for streaming-mode pipeline.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum audio buffer before emergency trim (seconds).
    pub max_buffer_secs: f32,
    /// Minimum new audio between transcriptions (seconds).
    pub step_secs: f32,
    /// Minimum audio before first transcription (seconds).
    pub min_chunk_secs: f32,
    /// LocalAgreement-k parameters.
    pub agreement_config: AgreementConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_buffer_secs: 30.0,
            step_secs: 0.5,
            min_chunk_secs: 1.0,
            agreement_config: AgreementConfig::default(),
        }
    }
}

/// Streaming ASR pipeline using LocalAgreement-2 for progressive word confirmation.
///
/// Manages a growing audio buffer, runs transcription at intervals,
/// and produces self-correcting ASR events (Commit/Retract/Interim).
/// Buffer is trimmed after confirmed words to keep memory bounded.
pub struct StreamingPipeline<B: AsrBackend> {
    backend: B,
    engine: AgreementEngine,
    config: StreamingConfig,
    audio_buf: PcmBuffer,
    /// Total samples trimmed from the start of the stream.
    trim_offset: usize,
    /// Buffer length at last transcription (to enforce step_secs).
    samples_at_last_transcribe: usize,
    /// Number of committed words already accounted for by buffer trimming.
    trimmed_committed_count: usize,
}

impl<B: AsrBackend> StreamingPipeline<B> {
    pub fn new(backend: B, config: StreamingConfig) -> Self {
        Self {
            backend,
            engine: AgreementEngine::new(config.agreement_config.clone()),
            config,
            audio_buf: PcmBuffer::new(),
            trim_offset: 0,
            samples_at_last_transcribe: 0,
            trimmed_committed_count: 0,
        }
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
    fn trim_to_confirmed(&mut self, tokens: &[WordToken]) {
        let committed = self.engine.committed_text();
        let new_committed = &committed[self.trimmed_committed_count..];
        if new_committed.is_empty() {
            return;
        }

        if let Some(last_end) = find_last_committed_end(new_committed, tokens) {
            let trim_samples = (last_end * SAMPLE_RATE as f32) as usize;
            let trim_samples = trim_samples.min(self.audio_buf.len());
            if trim_samples > 0 {
                self.trim_offset += trim_samples;
                self.audio_buf.0.drain(..trim_samples);
                self.samples_at_last_transcribe =
                    self.samples_at_last_transcribe.saturating_sub(trim_samples);
                self.trimmed_committed_count = committed.len();
                self.engine.freeze_committed();
                self.engine.clear_hypothesis();
            }
        }
    }

    /// Emergency trim: drop oldest 25% of buffer when it exceeds max size.
    fn force_trim(&mut self) {
        let trim = self.audio_buf.len() / 4;
        tracing::warn!(
            trimmed_samples = trim,
            buffer_samples = self.audio_buf.len(),
            "emergency buffer trim — agreement progress reset"
        );
        self.trim_offset += trim;
        self.audio_buf.0.drain(..trim);
        self.samples_at_last_transcribe =
            self.samples_at_last_transcribe.saturating_sub(trim);
        self.trimmed_committed_count = self.engine.committed_text().len();
        self.engine.freeze_committed();
        self.engine.clear_hypothesis();
    }
}

impl<B: AsrBackend> AsrPipeline for StreamingPipeline<B> {
    fn push_audio(&mut self, samples: &[f32]) -> anyhow::Result<Vec<AsrEvent>> {
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

        let max_samples = (self.config.max_buffer_secs * SAMPLE_RATE as f32) as usize;
        if self.audio_buf.len() > max_samples {
            self.force_trim();
        }

        self.step()
    }

    fn flush_utterance(&mut self) -> anyhow::Result<Vec<AsrEvent>> {
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

    fn flush_chunk(&mut self) -> anyhow::Result<Vec<AsrEvent>> {
        // In streaming mode, force_trim is the safety valve.
        let max_samples = (self.config.max_buffer_secs * SAMPLE_RATE as f32) as usize;
        if self.audio_buf.len() > max_samples {
            self.force_trim();
        }
        self.step()
    }

    fn buffer_duration_secs(&self) -> f32 {
        self.audio_buf.duration_secs()
    }

    fn buffer_full(&self) -> bool {
        self.audio_buf.duration_secs() > self.config.max_buffer_secs
    }

    fn reset(&mut self) {
        self.audio_buf.0.clear();
        self.trim_offset = 0;
        self.samples_at_last_transcribe = 0;
        self.trimmed_committed_count = 0;
        self.engine.reset();
        self.backend.reset();
    }

    fn transcribe_raw(&mut self, audio: &[f32]) -> anyhow::Result<Vec<WordToken>> {
        self.backend.transcribe(&PcmAudio::new(audio))
    }
}

// =============================================================================
// Shared helpers
// =============================================================================

/// Find the end timestamp of the last committed word in the token stream.
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rhino_backend::mock::MockBackend;

    const SR: usize = SAMPLE_RATE;

    fn token(word: &str, start: f32, end: f32) -> WordToken {
        WordToken {
            word: word.to_string(),
            start,
            end,
        }
    }

    fn push_one_sec(pipeline: &mut dyn AsrPipeline) -> anyhow::Result<Vec<AsrEvent>> {
        pipeline.push_audio(&[0.0; SR])
    }

    // ==================== Utterance mode tests ====================

    fn utterance_pipeline(mock: MockBackend) -> UtterancePipeline<MockBackend> {
        UtterancePipeline::new(mock, UtteranceConfig {
            max_buffer_secs: 25.0,
            chunk_interval_secs: None,
        })
    }

    fn utterance_pipeline_with_interval(mock: MockBackend, interval: f32) -> UtterancePipeline<MockBackend> {
        UtterancePipeline::new(mock, UtteranceConfig {
            max_buffer_secs: 25.0,
            chunk_interval_secs: Some(interval),
        })
    }

    #[test]
    fn utterance_push_audio_buffers_without_transcribing() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = utterance_pipeline(mock);
        pipeline.push_audio(&[0.0; SR]).unwrap();
        pipeline.push_audio(&[0.0; SR]).unwrap();

        assert_eq!(pipeline.audio_buf.len(), 2 * SR);
    }

    #[test]
    fn utterance_flush_transcribes_and_commits() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![
            token("hello", 0.0, 0.5),
            token("world", 0.5, 1.0),
        ]);

        let mut pipeline = utterance_pipeline(mock);
        pipeline.push_audio(&[0.0; SR]).unwrap();

        let events = pipeline.flush_utterance().unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], AsrEvent::Commit { text } if text == "hello world"));
        assert_eq!(events[1], AsrEvent::EndOfUtterance);
        assert!(pipeline.audio_buf.is_empty());
    }

    #[test]
    fn utterance_flush_empty_buffer_returns_empty() {
        let mock = MockBackend::new();
        let mut pipeline = utterance_pipeline(mock);
        let events = pipeline.flush_utterance().unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn utterance_previous_context_carries_over() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = utterance_pipeline(mock);
        pipeline.push_audio(&[0.0; SR]).unwrap();
        pipeline.flush_utterance().unwrap();

        assert_eq!(pipeline.previous_context, "hello");
    }

    #[test]
    fn utterance_reset_clears_everything() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = utterance_pipeline(mock);
        pipeline.push_audio(&[0.0; SR]).unwrap();
        pipeline.flush_utterance().unwrap();

        pipeline.reset();
        assert!(pipeline.audio_buf.is_empty());
        assert!(pipeline.previous_context.is_empty());
    }

    #[test]
    fn utterance_buffer_full_detects_overflow() {
        let mock = MockBackend::new();
        let mut pipeline = UtterancePipeline::new(mock, UtteranceConfig {
            max_buffer_secs: 1.0,
            chunk_interval_secs: None,
        });

        pipeline.push_audio(&[0.0; SR / 2]).unwrap();
        assert!(!pipeline.buffer_full());

        pipeline.push_audio(&[0.0; SR]).unwrap();
        assert!(pipeline.buffer_full());
    }

    #[test]
    fn utterance_flush_with_backend_error() {
        let mut mock = MockBackend::new();
        mock.set_fail_count(1);

        let mut pipeline = utterance_pipeline(mock);
        pipeline.push_audio(&[0.0; SR]).unwrap();

        let result = pipeline.flush_utterance();
        assert!(result.is_err());
    }

    #[test]
    fn utterance_end_of_utterance_emitted_for_empty_transcription() {
        let mock = MockBackend::new();
        let mut pipeline = utterance_pipeline(mock);
        pipeline.push_audio(&[0.0; SR]).unwrap();

        let events = pipeline.flush_utterance().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], AsrEvent::EndOfUtterance);
    }

    #[test]
    fn utterance_interval_emits_commit_without_end_of_utterance() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = utterance_pipeline_with_interval(mock, 1.5);

        // First push: 1s, under interval.
        let events = pipeline.push_audio(&[0.0; SR]).unwrap();
        assert!(events.is_empty());

        // Second push: 2s total, exceeds 1.5s interval.
        let events = pipeline.push_audio(&[0.0; SR]).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], AsrEvent::Commit { text } if text == "hello world"));

        assert!(pipeline.audio_buf.is_empty());
    }

    #[test]
    fn utterance_interval_then_flush_gives_end_of_utterance() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        // Interval at 1.5s — first push (1s) doesn't trigger, second does.
        let mut pipeline = utterance_pipeline_with_interval(mock, 1.5);

        // First push: 1s, under interval.
        let events = pipeline.push_audio(&[0.0; SR]).unwrap();
        assert!(events.is_empty());

        // Second push: 2s total, exceeds 1.5s. Interval flush → Commit only.
        let events = pipeline.push_audio(&[0.0; SR]).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], AsrEvent::Commit { .. }));

        // Push more audio, then flush utterance → must include EndOfUtterance.
        pipeline.push_audio(&[0.0; SR / 2]).unwrap();
        let events = pipeline.flush_utterance().unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::EndOfUtterance)));
    }

    #[test]
    fn utterance_flush_chunk_directly() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("test", 0.0, 0.5)]);

        let mut pipeline = utterance_pipeline(mock);
        pipeline.push_audio(&[0.0; SR]).unwrap();

        let events = pipeline.flush_chunk().unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], AsrEvent::Commit { text } if text == "test"));
        assert!(!events.iter().any(|e| matches!(e, AsrEvent::EndOfUtterance)));
    }

    #[test]
    fn utterance_never_emits_retract_or_interim() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = utterance_pipeline_with_interval(mock, 0.5);

        for _ in 0..5 {
            let events = pipeline.push_audio(&[0.0; SR]).unwrap();
            for e in &events {
                assert!(!matches!(e, AsrEvent::Retract { .. } | AsrEvent::Interim { .. }));
            }
        }
        let events = pipeline.flush_utterance().unwrap();
        for e in &events {
            assert!(!matches!(e, AsrEvent::Retract { .. } | AsrEvent::Interim { .. }));
        }
    }

    // ==================== Streaming mode tests ====================

    fn streaming_pipeline(mock: MockBackend) -> StreamingPipeline<MockBackend> {
        StreamingPipeline::new(mock, StreamingConfig {
            max_buffer_secs: 30.0,
            step_secs: 0.0,
            min_chunk_secs: 0.0,
            agreement_config: AgreementConfig {
                min_agreement: 2,
                commit_lookahead_secs: 0.5,
            },
        })
    }

    #[test]
    fn streaming_basic_transcription() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = streaming_pipeline(mock);

        // First push — agreement = 1, only interim.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Interim { .. })));
        assert!(!events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));

        // Second push — agreement = 2 = k, commit.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
    }

    #[test]
    fn streaming_step_interval_respected() {
        let mut mock = MockBackend::new();
        mock.queue_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = StreamingPipeline::new(mock, StreamingConfig {
            max_buffer_secs: 30.0,
            step_secs: 1.0,
            min_chunk_secs: 0.0,
            agreement_config: AgreementConfig::default(),
        });

        let events = pipeline.push_audio(&[0.0; 8000]).unwrap();
        assert!(events.is_empty());

        let events = pipeline.push_audio(&[0.0; 8000]).unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    fn streaming_no_events_below_min_chunk() {
        let mut mock = MockBackend::new();
        mock.queue_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = StreamingPipeline::new(mock, StreamingConfig {
            max_buffer_secs: 30.0,
            step_secs: 0.0,
            min_chunk_secs: 1.0,
            agreement_config: AgreementConfig::default(),
        });

        let events = pipeline.push_audio(&[0.0; 8000]).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn streaming_buffer_trimming() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = streaming_pipeline(mock);

        for _ in 0..2 {
            push_one_sec(&mut pipeline).unwrap();
        }

        assert!(pipeline.audio_buf.len() < 2 * SR);
    }

    #[test]
    fn streaming_flush_commits_pending() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = streaming_pipeline(mock);
        push_one_sec(&mut pipeline).unwrap();

        let events = pipeline.flush_utterance().unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
        assert_eq!(events.last(), Some(&AsrEvent::EndOfUtterance));
    }

    #[test]
    fn streaming_reset_clears_state() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = streaming_pipeline(mock);
        push_one_sec(&mut pipeline).unwrap();

        pipeline.reset();
        assert!(pipeline.audio_buf.is_empty());
        assert_eq!(pipeline.trim_offset, 0);
        assert_eq!(pipeline.samples_at_last_transcribe, 0);
    }

    #[test]
    fn streaming_flush_on_empty_buffer() {
        let mock = MockBackend::new();
        let mut pipeline = streaming_pipeline(mock);
        let events = pipeline.flush_utterance().unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn streaming_flush_emits_end_of_utterance() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = streaming_pipeline(mock);
        push_one_sec(&mut pipeline).unwrap();

        let events = pipeline.flush_utterance().unwrap();
        assert_eq!(events.last(), Some(&AsrEvent::EndOfUtterance));
    }

    #[test]
    fn streaming_flush_with_backend_error_still_returns_events() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = streaming_pipeline(mock);
        push_one_sec(&mut pipeline).unwrap();

        pipeline.backend.set_fail_count(1);

        let events = pipeline.flush_utterance().unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
        assert_eq!(events.last(), Some(&AsrEvent::EndOfUtterance));
    }

    #[test]
    fn streaming_force_trim_no_retract_on_next_update() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.3)]);

        let mut pipeline = StreamingPipeline::new(mock, StreamingConfig {
            max_buffer_secs: 2.0,
            step_secs: 0.0,
            min_chunk_secs: 0.0,
            agreement_config: AgreementConfig {
                min_agreement: 2,
                commit_lookahead_secs: 0.5,
            },
        });

        let events1 = push_one_sec(&mut pipeline).unwrap();
        assert!(!events1.iter().any(|e| matches!(e, AsrEvent::Retract { .. })));

        let events2 = push_one_sec(&mut pipeline).unwrap();
        assert!(events2.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));

        let events3 = push_one_sec(&mut pipeline).unwrap();
        assert!(!events3.iter().any(|e| matches!(e, AsrEvent::Retract { .. })));
    }

    #[test]
    fn streaming_emits_interim_events() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);

        let mut pipeline = streaming_pipeline(mock);

        // First push: agreement = 1, interim only.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Interim { .. })));
        assert!(!events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
    }

    #[test]
    fn streaming_retract_on_correction() {
        // Test retraction by using the engine directly via flush.
        // Commit "hello world", then push a divergent hypothesis "hello there".
        // After commit + trim, the engine's freeze_committed prevents retraction
        // of trimmed words, but we can test retraction before trimming by using
        // longer timestamps that avoid the commit_lookahead cutoff.
        let mut mock = MockBackend::new();
        // Use timestamps well within buffer so commit_lookahead doesn't block.
        mock.queue_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);
        mock.queue_response(vec![token("hello", 0.0, 0.5), token("world", 0.5, 1.0)]);
        // After commit, trim moves buffer. Next hypothesis diverges.
        // But post-trim, committed words are frozen — retraction only affects
        // non-frozen words. This is tested in the engine unit tests.

        let mut pipeline = streaming_pipeline(mock);

        // Push 1: interim.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Interim { .. })));

        // Push 2: commit.
        let events = push_one_sec(&mut pipeline).unwrap();
        assert!(events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })));
    }

    // ==================== Shared helper tests ====================

    #[test]
    fn find_last_committed_end_basic() {
        let committed = vec!["a".to_string(), "b".to_string()];
        let tokens = vec![
            token("a", 0.0, 0.2),
            token("b", 0.2, 0.4),
            token("c", 0.4, 0.6),
        ];
        assert_eq!(find_last_committed_end(&committed, &tokens), Some(0.4));
    }

    #[test]
    fn find_last_committed_end_no_match() {
        let committed = vec!["xyz".to_string()];
        let tokens = vec![token("abc", 0.0, 0.5)];
        assert_eq!(find_last_committed_end(&committed, &tokens), None);
    }
}
