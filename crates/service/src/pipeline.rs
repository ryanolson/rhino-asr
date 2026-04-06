use rhino_backend::{AsrBackend, PcmAudio, PcmBuffer, WordToken};
use rhino_protocol::AsrEvent;

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum audio buffer before forced split (seconds).
    pub max_buffer_secs: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_buffer_secs: 25.0,
        }
    }
}

/// Single-pass ASR pipeline for VAD-chunked transcription.
///
/// Audio accumulates during speech. On flush (VAD SpeechEnd), the full
/// buffer is transcribed once and emitted as a single Commit. Previous
/// utterance text is passed as `initial_prompt` for cross-chunk context.
///
/// VAD is **not** inside the pipeline — the session layer gates which
/// audio reaches here and triggers flush at speech boundaries.
pub struct AsrPipeline<B: AsrBackend> {
    backend: B,
    config: PipelineConfig,
    audio_buf: PcmBuffer,
    /// Previous utterance text, passed as initial_prompt to whisper
    /// for cross-chunk context continuity.
    previous_context: String,
}

impl<B: AsrBackend> AsrPipeline<B> {
    pub fn new(backend: B, config: PipelineConfig) -> Self {
        Self {
            backend,
            config,
            audio_buf: PcmBuffer::new(),
            previous_context: String::new(),
        }
    }

    /// Append audio samples to the buffer. No transcription happens here —
    /// audio just accumulates until `flush_utterance()` is called.
    pub fn push_audio(&mut self, samples: &[f32]) {
        self.audio_buf.0.extend_from_slice(samples);
    }

    /// Current buffer duration in seconds.
    pub fn buffer_duration_secs(&self) -> f32 {
        self.audio_buf.duration_secs()
    }

    /// Whether the buffer exceeds the maximum chunk size.
    pub fn buffer_full(&self) -> bool {
        self.audio_buf.duration_secs() > self.config.max_buffer_secs
    }

    /// Transcribe the full buffer in a single pass, emit Commit + EndOfUtterance,
    /// store text as context for the next chunk, and clear the buffer.
    ///
    /// Returns empty if no audio was buffered.
    pub fn flush_utterance(&mut self) -> anyhow::Result<Vec<AsrEvent>> {
        if self.audio_buf.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!(
            buf_secs = format!("{:.2}", self.audio_buf.duration_secs()),
            buf_samples = self.audio_buf.len(),
            context_len = self.previous_context.len(),
            "transcribing utterance"
        );

        let tokens = if self.previous_context.is_empty() {
            self.backend.transcribe(&self.audio_buf)?
        } else {
            self.backend
                .transcribe_with_prompt(&self.audio_buf, &self.previous_context)?
        };

        let text: String = tokens
            .iter()
            .map(|t| t.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if !text.is_empty() {
            tracing::info!(words = tokens.len(), text = %text, "utterance committed");
        }

        // Store for next chunk's initial_prompt context.
        self.previous_context = text.clone();

        // Clear buffer for next utterance.
        self.audio_buf.0.clear();

        let mut events = Vec::new();
        if !text.is_empty() {
            events.push(AsrEvent::Commit { text });
        }
        events.push(AsrEvent::EndOfUtterance);

        Ok(events)
    }

    /// Run a single transcription on arbitrary audio, bypassing buffer
    /// management. For diagnostics only.
    pub fn transcribe_raw(&mut self, audio: &[f32]) -> anyhow::Result<Vec<WordToken>> {
        self.backend.transcribe(&PcmAudio::new(audio))
    }

    /// Full reset for new session.
    pub fn reset(&mut self) {
        self.audio_buf.0.clear();
        self.previous_context.clear();
        self.backend.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rhino_backend::mock::MockBackend;

    const SAMPLE_RATE: usize = 16_000;

    fn token(word: &str, start: f32, end: f32) -> WordToken {
        WordToken {
            word: word.to_string(),
            start,
            end,
        }
    }

    fn pipeline_with_mock(mock: MockBackend) -> AsrPipeline<MockBackend> {
        AsrPipeline::new(mock, PipelineConfig::default())
    }

    #[test]
    fn push_audio_buffers_without_transcribing() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);
        pipeline.push_audio(&[0.0; SAMPLE_RATE]);
        pipeline.push_audio(&[0.0; SAMPLE_RATE]);

        // Buffer grows, no events emitted during push.
        assert_eq!(pipeline.audio_buf.len(), 2 * SAMPLE_RATE);
        // Backend was never called (queued response still available).
        assert_eq!(pipeline.backend.queued_count(), 0); // default, not queued
    }

    #[test]
    fn flush_transcribes_and_commits() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![
            token("hello", 0.0, 0.5),
            token("world", 0.5, 1.0),
        ]);

        let mut pipeline = pipeline_with_mock(mock);
        pipeline.push_audio(&[0.0; SAMPLE_RATE]);

        let events = pipeline.flush_utterance().unwrap();

        // Single Commit with full text + EndOfUtterance.
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], AsrEvent::Commit { text } if text == "hello world"));
        assert_eq!(events[1], AsrEvent::EndOfUtterance);

        // Buffer cleared after flush.
        assert!(pipeline.audio_buf.is_empty());
    }

    #[test]
    fn flush_empty_buffer_returns_empty() {
        let mock = MockBackend::new();
        let mut pipeline = pipeline_with_mock(mock);

        let events = pipeline.flush_utterance().unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn previous_context_carries_over() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);

        // First utterance.
        pipeline.push_audio(&[0.0; SAMPLE_RATE]);
        pipeline.flush_utterance().unwrap();
        assert_eq!(pipeline.previous_context, "hello");

        // Second utterance — context should be "hello".
        pipeline.push_audio(&[0.0; SAMPLE_RATE]);
        pipeline.flush_utterance().unwrap();
        // MockBackend ignores prompt, but context is still tracked.
        assert_eq!(pipeline.previous_context, "hello");
    }

    #[test]
    fn reset_clears_everything() {
        let mut mock = MockBackend::new();
        mock.set_default_response(vec![token("hello", 0.0, 0.5)]);

        let mut pipeline = pipeline_with_mock(mock);
        pipeline.push_audio(&[0.0; SAMPLE_RATE]);
        pipeline.flush_utterance().unwrap();

        pipeline.reset();
        assert!(pipeline.audio_buf.is_empty());
        assert!(pipeline.previous_context.is_empty());
    }

    #[test]
    fn buffer_full_detects_overflow() {
        let mock = MockBackend::new();
        let mut pipeline = AsrPipeline::new(
            mock,
            PipelineConfig {
                max_buffer_secs: 1.0,
            },
        );

        pipeline.push_audio(&[0.0; SAMPLE_RATE / 2]); // 0.5s
        assert!(!pipeline.buffer_full());

        pipeline.push_audio(&[0.0; SAMPLE_RATE]); // 1.5s total
        assert!(pipeline.buffer_full());
    }

    #[test]
    fn flush_with_backend_error() {
        let mut mock = MockBackend::new();
        mock.set_fail_count(1);

        let mut pipeline = pipeline_with_mock(mock);
        pipeline.push_audio(&[0.0; SAMPLE_RATE]);

        let result = pipeline.flush_utterance();
        assert!(result.is_err());
    }

    #[test]
    fn end_of_utterance_emitted_even_for_empty_transcription() {
        let mock = MockBackend::new(); // returns empty by default
        let mut pipeline = pipeline_with_mock(mock);
        pipeline.push_audio(&[0.0; SAMPLE_RATE]);

        let events = pipeline.flush_utterance().unwrap();
        // No Commit (empty transcription), but EndOfUtterance still emitted.
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], AsrEvent::EndOfUtterance);
    }
}
