use std::collections::VecDeque;

use crate::{AsrBackend, AudioData, WordToken};

/// Mock backend that returns pre-queued responses.
///
/// Responses are dequeued FIFO on each `transcribe()` call.
/// When the queue is empty, returns `default_response` (empty by default).
#[derive(Debug, Default)]
pub struct MockBackend {
    responses: VecDeque<Vec<WordToken>>,
    default_response: Vec<WordToken>,
    fail_next: usize,
}

impl MockBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a response to be returned by the next `transcribe()` call.
    pub fn queue_response(&mut self, tokens: Vec<WordToken>) {
        self.responses.push_back(tokens);
    }

    /// Set the fallback response returned when the queue is empty.
    pub fn set_default_response(&mut self, tokens: Vec<WordToken>) {
        self.default_response = tokens;
    }

    /// Number of queued responses remaining.
    pub fn queued_count(&self) -> usize {
        self.responses.len()
    }

    /// Make the next `n` calls to `transcribe()` return an error.
    pub fn set_fail_count(&mut self, n: usize) {
        self.fail_next = n;
    }
}

impl AsrBackend for MockBackend {
    fn transcribe(&mut self, _audio: &impl AudioData) -> anyhow::Result<Vec<WordToken>> {
        if self.fail_next > 0 {
            self.fail_next -= 1;
            return Err(anyhow::anyhow!("mock transcription error"));
        }
        Ok(self
            .responses
            .pop_front()
            .unwrap_or_else(|| self.default_response.clone()))
    }

    fn reset(&mut self) {
        // No-op for mock — queue is intentional test state, not runtime state.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PcmAudio;

    fn token(word: &str, start: f32, end: f32) -> WordToken {
        WordToken {
            word: word.to_string(),
            start,
            end,
        }
    }

    #[test]
    fn queue_and_transcribe() {
        let mut backend = MockBackend::new();
        let audio = PcmAudio::new(&[0.0; 16000]);

        backend.queue_response(vec![token("hello", 0.0, 0.5)]);
        backend.queue_response(vec![token("world", 0.5, 1.0)]);
        backend.queue_response(vec![
            token("hello", 0.0, 0.5),
            token("world", 0.5, 1.0),
        ]);

        let r1 = backend.transcribe(&audio).unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].word, "hello");

        let r2 = backend.transcribe(&audio).unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].word, "world");

        let r3 = backend.transcribe(&audio).unwrap();
        assert_eq!(r3.len(), 2);

        assert_eq!(backend.queued_count(), 0);
    }

    #[test]
    fn empty_queue_returns_default() {
        let mut backend = MockBackend::new();
        let audio = PcmAudio::new(&[0.0; 16000]);

        // Empty queue, no default set → empty vec.
        let r = backend.transcribe(&audio).unwrap();
        assert!(r.is_empty());

        // Set a default.
        backend.set_default_response(vec![token("fallback", 0.0, 1.0)]);
        let r = backend.transcribe(&audio).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].word, "fallback");
    }

    #[test]
    fn queued_responses_take_priority_over_default() {
        let mut backend = MockBackend::new();
        let audio = PcmAudio::new(&[0.0; 16000]);

        backend.set_default_response(vec![token("default", 0.0, 1.0)]);
        backend.queue_response(vec![token("queued", 0.0, 0.5)]);

        let r1 = backend.transcribe(&audio).unwrap();
        assert_eq!(r1[0].word, "queued");

        let r2 = backend.transcribe(&audio).unwrap();
        assert_eq!(r2[0].word, "default");
    }

    #[test]
    fn reset_does_not_panic() {
        let mut backend = MockBackend::new();
        backend.queue_response(vec![token("test", 0.0, 1.0)]);
        backend.reset();
        // Queue is preserved — reset is about runtime state, not test fixtures.
        assert_eq!(backend.queued_count(), 1);
    }
}
