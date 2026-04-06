pub mod mock;

/// Trait for typed audio input. Wraps raw `&[f32]` so the backend API
/// is self-documenting and extensible to other audio representations.
pub trait AudioData {
    fn samples(&self) -> &[f32];
}

/// Borrowed 16kHz mono f32 PCM slice.
#[derive(Debug)]
pub struct PcmAudio<'a>(&'a [f32]);

impl<'a> PcmAudio<'a> {
    pub fn new(samples: &'a [f32]) -> Self {
        Self(samples)
    }

    pub fn duration_secs(&self) -> f32 {
        self.0.len() as f32 / 16_000.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl AudioData for PcmAudio<'_> {
    fn samples(&self) -> &[f32] {
        self.0
    }
}

/// Owned 16kHz mono f32 PCM buffer.
#[derive(Debug, Clone, Default)]
pub struct PcmBuffer(pub Vec<f32>);

impl PcmBuffer {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn duration_secs(&self) -> f32 {
        self.0.len() as f32 / 16_000.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl AudioData for PcmBuffer {
    fn samples(&self) -> &[f32] {
        &self.0
    }
}

/// A word with timing from the ASR backend.
#[derive(Debug, Clone)]
pub struct WordToken {
    pub word: String,
    /// Start time in seconds from buffer start.
    pub start: f32,
    /// End time in seconds from buffer start.
    pub end: f32,
}

/// Trait for ASR backends.
///
/// - **Sync**: caller wraps in `spawn_blocking` for async contexts.
/// - **`&mut self`**: backend holds mutable state (whisper context, decode buffers).
/// - **`Send`**: must cross thread boundary for `spawn_blocking`.
/// - **Generic over `AudioData`**: callers pass `PcmAudio`, `PcmBuffer`, etc.
pub trait AsrBackend: Send {
    /// Transcribe audio. Returns word tokens with timestamps relative to buffer start.
    fn transcribe(&mut self, audio: &impl AudioData) -> anyhow::Result<Vec<WordToken>>;

    /// Reset internal state (new utterance / new session).
    fn reset(&mut self);
}

pub use mock::MockBackend;
