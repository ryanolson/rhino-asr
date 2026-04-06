pub mod gate;

#[cfg(feature = "silero")]
pub mod silero;

pub use gate::{VadConfig, VadGate, VadTransition};

/// Trait for VAD probability sources.
///
/// Implementations process fixed-size audio chunks and return a speech
/// probability [0.0, 1.0] that feeds into [`VadGate`].
pub trait VadProcessor: Send {
    /// Process a single audio chunk and return speech probability.
    /// The chunk must be exactly [`chunk_size()`](Self::chunk_size) samples.
    fn process_chunk(&mut self, audio: &[f32]) -> anyhow::Result<f32>;

    /// Expected number of samples per chunk.
    fn chunk_size(&self) -> usize;

    /// Reset internal state (new session / new utterance).
    fn reset(&mut self);
}

#[cfg(feature = "silero")]
impl VadProcessor for silero::SileroVad {
    fn process_chunk(&mut self, audio: &[f32]) -> anyhow::Result<f32> {
        self.process_chunk(audio)
    }

    fn chunk_size(&self) -> usize {
        silero::CHUNK_SAMPLES
    }

    fn reset(&mut self) {
        self.reset();
    }
}
