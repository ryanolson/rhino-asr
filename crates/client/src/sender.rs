use anyhow::{Context, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use velo::StreamSender;

use rhino_protocol::AudioChunk;

const TARGET_RATE: u32 = 16_000;
const CHUNK_SAMPLES: usize = 1_600; // 100ms at 16kHz

/// Sends audio to the ASR server. Handles resampling to 16kHz, buffering into
/// fixed-size chunks (100ms), and sequencing. Callers push raw samples at their
/// native sample rate.
pub struct AudioSender {
    inner: StreamSender<AudioChunk>,
    resampler: Option<SincFixedIn<f32>>,
    buffer: Vec<f32>,
    sequence: u64,
    /// Input chunk size the resampler expects (fixed by rubato).
    resample_chunk_size: usize,
    /// Leftover input samples that didn't fill a complete resampler input chunk.
    resample_pending: Vec<f32>,
}

impl AudioSender {
    pub(crate) fn new(inner: StreamSender<AudioChunk>, input_sample_rate: u32) -> Result<Self> {
        let (resampler, resample_chunk_size) = if input_sample_rate == TARGET_RATE {
            (None, 0)
        } else {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            let ratio = TARGET_RATE as f64 / input_sample_rate as f64;
            // Process in chunks of ~10ms at input rate.
            let chunk_size = (input_sample_rate as usize) / 100;
            let resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)
                .context("failed to create resampler")?;
            (Some(resampler), chunk_size)
        };

        Ok(Self {
            inner,
            resampler,
            buffer: Vec::with_capacity(CHUNK_SAMPLES * 2),
            sequence: 0,
            resample_chunk_size,
            resample_pending: Vec::new(),
        })
    }

    /// Push audio samples at the session's configured sample rate.
    /// Internally resamples to 16kHz, buffers, and sends complete chunks.
    pub async fn send(&mut self, samples: &[f32]) -> Result<()> {
        tracing::trace!(
            input_samples = samples.len(),
            buffer = self.buffer.len(),
            pending = self.resample_pending.len(),
            seq = self.sequence,
            has_resampler = self.resampler.is_some(),
            "audio_sender.send"
        );
        match &mut self.resampler {
            Some(resampler) => {
                self.resample_pending.extend_from_slice(samples);
                // Process all complete input chunks through the resampler.
                while self.resample_pending.len() >= self.resample_chunk_size {
                    let input_chunk: Vec<f32> = self
                        .resample_pending
                        .drain(..self.resample_chunk_size)
                        .collect();
                    let output = resampler
                        .process(&[input_chunk], None)
                        .context("resampler failed")?;
                    self.buffer.extend_from_slice(&output[0]);
                }
            }
            None => {
                self.buffer.extend_from_slice(samples);
            }
        }

        self.flush_chunks().await
    }

    /// Flush remaining buffered audio and finalize the stream.
    pub async fn finalize(mut self) -> Result<()> {
        // Flush remaining resampler input.
        if let Some(resampler) = &mut self.resampler {
            if !self.resample_pending.is_empty() {
                // Pad to the expected chunk size for the final block.
                self.resample_pending
                    .resize(self.resample_chunk_size, 0.0);
                let output = resampler
                    .process(&[std::mem::take(&mut self.resample_pending)], None)
                    .context("resampler flush failed")?;
                self.buffer.extend_from_slice(&output[0]);
            }
        }

        // Send any remaining buffer as a final short chunk.
        if !self.buffer.is_empty() {
            let samples = std::mem::take(&mut self.buffer);
            let chunk = AudioChunk {
                samples,
                sequence: self.sequence,
            };
            self.inner
                .send(chunk)
                .await
                .map_err(|e| anyhow::anyhow!("send failed: {e}"))?;
        }

        self.inner
            .finalize()
            .map_err(|e| anyhow::anyhow!("finalize failed: {e}"))
    }

    /// Send all complete CHUNK_SAMPLES-sized chunks from the buffer.
    async fn flush_chunks(&mut self) -> Result<()> {
        while self.buffer.len() >= CHUNK_SAMPLES {
            let samples: Vec<f32> = self.buffer.drain(..CHUNK_SAMPLES).collect();
            let chunk = AudioChunk {
                samples,
                sequence: self.sequence,
            };
            self.sequence += 1;
            tracing::debug!(seq = self.sequence, buffer_remaining = self.buffer.len(), "sending audio chunk to server");
            self.inner
                .send(chunk)
                .await
                .map_err(|e| anyhow::anyhow!("send failed: {e}"))?;
        }
        Ok(())
    }
}
