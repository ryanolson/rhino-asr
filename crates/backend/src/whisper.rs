use std::ffi::c_int;
use std::sync::Arc;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::{AsrBackend, AudioData, WordToken};

/// Configuration for the Whisper backend.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Language code (e.g. "en"). None = auto-detect.
    pub language: Option<String>,
    /// Beam search width. 1 = greedy.
    pub beam_size: i32,
    /// Sampling temperature. Lower = more deterministic.
    pub temperature: f32,
    /// Number of CPU threads for inference.
    pub n_threads: i32,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            language: Some("en".to_string()),
            beam_size: 5,
            temperature: 0.0,
            n_threads: 4,
        }
    }
}

/// Load a WhisperContext from a GGML model file.
///
/// The returned context is `Send + Sync` and should be wrapped in `Arc`
/// for sharing across sessions.
pub fn load_whisper_context(
    model_path: &str,
    use_gpu: bool,
    gpu_device: i32,
) -> anyhow::Result<WhisperContext> {
    let mut params = WhisperContextParameters::new();
    params.use_gpu(use_gpu);
    params.gpu_device(gpu_device as c_int);

    WhisperContext::new_with_params(model_path, params)
        .map_err(|e| anyhow::anyhow!("failed to load whisper model: {e}"))
}

/// GPU-accelerated Whisper ASR backend.
///
/// Each instance holds its own `WhisperState` (decode buffers) created from
/// a shared `WhisperContext` (model weights). This means one model load in
/// VRAM, with per-session decode state.
pub struct WhisperBackend {
    ctx: Arc<WhisperContext>,
    state: WhisperState,
    config: WhisperConfig,
}

impl WhisperBackend {
    pub fn new(ctx: Arc<WhisperContext>, config: WhisperConfig) -> anyhow::Result<Self> {
        let state = ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("failed to create whisper state: {e}"))?;
        Ok(Self { ctx, state, config })
    }
}

fn build_params(config: &WhisperConfig) -> FullParams<'_, '_> {
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: config.beam_size,
        patience: -1.0,
    });
    params.set_token_timestamps(true);
    params.set_n_threads(config.n_threads);
    params.set_temperature(config.temperature);

    if let Some(ref lang) = config.language {
        params.set_language(Some(lang));
    } else {
        params.set_language(None);
    }

    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_print_special(false);

    params
}

impl AsrBackend for WhisperBackend {
    fn transcribe(&mut self, audio: &impl AudioData) -> anyhow::Result<Vec<WordToken>> {
        let samples = audio.samples();
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let params = build_params(&self.config);
        self.state
            .full(params, samples)
            .map_err(|e| anyhow::anyhow!("whisper inference failed: {e}"))?;
        extract_words(&self.state)
    }

    fn transcribe_with_prompt(
        &mut self,
        audio: &impl AudioData,
        prompt: &str,
    ) -> anyhow::Result<Vec<WordToken>> {
        let samples = audio.samples();
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let mut params = build_params(&self.config);
        if !prompt.is_empty() {
            params.set_initial_prompt(prompt);
        }
        self.state
            .full(params, samples)
            .map_err(|e| anyhow::anyhow!("whisper inference failed: {e}"))?;
        extract_words(&self.state)
    }

    fn reset(&mut self) {
        if let Ok(state) = self.ctx.create_state() {
            self.state = state;
        }
    }
}

/// Extract word tokens with timestamps from completed whisper inference.
///
/// Uses segment-level timestamps with linear interpolation for per-word timing.
/// Token-level timestamps (`set_token_timestamps`) are experimental in whisper.cpp
/// and can produce erratic values; segment interpolation gives reliable monotonic
/// word ordering for the agreement engine.
fn extract_words(state: &WhisperState) -> anyhow::Result<Vec<WordToken>> {
    let n_segments = state.full_n_segments();
    let mut words = Vec::new();

    for seg_idx in 0..n_segments {
        let segment = match state.get_segment(seg_idx) {
            Some(s) => s,
            None => continue,
        };

        let text = segment
            .to_str_lossy()
            .map_err(|e| anyhow::anyhow!("failed to get segment text: {e}"))?;

        let t0 = segment.start_timestamp() as f32 / 100.0; // centiseconds → seconds
        let t1 = segment.end_timestamp() as f32 / 100.0;

        let segment_words: Vec<&str> = text.split_whitespace().collect();
        if segment_words.is_empty() {
            continue;
        }

        let duration = (t1 - t0).max(0.0);
        let word_duration = duration / segment_words.len() as f32;

        for (i, word) in segment_words.iter().enumerate() {
            words.push(WordToken {
                word: word.to_string(),
                start: t0 + i as f32 * word_duration,
                end: t0 + (i + 1) as f32 * word_duration,
            });
        }
    }

    Ok(words)
}
