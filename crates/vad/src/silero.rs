use ndarray::{Array0, Array2, Array3};
use ort::session::Session;
use ort::value::TensorRef;

const SAMPLE_RATE: i64 = 16_000;
/// Silero VAD expects 512-sample chunks at 16kHz (32ms per chunk).
pub const CHUNK_SAMPLES: usize = 512;
/// LSTM hidden state dimensions for Silero VAD v5 at 16kHz.
const STATE_DIM: usize = 128;

/// Silero VAD v5 wrapper using ONNX Runtime.
///
/// Processes 512-sample (32ms) chunks of 16kHz mono f32 PCM and returns
/// a speech probability [0.0, 1.0]. Maintains LSTM hidden state across
/// calls for streaming inference.
///
/// Model inputs:  `input` [1, 512], `state` [2, 1, 128], `sr` scalar i64
/// Model outputs: `output` [1, 1], `stateN` [2, 1, 128]
pub struct SileroVad {
    session: Session,
    state: Array3<f32>,
    sr: Array0<i64>,
}

impl SileroVad {
    /// Load Silero VAD from an ONNX model file.
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("ort optimization level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("ort intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("ort commit from file: {e}"))?;

        Ok(Self {
            session,
            state: Array3::zeros((2, 1, STATE_DIM)),
            sr: Array0::from_elem((), SAMPLE_RATE),
        })
    }

    /// Process a single audio chunk and return speech probability.
    ///
    /// `audio` must be exactly `CHUNK_SAMPLES` (512) f32 samples at 16kHz.
    /// Panics if the length doesn't match.
    pub fn process_chunk(&mut self, audio: &[f32]) -> anyhow::Result<f32> {
        assert_eq!(
            audio.len(),
            CHUNK_SAMPLES,
            "silero VAD expects exactly {CHUNK_SAMPLES} samples, got {}",
            audio.len()
        );

        let input =
            Array2::from_shape_vec((1, CHUNK_SAMPLES), audio.to_vec()).expect("shape mismatch");

        let input_tensor = TensorRef::from_array_view(&input)
            .map_err(|e| anyhow::anyhow!("input tensor: {e}"))?;
        let sr_tensor = TensorRef::from_array_view(&self.sr)
            .map_err(|e| anyhow::anyhow!("sr tensor: {e}"))?;
        let state_tensor = TensorRef::from_array_view(&self.state)
            .map_err(|e| anyhow::anyhow!("state tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sr_tensor,
            ])
            .map_err(|e| anyhow::anyhow!("ort run: {e}"))?;

        // output: speech probability [1, 1]
        let (_shape, prob_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract prob: {e}"))?;
        let speech_prob = prob_data[0];

        // stateN: updated LSTM state [2, 1, 128]
        let (_shape, state_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("extract state: {e}"))?;
        self.state = Array3::from_shape_vec((2, 1, STATE_DIM), state_data.to_vec())
            .expect("state shape mismatch");

        Ok(speech_prob)
    }

    /// Reset LSTM hidden state (new utterance / new session).
    pub fn reset(&mut self) {
        self.state = Array3::zeros((2, 1, STATE_DIM));
    }
}
