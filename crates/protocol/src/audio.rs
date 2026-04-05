use serde::{Deserialize, Serialize};

/// Audio chunk streamed from client to server via velo-streaming.
///
/// 16kHz mono f32 PCM. Client resamples to 16kHz before sending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunk {
    /// PCM samples, f32 little-endian, 16kHz mono.
    pub samples: Vec<f32>,

    /// Monotonically increasing frame counter.
    /// Used for ordering verification and debugging.
    pub sequence: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip() {
        let chunk = AudioChunk {
            samples: vec![0.0, 0.1, -0.5, 1.0],
            sequence: 42,
        };

        let packed = rmp_serde::to_vec(&chunk).unwrap();
        let back: AudioChunk = rmp_serde::from_slice(&packed).unwrap();
        assert_eq!(back.sequence, 42);
        assert_eq!(back.samples.len(), 4);
        assert!((back.samples[2] - (-0.5)).abs() < f32::EPSILON);
    }
}
