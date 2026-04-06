/// Hysteresis configuration for the VAD gate.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Speech probability threshold to open gate.
    pub threshold_on: f32,
    /// Speech probability threshold to close gate.
    /// Lower than `threshold_on` to prevent chattering.
    pub threshold_off: f32,
    /// Consecutive speech chunks required to open gate.
    pub min_speech_chunks: usize,
    /// Consecutive silence chunks required to close gate.
    pub min_silence_chunks: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold_on: 0.5,
            threshold_off: 0.35,
            min_speech_chunks: 3,    // ~96ms at 32ms/chunk
            min_silence_chunks: 20,  // ~640ms at 32ms/chunk
        }
    }
}

/// VAD state transition edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadTransition {
    SpeechStart,
    SpeechEnd,
}

/// Hysteresis gate that converts a stream of speech probabilities
/// into clean speech/silence state transitions.
///
/// Requires `min_speech_chunks` consecutive above-threshold readings
/// to open, and `min_silence_chunks` consecutive below-threshold readings
/// to close. The on/off thresholds differ to prevent chattering.
#[derive(Debug)]
pub struct VadGate {
    config: VadConfig,
    is_speech: bool,
    speech_count: usize,
    silence_count: usize,
}

impl VadGate {
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            is_speech: false,
            speech_count: 0,
            silence_count: 0,
        }
    }

    /// Feed a speech probability. Returns a transition if the gate state changed.
    pub fn update(&mut self, probability: f32) -> Option<VadTransition> {
        let threshold = if self.is_speech {
            self.config.threshold_off
        } else {
            self.config.threshold_on
        };

        let is_speech_chunk = probability >= threshold;
        let prev = self.is_speech;

        if is_speech_chunk {
            self.speech_count += 1;
            self.silence_count = 0;
            if !self.is_speech && self.speech_count >= self.config.min_speech_chunks {
                self.is_speech = true;
            }
        } else {
            self.silence_count += 1;
            self.speech_count = 0;
            if self.is_speech && self.silence_count >= self.config.min_silence_chunks {
                self.is_speech = false;
            }
        }

        match (prev, self.is_speech) {
            (false, true) => Some(VadTransition::SpeechStart),
            (true, false) => Some(VadTransition::SpeechEnd),
            _ => None,
        }
    }

    /// Current gate state.
    pub fn is_speech(&self) -> bool {
        self.is_speech
    }

    /// Reset to silence state.
    pub fn reset(&mut self) {
        self.is_speech = false;
        self.speech_count = 0;
        self.silence_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> VadGate {
        VadGate::new(VadConfig {
            threshold_on: 0.5,
            threshold_off: 0.35,
            min_speech_chunks: 3,
            min_silence_chunks: 3, // shorter for tests
        })
    }

    #[test]
    fn silence_stays_silent() {
        let mut g = gate();
        for _ in 0..20 {
            assert_eq!(g.update(0.1), None);
        }
        assert!(!g.is_speech());
    }

    #[test]
    fn speech_opens_after_min_chunks() {
        let mut g = gate();

        // 2 chunks above threshold — not enough.
        assert_eq!(g.update(0.8), None);
        assert_eq!(g.update(0.8), None);
        assert!(!g.is_speech());

        // 3rd chunk triggers SpeechStart.
        assert_eq!(g.update(0.8), Some(VadTransition::SpeechStart));
        assert!(g.is_speech());
    }

    #[test]
    fn speech_closes_after_silence() {
        let mut g = gate();

        // Open the gate.
        for _ in 0..3 {
            g.update(0.8);
        }
        assert!(g.is_speech());

        // 2 silence chunks — not enough.
        assert_eq!(g.update(0.1), None);
        assert_eq!(g.update(0.1), None);
        assert!(g.is_speech());

        // 3rd silence chunk triggers SpeechEnd.
        assert_eq!(g.update(0.1), Some(VadTransition::SpeechEnd));
        assert!(!g.is_speech());
    }

    #[test]
    fn hysteresis_prevents_chatter() {
        let mut g = gate();

        // Open the gate.
        for _ in 0..3 {
            g.update(0.8);
        }
        assert!(g.is_speech());

        // Probabilities between threshold_off (0.35) and threshold_on (0.5).
        // While in speech state, threshold is threshold_off (0.35).
        // 0.4 >= 0.35, so these count as speech — gate stays open.
        for _ in 0..10 {
            assert_eq!(g.update(0.4), None);
        }
        assert!(g.is_speech());
    }

    #[test]
    fn hysteresis_prevents_premature_open() {
        let mut g = gate();

        // While in silence state, threshold is threshold_on (0.5).
        // 0.4 < 0.5, so these don't count as speech — gate stays closed.
        for _ in 0..10 {
            assert_eq!(g.update(0.4), None);
        }
        assert!(!g.is_speech());
    }

    #[test]
    fn reset_returns_to_silence() {
        let mut g = gate();

        // Open the gate.
        for _ in 0..3 {
            g.update(0.8);
        }
        assert!(g.is_speech());

        g.reset();
        assert!(!g.is_speech());

        // Must accumulate min_speech_chunks again to reopen.
        assert_eq!(g.update(0.8), None);
        assert!(!g.is_speech());
    }

    #[test]
    fn interrupted_speech_resets_counter() {
        let mut g = gate();

        // 2 speech chunks, then 1 silence — resets speech counter.
        g.update(0.8);
        g.update(0.8);
        g.update(0.1); // resets speech_count to 0
        assert!(!g.is_speech());

        // Need 3 fresh speech chunks to open.
        g.update(0.8);
        g.update(0.8);
        assert!(!g.is_speech());
        assert_eq!(g.update(0.8), Some(VadTransition::SpeechStart));
    }

    #[test]
    fn default_config_values() {
        let config = VadConfig::default();
        assert_eq!(config.threshold_on, 0.5);
        assert_eq!(config.threshold_off, 0.35);
        assert_eq!(config.min_speech_chunks, 3);
        assert_eq!(config.min_silence_chunks, 20);
    }
}
