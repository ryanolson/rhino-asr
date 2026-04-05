/// A word hypothesis from the ASR backend with timing information.
#[derive(Debug, Clone)]
pub struct WordHypothesis {
    pub word: String,
    /// End time in seconds from the start of the current audio buffer.
    pub end_time: f32,
}

/// Events produced by the agreement engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    /// Newly confirmed words — append to committed buffer.
    Commit(String),
    /// Retract `n` previously committed words.
    Retract(usize),
    /// Current unconfirmed hypothesis.
    Interim(String),
}

/// Configuration for the LocalAgreement-k algorithm.
#[derive(Debug, Clone)]
pub struct AgreementConfig {
    /// Number of consecutive agreements required to confirm a word (k).
    /// Standard value is 2.
    pub min_agreement: usize,

    /// Words within this many seconds of buffer end are NOT committed.
    /// They're too close to the edge and likely to change.
    pub commit_lookahead_secs: f32,
}

impl Default for AgreementConfig {
    fn default() -> Self {
        Self {
            min_agreement: 2,
            commit_lookahead_secs: 1.0,
        }
    }
}

/// LocalAgreement-k streaming engine.
///
/// Algorithm (Macháček et al., 2023 — "Turning Whisper into Real-Time"):
///
/// Each inference step over the growing audio buffer produces a hypothesis H_t.
/// A word at position i is "confirmed" if it agrees across `min_agreement`
/// consecutive hypotheses. Confirmed words that aren't too close to the
/// buffer edge (lookahead) are committed.
///
/// Self-correction: if the confirmed prefix diverges from previously
/// committed words, Retract(n) + Commit(corrected) events are emitted.
pub struct AgreementEngine {
    config: AgreementConfig,
    prev_hypothesis: Vec<WordHypothesis>,
    agreement: Vec<usize>,
    committed: Vec<String>,
}

impl AgreementEngine {
    pub fn new(config: AgreementConfig) -> Self {
        Self {
            config,
            prev_hypothesis: Vec::new(),
            agreement: Vec::new(),
            committed: Vec::new(),
        }
    }

    /// Feed a new hypothesis from the ASR backend.
    ///
    /// `buffer_duration` is the current audio buffer length in seconds,
    /// used to compute the lookahead cutoff.
    ///
    /// Returns events to emit to the client.
    pub fn push_hypothesis(
        &mut self,
        words: Vec<WordHypothesis>,
        buffer_duration: f32,
    ) -> Vec<EngineEvent> {
        // Resize agreement counters to match new hypothesis length.
        self.agreement.resize(words.len(), 0);

        // Compare new hypothesis against previous, position by position.
        for i in 0..words.len() {
            if i < self.prev_hypothesis.len()
                && words_agree(&words[i], &self.prev_hypothesis[i])
            {
                self.agreement[i] += 1;
            } else {
                // Position diverges — reset everything from here onwards.
                for j in i..words.len() {
                    self.agreement[j] = 0;
                }
                break;
            }
        }

        // Truncate if new hypothesis is shorter.
        if words.len() < self.prev_hypothesis.len() {
            self.agreement.truncate(words.len());
        }

        self.prev_hypothesis = words.clone();

        // Determine confirmed prefix: agreement >= k AND end_time within safe zone.
        let commit_cutoff = buffer_duration - self.config.commit_lookahead_secs;
        let confirmed_end = self
            .agreement
            .iter()
            .zip(words.iter())
            .take_while(|(count, word)| {
                **count >= self.config.min_agreement && word.end_time < commit_cutoff
            })
            .count();

        let confirmed_words: Vec<&str> = words[..confirmed_end]
            .iter()
            .map(|w| w.word.as_str())
            .collect();

        // Diff confirmed against committed → Retract/Commit events.
        let mut events = self.reconcile_commit(&confirmed_words);

        // Emit interim for the unconfirmed suffix.
        if confirmed_end < words.len() {
            let interim: String = words[confirmed_end..]
                .iter()
                .map(|w| w.word.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            if !interim.is_empty() {
                events.push(EngineEvent::Interim(interim));
            }
        }

        events
    }

    /// Commit all remaining words regardless of agreement.
    /// Call at utterance boundary (VAD silence).
    pub fn flush(&mut self) -> Vec<EngineEvent> {
        let all_words: Vec<String> = self
            .prev_hypothesis
            .iter()
            .map(|w| w.word.clone())
            .collect();
        let refs: Vec<&str> = all_words.iter().map(|s| s.as_str()).collect();
        let events = self.reconcile_commit(&refs);
        self.prev_hypothesis.clear();
        self.agreement.clear();
        events
    }

    /// Reset all state for a new utterance.
    pub fn reset(&mut self) {
        self.prev_hypothesis.clear();
        self.agreement.clear();
        self.committed.clear();
    }

    /// Currently committed words.
    pub fn committed_text(&self) -> &[String] {
        &self.committed
    }

    /// Compute Commit/Retract events to bring `self.committed` in line with `target`.
    fn reconcile_commit(&mut self, target: &[&str]) -> Vec<EngineEvent> {
        // Find common prefix between committed and target.
        let common = self
            .committed
            .iter()
            .zip(target.iter())
            .take_while(|(a, b)| a.as_str() == **b)
            .count();

        let mut events = Vec::new();

        // Retract divergent words.
        let retract_n = self.committed.len() - common;
        if retract_n > 0 {
            events.push(EngineEvent::Retract(retract_n));
            self.committed.truncate(common);
        }

        // Commit new words.
        if target.len() > common {
            let new_words: Vec<String> = target[common..].iter().map(|s| s.to_string()).collect();
            let text = new_words.join(" ");
            self.committed.extend(new_words);
            events.push(EngineEvent::Commit(text));
        }

        events
    }
}

/// Two word hypotheses "agree" if they have the same normalized text
/// and similar timestamps (within 200ms).
fn words_agree(a: &WordHypothesis, b: &WordHypothesis) -> bool {
    normalize(&a.word) == normalize(&b.word) && (a.end_time - b.end_time).abs() < 0.2
}

fn normalize(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hyp(words: &[(&str, f32)]) -> Vec<WordHypothesis> {
        words
            .iter()
            .map(|(w, t)| WordHypothesis {
                word: w.to_string(),
                end_time: *t,
            })
            .collect()
    }

    #[test]
    fn stable_words_committed_after_k_agreements() {
        let mut engine = AgreementEngine::new(AgreementConfig {
            min_agreement: 2,
            commit_lookahead_secs: 1.0,
        });

        // First hypothesis — nothing committed yet (agreement = 0).
        let events = engine.push_hypothesis(
            hyp(&[("hello", 0.5), ("world", 1.0), ("test", 2.5)]),
            4.0,
        );
        // Only interim since no agreement yet.
        assert!(events.iter().all(|e| matches!(e, EngineEvent::Interim(_))));

        // Second hypothesis — same words → agreement = 1 (still < k=2).
        let events = engine.push_hypothesis(
            hyp(&[("hello", 0.5), ("world", 1.0), ("test", 2.5)]),
            4.0,
        );
        // agreement hits 1 for all, but k=2 so still not committed.
        assert!(events.iter().all(|e| matches!(e, EngineEvent::Interim(_))));

        // Third hypothesis — agreement = 2 = k, commit words within lookahead.
        // buffer_duration=4.0, lookahead=1.0, cutoff=3.0
        // "hello" end=0.5 < 3.0 ✓, "world" end=1.0 < 3.0 ✓, "test" end=2.5 < 3.0 ✓
        let events = engine.push_hypothesis(
            hyp(&[("hello", 0.5), ("world", 1.0), ("test", 2.5)]),
            4.0,
        );
        assert!(events.contains(&EngineEvent::Commit("hello world test".into())));
        assert_eq!(engine.committed_text(), &["hello", "world", "test"]);
    }

    #[test]
    fn lookahead_prevents_edge_commits() {
        let mut engine = AgreementEngine::new(AgreementConfig {
            min_agreement: 2,
            commit_lookahead_secs: 1.0,
        });

        // Three identical hypotheses → agreement = 2.
        // buffer_duration=2.0, cutoff=1.0
        // "hello" end=0.5 < 1.0 ✓, "world" end=1.5 >= 1.0 ✗
        for _ in 0..3 {
            engine.push_hypothesis(
                hyp(&[("hello", 0.5), ("world", 1.5)]),
                2.0,
            );
        }

        assert_eq!(engine.committed_text(), &["hello"]);
    }

    #[test]
    fn retract_on_correction() {
        let mut engine = AgreementEngine::new(AgreementConfig {
            min_agreement: 2,
            commit_lookahead_secs: 0.5,
        });

        // Build up agreement and commit "hello world".
        for _ in 0..3 {
            engine.push_hypothesis(
                hyp(&[("hello", 0.5), ("world", 1.0)]),
                5.0,
            );
        }
        assert_eq!(engine.committed_text(), &["hello", "world"]);

        // Now whisper changes its mind — "world" becomes "there".
        // The confirmed prefix shrinks to just ["hello"], so "world" is
        // immediately retracted even though "there" isn't committed yet.
        let events = engine.push_hypothesis(
            hyp(&[("hello", 0.5), ("there", 1.0)]),
            5.0,
        );
        assert!(events.contains(&EngineEvent::Retract(1)));
        assert_eq!(engine.committed_text(), &["hello"]);

        // Second push with "there" — agreement[1] = 1, not enough yet.
        engine.push_hypothesis(
            hyp(&[("hello", 0.5), ("there", 1.0)]),
            5.0,
        );
        assert_eq!(engine.committed_text(), &["hello"]);

        // Third push — agreement[1] = 2 = k, "there" gets committed.
        let events = engine.push_hypothesis(
            hyp(&[("hello", 0.5), ("there", 1.0)]),
            5.0,
        );
        assert!(events.contains(&EngineEvent::Commit("there".into())));
        assert_eq!(engine.committed_text(), &["hello", "there"]);
    }

    #[test]
    fn flush_commits_everything() {
        let mut engine = AgreementEngine::new(AgreementConfig::default());

        // Single hypothesis, no agreement yet.
        engine.push_hypothesis(
            hyp(&[("hello", 0.5), ("world", 1.0)]),
            5.0,
        );

        let events = engine.flush();
        assert!(events.contains(&EngineEvent::Commit("hello world".into())));
        assert_eq!(engine.committed_text(), &["hello", "world"]);
    }

    #[test]
    fn reset_clears_state() {
        let mut engine = AgreementEngine::new(AgreementConfig::default());

        for _ in 0..3 {
            engine.push_hypothesis(
                hyp(&[("hello", 0.5)]),
                5.0,
            );
        }
        assert!(!engine.committed_text().is_empty());

        engine.reset();
        assert!(engine.committed_text().is_empty());
    }

    #[test]
    fn words_agree_normalizes() {
        let a = WordHypothesis {
            word: "Hello!".into(),
            end_time: 1.0,
        };
        let b = WordHypothesis {
            word: "hello".into(),
            end_time: 1.1,
        };
        assert!(words_agree(&a, &b));
    }

    #[test]
    fn words_disagree_on_timestamp() {
        let a = WordHypothesis {
            word: "hello".into(),
            end_time: 1.0,
        };
        let b = WordHypothesis {
            word: "hello".into(),
            end_time: 1.5,
        };
        assert!(!words_agree(&a, &b));
    }

    #[test]
    fn empty_hypothesis() {
        let mut engine = AgreementEngine::new(AgreementConfig::default());
        let events = engine.push_hypothesis(vec![], 1.0);
        assert!(events.is_empty());
    }
}
