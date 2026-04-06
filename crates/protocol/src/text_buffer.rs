use crate::events::AsrEvent;

/// Client-side text buffer that applies ASR events to maintain
/// a committed + interim view of the transcription.
///
/// Portable logic — can be reimplemented in JS for browser clients.
#[derive(Default, Debug, Clone)]
pub struct TextBuffer {
    committed: Vec<String>,
    interim: Option<String>,
}

impl TextBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply an ASR event to update buffer state.
    pub fn apply(&mut self, event: &AsrEvent) {
        match event {
            AsrEvent::Commit { text } => {
                self.committed
                    .extend(text.split_whitespace().map(str::to_owned));
                self.interim = None;
            }
            AsrEvent::Retract { count } => {
                let len = self.committed.len();
                self.committed.truncate(len.saturating_sub(*count));
                self.interim = None;
            }
            AsrEvent::Interim { text } => {
                self.interim = Some(text.clone());
            }
            AsrEvent::EndOfUtterance => {
                self.interim = None;
            }
        }
    }

    /// Committed words as a slice.
    pub fn committed_words(&self) -> &[String] {
        &self.committed
    }

    /// Current interim hypothesis, if any.
    pub fn interim(&self) -> Option<&str> {
        self.interim.as_deref()
    }

    /// Render display string: committed text + [interim] suffix.
    pub fn display(&self) -> String {
        let committed = self.committed.join(" ");
        match &self.interim {
            Some(i) if committed.is_empty() => format!("[{i}]"),
            Some(i) => format!("{committed} [{i}]"),
            None => committed,
        }
    }

    /// Reset all state.
    pub fn clear(&mut self) {
        self.committed.clear();
        self.interim = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_appends_words() {
        let mut buf = TextBuffer::new();
        buf.apply(&AsrEvent::Commit {
            text: "hello world".into(),
        });
        assert_eq!(buf.committed_words(), &["hello", "world"]);
        assert_eq!(buf.display(), "hello world");
    }

    #[test]
    fn retract_then_commit_corrects() {
        let mut buf = TextBuffer::new();
        buf.apply(&AsrEvent::Commit {
            text: "hello world".into(),
        });
        buf.apply(&AsrEvent::Retract { count: 1 });
        assert_eq!(buf.committed_words(), &["hello"]);

        buf.apply(&AsrEvent::Commit {
            text: "there".into(),
        });
        assert_eq!(buf.display(), "hello there");
    }

    #[test]
    fn interim_shown_in_display() {
        let mut buf = TextBuffer::new();
        buf.apply(&AsrEvent::Commit {
            text: "hello".into(),
        });
        buf.apply(&AsrEvent::Interim {
            text: "world maybe".into(),
        });
        assert_eq!(buf.display(), "hello [world maybe]");
    }

    #[test]
    fn interim_cleared_on_commit() {
        let mut buf = TextBuffer::new();
        buf.apply(&AsrEvent::Interim {
            text: "speculative".into(),
        });
        assert!(buf.interim().is_some());

        buf.apply(&AsrEvent::Commit {
            text: "actual".into(),
        });
        assert!(buf.interim().is_none());
    }

    #[test]
    fn end_of_utterance_clears_interim() {
        let mut buf = TextBuffer::new();
        buf.apply(&AsrEvent::Interim {
            text: "partial".into(),
        });
        buf.apply(&AsrEvent::EndOfUtterance);
        assert!(buf.interim().is_none());
        assert_eq!(buf.display(), "");
    }

    #[test]
    fn interim_only_no_leading_space() {
        let mut buf = TextBuffer::new();
        buf.apply(&AsrEvent::Interim {
            text: "thinking".into(),
        });
        assert_eq!(buf.display(), "[thinking]");
    }

    #[test]
    fn retract_saturates_at_zero() {
        let mut buf = TextBuffer::new();
        buf.apply(&AsrEvent::Commit {
            text: "hi".into(),
        });
        buf.apply(&AsrEvent::Retract { count: 100 });
        assert!(buf.committed_words().is_empty());
    }
}
