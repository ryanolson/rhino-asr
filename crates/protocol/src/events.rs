use serde::{Deserialize, Serialize};

/// Wire events streamed from server to client via velo-streaming.
///
/// Self-correcting stream semantics:
///   - `Interim`  — speculative, replaced by next Interim or Commit
///   - `Commit`   — stable words, append to client buffer
///   - `Retract`  — pop N words from client buffer, then a Commit follows
///   - `EndOfUtterance` — VAD silence boundary
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AsrEvent {
    /// Stable words — append to committed buffer.
    Commit { text: String },

    /// Walk back `count` words from committed buffer.
    /// A corrected `Commit` will follow.
    Retract { count: usize },

    /// Speculative hypothesis for unconfirmed suffix.
    /// Display but mark as tentative.
    Interim { text: String },

    /// End of utterance (VAD silence boundary).
    EndOfUtterance,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_json_roundtrip() {
        let events = vec![
            AsrEvent::Commit {
                text: "hello world".into(),
            },
            AsrEvent::Retract { count: 2 },
            AsrEvent::Interim {
                text: "testing".into(),
            },
            AsrEvent::EndOfUtterance,
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let back: AsrEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, event);
        }
    }

    #[test]
    fn serde_msgpack_roundtrip() {
        let events = vec![
            AsrEvent::Commit {
                text: "hello world".into(),
            },
            AsrEvent::Retract { count: 2 },
            AsrEvent::Interim {
                text: "testing".into(),
            },
            AsrEvent::EndOfUtterance,
        ];

        for event in &events {
            let packed = rmp_serde::to_vec(event).unwrap();
            let back: AsrEvent = rmp_serde::from_slice(&packed).unwrap();
            assert_eq!(&back, event);
        }
    }

    #[test]
    fn json_tag_format() {
        let commit = AsrEvent::Commit {
            text: "hello".into(),
        };
        let json: serde_json::Value = serde_json::to_value(&commit).unwrap();
        assert_eq!(json["type"], "commit");
        assert_eq!(json["text"], "hello");
    }
}
