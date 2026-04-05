use serde::{Deserialize, Serialize};
use uuid::Uuid;
use velo_streaming::StreamAnchorHandle;

/// Session configuration sent by client when creating a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Language hint (e.g. "en"). None = auto-detect.
    pub language: Option<String>,

    /// Client's native sample rate in Hz.
    /// Server resamples if != 16000.
    pub sample_rate: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            language: Some("en".into()),
            sample_rate: 16_000,
        }
    }
}

/// Request to create a new ASR session.
///
/// Sent via velo-messenger typed unary to the "create_session" handler.
/// Client creates a `StreamAnchor<AsrEvent>` and passes its handle here
/// so the server can attach as sender.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    pub config: SessionConfig,
    pub event_stream_handle: StreamAnchorHandle,
}

/// Response from "create_session" handler.
///
/// Server creates a `StreamAnchor<AudioChunk>` and returns its handle
/// so the client can attach as sender.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session_id: Uuid,
    pub audio_stream_handle: StreamAnchorHandle,
}

/// Request to destroy an ASR session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestroySessionRequest {
    pub session_id: Uuid,
}

/// Response from "destroy_session" handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestroySessionResponse {
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.language, Some("en".into()));
        assert_eq!(config.sample_rate, 16_000);
    }
}
