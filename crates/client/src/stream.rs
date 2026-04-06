use futures::StreamExt;
use velo::{StreamAnchor, StreamError, StreamFrame};

use rhino_protocol::AsrEvent;

/// Receives ASR events from the server. Wraps velo streaming internals.
pub struct EventStream {
    inner: StreamAnchor<AsrEvent>,
}

impl EventStream {
    pub(crate) fn new(anchor: StreamAnchor<AsrEvent>) -> Self {
        Self { inner: anchor }
    }

    /// Returns the next ASR event, or `None` when the stream ends
    /// (server finalized, sender dropped, or transport error).
    pub async fn next(&mut self) -> Option<AsrEvent> {
        loop {
            match self.inner.next().await? {
                Ok(StreamFrame::Item(event)) => return Some(event),
                Ok(StreamFrame::Finalized) => return None,
                Err(StreamError::SenderDropped) => return None,
                Err(e) => {
                    tracing::warn!("event stream error: {e}");
                    return None;
                }
                _ => continue,
            }
        }
    }
}
