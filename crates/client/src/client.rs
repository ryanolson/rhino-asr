use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use uuid::Uuid;
use velo::{InstanceId, PeerInfo, StreamAnchorHandle, StreamConfig, Velo};
use velo_transports::tcp::TcpTransportBuilder;

use rhino_protocol::{
    AsrEvent, AudioChunk, CreateSessionRequest, CreateSessionResponse, DestroySessionRequest,
    DestroySessionResponse, SessionConfig,
};

use crate::sender::AudioSender;
use crate::stream::EventStream;

/// An active ASR session with split sender and optional receiver.
///
/// The event stream is `Some` when the client created the anchor (default),
/// or `None` when an external `StreamAnchorHandle` was provided via the builder.
pub struct AsrSession {
    /// Session identifier (for destroy_session).
    pub id: Uuid,
    /// Send audio samples to the server. Handles resampling + chunking.
    pub audio: AudioSender,
    events: Option<EventStream>,
}

impl AsrSession {
    /// Take the event stream. Returns `Some` if the client owns the anchor
    /// (no external handle was provided), `None` otherwise.
    ///
    /// Can only be called once — subsequent calls return `None`.
    pub fn take_event_stream(&mut self) -> Option<EventStream> {
        self.events.take()
    }
}

/// Builder for creating ASR sessions with optional configuration.
pub struct SessionBuilder<'a> {
    client: &'a AsrClient,
    config: SessionConfig,
    event_stream_handle: Option<StreamAnchorHandle>,
}

impl<'a> SessionBuilder<'a> {
    /// Provide an external event stream handle. The server will send ASR events
    /// to this anchor instead of one created by the client.
    ///
    /// When set, `AsrSession::take_event_stream()` will return `None`.
    pub fn event_stream_handle(mut self, handle: StreamAnchorHandle) -> Self {
        self.event_stream_handle = Some(handle);
        self
    }

    /// Build the session — connects to the server and sets up audio/event streams.
    pub async fn build(self) -> Result<AsrSession> {
        let input_sample_rate = self.config.sample_rate;

        // Determine event stream handle: use provided or create local anchor.
        let (event_handle, local_event_stream) = match self.event_stream_handle {
            Some(handle) => (handle, None),
            None => {
                let anchor = self.client.velo.create_anchor::<AsrEvent>();
                let handle = anchor.handle();
                (handle, Some(EventStream::new(anchor)))
            }
        };

        // Request session from server.
        let response: CreateSessionResponse = self
            .client
            .velo
            .typed_unary("create_session")
            .context("failed to build create_session request")?
            .payload(&CreateSessionRequest {
                config: self.config,
                event_stream_handle: event_handle,
            })
            .context("failed to serialize create_session payload")?
            .instance(self.client.server_instance)
            .send()
            .await
            .context("create_session RPC failed")?;

        // Attach as sender to the server's audio anchor.
        let audio_sender = self
            .client
            .velo
            .attach_anchor::<AudioChunk>(response.audio_stream_handle)
            .await
            .context("failed to attach to audio anchor")?;

        let sender = AudioSender::new(audio_sender, input_sample_rate)
            .context("failed to create AudioSender")?;

        Ok(AsrSession {
            id: response.session_id,
            audio: sender,
            events: local_event_stream,
        })
    }
}

/// Connection to an ASR server. Cheap to clone (Arc<Velo> internally).
#[derive(Clone)]
pub struct AsrClient {
    velo: Arc<Velo>,
    server_instance: InstanceId,
}

impl AsrClient {
    /// Connect by reading the server's PeerInfo from a JSON file.
    pub async fn connect(connect_file: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(connect_file.as_ref()).with_context(|| {
            format!(
                "failed to read connect file: {}",
                connect_file.as_ref().display()
            )
        })?;
        let peer_info: PeerInfo =
            serde_json::from_str(&content).context("failed to parse PeerInfo from connect file")?;
        let server_instance = peer_info.instance_id();

        let listener = std::net::TcpListener::bind("0.0.0.0:0")
            .context("failed to bind client TCP listener")?;
        let transport = Arc::new(
            TcpTransportBuilder::new()
                .from_listener(listener)
                .context("failed to create TCP transport")?
                .build()
                .context("failed to build TCP transport")?,
        );

        let velo = Velo::builder()
            .add_transport(transport)
            .stream_config(StreamConfig::Tcp(None))
            .context("failed to configure stream")?
            .build()
            .await
            .context("failed to build Velo instance")?;

        velo.register_peer(peer_info)
            .context("failed to register server peer")?;

        // Brief delay for peer handshake (matches integration test pattern).
        tokio::time::sleep(Duration::from_millis(200)).await;

        Ok(Self {
            velo,
            server_instance,
        })
    }

    /// Construct from an existing Velo instance (for testing).
    #[doc(hidden)]
    pub fn from_velo(velo: Arc<Velo>, server_instance: InstanceId) -> Self {
        Self {
            velo,
            server_instance,
        }
    }

    /// Create a session builder with the given config.
    pub fn session_builder(&self, config: SessionConfig) -> SessionBuilder<'_> {
        SessionBuilder {
            client: self,
            config,
            event_stream_handle: None,
        }
    }

    /// Convenience: create a session with a client-owned event stream.
    ///
    /// Equivalent to `self.session_builder(config).build().await`.
    pub async fn session(&self, config: SessionConfig) -> Result<AsrSession> {
        self.session_builder(config).build().await
    }

    /// Explicitly destroy a session. Optional — finalizing the AudioSender is the
    /// happy-path shutdown (server flushes, sends EndOfUtterance, finalizes events).
    pub async fn destroy_session(&self, session_id: Uuid) -> Result<()> {
        let _response: DestroySessionResponse = self
            .velo
            .typed_unary("destroy_session")
            .context("failed to build destroy_session request")?
            .payload(&DestroySessionRequest { session_id })
            .context("failed to serialize destroy_session payload")?
            .instance(self.server_instance)
            .send()
            .await
            .context("destroy_session RPC failed")?;
        Ok(())
    }
}
