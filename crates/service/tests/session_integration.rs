use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use uuid::Uuid;
use velo::*;
use velo_transports::tcp::{TcpTransport, TcpTransportBuilder};

use rhino_backend::mock::MockBackend;
use rhino_backend::{AsrBackend, AudioData, WordToken};
use rhino_engine::AgreementConfig;
use rhino_protocol::{
    AsrEvent, AudioChunk, CreateSessionRequest, CreateSessionResponse, DestroySessionRequest,
    DestroySessionResponse, SessionConfig,
};
use rhino_service::{PipelineConfig, SessionManager, register_handlers};

fn new_transport() -> Arc<TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

fn token(word: &str, start: f32, end: f32) -> WordToken {
    WordToken {
        word: word.to_string(),
        start,
        end,
    }
}

fn test_pipeline_config() -> PipelineConfig {
    PipelineConfig {
        step_secs: 0.0,
        max_buffer_secs: 30.0,
        min_chunk_secs: 0.0,
    }
}

fn test_engine_config() -> AgreementConfig {
    AgreementConfig {
        min_agreement: 2,
        commit_lookahead_secs: 0.5,
    }
}

/// Build a server and client Velo pair connected over TCP loopback.
async fn make_pair() -> (Arc<Velo>, Arc<Velo>) {
    let server = Velo::builder()
        .add_transport(new_transport())
        .stream_config(StreamConfig::Tcp(None))
        .unwrap()
        .build()
        .await
        .unwrap();

    let client = Velo::builder()
        .add_transport(new_transport())
        .stream_config(StreamConfig::Tcp(None))
        .unwrap()
        .build()
        .await
        .unwrap();

    client.register_peer(server.peer_info()).unwrap();
    server.register_peer(client.peer_info()).unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    (server, client)
}

/// Collect events from an anchor until Finalized, with a timeout.
async fn collect_events(anchor: &mut StreamAnchor<AsrEvent>) -> Vec<AsrEvent> {
    let mut events = Vec::new();
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(5), anchor.next()).await;
        match frame {
            Ok(Some(Ok(StreamFrame::Item(event)))) => events.push(event),
            Ok(Some(Ok(StreamFrame::Finalized))) => break,
            Ok(Some(Err(StreamError::SenderDropped))) => break,
            Ok(None) => break,
            Ok(other) => panic!("unexpected frame: {other:?}"),
            Err(_) => panic!("timeout waiting for events"),
        }
    }
    events
}

/// Helper: create session and return (session_id, audio_sender, event_anchor).
async fn create_test_session<B: AsrBackend + 'static>(
    server: &Arc<Velo>,
    client: &Arc<Velo>,
    _manager: &Arc<SessionManager<B>>,
) -> (Uuid, StreamSender<AudioChunk>, StreamAnchor<AsrEvent>) {
    let event_anchor = client.create_anchor::<AsrEvent>();
    let event_handle = event_anchor.handle();

    let response: CreateSessionResponse = client
        .typed_unary("create_session")
        .unwrap()
        .payload(&CreateSessionRequest {
            config: SessionConfig::default(),
            event_stream_handle: event_handle,
        })
        .unwrap()
        .instance(server.instance_id())
        .send()
        .await
        .unwrap();

    let audio_sender: StreamSender<AudioChunk> = client
        .attach_anchor(response.audio_stream_handle)
        .await
        .unwrap();

    (response.session_id, audio_sender, event_anchor)
}

fn one_sec_chunk(seq: u64) -> AudioChunk {
    AudioChunk {
        samples: vec![0.0f32; 16_000],
        sequence: seq,
    }
}

#[test]
fn session_lifecycle() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-lifecycle")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (server, client) = make_pair().await;

        let manager = Arc::new(SessionManager::new(
            Arc::new(|| {
                let mut mock = MockBackend::new();
                mock.set_default_response(vec![
                    token("hello", 0.0, 0.5),
                    token("world", 0.5, 1.0),
                ]);
                mock
            }),
            test_pipeline_config(),
            test_engine_config(),
        ));
        register_handlers(&server, &manager).unwrap();

        let (_session_id, audio_sender, mut event_anchor) =
            create_test_session(&server, &client, &manager).await;

        assert_eq!(manager.session_count(), 1);

        // Send 3 audio chunks (1s each). With LA-2, first produces Interim,
        // second produces Commit, third continues.
        for seq in 0..3u64 {
            audio_sender.send(one_sec_chunk(seq)).await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Finalize audio — triggers flush on server side.
        audio_sender.finalize().unwrap();

        // Collect all events.
        let events = collect_events(&mut event_anchor).await;

        // Verify event ordering contract:
        // 1. Interims come before any Commit (first observation, agreement < k)
        // 2. At least one Commit (agreement reaches k)
        // 3. No Retract (stable mock words don't change)
        // 4. EndOfUtterance is always the last event (natural close → flush)
        assert!(
            !events.is_empty(),
            "should produce events"
        );
        assert!(
            matches!(events.last(), Some(AsrEvent::EndOfUtterance)),
            "last event must be EndOfUtterance on natural close: {events:?}"
        );
        let first_commit_idx = events
            .iter()
            .position(|e| matches!(e, AsrEvent::Commit { .. }));
        let first_interim_idx = events
            .iter()
            .position(|e| matches!(e, AsrEvent::Interim { .. }));
        assert!(
            first_interim_idx.is_some(),
            "should have interim events: {events:?}"
        );
        assert!(
            first_commit_idx.is_some(),
            "should have commit events: {events:?}"
        );
        assert!(
            first_interim_idx.unwrap() < first_commit_idx.unwrap(),
            "first interim should precede first commit: {events:?}"
        );
        assert!(
            !events.iter().any(|e| matches!(e, AsrEvent::Retract { .. })),
            "should not retract with stable words: {events:?}"
        );

        // Session loop should have self-cleaned from the map after natural completion.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(
            manager.session_count(),
            0,
            "session should self-remove after natural completion"
        );
    });
}

#[test]
fn destroy_session_without_audio() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-destroy-no-audio")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (server, client) = make_pair().await;

        let manager = Arc::new(SessionManager::new(
            Arc::new(MockBackend::new),
            test_pipeline_config(),
            test_engine_config(),
        ));
        register_handlers(&server, &manager).unwrap();

        let (session_id, _audio_sender, mut event_anchor) =
            create_test_session(&server, &client, &manager).await;

        assert_eq!(manager.session_count(), 1);

        // Destroy session without sending any audio.
        let destroy_response: DestroySessionResponse = client
            .typed_unary("destroy_session")
            .unwrap()
            .payload(&DestroySessionRequest { session_id })
            .unwrap()
            .instance(server.instance_id())
            .send()
            .await
            .unwrap();

        assert!(destroy_response.success);
        assert_eq!(manager.session_count(), 0);

        // Destroy = abort: event stream terminates without flush/EndOfUtterance.
        let events = collect_events(&mut event_anchor).await;
        assert!(
            !events.iter().any(|e| matches!(e, AsrEvent::EndOfUtterance)),
            "destroy must not emit EndOfUtterance: {events:?}"
        );
    });
}

#[test]
fn destroy_during_active_audio() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-destroy-active")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (server, client) = make_pair().await;

        let manager = Arc::new(SessionManager::new(
            Arc::new(|| {
                let mut mock = MockBackend::new();
                mock.set_default_response(vec![
                    token("hello", 0.0, 0.5),
                    token("world", 0.5, 1.0),
                ]);
                mock
            }),
            test_pipeline_config(),
            test_engine_config(),
        ));
        register_handlers(&server, &manager).unwrap();

        let (session_id, audio_sender, mut event_anchor) =
            create_test_session(&server, &client, &manager).await;

        assert_eq!(manager.session_count(), 1);

        // Send audio to trigger pipeline processing.
        for seq in 0..3u64 {
            audio_sender.send(one_sec_chunk(seq)).await.unwrap();
        }

        // Small delay to ensure at least one chunk enters spawn_compute.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Destroy while audio has been sent (may be in-flight or between computes).
        let destroy_response: DestroySessionResponse = client
            .typed_unary("destroy_session")
            .unwrap()
            .payload(&DestroySessionRequest { session_id })
            .unwrap()
            .instance(server.instance_id())
            .send()
            .await
            .unwrap();

        assert!(destroy_response.success);
        assert_eq!(manager.session_count(), 0);

        // Destroy = abort: may contain partial events from chunks processed
        // before cancellation, but must NOT have EndOfUtterance (no flush).
        let events = collect_events(&mut event_anchor).await;
        assert!(
            !events.iter().any(|e| matches!(e, AsrEvent::EndOfUtterance)),
            "destroy must not emit EndOfUtterance even with in-flight audio: {events:?}"
        );
    });
}

/// A backend that blocks for a configurable duration on each transcribe call.
/// Used to test destroy timeout behavior with in-flight compute.
struct SlowBackend {
    delay: Duration,
}

impl AsrBackend for SlowBackend {
    fn transcribe(&mut self, _audio: &impl AudioData) -> anyhow::Result<Vec<WordToken>> {
        std::thread::sleep(self.delay);
        Ok(vec![token("slow", 0.0, 0.5)])
    }

    fn reset(&mut self) {}
}

#[test]
fn destroy_times_out_with_slow_backend() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-slow-destroy")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (server, client) = make_pair().await;

        // Backend blocks for 8s per transcribe — longer than DESTROY_TIMEOUT (5s).
        let manager = Arc::new(SessionManager::new(
            Arc::new(|| SlowBackend {
                delay: Duration::from_secs(8),
            }),
            PipelineConfig {
                step_secs: 0.0,
                max_buffer_secs: 30.0,
                min_chunk_secs: 0.0,
            },
            test_engine_config(),
        ));
        register_handlers(&server, &manager).unwrap();

        let (session_id, audio_sender, event_anchor) =
            create_test_session(&server, &client, &manager).await;

        // Send one chunk to put the session into a long spawn_compute.
        audio_sender.send(one_sec_chunk(0)).await.unwrap();

        // Wait for the chunk to enter the pipeline (start blocking on rayon).
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Destroy should hit the 5s timeout and detach rather than waiting 8s.
        let start = tokio::time::Instant::now();
        let destroy_response: DestroySessionResponse = client
            .typed_unary("destroy_session")
            .unwrap()
            .payload(&DestroySessionRequest { session_id })
            .unwrap()
            .instance(server.instance_id())
            .send()
            .await
            .unwrap();
        let elapsed = start.elapsed();

        assert!(destroy_response.success);
        // Should complete around 5s (timeout), not 8s (full compute).
        // Allow some margin but verify it didn't wait for the full backend delay.
        assert!(
            elapsed < Duration::from_secs(7),
            "destroy should timeout, not wait for full compute: {elapsed:?}"
        );

        // Session count should be 0 — removed from map before waiting.
        assert_eq!(manager.session_count(), 0);

        // Drop the event anchor — the detached session task hasn't finalized
        // the event sender yet (still in slow compute), so we can't collect
        // events. Runtime teardown will cancel the detached task.
        drop(event_anchor);
    });
}
