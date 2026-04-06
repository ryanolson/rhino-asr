use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use velo::*;
use velo_transports::tcp::TcpTransportBuilder;

use rhino_backend::mock::MockBackend;
use rhino_backend::WordToken;
use rhino_engine::AgreementConfig;
use rhino_protocol::{AsrEvent, SessionConfig};
use rhino_service::{PipelineConfig, SessionManager, register_handlers};

use rhino_client::AsrClient;

fn new_transport() -> Arc<velo_transports::tcp::TcpTransport> {
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

/// Collect events from a StreamAnchor until Finalized/Dropped, with a timeout.
async fn collect_events_from_anchor(anchor: &mut StreamAnchor<AsrEvent>) -> Vec<AsrEvent> {
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

#[test]
fn client_library_session_lifecycle() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-client-lib")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (server, client_velo) = make_pair().await;

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

        let client = AsrClient::from_velo(client_velo, server.instance_id());

        let mut session = client
            .session(SessionConfig {
                language: Some("en".to_string()),
                sample_rate: 16_000,
            })
            .await
            .unwrap();

        let mut events = session.take_event_stream().unwrap();
        let mut audio = session.audio;

        assert_eq!(manager.session_count(), 1);

        for _ in 0..3u64 {
            audio.send(&vec![0.0f32; 16_000]).await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        audio.finalize().await.unwrap();

        let mut all_events = Vec::new();
        let timeout = tokio::time::timeout(Duration::from_secs(5), async {
            while let Some(event) = events.next().await {
                all_events.push(event);
            }
        });
        timeout.await.expect("timed out waiting for events");

        // Verify ordering contract (same as service-level test):
        // Interims before first Commit, no Retract, EndOfUtterance last.
        assert!(
            !all_events.is_empty(),
            "should produce events"
        );
        assert!(
            matches!(all_events.last(), Some(AsrEvent::EndOfUtterance)),
            "last event must be EndOfUtterance on natural close: {all_events:?}"
        );
        let first_commit = all_events
            .iter()
            .position(|e| matches!(e, AsrEvent::Commit { .. }));
        let first_interim = all_events
            .iter()
            .position(|e| matches!(e, AsrEvent::Interim { .. }));
        assert!(first_interim.is_some(), "should have interim: {all_events:?}");
        assert!(first_commit.is_some(), "should have commit: {all_events:?}");
        assert!(
            first_interim.unwrap() < first_commit.unwrap(),
            "interim before commit: {all_events:?}"
        );
        assert!(
            !all_events.iter().any(|e| matches!(e, AsrEvent::Retract { .. })),
            "no retract with stable words: {all_events:?}"
        );

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(manager.session_count(), 0);
    });
}

#[test]
fn client_library_with_resampling() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-client-resample")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (server, client_velo) = make_pair().await;

        let manager = Arc::new(SessionManager::new(
            Arc::new(|| {
                let mut mock = MockBackend::new();
                mock.set_default_response(vec![
                    token("resampled", 0.0, 0.5),
                    token("audio", 0.5, 1.0),
                ]);
                mock
            }),
            test_pipeline_config(),
            test_engine_config(),
        ));
        register_handlers(&server, &manager).unwrap();

        let client = AsrClient::from_velo(client_velo, server.instance_id());

        let mut session = client
            .session(SessionConfig {
                language: Some("en".to_string()),
                sample_rate: 48_000,
            })
            .await
            .unwrap();

        let mut events = session.take_event_stream().unwrap();
        let mut audio = session.audio;

        for _ in 0..3u64 {
            audio.send(&vec![0.0f32; 48_000]).await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        audio.finalize().await.unwrap();

        let mut all_events = Vec::new();
        let timeout = tokio::time::timeout(Duration::from_secs(5), async {
            while let Some(event) = events.next().await {
                all_events.push(event);
            }
        });
        timeout.await.expect("timed out waiting for events");

        assert!(
            all_events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })),
            "should have commit events: {all_events:?}"
        );
        assert!(
            all_events.iter().any(|e| matches!(e, AsrEvent::EndOfUtterance)),
            "should have EndOfUtterance: {all_events:?}"
        );
    });
}

#[test]
fn client_library_destroy_session() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-client-destroy")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (server, client_velo) = make_pair().await;

        let manager = Arc::new(SessionManager::new(
            Arc::new(MockBackend::new),
            test_pipeline_config(),
            test_engine_config(),
        ));
        register_handlers(&server, &manager).unwrap();

        let client = AsrClient::from_velo(client_velo, server.instance_id());

        let mut session = client.session(SessionConfig::default()).await.unwrap();

        let session_id = session.id;
        let mut events = session.take_event_stream().unwrap();
        assert_eq!(manager.session_count(), 1);

        // Drop audio sender (simulates client disconnect from audio side).
        drop(session.audio);

        // Explicit destroy — should abort without flush.
        client.destroy_session(session_id).await.unwrap();
        assert_eq!(manager.session_count(), 0);

        // Collect events — destroy means no EndOfUtterance.
        let mut all_events = Vec::new();
        let timeout = tokio::time::timeout(Duration::from_secs(2), async {
            while let Some(event) = events.next().await {
                all_events.push(event);
            }
        });
        timeout.await.expect("event stream should terminate promptly");
        assert!(
            !all_events.iter().any(|e| matches!(e, AsrEvent::EndOfUtterance)),
            "destroy must not emit EndOfUtterance: {all_events:?}"
        );
    });
}

/// Test the external event stream handle path: the caller creates the anchor
/// elsewhere, passes the handle via the builder, and receives events directly.
#[test]
fn external_event_stream_handle() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-ext-handle")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (server, client_velo) = make_pair().await;

        let manager = Arc::new(SessionManager::new(
            Arc::new(|| {
                let mut mock = MockBackend::new();
                mock.set_default_response(vec![
                    token("external", 0.0, 0.5),
                    token("handle", 0.5, 1.0),
                ]);
                mock
            }),
            test_pipeline_config(),
            test_engine_config(),
        ));
        register_handlers(&server, &manager).unwrap();

        let client = AsrClient::from_velo(client_velo.clone(), server.instance_id());

        // Create the event anchor externally (e.g. on the same or different Velo instance).
        let mut external_anchor = client_velo.create_anchor::<AsrEvent>();
        let external_handle = external_anchor.handle();

        // Use the builder with an external event stream handle.
        let mut session = client
            .session_builder(SessionConfig {
                language: Some("en".to_string()),
                sample_rate: 16_000,
            })
            .event_stream_handle(external_handle)
            .build()
            .await
            .unwrap();

        // take_event_stream should return None — we provided an external handle.
        assert!(
            session.take_event_stream().is_none(),
            "should be None when external handle is provided"
        );

        let mut audio = session.audio;

        for _ in 0..3u64 {
            audio.send(&vec![0.0f32; 16_000]).await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        audio.finalize().await.unwrap();

        // Events should arrive on the external anchor, not through the client.
        let events = collect_events_from_anchor(&mut external_anchor).await;

        assert!(
            events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })),
            "external anchor should receive commit events: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e, AsrEvent::EndOfUtterance)),
            "external anchor should receive EndOfUtterance: {events:?}"
        );
    });
}
