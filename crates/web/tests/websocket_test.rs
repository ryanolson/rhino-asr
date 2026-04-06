use std::sync::Arc;
use std::time::Duration;

use futures::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use velo::*;
use velo_transports::tcp::TcpTransportBuilder;

use rhino_backend::mock::MockBackend;
use rhino_backend::WordToken;

use rhino_protocol::AsrEvent;
use rhino_service::{PipelineConfig, SessionManager, register_handlers};

use rhino_client::AsrClient;
use rhino_web::{AppState, build_router};

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
        max_buffer_secs: 30.0,
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

/// Start an ASR server + Axum web server, return the web server URL.
async fn start_web_server(
    backend_factory: Arc<dyn Fn() -> MockBackend + Send + Sync>,
) -> (String, Arc<SessionManager<MockBackend>>) {
    let (server_velo, client_velo) = make_pair().await;

    let manager = Arc::new(SessionManager::new(
        backend_factory,
        test_pipeline_config(),
    ));
    register_handlers(&server_velo, &manager).unwrap();

    let client = AsrClient::from_velo(client_velo, server_velo.instance_id());
    let state = AppState { client, model_info: "test".into() };

    // Use a temp dir for static files (we don't need them for WS tests)
    let static_dir = std::env::temp_dir();
    let app = build_router(state, &static_dir);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let url = format!("ws://127.0.0.1:{}/ws", addr.port());
    (url, manager)
}

fn f32_to_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// Collect all text messages as AsrEvents until the WS closes.
async fn collect_events(
    ws_rx: &mut (impl StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin),
) -> Vec<AsrEvent> {
    let mut events = Vec::new();
    while let Ok(Some(Ok(msg))) =
        tokio::time::timeout(Duration::from_secs(5), ws_rx.next()).await
    {
        match msg {
            Message::Text(text) => {
                let event: AsrEvent = serde_json::from_str(&text).unwrap();
                events.push(event);
            }
            Message::Close(_) => break,
            _ => continue,
        }
    }
    events
}

/// Full lifecycle: connect WS, send config, stream audio, end_audio, verify events.
#[test]
fn websocket_session_lifecycle() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-ws-lifecycle")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (url, manager) = start_web_server(Arc::new(|| {
            let mut mock = MockBackend::new();
            mock.set_default_response(vec![
                token("hello", 0.0, 0.5),
                token("world", 0.5, 1.0),
            ]);
            mock
        }))
        .await;

        let (ws, _) = connect_async(&url).await.unwrap();
        let (mut ws_tx, mut ws_rx) = ws.split();

        // Send config
        ws_tx
            .send(Message::Text(
                r#"{"type":"config","sample_rate":16000,"language":"en"}"#.into(),
            ))
            .await
            .unwrap();

        // Send 3 seconds of silence as f32 PCM
        for _ in 0..3 {
            let chunk = vec![0.0f32; 16_000];
            ws_tx
                .send(Message::Binary(f32_to_bytes(&chunk).into()))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Signal end of audio — keeps WS open for server to drain events
        ws_tx
            .send(Message::Text(r#"{"type":"end_audio"}"#.into()))
            .await
            .unwrap();

        let events = collect_events(&mut ws_rx).await;

        assert!(!events.is_empty(), "should produce events");
        assert!(
            matches!(events.last(), Some(AsrEvent::EndOfUtterance)),
            "last event must be EndOfUtterance: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })),
            "should have commit: {events:?}"
        );

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(manager.session_count(), 0, "session should self-clean");
    });
}

/// Verify the WS bridge correctly passes audio at non-16kHz sample rates
/// (the AudioSender resamples internally).
#[test]
fn websocket_with_48khz_audio() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-ws-48khz")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (url, _manager) = start_web_server(Arc::new(|| {
            let mut mock = MockBackend::new();
            mock.set_default_response(vec![
                token("resampled", 0.0, 0.5),
                token("audio", 0.5, 1.0),
            ]);
            mock
        }))
        .await;

        let (ws, _) = connect_async(&url).await.unwrap();
        let (mut ws_tx, mut ws_rx) = ws.split();

        // Config with 48kHz — browser-typical sample rate
        ws_tx
            .send(Message::Text(
                r#"{"type":"config","sample_rate":48000,"language":"en"}"#.into(),
            ))
            .await
            .unwrap();

        // Send 3 seconds at 48kHz
        for _ in 0..3 {
            let chunk = vec![0.0f32; 48_000];
            ws_tx
                .send(Message::Binary(f32_to_bytes(&chunk).into()))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        // Signal end of audio
        ws_tx
            .send(Message::Text(r#"{"type":"end_audio"}"#.into()))
            .await
            .unwrap();

        let events = collect_events(&mut ws_rx).await;

        assert!(
            events.iter().any(|e| matches!(e, AsrEvent::Commit { .. })),
            "should have commit events with resampled audio: {events:?}"
        );
        assert!(
            matches!(events.last(), Some(AsrEvent::EndOfUtterance)),
            "should end with EndOfUtterance: {events:?}"
        );
    });
}

/// Close the WS immediately after config (no audio) — should not panic.
#[test]
fn websocket_close_without_audio() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-ws-no-audio")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (url, manager) = start_web_server(Arc::new(MockBackend::new)).await;

        let (ws, _) = connect_async(&url).await.unwrap();
        let (mut ws_tx, mut ws_rx) = ws.split();

        ws_tx
            .send(Message::Text(
                r#"{"type":"config","sample_rate":16000,"language":"en"}"#.into(),
            ))
            .await
            .unwrap();

        // Close immediately — no audio sent
        ws_tx.close().await.unwrap();

        let events = collect_events(&mut ws_rx).await;

        // No audio means no transcription, but should not crash.
        // May or may not get EndOfUtterance depending on pipeline flush behavior.
        // The key assertion is that it doesn't hang or panic.
        let _ = events;

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(manager.session_count(), 0, "session should self-clean");
    });
}

/// Close the WS before sending config — should exit cleanly.
#[test]
fn websocket_close_before_config() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-ws-pre-config")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (url, manager) = start_web_server(Arc::new(MockBackend::new)).await;

        let (ws, _) = connect_async(&url).await.unwrap();
        let (mut ws_tx, _ws_rx) = ws.split();

        // Close without ever sending config
        ws_tx.close().await.unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(manager.session_count(), 0, "no session should be created");
    });
}

/// Start an HTTP server with the real static files directory and verify serving.
async fn start_http_server_with_static() -> String {
    let (server_velo, client_velo) = make_pair().await;

    let manager = Arc::new(SessionManager::new(
        Arc::new(MockBackend::new),
        test_pipeline_config(),
    ));
    register_handlers(&server_velo, &manager).unwrap();

    let client = AsrClient::from_velo(client_velo, server_velo.instance_id());
    let state = AppState { client, model_info: "test".into() };

    // Use the real static directory
    let static_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static");
    let app = build_router(state, &static_dir);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    format!("http://127.0.0.1:{}", addr.port())
}

/// Verify static files are served correctly.
#[test]
fn static_files_served() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-static")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let base_url = start_http_server_with_static().await;
        let http = reqwest::Client::new();

        // index.html served at root
        let resp = http.get(&format!("{base_url}/index.html")).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("Dynamo Whisper"), "index.html should contain title");

        // app.js
        let resp = http.get(&format!("{base_url}/app.js")).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("TextBuffer"), "app.js should contain TextBuffer class");

        // style.css
        let resp = http.get(&format!("{base_url}/style.css")).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("#76b900"), "style.css should contain NVIDIA green");

        // audio-worklet.js
        let resp = http.get(&format!("{base_url}/audio-worklet.js")).send().await.unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.text().await.unwrap();
        assert!(body.contains("AudioSenderProcessor"), "audio-worklet.js should contain processor");
    });
}

/// Simulate session replacement: start session A, send audio, then start session B
/// before A's WS closes. Verify B completes independently and the server handles
/// both sessions without interference.
#[test]
fn session_replacement_no_interference() {
    let _ = tracing_subscriber::fmt::try_init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("test-ws-replace")
        .pin_threads(false)
        .tokio_threads(4)
        .rayon_threads(2)
        .build()
        .unwrap();

    runtime.block_on(async {
        let (url, manager) = start_web_server(Arc::new(|| {
            let mut mock = MockBackend::new();
            mock.set_default_response(vec![
                token("hello", 0.0, 0.5),
                token("world", 0.5, 1.0),
            ]);
            mock
        }))
        .await;

        // --- Session A: start streaming, send some audio ---
        let (ws_a, _) = connect_async(&url).await.unwrap();
        let (mut tx_a, mut rx_a) = ws_a.split();

        tx_a.send(Message::Text(
            r#"{"type":"config","sample_rate":16000,"language":"en"}"#.into(),
        ))
        .await
        .unwrap();

        tx_a.send(Message::Binary(f32_to_bytes(&vec![0.0f32; 16_000]).into()))
            .await
            .unwrap();

        // Wait for session A to be fully created
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(manager.session_count(), 1);

        // --- Session B: start immediately (simulates user clicking new file) ---
        let (ws_b, _) = connect_async(&url).await.unwrap();
        let (mut tx_b, mut rx_b) = ws_b.split();

        tx_b.send(Message::Text(
            r#"{"type":"config","sample_rate":16000,"language":"en"}"#.into(),
        ))
        .await
        .unwrap();

        // Wait for session B to be fully created
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(manager.session_count(), 2);

        // --- Close session A (old session) via end_audio ---
        tx_a.send(Message::Text(r#"{"type":"end_audio"}"#.into()))
            .await
            .unwrap();

        // Drain A's events
        let _events_a = collect_events(&mut rx_a).await;

        // --- Session B: send audio and complete independently ---
        for _ in 0..3 {
            tx_b.send(Message::Binary(f32_to_bytes(&vec![0.0f32; 16_000]).into()))
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        tx_b.send(Message::Text(r#"{"type":"end_audio"}"#.into()))
            .await
            .unwrap();

        let events_b = collect_events(&mut rx_b).await;

        // Session B must complete normally with full event lifecycle
        assert!(!events_b.is_empty(), "session B should produce events");
        assert!(
            matches!(events_b.last(), Some(AsrEvent::EndOfUtterance)),
            "session B must end with EndOfUtterance: {events_b:?}"
        );
        assert!(
            events_b.iter().any(|e| matches!(e, AsrEvent::Commit { .. })),
            "session B should have commits: {events_b:?}"
        );

        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(manager.session_count(), 0, "all sessions should self-clean");
    });
}
