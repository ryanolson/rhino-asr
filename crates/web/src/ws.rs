use anyhow::{Result, bail};
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use rhino_client::AsrClient;
use rhino_protocol::SessionConfig;
use serde::Deserialize;
use tracing::{error, info, warn};

/// Typed client-to-server messages over the WebSocket.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClientMessage {
    /// Initial handshake: session configuration.
    Config {
        sample_rate: u32,
        language: Option<String>,
    },
    /// Signal that audio streaming is complete. Server drains remaining
    /// events (including EndOfUtterance) before closing the WebSocket.
    EndAudio,
}

pub async fn handle_socket(socket: WebSocket, client: AsrClient) {
    if let Err(e) = handle_socket_inner(socket, client).await {
        error!("websocket session error: {e:#}");
    }
}

async fn handle_socket_inner(socket: WebSocket, client: AsrClient) -> Result<()> {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Wait for Config message
    let config: SessionConfig = loop {
        match ws_rx.next().await {
            Some(Ok(Message::Text(text))) => {
                let msg: ClientMessage = serde_json::from_str(&text)?;
                match msg {
                    ClientMessage::Config {
                        sample_rate,
                        language,
                    } => {
                        break SessionConfig {
                            sample_rate,
                            language,
                        };
                    }
                    other => {
                        bail!("expected Config message, got {other:?}");
                    }
                }
            }
            Some(Ok(Message::Binary(_))) => {
                warn!("ignoring binary message before config");
                continue;
            }
            Some(Ok(Message::Close(_))) | None => return Ok(()),
            Some(Ok(_)) => continue,
            Some(Err(e)) => return Err(e.into()),
        }
    };

    info!(
        sample_rate = config.sample_rate,
        language = ?config.language,
        "creating ASR session"
    );

    let mut session = client.session(config).await?;
    let mut events = session
        .take_event_stream()
        .expect("event stream must be available");
    let mut audio = session.audio;

    // Audio ingestion loop: receives binary PCM and the EndAudio control message.
    let audio_loop = async {
        while let Some(msg) = ws_rx.next().await {
            match msg {
                Ok(Message::Binary(bytes)) => {
                    let samples = bytes_to_f32(&bytes);
                    if let Err(e) = audio.send(&samples).await {
                        warn!("audio send error: {e:#}");
                        break;
                    }
                }
                Ok(Message::Text(text)) => match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::EndAudio) => {
                        info!("end_audio received, finalizing");
                        if let Err(e) = audio.finalize().await {
                            warn!("audio finalize error: {e:#}");
                        }
                        return;
                    }
                    Ok(other) => {
                        warn!("unexpected message during audio: {other:?}");
                        continue;
                    }
                    Err(e) => {
                        warn!("ignoring malformed text message: {e}");
                        continue;
                    }
                },
                Ok(Message::Close(_)) => break,
                Ok(_) => continue,
                Err(e) => {
                    warn!("ws recv error: {e}");
                    break;
                }
            }
        }
        // WS closed or errored — still finalize audio for clean shutdown
        if let Err(e) = audio.finalize().await {
            warn!("audio finalize error: {e:#}");
        }
    };

    // Event forwarding loop: sends ASR events as JSON text over the WebSocket.
    let event_loop = async {
        while let Some(event) = events.next().await {
            let json = match serde_json::to_string(&event) {
                Ok(j) => j,
                Err(e) => {
                    error!("event serialize error: {e}");
                    continue;
                }
            };
            if ws_tx.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
        let _ = ws_tx.close().await;
    };

    // Both loops run to completion: audio_loop finalizes the sender,
    // then the server flushes + sends EndOfUtterance, which event_loop forwards.
    // The server closes the WS after event stream drains.
    tokio::join!(audio_loop, event_loop);

    Ok(())
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config_message() {
        let json = r#"{"type":"config","sample_rate":48000,"language":"en"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(
            msg,
            ClientMessage::Config {
                sample_rate: 48000,
                language: Some(_)
            }
        ));
    }

    #[test]
    fn parse_config_message_no_language() {
        let json = r#"{"type":"config","sample_rate":16000}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(
            msg,
            ClientMessage::Config {
                sample_rate: 16000,
                language: None
            }
        ));
    }

    #[test]
    fn parse_end_audio_message() {
        let json = r#"{"type":"end_audio"}"#;
        let msg: ClientMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, ClientMessage::EndAudio));
    }

    #[test]
    fn reject_unknown_type() {
        let json = r#"{"type":"unknown_thing"}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(result.is_err());
    }

    #[test]
    fn bytes_to_f32_roundtrip() {
        let samples: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, 42.42];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let result = bytes_to_f32(&bytes);
        assert_eq!(result, samples);
    }

    #[test]
    fn bytes_to_f32_empty() {
        assert!(bytes_to_f32(&[]).is_empty());
    }

    #[test]
    fn bytes_to_f32_truncates_remainder() {
        let mut bytes = 1.0f32.to_le_bytes().to_vec();
        bytes.push(0xFF);
        let result = bytes_to_f32(&bytes);
        assert_eq!(result, vec![1.0]);
    }
}
