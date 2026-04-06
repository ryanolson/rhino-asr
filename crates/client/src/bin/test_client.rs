use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;

use rhino_client::AsrClient;
use rhino_protocol::{AsrEvent, SessionConfig, TextBuffer};

#[derive(Parser)]
#[command(name = "rhino-test-client", about = "Test client for streaming ASR")]
struct Args {
    /// Path to WAV audio file
    #[arg(long)]
    file: PathBuf,

    /// Server connection file (PeerInfo JSON)
    #[arg(long, default_value = "/tmp/rhino-server.json")]
    connect: PathBuf,

    /// Language hint
    #[arg(long, default_value = "en")]
    language: String,

    /// Send audio as fast as possible (no real-time pacing)
    #[arg(long)]
    no_pace: bool,
}

/// Read a WAV file into mono f32 samples + sample rate.
fn read_wav(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to read WAV samples")?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("failed to read WAV samples")?,
    };

    // Mix to mono if stereo.
    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    if mono.is_empty() {
        bail!("WAV file is empty");
    }

    tracing::info!(
        sample_rate,
        channels,
        duration_secs = mono.len() as f32 / sample_rate as f32,
        "loaded WAV file"
    );

    Ok((mono, sample_rate))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    // Read audio file.
    let (samples, sample_rate) = read_wav(&args.file)?;

    // Connect to server.
    let client = AsrClient::connect(&args.connect).await?;
    tracing::info!("connected to server");

    // Create session.
    let config = SessionConfig {
        language: Some(args.language),
        sample_rate,
    };
    let mut session = client.session(config).await?;
    let session_id = session.id;
    let events = session
        .take_event_stream()
        .expect("client-created session always has event stream");
    let audio = session.audio;
    tracing::info!(%session_id, "session created");

    // Chunk size at input sample rate matching ~100ms.
    let chunk_size = (sample_rate as usize) / 10;
    let is_tty = std::io::stdout().is_terminal();

    // Sender task: push audio through the library.
    let sender_task = tokio::spawn(async move {
        let mut audio = audio;
        for chunk in samples.chunks(chunk_size) {
            audio.send(chunk).await?;
            if !args.no_pace {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
        audio.finalize().await?;
        tracing::info!("audio stream finalized");
        Ok::<(), anyhow::Error>(())
    });

    // Receiver task: display events.
    let receiver_task = tokio::spawn(async move {
        let mut events = events;
        let mut buffer = TextBuffer::default();
        while let Some(event) = events.next().await {
            buffer.apply(&event);
            if is_tty {
                match &event {
                    AsrEvent::EndOfUtterance => {
                        print!("\r{}\x1b[K\n", buffer.display());
                    }
                    _ => {
                        print!("\r{}\x1b[K", buffer.display());
                    }
                }
                use std::io::Write;
                let _ = std::io::stdout().flush();
            } else {
                // Non-TTY: print committed text on commit events.
                if matches!(event, AsrEvent::Commit { .. } | AsrEvent::EndOfUtterance) {
                    println!("{}", buffer.display());
                }
            }
        }
        // Final newline if needed.
        if is_tty {
            println!();
        }
    });

    tokio::select! {
        result = sender_task => {
            result??;
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("ctrl-c, destroying session");
            let _ = client.destroy_session(session_id).await;
        }
    }

    // Wait for receiver to finish (server finalizes events after audio ends).
    receiver_task.await?;

    Ok(())
}
