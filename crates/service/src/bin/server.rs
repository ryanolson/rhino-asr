use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use velo::{StreamConfig, Velo};
use velo_transports::tcp::TcpTransportBuilder;


use rhino_service::{PipelineConfig, SessionManager, register_handlers};

#[derive(Parser)]
#[command(name = "rhino-server", about = "Streaming ASR server")]
struct Args {
    /// Address to bind to
    #[arg(long, default_value = "0.0.0.0:0")]
    bind: SocketAddr,

    /// Path to write connection file (PeerInfo JSON for clients)
    #[arg(long, default_value = "/tmp/rhino-server.json")]
    connect_file: PathBuf,

    /// Path to whisper GGML model. If omitted, uses MockBackend.
    #[arg(long)]
    model_path: Option<String>,

    /// Language code for transcription (e.g. "en")
    #[arg(long, default_value = "en")]
    language: String,

    /// GPU device index (0-based)
    #[arg(long, default_value = "0")]
    gpu_device: i32,

    /// Disable GPU (CPU-only inference)
    #[arg(long)]
    no_gpu: bool,

    /// Path to Silero VAD ONNX model. If omitted, VAD is disabled.
    #[arg(long)]
    vad_model_path: Option<String>,

    /// Beam search width (1 = greedy)
    #[arg(long, default_value = "5")]
    beam_size: i32,

    /// Enable diagnostic one-shot transcription on each session flush
    /// (captures audio to /tmp and compares against streaming output)
    #[arg(long)]
    diagnostic: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("rhino")
        .tokio_threads(2)
        .rayon_threads(6)
        .build()?;

    runtime.block_on(async {
        let listener = std::net::TcpListener::bind(args.bind)?;
        let addr = listener.local_addr()?;
        let transport = Arc::new(
            TcpTransportBuilder::new()
                .from_listener(listener)?
                .build()?,
        );

        let velo = Velo::builder()
            .add_transport(transport)
            .stream_config(StreamConfig::Tcp(None))?
            .build()
            .await?;

        // Write connection file for clients.
        let peer_info_json = serde_json::to_string_pretty(&velo.peer_info())?;
        std::fs::write(&args.connect_file, &peer_info_json)?;
        tracing::info!(
            %addr,
            instance_id = %velo.instance_id(),
            connect_file = %args.connect_file.display(),
            "server listening"
        );

        // Build VAD factory if model path is provided.
        let vad_factory = match &args.vad_model_path {
            #[cfg(feature = "silero")]
            Some(path) => {
                let vad_path = path.clone();
                let factory: rhino_service::VadFactory = Arc::new(move || {
                    let vad = rhino_vad::silero::SileroVad::new(&vad_path)?;
                    Ok(Box::new(vad) as Box<dyn rhino_vad::VadProcessor>)
                });
                tracing::info!(vad_model = %path, "Silero VAD enabled");
                Some(factory)
            }
            #[cfg(not(feature = "silero"))]
            Some(_) => {
                anyhow::bail!("--vad-model-path requires the 'silero' feature. Rebuild with: cargo build --features silero");
            }
            None => {
                tracing::info!("VAD disabled (no --vad-model-path)");
                None
            }
        };

        match &args.model_path {
            #[cfg(feature = "whisper")]
            Some(model_path) => {
                use rhino_backend::whisper::{WhisperBackend, WhisperConfig, load_whisper_context};

                let ctx = Arc::new(load_whisper_context(
                    model_path,
                    !args.no_gpu,
                    args.gpu_device,
                )?);
                tracing::info!(
                    model = %model_path,
                    gpu = !args.no_gpu,
                    gpu_device = args.gpu_device,
                    "whisper model loaded"
                );

                let config = WhisperConfig {
                    language: Some(args.language.clone()),
                    beam_size: args.beam_size,
                    ..WhisperConfig::default()
                };

                let ctx_clone = Arc::clone(&ctx);
                let manager = Arc::new({
                    let mut m = SessionManager::new(
                        Arc::new(move || {
                            WhisperBackend::new(Arc::clone(&ctx_clone), config.clone())
                                .expect("failed to create whisper backend")
                        }),
                        PipelineConfig::default(),
                    )
                    .with_diagnostic(args.diagnostic);
                    if let Some(factory) = vad_factory {
                        m = m.with_vad(factory, rhino_vad::VadConfig::default());
                    }
                    m
                });

                register_handlers(&velo, &manager)?;
            }
            #[cfg(not(feature = "whisper"))]
            Some(_) => {
                anyhow::bail!("--model-path requires the 'whisper' feature. Rebuild with: cargo build --features whisper");
            }
            None => {
                use rhino_backend::MockBackend;
                tracing::info!("no --model-path, using MockBackend");

                let manager = Arc::new({
                    let mut m = SessionManager::new(
                        Arc::new(MockBackend::new),
                        PipelineConfig::default(),
                    )
                    .with_diagnostic(args.diagnostic);
                    if let Some(factory) = vad_factory {
                        m = m.with_vad(factory, rhino_vad::VadConfig::default());
                    }
                    m
                });

                register_handlers(&velo, &manager)?;
            }
        }

        tokio::signal::ctrl_c().await?;
        tracing::info!("shutting down");

        // Clean up connection file.
        let _ = std::fs::remove_file(&args.connect_file);

        Ok(())
    })
}
