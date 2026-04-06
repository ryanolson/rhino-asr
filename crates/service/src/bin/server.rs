use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use velo::{StreamConfig, Velo};
use velo_transports::tcp::TcpTransportBuilder;

use rhino_backend::MockBackend;
use rhino_engine::AgreementConfig;
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

        let manager = Arc::new(SessionManager::new(
            Arc::new(MockBackend::new),
            PipelineConfig::default(),
            AgreementConfig::default(),
        ));

        register_handlers(&velo, &manager)?;

        tokio::signal::ctrl_c().await?;
        tracing::info!("shutting down");

        // Clean up connection file.
        let _ = std::fs::remove_file(&args.connect_file);

        Ok(())
    })
}
