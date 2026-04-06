use std::sync::Arc;

use anyhow::Result;
use velo::{StreamConfig, Velo};
use velo_transports::tcp::TcpTransportBuilder;

use rhino_backend::MockBackend;
use rhino_engine::AgreementConfig;
use rhino_service::{PipelineConfig, SessionManager, register_handlers};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let runtime = loom_rs::LoomBuilder::new()
        .prefix("rhino")
        .tokio_threads(2)
        .rayon_threads(6)
        .build()?;

    runtime.block_on(async {
        let listener = std::net::TcpListener::bind("0.0.0.0:0")?;
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

        tracing::info!(%addr, instance_id = %velo.instance_id(), "server listening");

        let manager = Arc::new(SessionManager::new(
            Arc::new(MockBackend::new),
            PipelineConfig::default(),
            AgreementConfig::default(),
        ));

        register_handlers(&velo, &manager)?;

        tokio::signal::ctrl_c().await?;
        tracing::info!("shutting down");

        Ok(())
    })
}
