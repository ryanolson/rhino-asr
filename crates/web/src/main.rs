use anyhow::Result;
use clap::Parser;
use rhino_client::AsrClient;
use rhino_web::{AppState, build_router};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "rhino-web", about = "Web frontend for Dynamo Whisper ASR")]
struct Args {
    /// Path to ASR server PeerInfo JSON
    #[arg(long, default_value = "/tmp/rhino-server.json")]
    connect_file: PathBuf,

    /// Address to bind HTTP server
    #[arg(long, default_value = "0.0.0.0:3000")]
    bind: String,

    /// Path to static files directory
    #[arg(long, default_value = "crates/web/static")]
    static_dir: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    info!(connect_file = %args.connect_file.display(), "connecting to ASR server");
    let client = AsrClient::connect(&args.connect_file).await?;
    info!("connected to ASR server");

    let state = AppState { client };
    let app = build_router(state, &args.static_dir);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    info!(addr = %args.bind, "serving web UI");
    axum::serve(listener, app).await?;

    Ok(())
}
