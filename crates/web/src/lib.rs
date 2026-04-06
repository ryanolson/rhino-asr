mod ws;

use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use rhino_client::AsrClient;
use std::path::Path;
use tower_http::services::ServeDir;

#[derive(Clone)]
pub struct AppState {
    pub client: AsrClient,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| ws::handle_socket(socket, state.client))
}

pub fn build_router(state: AppState, static_dir: &Path) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new(static_dir))
        .with_state(state)
}
