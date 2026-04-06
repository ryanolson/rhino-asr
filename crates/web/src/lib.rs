mod ws;

use axum::{
    Json,
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use rhino_client::AsrClient;
use serde::Serialize;
use std::path::Path;
use tower_http::services::ServeDir;

#[derive(Clone)]
pub struct AppState {
    pub client: AsrClient,
    pub model_info: String,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| ws::handle_socket(socket, state.client))
}

#[derive(Serialize)]
struct InfoResponse {
    model: String,
}

async fn info_handler(State(state): State<AppState>) -> Json<InfoResponse> {
    Json(InfoResponse {
        model: state.model_info.clone(),
    })
}

pub fn build_router(state: AppState, static_dir: &Path) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/info", get(info_handler))
        .fallback_service(ServeDir::new(static_dir))
        .with_state(state)
}
