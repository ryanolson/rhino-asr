use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use velo::{Handler, StreamAnchor, StreamFrame, StreamSender, TypedContext, Velo};

use rhino_backend::AsrBackend;
use rhino_engine::AgreementConfig;
use rhino_protocol::{
    AsrEvent, AudioChunk, CreateSessionRequest, CreateSessionResponse, DestroySessionRequest,
    DestroySessionResponse,
};

use crate::pipeline::{AsrPipeline, PipelineConfig};

/// How long `destroy_session` waits for the session task before detaching it.
/// In-flight `spawn_compute` (Whisper inference) cannot be cancelled, so this
/// bounds the worst-case destroy latency. The detached task will self-remove
/// from the session map when it eventually completes.
const DESTROY_TIMEOUT: Duration = Duration::from_secs(5);

/// Handle to a running session, stored in SessionManager's DashMap.
struct SessionHandle {
    cancel: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}

/// Manages the lifecycle of all active ASR sessions.
pub struct SessionManager<B: AsrBackend> {
    sessions: Arc<DashMap<Uuid, SessionHandle>>,
    backend_factory: Arc<dyn Fn() -> B + Send + Sync>,
    pipeline_config: PipelineConfig,
    engine_config: AgreementConfig,
}

impl<B: AsrBackend + 'static> SessionManager<B> {
    pub fn new(
        backend_factory: Arc<dyn Fn() -> B + Send + Sync>,
        pipeline_config: PipelineConfig,
        engine_config: AgreementConfig,
    ) -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            backend_factory,
            pipeline_config,
            engine_config,
        }
    }

    /// Create a new ASR session.
    ///
    /// Creates an audio anchor for receiving audio, attaches as sender to the
    /// client's event anchor, spawns the session loop, and returns handles.
    ///
    /// `request.config` is accepted for forward compatibility but not yet applied:
    /// `language` requires WhisperBackend (Phase 6), `sample_rate` requires
    /// resampling (Phase 5). The values are logged for observability.
    pub async fn create_session(
        &self,
        velo: &Arc<Velo>,
        request: CreateSessionRequest,
    ) -> Result<CreateSessionResponse> {
        let audio_anchor = velo.create_anchor::<AudioChunk>();
        let audio_handle = audio_anchor.handle();

        let event_sender = velo
            .attach_anchor::<AsrEvent>(request.event_stream_handle)
            .await
            .map_err(|e| anyhow::anyhow!("failed to attach event sender: {e}"))?;

        let pipeline = AsrPipeline::new(
            (self.backend_factory)(),
            self.engine_config.clone(),
            self.pipeline_config.clone(),
        );

        let cancel = CancellationToken::new();
        let session_id = Uuid::new_v4();

        tracing::info!(
            %session_id,
            language = ?request.config.language,
            sample_rate = request.config.sample_rate,
            "session created (config accepted but not yet applied)",
        );

        let task = tokio::spawn(session_loop(
            session_id,
            Arc::clone(&self.sessions),
            audio_anchor,
            event_sender,
            pipeline,
            cancel.clone(),
        ));

        self.sessions.insert(
            session_id,
            SessionHandle { cancel, task },
        );

        Ok(CreateSessionResponse {
            session_id,
            audio_stream_handle: audio_handle,
        })
    }

    /// Destroy an active session. Cancels the pipeline and waits for cleanup.
    ///
    /// Idempotent: returns success if the session already completed naturally
    /// (the session loop self-removes on exit). If in-flight inference blocks
    /// the task beyond `DESTROY_TIMEOUT`, the task is detached and will
    /// self-remove when it eventually completes.
    pub async fn destroy_session(&self, session_id: Uuid) -> Result<DestroySessionResponse> {
        if let Some((_, handle)) = self.sessions.remove(&session_id) {
            handle.cancel.cancel();

            match tokio::time::timeout(DESTROY_TIMEOUT, handle.task).await {
                Ok(_) => {
                    tracing::info!(%session_id, "session destroyed");
                }
                Err(_) => {
                    tracing::warn!(
                        %session_id,
                        timeout_secs = DESTROY_TIMEOUT.as_secs(),
                        "session destroy timed out waiting for in-flight compute; task detached"
                    );
                }
            }
        } else {
            tracing::debug!(%session_id, "session already completed, nothing to destroy");
        }

        Ok(DestroySessionResponse { success: true })
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

/// Register create_session and destroy_session handlers on a Velo instance.
pub fn register_handlers<B: AsrBackend + 'static>(
    velo: &Arc<Velo>,
    manager: &Arc<SessionManager<B>>,
) -> Result<()> {
    // create_session handler
    let v = Arc::clone(velo);
    let m = Arc::clone(manager);
    let create = Handler::typed_unary_async(
        "create_session",
        move |ctx: TypedContext<CreateSessionRequest>| {
            let v = Arc::clone(&v);
            let m = Arc::clone(&m);
            async move { m.create_session(&v, ctx.input).await }
        },
    )
    .spawn()
    .build();
    velo.register_handler(create)?;

    // destroy_session handler
    let m = Arc::clone(manager);
    let destroy = Handler::typed_unary_async(
        "destroy_session",
        move |ctx: TypedContext<DestroySessionRequest>| {
            let m = Arc::clone(&m);
            async move { m.destroy_session(ctx.input.session_id).await }
        },
    )
    .spawn()
    .build();
    velo.register_handler(destroy)?;

    Ok(())
}

/// Per-session async loop: reads audio, runs pipeline via spawn_compute, sends events.
///
/// On exit (any path), the session removes itself from the session map and
/// finalizes the event stream. This ensures natural completion (audio finalized,
/// sender dropped) cleans up without requiring an explicit `destroy_session` call.
async fn session_loop<B: AsrBackend + 'static>(
    session_id: Uuid,
    sessions: Arc<DashMap<Uuid, SessionHandle>>,
    mut audio_anchor: StreamAnchor<AudioChunk>,
    event_sender: StreamSender<AsrEvent>,
    mut pipeline: AsrPipeline<B>,
    cancel: CancellationToken,
) {
    tracing::info!(%session_id, "session loop started");

    let mut should_flush = true;

    'session: loop {
        let frame = tokio::select! {
            _ = cancel.cancelled() => {
                // Explicit destroy — abort without flushing pending text.
                tracing::info!(%session_id, "session cancelled");
                should_flush = false;
                break 'session;
            }
            frame = audio_anchor.next() => {
                match frame {
                    Some(f) => f,
                    None => {
                        tracing::info!(%session_id, "audio stream channel closed");
                        break 'session;
                    }
                }
            }
        };

        match frame {
            Ok(StreamFrame::Item(chunk)) => {
                let samples = chunk.samples;

                // Offload sync pipeline work to rayon via loom-rs.
                // Pipeline moves in, computes, moves back out.
                let (p, result) = loom_rs::spawn_compute(move || {
                    let events = pipeline.push_audio(&samples);
                    (pipeline, events)
                })
                .await;
                pipeline = p;

                match result {
                    Ok(events) => {
                        for event in events {
                            if event_sender.send(event).await.is_err() {
                                tracing::error!(%session_id, "event sender failed");
                                break 'session;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(%session_id, %e, "pipeline error, continuing");
                    }
                }
            }
            Ok(StreamFrame::Finalized) => {
                tracing::info!(%session_id, "audio stream finalized");
                break 'session;
            }
            Err(velo::StreamError::SenderDropped) => {
                tracing::warn!(%session_id, "audio sender dropped");
                break 'session;
            }
            Err(e) => {
                tracing::warn!(%session_id, %e, "audio stream error, continuing");
            }
            _ => {}
        }
    }

    // Flush only on natural close (audio finalized, sender dropped, channel closed).
    // Explicit destroy (cancel) skips flush — destroy means abort.
    if should_flush {
        let (_pipeline, flush_result) = loom_rs::spawn_compute(move || {
            let events = pipeline.flush_utterance();
            (pipeline, events)
        })
        .await;

        if let Ok(events) = flush_result {
            for event in events {
                if event_sender.send(event).await.is_err() {
                    break;
                }
            }
        }
    }

    if let Err(e) = event_sender.finalize() {
        tracing::warn!(%session_id, %e, "event sender finalize failed");
    }

    // Self-remove from session map. May be None if destroy_session already removed us.
    sessions.remove(&session_id);

    tracing::info!(%session_id, "session loop ended");
}
