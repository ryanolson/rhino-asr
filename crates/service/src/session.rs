use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dashmap::DashMap;
use futures::StreamExt;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
use velo::{Handler, StreamAnchor, StreamFrame, StreamSender, TypedContext, Velo};

use rhino_protocol::{
    AsrEvent, AudioChunk, CreateSessionRequest, CreateSessionResponse, DestroySessionRequest,
    DestroySessionResponse,
};
use rhino_vad::{VadConfig, VadGate, VadProcessor, VadTransition};

use crate::pipeline::AsrPipeline;

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

/// Bundles all sync state that moves in/out of `spawn_compute`.
struct PipelineState {
    pipeline: Box<dyn AsrPipeline>,
    vad: Option<Box<dyn VadProcessor>>,
    gate: VadGate,
    vad_buffer: Vec<f32>,
    /// Captures ALL audio (pre-VAD) for diagnostic one-shot evaluation.
    capture_buffer: Vec<f32>,
    /// Enable diagnostic one-shot comparison on flush.
    diagnostic: bool,
}

impl PipelineState {
    /// Process an audio chunk through VAD gate + pipeline.
    ///
    /// When VAD is present, audio is chunked into VAD-sized pieces and gated
    /// through the hysteresis state machine. Only speech audio reaches the
    /// pipeline buffer. Transcription happens on flush (SpeechEnd) or when
    /// the buffer exceeds max duration (long utterance split).
    ///
    /// When VAD is None, all audio buffers directly in the pipeline.
    fn process_audio(&mut self, samples: &[f32]) -> Result<Vec<AsrEvent>> {
        if self.diagnostic {
            self.capture_buffer.extend_from_slice(samples);
        }

        let Some(ref mut vad) = self.vad else {
            // No VAD — buffer everything directly.
            self.pipeline.push_audio(samples)?;
            return Ok(vec![]);
        };

        let chunk_size = vad.chunk_size();
        let mut events = Vec::new();

        // Append to VAD buffer for chunking.
        self.vad_buffer.extend_from_slice(samples);

        // Process complete VAD-sized chunks.
        while self.vad_buffer.len() >= chunk_size {
            let chunk: Vec<f32> = self.vad_buffer.drain(..chunk_size).collect();

            let prob = vad.process_chunk(&chunk)?;
            tracing::trace!(prob = format!("{prob:.3}"), is_speech = self.gate.is_speech(), "VAD chunk");

            if let Some(transition) = self.gate.update(prob) {
                match transition {
                    VadTransition::SpeechStart => {
                        tracing::info!("VAD: speech start");
                    }
                    VadTransition::SpeechEnd => {
                        tracing::info!("VAD: speech end, flushing utterance");
                        events.extend(self.pipeline.flush_utterance()?);
                    }
                }
            }

            if self.gate.is_speech() {
                // push_audio may return events (streaming mode: LA-2 Commit/Retract/Interim).
                events.extend(self.pipeline.push_audio(&chunk)?);

                // Safety net: if buffer exceeds max, flush chunk (no EndOfUtterance).
                if self.pipeline.buffer_full() {
                    tracing::info!(
                        buf_secs = format!("{:.1}", self.pipeline.buffer_duration_secs()),
                        "long utterance split"
                    );
                    events.extend(self.pipeline.flush_chunk()?);
                }
            }
        }

        Ok(events)
    }

    /// Flush any remaining audio and finalize the current utterance.
    fn flush(&mut self) -> Result<Vec<AsrEvent>> {
        // Process any leftover audio in the VAD buffer.
        if self.vad.is_some() && !self.vad_buffer.is_empty() {
            let chunk_size = self.vad.as_ref().unwrap().chunk_size();
            let remaining = std::mem::take(&mut self.vad_buffer);
            if remaining.len() < chunk_size {
                let mut padded = remaining;
                padded.resize(chunk_size, 0.0);
                let _ = self.vad.as_mut().unwrap().process_chunk(&padded);
            }
        }

        let events = self.pipeline.flush_utterance()?;

        if self.diagnostic {
            self.diagnostic_oneshot();
        }

        Ok(events)
    }

    /// Write all captured audio to a WAV file and run a single-shot
    /// transcription for quality comparison against streaming output.
    fn diagnostic_oneshot(&mut self) {
        let samples = std::mem::take(&mut self.capture_buffer);
        if samples.is_empty() {
            return;
        }

        let duration_secs = samples.len() as f32 / 16_000.0;
        tracing::info!(
            samples = samples.len(),
            duration_secs = format!("{duration_secs:.2}"),
            "diagnostic: captured audio"
        );

        let wav_path = format!("/tmp/rhino-capture-{}.wav", std::process::id());
        match write_wav_f32(&wav_path, &samples, 16_000) {
            Ok(()) => tracing::info!(path = %wav_path, "diagnostic: wrote capture WAV"),
            Err(e) => {
                tracing::warn!("diagnostic: failed to write WAV: {e}");
                return;
            }
        }

        match self.pipeline.transcribe_raw(&samples) {
            Ok(tokens) => {
                let text: String = tokens.iter().map(|t| t.word.as_str()).collect::<Vec<_>>().join(" ");
                tracing::info!(
                    words = tokens.len(),
                    text = %text,
                    "diagnostic: ONE-SHOT transcription"
                );
            }
            Err(e) => {
                tracing::warn!("diagnostic: one-shot transcription failed: {e}");
            }
        }
    }
}

/// Factory type for creating VAD processors per session.
pub type VadFactory = Arc<dyn Fn() -> Result<Box<dyn VadProcessor>> + Send + Sync>;

/// Factory type for creating pipeline instances per session.
pub type PipelineFactory = Arc<dyn Fn() -> Box<dyn AsrPipeline> + Send + Sync>;

/// Manages the lifecycle of all active ASR sessions.
pub struct SessionManager {
    sessions: Arc<DashMap<Uuid, SessionHandle>>,
    pipeline_factory: PipelineFactory,
    vad_factory: Option<VadFactory>,
    vad_config: VadConfig,
    diagnostic: bool,
}

impl SessionManager {
    pub fn new(pipeline_factory: PipelineFactory) -> Self {
        Self {
            sessions: Arc::new(DashMap::new()),
            pipeline_factory,
            vad_factory: None,
            vad_config: VadConfig::default(),
            diagnostic: false,
        }
    }

    /// Set a VAD factory. When set, each session creates a VAD processor
    /// that gates audio before it reaches the pipeline.
    pub fn with_vad(mut self, factory: VadFactory, config: VadConfig) -> Self {
        self.vad_factory = Some(factory);
        self.vad_config = config;
        self
    }

    /// Enable diagnostic one-shot comparison (captures all audio, runs
    /// full-buffer transcription on flush for quality comparison).
    pub fn with_diagnostic(mut self, enabled: bool) -> Self {
        self.diagnostic = enabled;
        self
    }

    /// Create a new ASR session.
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

        let pipeline = (self.pipeline_factory)();

        let vad = match &self.vad_factory {
            Some(factory) => Some(factory()?),
            None => None,
        };

        let state = PipelineState {
            pipeline,
            vad,
            gate: VadGate::new(self.vad_config.clone()),
            vad_buffer: Vec::new(),
            capture_buffer: Vec::new(),
            diagnostic: self.diagnostic,
        };

        let cancel = CancellationToken::new();
        let session_id = Uuid::new_v4();

        tracing::info!(
            %session_id,
            language = ?request.config.language,
            sample_rate = request.config.sample_rate,
            vad_enabled = self.vad_factory.is_some(),
            "session created (config accepted but not yet applied)",
        );

        let task = tokio::spawn(session_loop(
            session_id,
            Arc::clone(&self.sessions),
            audio_anchor,
            event_sender,
            state,
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
pub fn register_handlers(
    velo: &Arc<Velo>,
    manager: &Arc<SessionManager>,
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
async fn session_loop(
    session_id: Uuid,
    sessions: Arc<DashMap<Uuid, SessionHandle>>,
    mut audio_anchor: StreamAnchor<AudioChunk>,
    event_sender: StreamSender<AsrEvent>,
    mut state: PipelineState,
    cancel: CancellationToken,
) {
    tracing::info!(%session_id, "session loop started");

    let mut should_flush = true;

    'session: loop {
        let frame = tokio::select! {
            _ = cancel.cancelled() => {
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
                let rms = if samples.is_empty() {
                    0.0
                } else {
                    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
                };
                tracing::trace!(%session_id, samples = samples.len(), rms = format!("{rms:.6}"), "audio chunk received");

                let (s, result) = loom_rs::spawn_compute(move || {
                    let events = state.process_audio(&samples);
                    (state, events)
                })
                .await;
                state = s;

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

    if should_flush {
        let (_state, flush_result) = loom_rs::spawn_compute(move || {
            let events = state.flush();
            (state, events)
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

    sessions.remove(&session_id);

    tracing::info!(%session_id, "session loop ended");
}

/// Write f32 PCM samples as a 16-bit WAV file.
fn write_wav_f32(path: &str, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    use std::io::Write;
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2;
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        f.write_all(&i16_val.to_le_bytes())?;
    }
    Ok(())
}
