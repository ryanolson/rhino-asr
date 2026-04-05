# Dynamo Whisper — Streaming ASR System

## Context

Build a streaming ASR (Automatic Speech Recognition) system in Rust that runs whisper.cpp via whisper-rs, uses Silero VAD for speech detection, and implements LocalAgreement-2 for real-time word confirmation. The system integrates with the velo distributed infrastructure: **velo-streaming** for bidirectional data transport (audio PCM in, ASR events out), **velo-messenger** for session control (create/destroy/configure).

## Architecture

### Crate Map

```
dynamo-whisper/
  crates/
    protocol/       — Wire types: AsrEvent, AudioChunk, control messages, TextBuffer
    engine/         — Pure LocalAgreement-2 algorithm (no transport deps)
    vad/            — Silero VAD wrapper + VadGate hysteresis (ONNX via ort)
    backend/        — AsrBackend trait + WhisperBackend + MockBackend
    service/        — Pipeline (VAD+engine+backend) + velo session management
    client/         — CLI binary: mic/file capture, streams audio, displays events
```

### Session Lifecycle (Dual Streams + Messenger)

Each session uses **two velo-streaming channels** and **messenger control messages**:

```
                  velo-messenger (typed unary)
Client ─────────────────────────────────────────── Server
  │  CreateSessionRequest{config, event_handle}  →   │
  │  ← CreateSessionResponse{session_id, audio_handle}│
  │                                                   │
  │  velo-streaming: AudioChunk                       │
  │  StreamSender<AudioChunk> ──────────────────→ StreamAnchor<AudioChunk>
  │                                                   │
  │  velo-streaming: AsrEvent                         │
  │  StreamAnchor<AsrEvent> ←────────────────── StreamSender<AsrEvent>
  │                                                   │
  │  DestroySessionRequest{session_id}           →   │
  │  ← DestroySessionResponse                        │
```

**Setup flow:**
1. Both client and server build `Velo` instances (each gets messenger + anchor_manager + streaming handlers)
2. Client creates `StreamAnchor<AsrEvent>` locally, gets `event_handle`
3. Client calls `velo.typed_unary("create_session")` with `{config, event_handle}` → server
4. Server creates `StreamAnchor<AudioChunk>` locally, gets `audio_handle`
5. Server attaches as sender to client's event anchor: `velo.attach_anchor::<AsrEvent>(event_handle)`
6. Server spawns pipeline task, returns `{session_id, audio_handle}`
7. Client attaches as sender to server's audio anchor: `velo.attach_anchor::<AudioChunk>(audio_handle)`
8. Client streams audio; server streams events

**Teardown:** Client calls `destroy_session` → server cancels pipeline, finalizes event sender → client finalizes audio sender.

### Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Audio transport | velo-streaming (not AM) | Backpressure, ordering, heartbeat monitoring, user requirement |
| Engine purity | Zero transport deps | Pure state-machine tests, no mocks needed |
| Service structure | Library + thin binary | Embeddable in dynamo runtime later |
| VAD location | Server-side only | Simpler client, server controls thresholds |

---

## Phase 1: Foundation — COMPLETE

**Goal:** Workspace skeleton + protocol types + streaming engine.

**What was built:**
- 6-crate workspace (`protocol`, `engine`, `vad`, `backend`, `service`, `client`)
- **Protocol crate** (`rhino-protocol`): `AsrEvent` (Commit/Retract/Interim/EndOfUtterance), `AudioChunk`, control messages (`CreateSessionRequest/Response`, `DestroySessionRequest/Response`, `SessionConfig`), `TextBuffer` client helper. 11 tests (serde roundtrips, text buffer logic).
- **Engine crate** (`rhino-engine`): `AgreementEngine` implementing LocalAgreement-2. `WordHypothesis`, `EngineEvent`, `AgreementConfig`. 8 tests (agreement, lookahead, retract on correction, flush, reset, word normalization).
- Remaining crates stubbed.

**Key files:**
- `crates/protocol/src/{lib,events,audio,control,text_buffer}.rs`
- `crates/engine/src/lib.rs`

---

## Phase 2: Audio Processing — PENDING

**Goal:** VAD and ASR backend crates with trait abstractions and mocks.

### 2a. Backend Crate (`crates/backend`)

```rust
pub trait AsrBackend: Send + Sync {
    fn transcribe(&self, pcm_16k_mono: &[f32]) -> anyhow::Result<Vec<WordToken>>;
    fn reset(&mut self);
}

pub struct WordToken { pub word: String, pub start: f32, pub end: f32 }

// Mock — always available
pub struct MockBackend { responses: VecDeque<Vec<WordToken>> }

// Real — behind feature "whisper"
#[cfg(feature = "whisper")]
pub struct WhisperBackend { ... }
```

`transcribe()` is sync — caller wraps in `spawn_blocking`. Keeps the trait simple.

### 2b. VAD Crate (`crates/vad`)

```rust
pub struct SileroVad { ... }  // ONNX inference, 512-sample chunks
pub struct VadGate { ... }    // Pure hysteresis state machine
pub enum VadTransition { SpeechStart, SpeechEnd }
```

`VadGate` separated from `SileroVad` — testable without ONNX model.

### Verification
- `cargo test -p rhino-backend` (mock tests)
- `cargo test -p rhino-vad` (VadGate unit tests with synthetic probabilities)

---

## Phase 3: Pipeline — PENDING

**Goal:** Core ASR pipeline wiring backend + engine. Testable with mocks, no transport.

```rust
pub struct AsrPipeline<B: AsrBackend> {
    backend: B,
    engine: AgreementEngine,
    audio_buf: Vec<f32>,
    trim_offset: usize,
}

impl<B: AsrBackend> AsrPipeline<B> {
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<Vec<AsrEvent>>;
    pub fn flush_utterance(&mut self) -> Vec<AsrEvent>;
    pub fn reset(&mut self);
}
```

Pipeline appends audio → runs transcribe when buffer is large enough → feeds engine → maps EngineEvent to AsrEvent → trims buffer. VAD runs in the session layer (Phase 4), not inside the pipeline.

### Verification
- Unit tests with `MockBackend`
- `cargo test -p rhino-service`

---

## Phase 4: Velo Integration — PENDING

**Goal:** Session manager with messenger handlers + streaming channels. Deployable server.

- `SessionManager` with `DashMap<Uuid, SessionHandle>`
- `"create_session"` typed unary handler — creates audio anchor, attaches event sender, spawns pipeline task
- `"destroy_session"` typed unary handler — cancels pipeline, finalizes streams
- Per-session async loop: reads audio anchor → VAD gate → pipeline → sends events
- Thin `main.rs` binary wrapper

### Verification
- Integration test: two Velo instances over TCP loopback with MockBackend
- `cargo test -p rhino-service`

---

## Phase 5: Client — PENDING

**Goal:** CLI binary streaming from mic or file, displaying real-time transcription.

- `--input mic|file`, `--file <path>`, `--server <addr>`, `--language <lang>`
- Builds Velo instance, creates event anchor, calls create_session, attaches audio sender
- Two tasks: audio sender (cpal/file + rubato resample) + event receiver (TextBuffer display)
- No client-side VAD

### Verification
- Manual: server with mock backend + client with file input
- Automated: integration test in-process

---

## Phase 6: End-to-End & Polish — PENDING

**Goal:** Real models, latency profiling, error recovery, documentation.

- End-to-end with Whisper small.en + Silero VAD
- Latency: audio chunk → committed word event
- Error recovery: client disconnect, server restart
- README with build/run instructions

---

## Living Document Protocol

- Status transitions: PENDING → IN PROGRESS → COMPLETE
- Completed phases are updated to reflect what was actually built
- Deviations from plan must be discussed with the developer before updating
- End of each session: review completed work against phase description, flag drift
- This document is never modified without developer discussion
