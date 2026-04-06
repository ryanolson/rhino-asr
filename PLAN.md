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

6-crate workspace with `rhino-protocol` (AsrEvent, AudioChunk, control messages, TextBuffer — 11 tests) and `rhino-engine` (LocalAgreement-2: AgreementEngine, WordHypothesis, EngineEvent — 8 tests). Remaining crates stubbed.

---

## Phase 2: Audio Processing — COMPLETE

**What was built:**
- **Backend crate** (`rhino-backend`): `AudioData` trait + `PcmAudio<'a>` (borrowed) + `PcmBuffer` (owned) newtypes. `AsrBackend` trait with `fn transcribe(&mut self, audio: &impl AudioData)`. `MockBackend` with FIFO queue + default fallback. `WordToken` with word/start/end. 4 tests.
- **VAD crate** (`rhino-vad`): `VadGate` pure hysteresis state machine with `VadConfig` (threshold_on/off, min_speech/silence_chunks). `VadTransition` (SpeechStart/SpeechEnd). 8 tests.
- WhisperBackend and SileroVad implementations deferred to Phase 6 — module structure and feature gates ready.

---

## Phase 3: Pipeline — COMPLETE

**What was built:**
- **`AsrPipeline<B: AsrBackend>`** in `crates/service/src/pipeline.rs`. Sync, generic over backend. Uses `PcmBuffer` for growing audio, `AgreementEngine` for word confirmation. `PipelineConfig` controls step/min/max intervals.
- `push_audio(&mut self, &[f32])` — appends, transcribes at intervals, maps WordToken→WordHypothesis→EngineEvent→AsrEvent, trims buffer.
- `flush_utterance()` — final transcribe + engine flush, commits all pending words.
- Buffer trimming after confirmed words keeps memory bounded.
- 7 tests: basic transcription, step interval, min chunk, buffer trimming, flush, reset, empty flush.

---

## Phase 4: Velo Integration + loom-rs — COMPLETE

**What was built:**
- **Session module** (`crates/service/src/session.rs`): `SessionManager<B: AsrBackend>` with `Arc<DashMap<Uuid, SessionHandle>>` (shared with session loops for self-cleanup), backend factory pattern (`Arc<dyn Fn() -> B + Send + Sync>`), `AgreementConfig` + `PipelineConfig` per manager.
- **Messenger handlers**: `"create_session"` (typed async — creates audio anchor, attaches event sender, spawns session task; logs config with "accepted but not yet applied") and `"destroy_session"` (typed async — idempotent, cancels via `CancellationToken`, 5s timeout on task await, detaches on timeout). Registered via `register_handlers()` free function.
- **Session loop**: async task per session. `tokio::select!` on cancel token vs audio anchor stream. Audio chunks offloaded to rayon via `loom_rs::spawn_compute` with move-in-move-out pipeline pattern. Events sent back via `StreamSender<AsrEvent>`. On exit: flush utterance, finalize event stream, self-remove from session map.
- **Server binary** (`rhino-server`): `LoomRuntime` + `Velo` with TCP transport + `StreamConfig::Tcp`, `SessionManager<MockBackend>`, ctrl-c shutdown.
- **VAD deferred** to Phase 6 — session loop passes all audio directly to pipeline (no probability source without Silero model).
- **Config handling**: `SessionConfig.language` and `sample_rate` accepted on wire and logged, but not applied until WhisperBackend (Phase 6) and resampling (Phase 5).
- **Hardening test**: `force_trim_no_retract_on_next_update` — verifies `force_trim()` does not emit `Retract` on the first post-trim update (reviewer-suggested item).
- 3 integration tests (session lifecycle with self-cleanup assertion, destroy without audio, destroy during active audio) + 1 pipeline hardening test + 1 slow-backend timeout test. Total workspace: 53 tests.

---

## Phase 5: Client Library + Test Binary — COMPLETE

**What was built:**
- **Client library** (`rhino-client`): `AsrClient` (connect via PeerInfo JSON file, create/destroy sessions), `AudioSender` (accepts audio at any sample rate, resamples to 16kHz via rubato `SincFixedIn`, buffers into 100ms/1600-sample chunks, auto-sequences), `EventStream` (yields `AsrEvent`, hides velo `StreamAnchor`/`StreamFrame` internals). `AsrSession` struct with split `audio`/`events` for concurrent task use. `from_velo()` constructor for in-process testing.
- **Test binary** (`rhino-test-client`): reads WAV file via hound (i16/f32, stereo→mono), sends audio through library with optional real-time pacing, prints events via TextBuffer with ANSI overwrite for TTY.
- **Server update**: clap CLI with `--connect-file` (default `/tmp/rhino-server.json`) and `--bind` args. Writes `PeerInfo` JSON on startup, cleans up on shutdown.
- **Pipeline fix**: `flush_utterance()` now emits `EndOfUtterance` when buffer was fully trimmed but audio was processed (checks `trim_offset > 0` in addition to buffer non-empty).
- 3 integration tests (session lifecycle, resampling 48kHz→16kHz, explicit destroy). Total workspace: 56 tests.

---

## Phase 5.1: Axum Web Service + WASM Frontend — PENDING

**Goal:** Browser-based streaming transcription via Axum + WASM.

- **Axum service**: hosts WASM app, accepts WebSocket connections, bridges WebSocket ↔ client library ↔ ASR server, manages per-connection ASR sessions
- **WASM app**: file drop (upload audio, see streaming transcription) + mic capture (browser MediaStream → live transcription in browser)

### Verification
- Open browser, drop audio file, see streaming transcription
- Click record, speak, see live transcription

---

## Phase 6: End-to-End & Polish — PENDING

**Goal:** Real models, latency profiling, error recovery, documentation.

- End-to-end with Whisper small.en + Silero VAD
- Latency: audio chunk → committed word event
- Error recovery: client disconnect, server restart
- README with build/run instructions

---

## Phase 7: macOS Native App — PENDING

**Goal:** Global-keybind macOS app for system-wide voice-to-text.

- Single global keybind to start audio recording
- Escape to close the stream
- Output text placed at the active cursor position
- Uses client library for ASR server communication

---

## Living Document Protocol

- Status transitions: PENDING → IN PROGRESS → COMPLETE
- Completed phases are updated to reflect what was actually built
- Deviations from plan must be discussed with the developer before updating
- End of each session: review completed work against phase description, flag drift
- This document is never modified without developer discussion
