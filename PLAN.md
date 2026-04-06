# Dynamo Whisper — Streaming ASR System

## Context

Build an ASR (Automatic Speech Recognition) system in Rust with selectable chunking strategies. Currently supports whisper.cpp via whisper-rs with Silero VAD for speech detection. Two pipeline modes are available: **utterance mode** (buffer audio, transcribe once on VAD silence, with interval chunking for long utterances) and **streaming mode** (LocalAgreement-2 for progressive word confirmation with self-correcting Commit/Retract/Interim events). The system integrates with the velo distributed infrastructure: **velo-streaming** for bidirectional data transport (audio PCM in, ASR events out), **velo-messenger** for session control (create/destroy/configure). Whisper is a placeholder backend — a true streaming ASR model will eventually replace it.

## Architecture

### Crate Map

```
dynamo-whisper/
  crates/
    protocol/       — Wire types: AsrEvent, AudioChunk, control messages, TextBuffer
    engine/         — Pure LocalAgreement-2 algorithm (no transport deps, used by streaming pipeline mode)
    vad/            — Silero VAD wrapper + VadGate hysteresis (ONNX via ort)
    backend/        — AsrBackend trait + WhisperBackend + MockBackend
    service/        — AsrPipeline trait + UtterancePipeline + StreamingPipeline + velo session management
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
| Pipeline abstraction | Trait + two impls | Utterance (Whisper-optimal) and Streaming (LA-2) are selectable; trait erases backend type from session layer |
| Whisper as placeholder | Utterance mode default | Whisper is encoder-decoder, fundamentally not streaming; will be replaced by a true streaming model |

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

## Phase 5.1: Axum Web Service + Browser Frontend — COMPLETE

**What was built:**
- **Web crate** (`rhino-web`): Library + binary. Axum HTTP server with WebSocket endpoint (`/ws`). Bridges each WebSocket connection 1:1 to an ASR session via `AsrClient`. Typed WS protocol: client sends `{"type":"config",...}` then binary f32 LE PCM audio, then `{"type":"end_audio"}` to signal completion (keeps WS open for server to drain events including `EndOfUtterance`, then server closes). `ClientMessage` serde enum parses all client text frames; unknown types rejected. `ServeDir` fallback serves static frontend. CLI: `--connect-file`, `--bind`, `--static-dir`.
- **Browser frontend** (vanilla JS, not WASM): Dynamo-blog-inspired card layout with dark header, NVIDIA green accents. Two cards: **File Upload** (drag-and-drop + file picker, `decodeAudioData` → stereo-to-mono mixing → 100ms-paced f32 chunks over WebSocket) and **Live Microphone** (`getUserMedia` → AudioWorklet with ScriptProcessor fallback → chunked streaming). JS `TextBuffer` port mirrors Rust `TextBuffer`. Committed text + italic interim display with streaming status indicator. Stop button sends graceful `end_audio`, waits for final transcription.
- **Session state management**: `activeSession` object created before `ws.onopen`; callbacks mutate properties in-place (`interval`, `processor`, `buffer`) so `stopSession()` always has live handles. Buffer stored on session object (not local variable) to avoid stale-closure bugs. `ws.onclose`/`ws.onerror` are session-scoped — only clear `activeSession` if the closing socket matches the current session, preventing old-socket close from clobbering a replacement session.
- **Design deviations**: (1) WASM replaced with vanilla JS — only logic to port was `TextBuffer` (~25 LOC). No wasm-pack/wasm-bindgen toolchain needed. (2) Explicit `end_audio` message instead of WS close for audio completion — WS close handshake prevents server from sending remaining events.
- 7 unit tests (`bytes_to_f32` roundtrip/empty/truncation, `ClientMessage` config/end_audio/no-language/reject-unknown) + 6 integration tests (full WS lifecycle, 48kHz resampling, close without audio, close before config, static file serving, session replacement without interference). Total workspace: 74 tests.

### Verification
- Open browser, drop audio file, see streaming transcription
- Click record, speak, see live transcription

---

## Phase 6: Real Models + Selectable Pipeline Modes — IN PROGRESS

**What was built (6a — models):**
- WhisperBackend with whisper-rs, shared `Arc<WhisperContext>` across sessions, per-session `WhisperState`
- SileroVad with ONNX Runtime (ort), `VadProcessor` trait, integrated into session layer via `VadFactory`
- Server CLI: `--model-path`, `--vad-model-path`, `--language`, `--beam-size`, `--gpu-device`, `--no-gpu`, `--diagnostic`
- Diagnostic mode: captures full session audio to WAV, runs one-shot transcription for quality comparison

**What was built (6b — selectable pipeline modes):**
- `AsrPipeline` trait (object-safe): `push_audio`, `flush_utterance`, `flush_chunk`, `buffer_full`, `reset`, `transcribe_raw`
- `UtterancePipeline<B>`: Single-pass VAD-chunked transcription. Optional interval chunking (`chunk_interval_secs`, default 8s) emits `Commit` events every N seconds during long continuous speech, bounding inference latency. `EndOfUtterance` emitted only on VAD SpeechEnd.
- `StreamingPipeline<B>`: Restored LocalAgreement-2 from Phase 3-5. Periodic transcription at `step_secs` intervals, `AgreementEngine` for progressive word confirmation (Commit/Retract/Interim), buffer trimming after confirmed words, `force_trim` safety valve.
- `SessionManager` simplified: no longer generic over `B: AsrBackend`, takes `PipelineFactory: Arc<dyn Fn() -> Box<dyn AsrPipeline>>`. Backend type encapsulated in concrete pipeline.
- Session-layer `buffer_full()` now calls `flush_chunk()` (no EndOfUtterance) instead of `flush_utterance()`. Only VAD SpeechEnd emits EndOfUtterance.
- Server CLI: `--mode utterance|streaming`, `--chunk-interval`, `--step-interval`
- 26 pipeline unit tests (13 utterance + 11 streaming + 2 shared helpers). 80 tests total across workspace.

**Remaining:**
- Per-session config application (language, sample_rate still logged-not-applied)
- Error recovery coverage (server restart)
- README with build/run instructions
- Latency profiling with real models

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
