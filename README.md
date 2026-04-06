# Dynamo Whisper

ASR (Automatic Speech Recognition) system in Rust with selectable chunking strategies, VAD-gated speech detection, and a browser-based UI. Built on [velo](https://github.com/ryanolson/velo) for distributed session management and streaming transport.

Two pipeline modes are available:

- **Utterance mode** (default): Buffers audio during speech, transcribes once on silence. Optional interval chunking (default 8s) bounds inference latency during long continuous speech.
- **Streaming mode**: LocalAgreement-2 for progressive word confirmation with self-correcting Commit/Retract/Interim events. Best suited for true streaming ASR models (not Whisper).

## Quick Start

### 1. Download Models

Create a `models/` directory and download the Whisper and Silero VAD models:

```bash
mkdir -p models

# Whisper large-v3-turbo (GGML format, ~1.6GB)
wget -O models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# Silero VAD v5 (ONNX, ~2.3MB)
# Use the 16kHz ONNX from the snakers4/silero-vad repo.
# Important: use the v5.1.2 release tag — earlier v5 ONNX exports had broken output shapes.
wget -O models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx
```

### 2. Start the ASR Server

```bash
cargo run --features whisper,silero,cuda --bin rhino-server --release -- \
  --model-path models/ggml-large-v3-turbo.bin \
  --vad-model-path models/silero_vad.onnx \
  --language en \
  --connect-file ./tmp/server.json \
  --mode utterance \
  --chunk-interval 8
```

Drop `cuda` from the features if you don't have a GPU:

```bash
cargo run --features whisper,silero --bin rhino-server --release -- \
  --model-path models/ggml-large-v3-turbo.bin \
  --vad-model-path models/silero_vad.onnx \
  --language en \
  --connect-file ./tmp/server.json \
  --mode utterance \
  --chunk-interval 8
```

### 3. Start the Web UI

```bash
cargo run --release --bin rhino-web -- \
  --connect-file ./tmp/server.json \
  --bind 0.0.0.0:3000 \
  --static-dir crates/web/static
```

Open `http://localhost:3000` in your browser. Drop an audio file or click the microphone to start transcribing.

## Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | *(none, uses MockBackend)* | Path to Whisper GGML model |
| `--vad-model-path` | *(none, VAD disabled)* | Path to Silero VAD ONNX model |
| `--language` | `en` | Language code for transcription |
| `--beam-size` | `5` | Beam search width (1 = greedy) |
| `--gpu-device` | `0` | GPU device index |
| `--no-gpu` | `false` | Force CPU-only inference |
| `--mode` | `utterance` | Pipeline mode: `utterance` or `streaming` |
| `--chunk-interval` | `8.0` | Utterance mode: seconds between interval flushes (0 to disable) |
| `--step-interval` | `0.5` | Streaming mode: seconds between transcription steps |
| `--diagnostic` | `false` | Capture session audio to `/tmp` and run one-shot comparison |
| `--connect-file` | `/tmp/rhino-server.json` | Path to write PeerInfo JSON for clients |
| `--bind` | `0.0.0.0:0` | Server bind address |

## Crates

### `rhino-protocol`

Wire types shared across all crates: `AsrEvent` (Commit, Retract, Interim, EndOfUtterance), `AudioChunk`, session control messages (Create/Destroy), `SessionConfig`, and `TextBuffer` (client-side event application logic, ported to JS for the browser frontend).

### `rhino-engine`

Pure LocalAgreement-2 algorithm with zero transport dependencies. `AgreementEngine` tracks word-level agreement across consecutive hypotheses — words are confirmed after k=2 matching observations. Supports self-correction (Retract + re-Commit when Whisper changes its mind), lookahead cutoff (words too close to the buffer edge aren't committed), and freeze/clear for buffer trim recovery.

### `rhino-vad`

`VadGate` hysteresis state machine that converts a stream of speech probabilities into clean SpeechStart/SpeechEnd transitions. Configurable thresholds (on/off with hysteresis gap to prevent chattering) and minimum consecutive chunk counts (3 speech chunks to open, 20 silence chunks to close). Silero VAD v5 ONNX wrapper behind the `silero` feature flag.

### `rhino-backend`

`AsrBackend` trait with `transcribe()` and `transcribe_with_prompt()`. `WhisperBackend` wraps whisper-rs with shared `Arc<WhisperContext>` (one model load) and per-session `WhisperState` (decode buffers). `MockBackend` with FIFO queue and default response for testing. Feature-gated: `whisper`, `cuda`.

### `rhino-service`

`AsrPipeline` trait with two implementations:

- `UtterancePipeline`: Accumulates audio during speech. Transcribes once on flush (VAD SpeechEnd). Optional interval chunking emits Commit every N seconds during long continuous speech. Uses `initial_prompt` for cross-chunk context continuity.
- `StreamingPipeline`: Transcribes at configurable intervals using the full growing buffer. Feeds hypotheses to `AgreementEngine` for progressive word confirmation. Trims committed audio from the buffer to bound memory.

`SessionManager` with per-session async loops, VAD gating, rayon offload for sync inference via `loom-rs`, and velo messenger handlers for session create/destroy.

### `rhino-client`

`AsrClient` for connecting to the ASR server over velo. `AudioSender` accepts audio at any sample rate, resamples to 16kHz via rubato, and buffers into 100ms chunks. `EventStream` yields `AsrEvent`s. A test binary (`rhino-test-client`) reads WAV files and prints streaming transcription to the terminal.

### `rhino-web`

Axum HTTP server with WebSocket endpoint. Bridges each WebSocket connection 1:1 to an ASR session. Typed WS protocol: client sends JSON config, then binary f32 LE PCM audio, then `end_audio` to signal completion. Vanilla JS frontend with drag-and-drop file upload and live microphone capture via AudioWorklet.
