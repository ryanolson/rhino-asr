// --- Fetch model info from server ---
fetch('/api/info')
  .then(r => r.json())
  .then(info => {
    const el = document.getElementById('subtitle');
    if (el && info.model) el.textContent = info.model;
  })
  .catch(() => {});

// --- TextBuffer: mirrors crates/protocol/src/text_buffer.rs ---

class TextBuffer {
  constructor() {
    this.committed = [];
    this.interim = null;
  }

  apply(event) {
    switch (event.type) {
      case "commit":
        this.committed.push(
          ...event.text.split(/\s+/).filter((w) => w.length > 0)
        );
        this.interim = null;
        break;
      case "retract":
        this.committed.splice(-event.count);
        this.interim = null;
        break;
      case "interim":
        this.interim = event.text;
        break;
      case "end_of_utterance":
        this.interim = null;
        break;
    }
  }

  committedText() {
    return this.committed.join(" ");
  }
}

// --- State ---

let activeSession = null;
const textBuffer = new TextBuffer();

const $transcriptArea = document.getElementById("transcript-area");
const $committed = document.getElementById("committed");
const $interim = document.getElementById("interim");
const $status = document.getElementById("status");
const $btnStop = document.getElementById("btn-stop");
const $btnMic = document.getElementById("btn-mic");
const $dropZone = document.getElementById("drop-zone");
const $fileInput = document.getElementById("file-input");

// --- UI Helpers ---

function updateDisplay() {
  $committed.textContent = textBuffer.committedText();
  $interim.textContent = textBuffer.interim ? " " + textBuffer.interim : "";
}

function showTranscript(mode) {
  $transcriptArea.classList.add("active");
  $status.textContent = mode === "file" ? "Streaming file..." : "Listening...";
  $status.className = "status streaming";
  textBuffer.committed = [];
  textBuffer.interim = null;
  updateDisplay();
}

function setDone() {
  $status.textContent = "Done";
  $status.className = "status done";
}

// --- WebSocket ---

function connectWs(sampleRate, language) {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        type: "config",
        sample_rate: sampleRate,
        language: language || "en",
      })
    );
  };

  ws.onmessage = (e) => {
    const event = JSON.parse(e.data);
    textBuffer.apply(event);
    updateDisplay();
  };

  // onclose/onerror are set per-session after activeSession is assigned.
  // connectWs only handles config send and event dispatch.
  return ws;
}

// --- Stop (graceful end-of-audio) ---

function stopSession() {
  if (!activeSession) return;
  const session = activeSession;

  if (session.type === "mic") {
    if (session.processor) session.processor.disconnect();
    if (session.source) session.source.disconnect();
    if (session.audioCtx) session.audioCtx.close();
    if (session.stream) {
      session.stream.getTracks().forEach((t) => t.stop());
    }
    // Flush remaining buffered audio
    if (
      session.buffer.length > 0 &&
      session.ws.readyState === WebSocket.OPEN
    ) {
      const tail = session.buffer;
      session.ws.send(
        tail.buffer.slice(tail.byteOffset, tail.byteOffset + tail.byteLength)
      );
      session.buffer = new Float32Array(0);
    }
  }

  if (session.type === "file" && session.interval != null) {
    clearInterval(session.interval);
    session.interval = null;
    if (session.audioCtx) session.audioCtx.close();
  }

  // Signal end of audio — server drains events then closes the WS
  if (session.ws && session.ws.readyState === WebSocket.OPEN) {
    session.ws.send(JSON.stringify({ type: "end_audio" }));
  }

  // Don't null activeSession here — ws.onclose does it after server closes
}

$btnStop.addEventListener("click", stopSession);

// --- Audio Helpers ---

/** Mix multi-channel AudioBuffer to mono Float32Array by averaging channels. */
function mixToMono(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  if (numChannels === 1) return audioBuffer.getChannelData(0);

  const mono = new Float32Array(length);
  for (let ch = 0; ch < numChannels; ch++) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) mono[i] += data[i];
  }
  for (let i = 0; i < length; i++) mono[i] /= numChannels;
  return mono;
}

// --- File Upload ---

$dropZone.addEventListener("click", () => $fileInput.click());

$dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  $dropZone.classList.add("dragover");
});

$dropZone.addEventListener("dragleave", () => {
  $dropZone.classList.remove("dragover");
});

$dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  $dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

$fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
  e.target.value = "";
});

async function handleFile(file) {
  stopSession();
  showTranscript("file");

  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new AudioContext();
  let audioBuffer;
  try {
    audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  } catch {
    $status.textContent = "Unsupported audio format";
    $status.className = "status";
    audioCtx.close();
    return;
  }

  const samples = mixToMono(audioBuffer);
  const sampleRate = audioBuffer.sampleRate;
  const chunkSize = Math.floor(sampleRate * 0.1); // 100ms

  const ws = connectWs(sampleRate, "en");

  // Create session object BEFORE ws.onopen so stopSession() has live handles
  const session = { type: "file", ws, interval: null, audioCtx };
  activeSession = session;

  // Session-scoped close: only clear activeSession if this session is still current
  ws.onclose = () => {
    if (activeSession === session) {
      setDone();
      activeSession = null;
    }
  };
  ws.onerror = () => {
    if (activeSession === session) {
      $status.textContent = "Connection error";
      $status.className = "status";
    }
  };

  let offset = 0;

  ws.onopen = () => {
    ws.send(
      JSON.stringify({ type: "config", sample_rate: sampleRate, language: "en" })
    );

    // Assign interval directly on session object — visible to stopSession()
    session.interval = setInterval(() => {
      if (offset >= samples.length) {
        clearInterval(session.interval);
        session.interval = null;
        ws.send(JSON.stringify({ type: "end_audio" }));
        audioCtx.close();
        return;
      }
      const end = Math.min(offset + chunkSize, samples.length);
      const chunk = samples.slice(offset, end);
      ws.send(
        chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength)
      );
      offset += chunkSize;
    }, 100);
  };
}

// --- Microphone ---

$btnMic.addEventListener("click", async () => {
  if (activeSession && activeSession.type === "mic") {
    stopSession();
    return;
  }

  stopSession();

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    alert("Microphone access denied");
    return;
  }

  const audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);
  const sampleRate = audioCtx.sampleRate;
  const chunkSize = Math.floor(sampleRate * 0.1); // 100ms

  showTranscript("mic");
  $btnMic.textContent = "Stop Recording";

  const ws = connectWs(sampleRate, "en");

  // Create session object BEFORE ws.onopen with mutable buffer and processor.
  // Callbacks mutate these properties in-place so stopSession() sees live state.
  const session = {
    type: "mic",
    ws,
    stream,
    audioCtx,
    source,
    processor: null,
    buffer: new Float32Array(0),
  };
  activeSession = session;

  // Session-scoped close: only clear activeSession if this session is still current
  ws.onclose = () => {
    if (activeSession === session) {
      setDone();
      $btnMic.textContent = "Start Recording";
      activeSession = null;
    }
  };
  ws.onerror = () => {
    if (activeSession === session) {
      $status.textContent = "Connection error";
      $status.className = "status";
      $btnMic.textContent = "Start Recording";
    }
  };

  ws.onopen = async () => {
    ws.send(
      JSON.stringify({ type: "config", sample_rate: sampleRate, language: "en" })
    );

    try {
      await audioCtx.audioWorklet.addModule("audio-worklet.js");
      const node = new AudioWorkletNode(audioCtx, "audio-sender");
      session.processor = node;

      node.port.onmessage = (e) => {
        const newSamples = e.data;
        const merged = new Float32Array(session.buffer.length + newSamples.length);
        merged.set(session.buffer);
        merged.set(newSamples, session.buffer.length);
        session.buffer = merged;

        while (session.buffer.length >= chunkSize) {
          const chunk = session.buffer.slice(0, chunkSize);
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(chunk.buffer);
          }
          session.buffer = session.buffer.slice(chunkSize);
        }
      };

      source.connect(node);
    } catch {
      // AudioWorklet not supported — fall back to ScriptProcessor
      const scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
      session.processor = scriptNode;

      scriptNode.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const merged = new Float32Array(session.buffer.length + input.length);
        merged.set(session.buffer);
        merged.set(input, session.buffer.length);
        session.buffer = merged;

        while (session.buffer.length >= chunkSize) {
          const chunk = session.buffer.slice(0, chunkSize);
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(chunk.buffer);
          }
          session.buffer = session.buffer.slice(chunkSize);
        }
      };
      source.connect(scriptNode);
      scriptNode.connect(audioCtx.destination);
    }
  };
});
