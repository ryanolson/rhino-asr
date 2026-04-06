class AudioSenderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch) {
      this.port.postMessage(new Float32Array(ch));
    }
    return true;
  }
}

registerProcessor("audio-sender", AudioSenderProcessor);
