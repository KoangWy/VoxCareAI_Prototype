// AudioWorklet processor: forwards raw Float32 mono frames to the main thread.
// Resampling to 16 kHz, VAD gating, and socket send stay on the main thread
// (shared with the ScriptProcessor fallback in mic-capture-node.js), so this
// file deliberately does no signal processing.

class VoxcarePcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs && inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      this.port.postMessage({ frames: [channel.slice(0)] });
    }
    return true;
  }
}

registerProcessor('voxcare-pcm-capture', VoxcarePcmCapture);
