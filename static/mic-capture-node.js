// 16 kHz mono PCM16 capture via AudioWorklet, with a ScriptProcessor fallback.
//
// The worklet processor (`pcm-worklet.js`) forwards raw Float32 frames to
// this node; resampling + VAD + socket send stay on the main thread in app.js
// (shared with the fallback path). Loading the worklet can fail (older
// browsers, non-secure contexts, MIME misconfig); the caller falls back to
// ScriptProcessor and surfaces the error instead of leaving "Listening".

export const WORKLET_NAME = 'voxcare-pcm-capture';
export const WORKLET_PATH = '/static/pcm-worklet.js';

/**
 * Attach capture to `audioContext` from `mediaStream`. Resolves with a handle
 * `{ kind: 'worklet'|'script', node, disconnect() }`. Never leaves a
 * half-connected graph on failure: errors propagate to the caller, which must
 * surface them instead of parking the UI in "Listening".
 */
export async function attachCapture({ audioContext, mediaStream, onFrames, workletPath = WORKLET_PATH }) {
  if (!audioContext || !mediaStream) throw new Error('Audio capture needs an audio context and a microphone stream.');
  if (typeof onFrames !== 'function') throw new Error('Audio capture needs a frame callback.');
  const source = audioContext.createMediaStreamSource(mediaStream);
  const tried = [];
  if (audioContext.audioWorklet && typeof audioContext.audioWorklet.addModule === 'function') {
    try {
      await audioContext.audioWorklet.addModule(workletPath);
      const node = new AudioWorkletNode(audioContext, WORKLET_NAME, { numberOfInputs: 1, numberOfOutputs: 1 });
      const muted = audioContext.createGain();
      muted.gain.value = 0;
      node.port.onmessage = (event) => {
        const frames = event.data && event.data.frames;
        if (frames && frames.length) onFrames(frames);
      };
      source.connect(node);
      node.connect(muted);
      muted.connect(audioContext.destination);
      const disconnect = () => {
        try { node.port.onmessage = null; } catch { /* already torn down */ }
        try { source.disconnect(); } catch { /* already torn down */ }
        try { node.disconnect(); } catch { /* already torn down */ }
        try { muted.disconnect(); } catch { /* already torn down */ }
      };
      return { kind: 'worklet', node, disconnect };
    } catch (error) {
      tried.push(error);
      try { source.disconnect(); } catch { /* never connected */ }
    }
  }
  // Fallback: deprecated ScriptProcessorNode. Still main-thread, but
  // better than no capture on older browsers.
  try {
    const fallback = attachScriptProcessor(audioContext, mediaStream, source, onFrames);
    if (tried.length) fallback.workletError = tried[0];
    return fallback;
  } catch (error) {
    try { source.disconnect(); } catch { /* never connected */ }
    throw error;
  }
}

function attachScriptProcessor(audioContext, mediaStream, source, onFrames) {
  if (typeof audioContext.createScriptProcessor !== 'function') {
    throw new Error('This browser does not provide microphone capture. Use a current browser over localhost or HTTPS.');
  }
  const processor = audioContext.createScriptProcessor(2048, 1, 1);
  const muted = audioContext.createGain();
  muted.gain.value = 0;
  processor.onaudioprocess = (event) => {
    onFrames([event.inputBuffer.getChannelData(0)]);
  };
  source.connect(processor);
  processor.connect(muted);
  muted.connect(audioContext.destination);
  const disconnect = () => {
    try { processor.onaudioprocess = null; } catch { /* already torn down */ }
    try { source.disconnect(); } catch { /* already torn down */ }
    try { processor.disconnect(); } catch { /* already torn down */ }
    try { muted.disconnect(); } catch { /* already torn down */ }
  };
  void mediaStream;
  return { kind: 'script', node: processor, disconnect };
}

/** Friendly English message for getUserMedia failures. */
export function micErrorMessage(error) {
  const name = error && error.name;
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Microphone access was denied. Allow microphone access in the browser site settings, then retry.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return 'No microphone was found. Connect a microphone, then retry.';
  }
  if (name === 'NotReadableError' || name === 'AbortError') {
    return 'The microphone is busy or stopped responding. Close other tabs using it, then retry.';
  }
  return (error && error.message) || 'Microphone startup failed. The session is still available for retry.';
}

/** Summarise which requested constraints the browser actually granted. */
export function grantedConstraints(stream) {
  const granted = {};
  try {
    const track = stream && typeof stream.getAudioTracks === 'function' ? stream.getAudioTracks()[0] : null;
    const settings = track && typeof track.getSettings === 'function' ? track.getSettings() : {};
    for (const key of ['echoCancellation', 'noiseSuppression', 'autoGainControl', 'channelCount', 'sampleRate']) {
      if (settings[key] !== undefined) granted[key] = settings[key];
    }
  } catch {
    /* settings unavailable — caller logs the absence */
  }
  return granted;
}
