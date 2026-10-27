(function (global) {
  'use strict';

  /********************************************************************
   * 1. PERSISTENT CUSTOM SONG LIBRARY (IndexedDB + localStorage fallback)
   ********************************************************************/
  const DB_NAME = 'KeyFlowSongDB';
  const DB_VERSION = 1;
  const STORE_NAME = 'custom_songs';
  const FALLBACK_KEY = 'keyflow_custom_songs_v1';

  function openSongDB() {
    return new Promise((resolve, reject) => {
      if (!global.indexedDB) {
        reject(new Error('IndexedDB unavailable'));
        return;
      }
      try {
        const req = global.indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = function (e) {
          const db = e.target.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          }
        };
        req.onsuccess = function (e) {
          resolve(e.target.result);
        };
        req.onerror = function () {
          reject(req.error || new Error('IndexedDB open failed'));
        };
      } catch (err) {
        reject(err);
      }
    });
  }

  const customSongStore = {
    async saveSong(songObj) {
      if (!songObj || !songObj.id) return false;
      const record = {
        id: songObj.id,
        title: songObj.title || 'Custom Song',
        notes: songObj.notes || [],
        duration: songObj.duration || 10,
        beats: songObj.beats || [],
        isCustom: true,
        source: songObj.source || 'custom',
        createdAt: songObj.createdAt || Date.now()
      };
      try {
        const db = await openSongDB();
        await new Promise((resolve, reject) => {
          const tx = db.transaction(STORE_NAME, 'readwrite');
          const store = tx.objectStore(STORE_NAME);
          store.put(record);
          tx.oncomplete = () => resolve(true);
          tx.onerror = () => reject(tx.error);
        });
        db.close();
        return true;
      } catch (e) {
        try {
          const list = JSON.parse(global.localStorage.getItem(FALLBACK_KEY) || '[]');
          const filtered = list.filter(item => item.id !== record.id);
          filtered.push(record);
          global.localStorage.setItem(FALLBACK_KEY, JSON.stringify(filtered));
          return true;
        } catch (err) {
          return false;
        }
      }
    },

    async getAllSongs() {
      try {
        const db = await openSongDB();
        const records = await new Promise((resolve, reject) => {
          const tx = db.transaction(STORE_NAME, 'readonly');
          const store = tx.objectStore(STORE_NAME);
          const req = store.getAll();
          req.onsuccess = () => resolve(req.result || []);
          req.onerror = () => reject(req.error);
        });
        db.close();
        records.sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
        return records;
      } catch (e) {
        try {
          return JSON.parse(global.localStorage.getItem(FALLBACK_KEY) || '[]');
        } catch (err) {
          return [];
        }
      }
    },

    async deleteSong(id) {
      if (!id) return false;
      try {
        const db = await openSongDB();
        await new Promise((resolve, reject) => {
          const tx = db.transaction(STORE_NAME, 'readwrite');
          const store = tx.objectStore(STORE_NAME);
          store.delete(id);
          tx.oncomplete = () => resolve(true);
          tx.onerror = () => reject(tx.error);
        });
        db.close();
      } catch (e) {}
      try {
        const list = JSON.parse(global.localStorage.getItem(FALLBACK_KEY) || '[]');
        const filtered = list.filter(item => item.id !== id);
        global.localStorage.setItem(FALLBACK_KEY, JSON.stringify(filtered));
      } catch (err) {}
      return true;
    }
  };

  /********************************************************************
   * 2. STANDARD MIDI FILE (.MID) FORMAT 1 EXPORTER (L/R HAND TRACKS)
   ********************************************************************/
  function writeVarLen(value) {
    let v = Math.max(0, Math.round(value));
    const bytes = [v & 0x7f];
    v >>= 7;
    while (v > 0) {
      bytes.unshift((v & 0x7f) | 0x80);
      v >>= 7;
    }
    return bytes;
  }

  function buildTrackChunk(events) {
    events.sort((a, b) => a.tick - b.tick || a.order - b.order);
    const body = [];
    let lastTick = 0;

    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const delta = Math.max(0, ev.tick - lastTick);
      lastTick = ev.tick;
      const vBytes = writeVarLen(delta);
      for (let j = 0; j < vBytes.length; j++) body.push(vBytes[j]);
      for (let j = 0; j < ev.bytes.length; j++) body.push(ev.bytes[j]);
    }

    // End of Track Meta Event (FF 2F 00)
    body.push(0x00, 0xff, 0x2f, 0x00);

    const len = body.length;
    const header = [
      0x4d, 0x54, 0x72, 0x6b, // 'MTrk'
      (len >>> 24) & 0xff,
      (len >>> 16) & 0xff,
      (len >>> 8) & 0xff,
      len & 0xff
    ];
    return header.concat(body);
  }

  function stringToAsciiBytes(str) {
    const out = [];
    const s = String(str || '');
    for (let i = 0; i < s.length; i++) {
      out.push(s.charCodeAt(i) & 0x7f);
    }
    return out;
  }

  function exportSongToMidiUint8Array(song) {
    const ticksPerBeat = 480; // 120 BPM => 1 beat = 0.5s => 960 ticks/sec
    const ticksPerSec = 960;

    // Track 0: Tempo (120 BPM = 500,000 us/beat) + Time Signature (4/4) + Title
    const titleBytes = stringToAsciiBytes(song.title || 'KeyFlow Export');
    const tempoTrackEvents = [
      {
        tick: 0,
        order: 0,
        bytes: [0xff, 0x03, ...writeVarLen(titleBytes.length), ...titleBytes]
      },
      {
        tick: 0,
        order: 1,
        bytes: [0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]
      },
      {
        tick: 0,
        order: 2,
        bytes: [0xff, 0x51, 0x03, 0x07, 0xa1, 0x20]
      }
    ];

    const rhName = stringToAsciiBytes('Right Hand');
    const lhName = stringToAsciiBytes('Left Hand');
    const rhEvents = [
      { tick: 0, order: 0, bytes: [0xff, 0x03, ...writeVarLen(rhName.length), ...rhName] },
      { tick: 0, order: 1, bytes: [0xc0, 0x00] } // Acoustic Grand Piano program change
    ];
    const lhEvents = [
      { tick: 0, order: 0, bytes: [0xff, 0x03, ...writeVarLen(lhName.length), ...lhName] },
      { tick: 0, order: 1, bytes: [0xc1, 0x00] }
    ];

    const notes = Array.isArray(song.notes) ? song.notes : [];
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const midi = Math.max(21, Math.min(108, Math.round(n.midi)));
      const startSec = Math.max(0, (n.time || 0) - 0.8);
      const durSec = Math.max(0.08, n.duration || 0.25);
      const startTick = Math.round(startSec * ticksPerSec);
      const endTick = Math.max(startTick + 24, Math.round((startSec + durSec) * ticksPerSec));
      const vel = Math.max(18, Math.min(127, Math.round((n.velocity || 0.78) * 127)));

      if (n.hand === 'L') {
        lhEvents.push({ tick: startTick, order: 2, bytes: [0x91, midi, vel] });
        lhEvents.push({ tick: endTick, order: 1, bytes: [0x81, midi, 0] });
      } else {
        rhEvents.push({ tick: startTick, order: 2, bytes: [0x90, midi, vel] });
        rhEvents.push({ tick: endTick, order: 1, bytes: [0x80, midi, 0] });
      }
    }

    const trk0 = buildTrackChunk(tempoTrackEvents);
    const trk1 = buildTrackChunk(rhEvents);
    const trk2 = buildTrackChunk(lhEvents);

    const header = [
      0x4d, 0x54, 0x68, 0x64, // 'MThd'
      0x00, 0x00, 0x00, 0x06, // Header length = 6
      0x00, 0x01,             // Format 1 (multi-track)
      0x00, 0x03,             // 3 tracks
      (ticksPerBeat >>> 8) & 0xff,
      ticksPerBeat & 0xff
    ];

    const allBytes = header.concat(trk0, trk1, trk2);
    return new Uint8Array(allBytes);
  }

  function exportSongToMidiBlob(song) {
    const bytes = exportSongToMidiUint8Array(song);
    return new Blob([bytes], { type: 'audio/midi' });
  }

  /********************************************************************
   * 3. PYTHON AI SERVER HEALTH CHECK & REMOTE TRANSCRIPTION
   ********************************************************************/
  function normalizeServerUrl(url) {
    return String(url || 'http://localhost:8000').trim().replace(/\/+$/, '');
  }

  async function checkServerHealth(serverUrl, timeoutMs = 2500) {
    const base = normalizeServerUrl(serverUrl);
    if (!base || !global.fetch) return { ok: false, url: base };
    const ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
    try {
      const res = await fetch(base + '/health', {
        method: 'GET',
        signal: ctrl ? ctrl.signal : undefined
      });
      if (timer) clearTimeout(timer);
      if (!res.ok) return { ok: false, url: base };
      const data = await res.json();
      return { ok: true, url: base, data: data };
    } catch (e) {
      if (timer) clearTimeout(timer);
      return { ok: false, url: base };
    }
  }

  function base64ToArrayBuffer(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = bin.charCodeAt(i);
    }
    return bytes.buffer;
  }

  async function transcribeViaPythonServer(file, options, onProgress) {
    const base = normalizeServerUrl(options.serverUrl);
    if (onProgress) {
      onProgress({
        stage: options.isolatePiano
          ? 'Uploading to Python AI Server (HT-Demucs + Neural Piano Transcription)...'
          : 'Uploading to Python AI Server (Neural Piano Transcription)...',
        percent: 25
      });
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('isolate_piano', options.isolatePiano ? 'true' : 'false');
    formData.append('sensitivity', String(options.sensitivity || 55));

    const res = await fetch(base + '/api/transcribe', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => 'Server error');
      throw new Error('AI Server error (' + res.status + '): ' + errText);
    }

    if (onProgress) {
      onProgress({ stage: 'Parsing neural MIDI & separating Left/Right hands...', percent: 88 });
    }

    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    let parsedSong = null;

    if (contentType.includes('application/json')) {
      const payload = await res.json();
      if (payload.midi_base64) {
        const ab = base64ToArrayBuffer(payload.midi_base64);
        parsedSong = global.parseStandardMidiFile(ab, file.name);
      } else if (Array.isArray(payload.notes) && payload.notes.length > 0) {
        const notes = payload.notes.map(n => ({
          midi: n.midi,
          name: global.midiToName(n.midi),
          time: (n.time || 0) + 0.8,
          duration: Math.max(0.1, n.duration || 0.25),
          hand: n.hand || 'R',
          velocity: n.velocity || 0.78
        }));
        global.assignHandsSmart(notes);
        const maxEnd = notes.reduce((m, n) => Math.max(m, n.time + n.duration), 0);
        const dur = maxEnd + 2.0;
        const bpm = payload.bpm || 120;
        parsedSong = {
          title: payload.title || file.name.replace(/\.[^/.]+$/, ''),
          notes: notes,
          duration: dur,
          beats: global.generateBeatGrid(dur, 60 / bpm, 4, 0.8)
        };
      }
    } else {
      const ab = await res.arrayBuffer();
      parsedSong = global.parseStandardMidiFile(ab, file.name);
    }

    if (!parsedSong || !Array.isArray(parsedSong.notes) || parsedSong.notes.length === 0) {
      throw new Error('No playable piano notes returned from AI server');
    }

    global.assignHandsSmart(parsedSong.notes);
    parsedSong.source = 'python-ai';
    if (onProgress) {
      onProgress({ stage: 'Neural AI transcription complete!', percent: 100 });
    }
    return parsedSong;
  }

  /********************************************************************
   * 4. IN-BROWSER WEB AUDIO POLYPHONIC PIANO TRANSCRIBER
   *    - Works 100% offline on iPad Safari / GitHub Pages
   *    - Pitch-adaptive Goertzel spectral analysis (C1..C7)
   *    - Harmonic overtone suppression (2nd, 3rd, 4th, 5th harmonics)
   *    - Spectral flux onset detection + BPM beat grid estimation
   *    - Smart Left/Right hand separation via assignHandsSmart
   ********************************************************************/
  function estimateTempoBpm(onsetTimes) {
    if (!onsetTimes || onsetTimes.length < 6) return 110;
    const intervals = [];
    for (let i = 1; i < onsetTimes.length; i++) {
      const dt = onsetTimes[i] - onsetTimes[i - 1];
      if (dt >= 0.18 && dt <= 1.2) {
        intervals.push(dt);
      }
    }
    if (intervals.length < 4) return 110;

    // Histogram candidate BPMs between 70 and 155 BPM
    let bestBpm = 110;
    let bestScore = -1;
    for (let bpm = 72; bpm <= 152; bpm += 2) {
      const beatSec = 60 / bpm;
      let score = 0;
      for (let i = 0; i < intervals.length; i++) {
        const ratio = intervals[i] / beatSec;
        // Check alignment with 1/2 beat, 1 beat, or 2 beats
        const nearest = Math.round(ratio * 2) / 2;
        if (nearest >= 0.5 && nearest <= 2.0) {
          const err = Math.abs(ratio - nearest);
          if (err < 0.14) {
            score += (0.14 - err);
          }
        }
      }
      if (score > bestScore) {
        bestScore = score;
        bestBpm = bpm;
      }
    }
    return bestBpm;
  }

  async function decodeAudioBufferFromFile(arrayBuffer) {
    const AudioCtx = global.AudioContext || global.webkitAudioContext;
    if (!AudioCtx) throw new Error('Web Audio API not supported in this browser');
    const tempCtx = new AudioCtx();
    try {
      const decoded = await tempCtx.decodeAudioData(arrayBuffer.slice(0));
      if (tempCtx.close) {
        try { await tempCtx.close(); } catch (e) {}
      }
      return decoded;
    } catch (err) {
      if (tempCtx.close) {
        try { await tempCtx.close(); } catch (e) {}
      }
      throw new Error('Could not decode audio file. Please use MP3, WAV, M4A, or OGG.');
    }
  }

  function computeGoertzelMagnitude(samples, startIdx, windowLen, freq, sampleRate, hannWindow) {
    const omega = (2 * Math.PI * freq) / sampleRate;
    const coeff = 2 * Math.cos(omega);
    let s1 = 0;
    let s2 = 0;

    for (let i = 0; i < windowLen; i++) {
      const w = hannWindow ? hannWindow[i] : (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (windowLen - 1)));
      const s0 = samples[startIdx + i] * w + coeff * s1 - s2;
      s2 = s1;
      s1 = s0;
    }

    const power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    return Math.sqrt(Math.max(0, power)) / (windowLen * 0.5);
  }

  async function transcribeInBrowser(file, options, onProgress) {
    if (onProgress) {
      onProgress({ stage: 'Reading & decoding audio file on device...', percent: 10 });
    }
    const arrayBuffer = await file.arrayBuffer();
    const audioBuffer = await decodeAudioBufferFromFile(arrayBuffer);

    const sr = audioBuffer.sampleRate;
    const numChannels = audioBuffer.numberOfChannels;
    const totalSamples = audioBuffer.length;

    // Mix to mono (and optionally emphasize harmonic midband if isolatePiano is enabled)
    const mono = new Float32Array(totalSamples);
    if (numChannels === 1) {
      mono.set(audioBuffer.getChannelData(0));
    } else {
      const ch0 = audioBuffer.getChannelData(0);
      const ch1 = audioBuffer.getChannelData(1);
      for (let i = 0; i < totalSamples; i++) {
        mono[i] = 0.5 * (ch0[i] + ch1[i]);
      }
    }

    // Normalize peak amplitude
    let peakAmp = 0.001;
    for (let i = 0; i < totalSamples; i++) {
      const a = Math.abs(mono[i]);
      if (a > peakAmp) peakAmp = a;
    }
    const normGain = 1.0 / peakAmp;
    for (let i = 0; i < totalSamples; i++) {
      mono[i] *= normGain;
    }

    // Analyze MIDI range 28 (E1) .. 96 (C7) where 99% of musical piano parts reside
    const MIN_MIDI = 28;
    const MAX_MIDI = 96;
    const numBins = MAX_MIDI - MIN_MIDI + 1;
    const freqs = new Float32Array(numBins);
    const windowLens = new Int32Array(numBins);

    const maxFrameSize = Math.min(8192, Math.max(2048, Math.round(sr * 0.092)));
    const hopSamples = Math.max(256, Math.round(sr * 0.028)); // ~28ms hop
    const hopSec = hopSamples / sr;

    // Precompute Hann windows for 3 octave bands (long for bass resolution, medium for mid, short for treble)
    const lenLow = maxFrameSize;
    const lenMid = Math.max(1024, Math.floor(maxFrameSize * 0.5));
    const lenHigh = Math.max(512, Math.floor(maxFrameSize * 0.25));

    function makeHann(N) {
      const w = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));
      }
      return w;
    }
    const hannLow = makeHann(lenLow);
    const hannMid = makeHann(lenMid);
    const hannHigh = makeHann(lenHigh);

    for (let b = 0; b < numBins; b++) {
      const midi = MIN_MIDI + b;
      freqs[b] = global.midiToFreq(midi);
      if (midi < 48) windowLens[b] = lenLow;
      else if (midi < 72) windowLens[b] = lenMid;
      else windowLens[b] = lenHigh;
    }

    const totalFrames = Math.max(1, Math.floor((totalSamples - maxFrameSize) / hopSamples));
    const frameMagnitudes = new Array(totalFrames);
    const frameEnergy = new Float32Array(totalFrames);
    let globalMaxMag = 0.0001;

    // Step through audio frames asynchronously so UI stays smooth
    for (let f = 0; f < totalFrames; f++) {
      const startIdx = f * hopSamples;
      const mags = new Float32Array(numBins);

      // Quick RMS gate on frame
      let rmsSum = 0;
      for (let i = 0; i < lenMid; i += 2) {
        const s = mono[startIdx + i];
        rmsSum += s * s;
      }
      const rms = Math.sqrt(rmsSum / (lenMid * 0.5));
      frameEnergy[f] = rms;

      if (rms > 0.012) {
        for (let b = 0; b < numBins; b++) {
          const wLen = windowLens[b];
          const hWin = (wLen === lenLow) ? hannLow : (wLen === lenMid ? hannMid : hannHigh);
          const mag = computeGoertzelMagnitude(mono, startIdx, wLen, freqs[b], sr, hWin);
          mags[b] = mag;
        }

        // Local spectral peak picking (suppress non-peak semitone leakage)
        const cleaned = new Float32Array(numBins);
        for (let b = 1; b < numBins - 1; b++) {
          const m = mags[b];
          if (m >= mags[b - 1] * 0.92 && m >= mags[b + 1] * 0.92) {
            cleaned[b] = m - 0.35 * Math.min(mags[b - 1], mags[b + 1]);
          }
        }

        // Harmonic Overtone Suppression: subtract 2nd (+12), 3rd (+19), 4th (+24), and 5th (+28) harmonics
        for (let b = 0; b < numBins; b++) {
          const fund = cleaned[b];
          if (fund <= 0.002) continue;
          if (b + 12 < numBins) cleaned[b + 12] = Math.max(0, cleaned[b + 12] - fund * 0.58);
          if (b + 19 < numBins) cleaned[b + 19] = Math.max(0, cleaned[b + 19] - fund * 0.36);
          if (b + 24 < numBins) cleaned[b + 24] = Math.max(0, cleaned[b + 24] - fund * 0.25);
          if (b + 28 < numBins) cleaned[b + 28] = Math.max(0, cleaned[b + 28] - fund * 0.18);
        }

        for (let b = 0; b < numBins; b++) {
          if (cleaned[b] > globalMaxMag) globalMaxMag = cleaned[b];
        }
        frameMagnitudes[f] = cleaned;
      } else {
        frameMagnitudes[f] = mags;
      }

      if (f % 45 === 0) {
        if (onProgress) {
          const pct = 15 + Math.round((f / totalFrames) * 65);
          onProgress({
            stage: 'Analyzing polyphonic piano spectrum (' + Math.round((f / totalFrames) * 100) + '%)...',
            percent: pct
          });
        }
        await new Promise(r => setTimeout(r, 0));
      }
    }

    if (onProgress) {
      onProgress({ stage: 'Tracking note onsets, durations & Left/Right hands...', percent: 85 });
    }

    // Sensitivity mapping (0..100, default 55)
    let rawSens = Number(options.sensitivity);
    if (!isNaN(rawSens) && rawSens > 0 && rawSens <= 1) {
      rawSens = Math.round(rawSens * 100);
    }
    const sens = Math.max(10, Math.min(90, rawSens || 55));
    const normSens = (sens - 10) / 80; // 0..1
    const activationThreshold = globalMaxMag * (0.28 - normSens * 0.17); // ~0.11..0.28 of peak
    const sustainThreshold = activationThreshold * 0.58;
    const maxPolyphonyPerFrame = Math.round(3 + normSens * 4); // 3..7 notes simultaneously

    // Limit each frame to top-K strongest peaks above threshold
    const activeMatrix = new Array(totalFrames);
    for (let f = 0; f < totalFrames; f++) {
      const row = frameMagnitudes[f];
      const activeRow = new Float32Array(numBins);
      const candidates = [];

      for (let b = 0; b < numBins; b++) {
        if (row[b] >= sustainThreshold) {
          candidates.push({ b, val: row[b] });
        }
      }
      candidates.sort((a, b) => b.val - a.val);
      const keep = Math.min(candidates.length, maxPolyphonyPerFrame);
      for (let k = 0; k < keep; k++) {
        activeRow[candidates[k].b] = candidates[k].val;
      }
      activeMatrix[f] = activeRow;
    }

    // Extract note events with onset re-articulation and minimum duration filtering
    const extractedNotes = [];
    const onsetTimes = [];
    const minNoteFrames = Math.max(3, Math.round(0.085 / hopSec)); // >= 85ms

    for (let b = 0; b < numBins; b++) {
      const midi = MIN_MIDI + b;
      let inNote = false;
      let startFrame = 0;
      let peakVal = 0;
      let prevVal = 0;
      let gapFrames = 0;

      for (let f = 0; f < totalFrames; f++) {
        const val = activeMatrix[f][b];

        if (!inNote) {
          if (val >= activationThreshold) {
            inNote = true;
            startFrame = f;
            peakVal = val;
            prevVal = val;
            gapFrames = 0;
          }
        } else {
          // Check for a fresh re-strike on the same key
          const isRestrike = (f - startFrame >= minNoteFrames) &&
            (val >= activationThreshold * 1.15) &&
            (val > prevVal * 1.75);

          if (isRestrike) {
            const durFrames = f - startFrame;
            if (durFrames >= minNoteFrames) {
              const t0 = startFrame * hopSec + 0.8;
              const dur = Math.min(4.5, durFrames * hopSec);
              const vel = Math.max(0.32, Math.min(0.95, Math.sqrt(peakVal / globalMaxMag)));
              extractedNotes.push({
                midi: midi,
                name: global.midiToName(midi),
                time: Math.round(t0 * 1000) / 1000,
                duration: Math.round(dur * 1000) / 1000,
                hand: midi < 60 ? 'L' : 'R',
                velocity: Math.round(vel * 100) / 100
              });
              onsetTimes.push(t0);
            }
            startFrame = f;
            peakVal = val;
            prevVal = val;
            gapFrames = 0;
          } else if (val >= sustainThreshold) {
            if (val > peakVal) peakVal = val;
            prevVal = val;
            gapFrames = 0;
          } else {
            gapFrames++;
            if (gapFrames >= 2 || f === totalFrames - 1) {
              const endFrame = f - gapFrames + 1;
              const durFrames = endFrame - startFrame;
              if (durFrames >= minNoteFrames) {
                const t0 = startFrame * hopSec + 0.8;
                const dur = Math.min(4.5, durFrames * hopSec);
                const vel = Math.max(0.32, Math.min(0.95, Math.sqrt(peakVal / globalMaxMag)));
                extractedNotes.push({
                  midi: midi,
                  name: global.midiToName(midi),
                  time: Math.round(t0 * 1000) / 1000,
                  duration: Math.round(dur * 1000) / 1000,
                  hand: midi < 60 ? 'L' : 'R',
                  velocity: Math.round(vel * 100) / 100
                });
                onsetTimes.push(t0);
              }
              inNote = false;
              gapFrames = 0;
            }
          }
        }
      }

      if (inNote) {
        const durFrames = totalFrames - startFrame;
        if (durFrames >= minNoteFrames) {
          const t0 = startFrame * hopSec + 0.8;
          const dur = Math.min(4.5, durFrames * hopSec);
          const vel = Math.max(0.32, Math.min(0.95, Math.sqrt(peakVal / globalMaxMag)));
          extractedNotes.push({
            midi: midi,
            name: global.midiToName(midi),
            time: Math.round(t0 * 1000) / 1000,
            duration: Math.round(dur * 1000) / 1000,
            hand: midi < 60 ? 'L' : 'R',
            velocity: Math.round(vel * 100) / 100
          });
          onsetTimes.push(t0);
        }
      }
    }

    if (extractedNotes.length === 0) {
      throw new Error('No clear musical notes detected in audio. Try increasing the Sensitivity slider.');
    }

    extractedNotes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    global.assignHandsSmart(extractedNotes);

    onsetTimes.sort((a, b) => a - b);
    const estimatedBpm = estimateTempoBpm(onsetTimes);
    const maxEnd = extractedNotes.reduce((m, n) => Math.max(m, n.time + n.duration), 0);
    const duration = maxEnd + 2.0;
    const cleanTitle = file.name.replace(/\.[^/.]+$/, '');

    if (onProgress) {
      onProgress({ stage: 'Transcription complete!', percent: 100 });
    }

    return {
      title: cleanTitle + ' (AI Audio)',
      notes: extractedNotes,
      duration: duration,
      beats: global.generateBeatGrid(duration, 60 / estimatedBpm, 4, 0.8),
      source: 'browser-ai'
    };
  }

  /********************************************************************
   * 5. UNIFIED HYBRID TRANSCRIPTION ENTRY POINT
   ********************************************************************/
  async function transcribeAudioFile(file, options = {}, onProgress) {
    const mode = options.mode || options.engineMode || 'auto'; // 'auto' | 'browser' | 'server'
    const serverUrl = normalizeServerUrl(options.serverUrl || 'http://localhost:8000');
    let normSensitivity = Number(options.sensitivity);
    if (!isNaN(normSensitivity) && normSensitivity > 0 && normSensitivity <= 1) {
      normSensitivity = Math.round(normSensitivity * 100);
    }
    const normalizedOptions = {
      ...options,
      mode: mode,
      serverUrl: serverUrl,
      sensitivity: normSensitivity || 55
    };

    if (mode === 'server' || mode === 'auto') {
      if (onProgress) {
        onProgress({ stage: 'Checking Python AI Server connection...', percent: 5 });
      }
      const health = await checkServerHealth(serverUrl, 2200);
      if (health.ok) {
        return await transcribeViaPythonServer(file, normalizedOptions, onProgress);
      } else if (mode === 'server') {
        throw new Error(
          'Python AI Server at ' + serverUrl + ' is offline. Start backend/server.py or switch Engine Mode to Auto / In-Browser.'
        );
      }
    }

    return await transcribeInBrowser(file, normalizedOptions, onProgress);
  }

  global.audioTranscriber = {
    transcribeAudioFile: transcribeAudioFile,
    checkServerHealth: checkServerHealth,
    exportSongToMidiBlob: exportSongToMidiBlob,
    exportSongToMidiUint8Array: exportSongToMidiUint8Array,
    customSongStore: customSongStore
  };
})(window);
