(function (global) {
  'use strict';

  /********************************************************************
   * PURE WEB AUDIO API ACOUSTIC & MELLOW FELT PIANO ENGINE
   * - Zero external library dependencies (never crashes in Google Sites)
   * - Combines multi-partial acoustic piano modeling + wooden soundboard
   *   body EQ + velocity-sensitive felt hammer transient + stereo hall
   *   impulse reverb + optional background Salamander Grand sample loader.
   * - Supports WebMIDI CC 64 Sustain Pedal extension & smooth damping.
   ********************************************************************/
  class PureAcousticPianoEngine {
    constructor() {
      this.ctx = null;
      this.muted = false;
      this.preset = 'mellow_felt'; // 'mellow_felt' | 'acoustic_grand' | 'warm_upright' | 'electric_piano'
      this.warmth = 78; // 0..100
      this.sustainPedal = false;
      this.sustainedVoices = [];

      this.masterGain = null;
      this.feltFilter = null;
      this.bodyLowEQ = null;
      this.bodyMidEQ = null;
      this.reverbSend = null;
      this.dryGain = null;
      this.convolver = null;
      this.noiseBuffer = null;

      // Optional background multi-sampled grand piano buffers
      this.sampleBuffers = {};
      this.sampleMidis = [21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96];
      this.samplesReadyCount = 0;
      this.samplesStartedLoading = false;
    }

    ensureContext() {
      if (this.ctx) {
        if (this.ctx.state === 'suspended') {
          try { this.ctx.resume(); } catch (e) {}
        }
        return this.ctx;
      }

      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return null;
        this.ctx = new AudioCtx();

        // Master Output & Dynamics Compressor
        const comp = this.ctx.createDynamicsCompressor();
        comp.threshold.value = -16;
        comp.knee.value = 18;
        comp.ratio.value = 3.5;
        comp.attack.value = 0.004;
        comp.release.value = 0.22;

        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.value = 0.85;

        // Real-time Felt Softness Low-pass Filter
        this.feltFilter = this.ctx.createBiquadFilter();
        this.feltFilter.type = 'lowpass';
        this.feltFilter.Q.value = 0.65;

        // Warm Soundboard Body Resonance Filters
        this.bodyLowEQ = this.ctx.createBiquadFilter();
        this.bodyLowEQ.type = 'lowshelf';
        this.bodyLowEQ.frequency.value = 210;

        this.bodyMidEQ = this.ctx.createBiquadFilter();
        this.bodyMidEQ.type = 'peaking';
        this.bodyMidEQ.frequency.value = 650;
        this.bodyMidEQ.Q.value = 0.9;

        // Algorithmic Stereo Hall Reverb Convolver
        this.convolver = this.ctx.createConvolver();
        this.convolver.buffer = this.createHallImpulseBuffer(2.4, 2.3);

        this.dryGain = this.ctx.createGain();
        this.reverbSend = this.ctx.createGain();

        // Routing: Voices -> bodyLowEQ -> bodyMidEQ -> feltFilter -> (Dry + Convolver) -> Compressor -> Master
        this.bodyLowEQ.connect(this.bodyMidEQ);
        this.bodyMidEQ.connect(this.feltFilter);

        this.feltFilter.connect(this.dryGain);
        this.feltFilter.connect(this.reverbSend);
        this.reverbSend.connect(this.convolver);

        this.dryGain.connect(comp);
        this.convolver.connect(comp);
        comp.connect(this.masterGain);
        this.masterGain.connect(this.ctx.destination);

        // Generate hammer felt noise buffer
        this.noiseBuffer = this.createHammerNoiseBuffer();

        this.applyToneSettings();
        this.tryLoadGrandSamplesInBackground();
      } catch (err) {
        // Audio blocked or unsupported
      }
      return this.ctx;
    }

    setSustainPedal(isDown) {
      const nextState = !!isDown;
      if (this.sustainPedal === nextState) return;
      this.sustainPedal = nextState;

      if (!nextState && this.ctx && this.sustainedVoices.length > 0) {
        const now = this.ctx.currentTime;
        for (let i = 0; i < this.sustainedVoices.length; i++) {
          const v = this.sustainedVoices[i];
          if (v && v.gainNode && now < v.endTime) {
            try {
              v.gainNode.gain.cancelScheduledValues(now);
              v.gainNode.gain.setValueAtTime(Math.max(0.0002, v.gainNode.gain.value || 0.05), now);
              v.gainNode.gain.setTargetAtTime(0.0001, now, 0.16);
            } catch (e) {}
          }
        }
        this.sustainedVoices = [];
      }
    }

    registerSustainedVoice(gainNode, endTime) {
      if (!this.sustainPedal || !this.ctx) return;
      const now = this.ctx.currentTime;
      this.sustainedVoices = this.sustainedVoices.filter(v => v.endTime > now);
      if (this.sustainedVoices.length > 48) {
        this.sustainedVoices.shift();
      }
      this.sustainedVoices.push({ gainNode, endTime });
    }

    createHallImpulseBuffer(durationSec, decayRate) {
      const sr = this.ctx.sampleRate;
      const length = Math.floor(sr * durationSec);
      const impulse = this.ctx.createBuffer(2, length, sr);
      for (let ch = 0; ch < 2; ch++) {
        const data = impulse.getChannelData(ch);
        let lp = 0;
        for (let i = 0; i < length; i++) {
          const t = i / length;
          const env = Math.pow(1 - t, decayRate);
          const white = (Math.random() * 2 - 1);
          // Warm studio damping on reverb tail
          lp = lp * 0.78 + white * 0.22;
          data[i] = lp * env * 0.45;
        }
      }
      return impulse;
    }

    createHammerNoiseBuffer() {
      const sr = this.ctx.sampleRate;
      const len = Math.floor(sr * 0.06);
      const buf = this.ctx.createBuffer(1, len, sr);
      const data = buf.getChannelData(0);
      let prev = 0;
      for (let i = 0; i < len; i++) {
        const env = Math.exp(-i / (sr * 0.012));
        const w = Math.random() * 2 - 1;
        prev = 0.65 * prev + 0.35 * w;
        data[i] = prev * env;
      }
      return buf;
    }

    tryLoadGrandSamplesInBackground() {
      if (this.samplesStartedLoading) return;
      this.samplesStartedLoading = true;

      const baseUrl = 'https://tonejs.github.io/audio/salamander/';
      const mapMidiToFile = {
        21: 'A0.mp3', 24: 'C1.mp3', 27: 'Ds1.mp3', 30: 'Fs1.mp3', 33: 'A1.mp3',
        36: 'C2.mp3', 39: 'Ds2.mp3', 42: 'Fs2.mp3', 45: 'A2.mp3',
        48: 'C3.mp3', 51: 'Ds3.mp3', 54: 'Fs3.mp3', 57: 'A3.mp3',
        60: 'C4.mp3', 63: 'Ds4.mp3', 66: 'Fs4.mp3', 69: 'A4.mp3',
        72: 'C5.mp3', 75: 'Ds5.mp3', 78: 'Fs5.mp3', 81: 'A5.mp3',
        84: 'C6.mp3', 87: 'Ds6.mp3', 90: 'Fs6.mp3', 93: 'A6.mp3', 96: 'C7.mp3'
      };

      Object.keys(mapMidiToFile).forEach((mStr) => {
        const midi = parseInt(mStr, 10);
        const url = baseUrl + mapMidiToFile[midi];
        fetch(url)
          .then(r => r.ok ? r.arrayBuffer() : Promise.reject())
          .then(ab => this.ctx.decodeAudioData(ab))
          .then(audioBuf => {
            this.sampleBuffers[midi] = audioBuf;
            this.samplesReadyCount++;
          })
          .catch(() => {
            // Silent fallback to built-in physical modeling acoustic engine if offline/blocked
          });
      });
    }

    setPreset(presetId, customWarmth) {
      this.preset = presetId;
      if (typeof customWarmth === 'number') {
        this.warmth = customWarmth;
      } else {
        if (presetId === 'mellow_felt') this.warmth = 78;
        else if (presetId === 'acoustic_grand') this.warmth = 24;
        else if (presetId === 'warm_upright') this.warmth = 58;
        else if (presetId === 'electric_piano') this.warmth = 48;
      }
      this.applyToneSettings();
    }

    applyToneSettings() {
      if (!this.ctx || !this.feltFilter) return;
      const now = this.ctx.currentTime;
      const normW = Math.max(0, Math.min(1, this.warmth / 100));

      // Exponentially map warmth (0..100%) to low-pass cutoff (9500Hz down to 850Hz)
      const cutoff = 9500 * Math.pow(0.09, normW);
      this.feltFilter.frequency.setTargetAtTime(cutoff, now, 0.04);

      if (this.preset === 'mellow_felt') {
        this.bodyLowEQ.gain.setTargetAtTime(4.2, now, 0.04);
        this.bodyMidEQ.gain.setTargetAtTime(1.8, now, 0.04);
        this.dryGain.gain.setTargetAtTime(0.76, now, 0.04);
        this.reverbSend.gain.setTargetAtTime(0.42, now, 0.04);
      } else if (this.preset === 'acoustic_grand') {
        this.bodyLowEQ.gain.setTargetAtTime(2.2, now, 0.04);
        this.bodyMidEQ.gain.setTargetAtTime(0.4, now, 0.04);
        this.dryGain.gain.setTargetAtTime(0.85, now, 0.04);
        this.reverbSend.gain.setTargetAtTime(0.28, now, 0.04);
      } else if (this.preset === 'warm_upright') {
        this.bodyLowEQ.gain.setTargetAtTime(3.0, now, 0.04);
        this.bodyMidEQ.gain.setTargetAtTime(3.4, now, 0.04);
        this.dryGain.gain.setTargetAtTime(0.88, now, 0.04);
        this.reverbSend.gain.setTargetAtTime(0.18, now, 0.04);
      } else if (this.preset === 'electric_piano') {
        this.bodyLowEQ.gain.setTargetAtTime(2.5, now, 0.04);
        this.bodyMidEQ.gain.setTargetAtTime(1.5, now, 0.04);
        this.dryGain.gain.setTargetAtTime(0.82, now, 0.04);
        this.reverbSend.gain.setTargetAtTime(0.34, now, 0.04);
      }
    }

    findClosestSample(midi) {
      let bestMidi = null;
      let minDiff = 999;
      for (let i = 0; i < this.sampleMidis.length; i++) {
        const sm = this.sampleMidis[i];
        if (this.sampleBuffers[sm]) {
          const diff = Math.abs(midi - sm);
          if (diff < minDiff) {
            minDiff = diff;
            bestMidi = sm;
          }
        }
      }
      return (minDiff <= 6) ? bestMidi : null;
    }

    playNote(midi, durationSec = 0.9, velocity = 0.78) {
      if (this.muted) return;
      const ctx = this.ensureContext();
      if (!ctx) return;

      const now = ctx.currentTime;
      const freq = global.midiToFreq(midi);
      const vel = Math.max(0.15, Math.min(1.0, velocity));
      const baseDur = this.sustainPedal ? Math.max(durationSec * 2.6, 2.8) : durationSec;
      const dur = Math.max(0.22, Math.min(6.5, baseDur));

      // Check if we have a decoded Salamander acoustic sample nearby and not in Electric Piano mode
      const sampleBaseMidi = (this.preset !== 'electric_piano') ? this.findClosestSample(midi) : null;
      if (sampleBaseMidi !== null && this.sampleBuffers[sampleBaseMidi]) {
        this.triggerSampleVoice(ctx, now, midi, sampleBaseMidi, dur, vel);
        return;
      }

      // Pure Web Audio Multi-Partial Acoustic / Felt / Rhodes Piano Synthesis
      this.triggerModeledAcousticVoice(ctx, now, midi, freq, dur, vel);
    }

    triggerSampleVoice(ctx, now, targetMidi, sampleMidi, dur, vel) {
      try {
        const src = ctx.createBufferSource();
        src.buffer = this.sampleBuffers[sampleMidi];
        src.playbackRate.value = Math.pow(2, (targetMidi - sampleMidi) / 12);

        const voiceFilter = ctx.createBiquadFilter();
        voiceFilter.type = 'lowpass';
        const normW = this.warmth / 100;
        voiceFilter.frequency.value = Math.max(600, (4800 - normW * 3100) * (0.6 + vel * 0.6));

        const env = ctx.createGain();
        const attack = (this.preset === 'mellow_felt') ? 0.016 : 0.003;
        const peakGain = vel * (this.preset === 'mellow_felt' ? 0.65 : 0.8);
        const releaseTC = this.sustainPedal ? 0.65 : 0.32;

        env.gain.setValueAtTime(0.0001, now);
        env.gain.linearRampToValueAtTime(peakGain, now + attack);
        env.gain.setTargetAtTime(peakGain * 0.45, now + attack + 0.08, 0.65);
        env.gain.setTargetAtTime(0.0001, now + dur, releaseTC);

        src.connect(voiceFilter);
        voiceFilter.connect(env);
        env.connect(this.bodyLowEQ);

        const stopTime = now + dur + (this.sustainPedal ? 2.4 : 1.5);
        src.start(now);
        src.stop(stopTime);
        this.registerSustainedVoice(env, stopTime);
      } catch (e) {}
    }

    triggerModeledAcousticVoice(ctx, now, midi, freq, dur, vel) {
      try {
        const normW = this.warmth / 100;
        const voiceGain = ctx.createGain();

        // Higher notes decay naturally faster on a real acoustic piano
        const pitchFactor = Math.max(0.3, Math.min(1.6, Math.pow(60 / Math.max(24, midi), 1.3)));
        const stringDecay = Math.max(0.7, (this.sustainPedal ? 3.2 : 2.2) * pitchFactor);
        const attackTime = (this.preset === 'mellow_felt') ? 0.016 : 0.005;
        const masterPeak = vel * 0.24;

        voiceGain.gain.setValueAtTime(0.0001, now);
        voiceGain.gain.linearRampToValueAtTime(masterPeak, now + attackTime);
        // Two-stage piano string decay (initial prompt soundboard decay -> longer sustain tail)
        voiceGain.gain.exponentialRampToValueAtTime(Math.max(0.001, masterPeak * 0.38), now + attackTime + 0.18);
        voiceGain.gain.exponentialRampToValueAtTime(0.0002, now + dur + stringDecay);

        // Per-note velocity-controlled lowpass filter
        const noteLP = ctx.createBiquadFilter();
        noteLP.type = 'lowpass';
        const startCutoff = freq * (this.preset === 'mellow_felt' ? 4.2 : 7.5) * (0.65 + vel * 0.55) * (1.15 - normW * 0.55);
        const endCutoff = Math.max(freq * 1.3, 320);
        noteLP.frequency.setValueAtTime(Math.min(12000, startCutoff), now);
        noteLP.frequency.exponentialRampToValueAtTime(endCutoff, now + dur + 0.5);

        noteLP.connect(voiceGain);
        voiceGain.connect(this.bodyLowEQ);

        const stopTime = now + dur + stringDecay + 0.05;
        this.registerSustainedVoice(voiceGain, stopTime);

        if (this.preset === 'electric_piano') {
          // Warm Vintage Rhodes Tine + Fundamental Voice
          const osc1 = ctx.createOscillator();
          const osc2 = ctx.createOscillator();
          const tineGain = ctx.createGain();

          osc1.type = 'sine';
          osc1.frequency.value = freq;

          // Bell tine overtone
          osc2.type = 'sine';
          osc2.frequency.value = freq * 7.0;
          tineGain.gain.setValueAtTime(0.18 * vel, now);
          tineGain.gain.exponentialRampToValueAtTime(0.001, now + 0.14);

          osc1.connect(noteLP);
          osc2.connect(tineGain);
          tineGain.connect(noteLP);

          osc1.start(now);
          osc2.start(now);
          osc1.stop(stopTime);
          osc2.stop(stopTime);
          return;
        }

        // Acoustic Piano String Partials with natural piano inharmonicity & unison string detuning
        const inharm = 0.00018 * Math.pow((midi - 21) / 67, 2);
        const partials = [
          { mult: 1.0, amp: 1.00, detuneCents: -1.8, type: 'triangle' },
          { mult: 1.0, amp: 0.85, detuneCents: +1.8, type: 'sine' },
          { mult: 2.0 * (1 + inharm), amp: 0.42 * (1 - normW * 0.35), detuneCents: 0.8, type: 'sine' },
          { mult: 3.0 * (1 + inharm * 2.4), amp: 0.16 * (1 - normW * 0.55), detuneCents: -1.2, type: 'sine' },
          { mult: 4.0 * (1 + inharm * 4.2), amp: 0.06 * (1 - normW * 0.75), detuneCents: 1.5, type: 'sine' }
        ];

        for (let i = 0; i < partials.length; i++) {
          const p = partials[i];
          const pFreq = freq * p.mult;
          if (pFreq > 11000 || p.amp <= 0.005) continue;

          const osc = ctx.createOscillator();
          const pGain = ctx.createGain();
          osc.type = p.type;
          osc.frequency.value = pFreq;
          osc.detune.value = (this.preset === 'warm_upright') ? p.detuneCents * 2.4 : p.detuneCents;

          // Upper harmonics decay much faster than the fundamental on an acoustic piano
          const pGainVal = p.amp;
          pGain.gain.setValueAtTime(pGainVal, now);
          if (i >= 2) {
            pGain.gain.exponentialRampToValueAtTime(0.001, now + Math.max(0.25, 0.9 / p.mult));
          }

          osc.connect(pGain);
          pGain.connect(noteLP);
          osc.start(now);
          osc.stop(stopTime);
        }

        // Soft Felt Hammer Thump Transient (adds authentic physical key-strike warmth)
        if (this.noiseBuffer) {
          const noiseSrc = ctx.createBufferSource();
          noiseSrc.buffer = this.noiseBuffer;

          const hammerBP = ctx.createBiquadFilter();
          hammerBP.type = 'bandpass';
          hammerBP.frequency.value = Math.min(1800, Math.max(240, freq * (this.preset === 'mellow_felt' ? 1.2 : 2.2)));
          hammerBP.Q.value = 1.4;

          const hammerGain = ctx.createGain();
          hammerGain.gain.value = vel * (this.preset === 'mellow_felt' ? 0.08 : 0.14);

          noiseSrc.connect(hammerBP);
          hammerBP.connect(hammerGain);
          hammerGain.connect(noteLP);
          noiseSrc.start(now);
        }
      } catch (e) {}
    }
  }

  const audioEngine = new PureAcousticPianoEngine();

  global.PureAcousticPianoEngine = PureAcousticPianoEngine;
  global.audioEngine = audioEngine;
})(window);
