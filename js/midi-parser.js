(function (global) {
  'use strict';

  /********************************************************************
   * NATIVE BINARY .MID FILE PARSER (ZERO EXTERNAL DEPENDENCIES)
   ********************************************************************/
  function parseStandardMidiFile(arrayBuffer, fileName) {
    const view = new DataView(arrayBuffer);
    let pos = 0;

    function readStr(len) {
      let s = '';
      for (let i = 0; i < len && pos < view.byteLength; i++) {
        s += String.fromCharCode(view.getUint8(pos++));
      }
      return s;
    }
    function readU16() { const v = view.getUint16(pos, false); pos += 2; return v; }
    function readU32() { const v = view.getUint32(pos, false); pos += 4; return v; }
    function readVarLen() {
      let val = 0;
      for (let i = 0; i < 4 && pos < view.byteLength; i++) {
        const b = view.getUint8(pos++);
        val = (val << 7) | (b & 0x7f);
        if ((b & 0x80) === 0) break;
      }
      return val;
    }

    if (readStr(4) !== 'MThd') throw new Error('Invalid MIDI header');
    const hdrLen = readU32();
    const format = readU16();
    const numTracks = readU16();
    const division = readU16();
    pos = 8 + hdrLen;

    const ticksPerBeat = (division & 0x8000) ? 480 : (division || 480);
    const tempoEvents = [{ tick: 0, usPerBeat: 500000 }]; // Default 120 BPM
    const rawTracks = [];

    for (let t = 0; t < numTracks && pos < view.byteLength; t++) {
      const chunkId = readStr(4);
      const chunkLen = readU32();
      const trackEnd = Math.min(view.byteLength, pos + chunkLen);
      if (chunkId !== 'MTrk') {
        pos = trackEnd;
        continue;
      }

      let tick = 0;
      let runningStatus = 0;
      const activeNotes = {};
      const trackNotes = [];

      while (pos < trackEnd) {
        tick += readVarLen();
        if (pos >= trackEnd) break;

        let status = view.getUint8(pos);
        if (status < 0x80) {
          status = runningStatus;
        } else {
          pos++;
          if (status < 0xf0) runningStatus = status;
        }

        const eventType = status & 0xf0;
        const channel = status & 0x0f;

        if (eventType === 0x90 || eventType === 0x80) {
          const noteNum = view.getUint8(pos++);
          const vel = view.getUint8(pos++);
          const isNoteOn = (eventType === 0x90 && vel > 0);
          const key = channel + '_' + noteNum;

          if (isNoteOn) {
            activeNotes[key] = { startTick: tick, vel: vel / 127, midi: noteNum };
          } else if (activeNotes[key]) {
            const startObj = activeNotes[key];
            delete activeNotes[key];
            trackNotes.push({
              midi: noteNum,
              startTick: startObj.startTick,
              endTick: Math.max(startObj.startTick + 20, tick),
              velocity: startObj.vel
            });
          }
        } else if (eventType === 0xa0 || eventType === 0xb0 || eventType === 0xe0) {
          pos += 2;
        } else if (eventType === 0xc0 || eventType === 0xd0) {
          pos += 1;
        } else if (status === 0xff) {
          const metaType = view.getUint8(pos++);
          const metaLen = readVarLen();
          if (metaType === 0x51 && metaLen === 3 && pos + 3 <= trackEnd) {
            const us = (view.getUint8(pos) << 16) | (view.getUint8(pos + 1) << 8) | view.getUint8(pos + 2);
            tempoEvents.push({ tick, usPerBeat: us });
          }
          pos += metaLen;
        } else if (status === 0xf0 || status === 0xf7) {
          const sysLen = readVarLen();
          pos += sysLen;
        } else {
          pos++;
        }
      }

      if (trackNotes.length > 0) {
        rawTracks.push(trackNotes);
      }
      pos = trackEnd;
    }

    tempoEvents.sort((a, b) => a.tick - b.tick);

    function tickToSeconds(targetTick) {
      let sec = 0;
      let prevTick = 0;
      let curUs = 500000;
      for (let i = 0; i < tempoEvents.length; i++) {
        const te = tempoEvents[i];
        if (te.tick >= targetTick) break;
        sec += ((te.tick - prevTick) / ticksPerBeat) * (curUs / 1000000);
        prevTick = te.tick;
        curUs = te.usPerBeat;
      }
      sec += ((targetTick - prevTick) / ticksPerBeat) * (curUs / 1000000);
      return sec;
    }

    const finalNotes = [];
    if (rawTracks.length >= 2) {
      const avgs = rawTracks.map(tr => tr.reduce((s, n) => s + n.midi, 0) / tr.length);
      const maxAvgIdx = avgs.indexOf(Math.max.apply(null, avgs));
      rawTracks.forEach((tr, idx) => {
        const hand = (idx === maxAvgIdx) ? 'R' : 'L';
        tr.forEach(n => {
          const t0 = tickToSeconds(n.startTick);
          const t1 = tickToSeconds(n.endTick);
          finalNotes.push({
            midi: n.midi,
            name: global.midiToName(n.midi),
            time: t0 + 0.8,
            duration: Math.max(0.12, t1 - t0),
            hand,
            velocity: n.velocity || 0.78
          });
        });
      });
    } else if (rawTracks.length === 1) {
      rawTracks[0].forEach(n => {
        const t0 = tickToSeconds(n.startTick);
        const t1 = tickToSeconds(n.endTick);
        finalNotes.push({
          midi: n.midi,
          name: global.midiToName(n.midi),
          time: t0 + 0.8,
          duration: Math.max(0.12, t1 - t0),
          hand: n.midi < 60 ? 'L' : 'R',
          velocity: n.velocity || 0.78
        });
      });
    }

    if (finalNotes.length === 0) throw new Error('No playable notes in MIDI');
    finalNotes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const maxEnd = finalNotes.reduce((m, n) => Math.max(m, n.time + n.duration), 0);

    return {
      title: fileName.replace(/\.(mid|midi)$/i, ''),
      notes: finalNotes,
      duration: maxEnd + 2.0
    };
  }

  /********************************************************************
   * AUTO-SCALING OCTAVE RANGE CALCULATOR
   ********************************************************************/
  function calculateAutoRange(state) {
    const notes = state.song.notes;
    if (state.rangeMode === 'full88') {
      state.minMidi = 21;
      state.maxMidi = 108;
    } else if (state.rangeMode === 'compact') {
      let sum = 0;
      for (let i = 0; i < notes.length; i++) sum += notes[i].midi;
      const avg = notes.length ? Math.round(sum / notes.length) : 60;
      const lowC = Math.max(24, Math.floor(avg / 12) * 12 - 12);
      state.minMidi = lowC;
      state.maxMidi = Math.min(108, lowC + 24);
    } else {
      // Auto-Scale Range: exact octaves used by the song (minimum 2 octaves, max 88 keys)
      let minM = 108;
      let maxM = 21;
      for (let i = 0; i < notes.length; i++) {
        const m = notes[i].midi;
        if (m < minM) minM = m;
        if (m > maxM) maxM = m;
      }
      if (minM > maxM) { minM = 48; maxM = 72; }

      let lowC = Math.floor(minM / 12) * 12;
      let highC = Math.ceil((maxM + 1) / 12) * 12;

      while (highC - lowC < 24) {
        if (lowC >= 36) lowC -= 12;
        if (highC - lowC < 24 && highC <= 96) highC += 12;
      }
      state.minMidi = Math.max(21, lowC);
      state.maxMidi = Math.min(108, highC);
    }

    const badge = document.getElementById('activeRangeBadge');
    if (badge) {
      badge.textContent = global.midiToName(state.minMidi) + '–' + global.midiToName(state.maxMidi);
    }
  }

  global.parseStandardMidiFile = parseStandardMidiFile;
  global.calculateAutoRange = calculateAutoRange;
})(window);
