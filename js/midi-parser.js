(function (global) {
  'use strict';

  /********************************************************************
   * SMART LEFT/RIGHT HAND VOICE-SPLITTING ALGORITHM (PHASE 2)
   * - Groups simultaneous notes within 35ms (0.035s) into onset clusters
   * - Enforces human hand-span limit (max 14 semitones = 10th interval)
   * - Splits wide clusters at the largest internal pitch gap
   * - Tracks moving pitch centers (EMA initialized at 48 for L, 67 for R)
   *   with a soft split bias around C4 (MIDI 60)
   ********************************************************************/
  function assignHandsSmart(notes) {
    if (!Array.isArray(notes) || notes.length === 0) return notes;

    const MAX_HAND_SPAN = 14; // 10th interval (14 semitones)
    const CLUSTER_WINDOW_SEC = 0.035; // 35ms simultaneous onset window
    const EMA_ALPHA = 0.28;

    // Sort notes chronologically, then ascending by pitch
    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);

    // 1. Group into simultaneous onset clusters (within 35ms)
    const clusters = [];
    let currentCluster = [notes[0]];

    for (let i = 1; i < notes.length; i++) {
      const n = notes[i];
      if (n.time - currentCluster[0].time <= CLUSTER_WINDOW_SEC) {
        currentCluster.push(n);
      } else {
        clusters.push(currentCluster);
        currentCluster = [n];
      }
    }
    if (currentCluster.length > 0) {
      clusters.push(currentCluster);
    }

    // 2. Track moving pitch centers for Left (L) and Right (R) hands
    let leftCenter = 48;  // C3
    let rightCenter = 67; // G4

    for (let cIdx = 0; cIdx < clusters.length; cIdx++) {
      const cNotes = clusters[cIdx];
      cNotes.sort((a, b) => a.midi - b.midi);

      const minPitch = cNotes[0].midi;
      const maxPitch = cNotes[cNotes.length - 1].midi;
      const totalSpan = maxPitch - minPitch;

      if (cNotes.length === 1) {
        const p = cNotes[0].midi;
        if (p < 46) {
          cNotes[0].hand = 'L';
        } else if (p > 74) {
          cNotes[0].hand = 'R';
        } else {
          const midPivot = (leftCenter + rightCenter) * 0.5;
          const distL = Math.abs(p - leftCenter) + (p >= 60 ? (p - 60) * 0.35 : 0);
          const distR = Math.abs(p - rightCenter) + (p < 60 ? (60 - p) * 0.35 : 0);

          if (Math.abs(distL - distR) < 1.2) {
            cNotes[0].hand = (p < 60 || p < midPivot) ? 'L' : 'R';
          } else {
            cNotes[0].hand = (distL < distR) ? 'L' : 'R';
          }
        }
      } else if (totalSpan > MAX_HAND_SPAN) {
        // Total span exceeds 14 semitones: must split between Left and Right hands
        // Find split index k (1..cNotes.length-1) that maximizes the internal pitch gap
        // while keeping both hands within MAX_HAND_SPAN if possible.
        const midPivot = (leftCenter + rightCenter) * 0.5;
        let bestSplit = 1;
        let bestScore = -9999;

        for (let k = 1; k < cNotes.length; k++) {
          const leftSpan = cNotes[k - 1].midi - cNotes[0].midi;
          const rightSpan = cNotes[cNotes.length - 1].midi - cNotes[k].midi;
          const gap = cNotes[k].midi - cNotes[k - 1].midi;
          const splitBoundaryPitch = (cNotes[k - 1].midi + cNotes[k].midi) * 0.5;

          const overflowPenalty =
            Math.max(0, leftSpan - MAX_HAND_SPAN) * 4.5 +
            Math.max(0, rightSpan - MAX_HAND_SPAN) * 4.5;
          const pivotPenalty = Math.abs(splitBoundaryPitch - midPivot) * 0.22 +
            Math.abs(splitBoundaryPitch - 60) * 0.15;

          const score = gap * 2.0 - overflowPenalty - pivotPenalty;
          if (score > bestScore) {
            bestScore = score;
            bestSplit = k;
          }
        }

        for (let i = 0; i < cNotes.length; i++) {
          cNotes[i].hand = (i < bestSplit) ? 'L' : 'R';
        }
      } else {
        // Compact cluster (totalSpan <= 14 semitones, 2+ notes)
        // Check if there is a wide internal gap (>= 7 semitones) straddling the L/R centers
        let maxGap = 0;
        let maxGapIdx = -1;
        for (let k = 1; k < cNotes.length; k++) {
          const gap = cNotes[k].midi - cNotes[k - 1].midi;
          if (gap > maxGap) {
            maxGap = gap;
            maxGapIdx = k;
          }
        }

        const lowAvg = cNotes.slice(0, maxGapIdx).reduce((s, n) => s + n.midi, 0) / Math.max(1, maxGapIdx);
        const highAvg = cNotes.slice(maxGapIdx).reduce((s, n) => s + n.midi, 0) / Math.max(1, cNotes.length - maxGapIdx);

        const straddlesHands =
          maxGap >= 7 &&
          lowAvg <= 60 &&
          highAvg >= 59 &&
          Math.abs(lowAvg - leftCenter) <= Math.abs(lowAvg - rightCenter) + 2 &&
          Math.abs(highAvg - rightCenter) <= Math.abs(highAvg - leftCenter) + 2;

        if (straddlesHands) {
          for (let i = 0; i < cNotes.length; i++) {
            cNotes[i].hand = (i < maxGapIdx) ? 'L' : 'R';
          }
        } else {
          // Assign the compact chord to a single hand based on moving centers + C4 soft bias
          const clusterAvg = cNotes.reduce((s, n) => s + n.midi, 0) / cNotes.length;
          const distL = Math.abs(clusterAvg - leftCenter) + (clusterAvg >= 60 ? (clusterAvg - 60) * 0.4 : 0);
          const distR = Math.abs(clusterAvg - rightCenter) + (clusterAvg < 60 ? (60 - clusterAvg) * 0.4 : 0);
          const assignedHand = (Math.abs(distL - distR) < 1.2)
            ? (clusterAvg < 60 ? 'L' : 'R')
            : (distL < distR ? 'L' : 'R');

          for (let i = 0; i < cNotes.length; i++) {
            cNotes[i].hand = assignedHand;
          }
        }
      }

      // 3. Update exponential moving average pitch centers
      let lSum = 0, lCount = 0;
      let rSum = 0, rCount = 0;
      for (let i = 0; i < cNotes.length; i++) {
        if (cNotes[i].hand === 'L') {
          lSum += cNotes[i].midi;
          lCount++;
        } else {
          rSum += cNotes[i].midi;
          rCount++;
        }
      }

      if (lCount > 0) {
        const lAvg = lSum / lCount;
        leftCenter = Math.max(33, Math.min(62, leftCenter * (1 - EMA_ALPHA) + lAvg * EMA_ALPHA));
      }
      if (rCount > 0) {
        const rAvg = rSum / rCount;
        rightCenter = Math.max(56, Math.min(90, rightCenter * (1 - EMA_ALPHA) + rAvg * EMA_ALPHA));
      }
      if (leftCenter > rightCenter - 6) {
        const mid = (leftCenter + rightCenter) * 0.5;
        leftCenter = Math.min(57, mid - 3);
        rightCenter = Math.max(63, mid + 3);
      }
    }

    return notes;
  }

  /********************************************************************
   * NATIVE BINARY .MID FILE PARSER (ZERO EXTERNAL DEPENDENCIES)
   * - Parses tempo events (0x51) and time signature events (0x58)
   * - Generates synchronized measure & beat gridlines
   * - Uses assignHandsSmart for single-track or dominant-track MIDIs
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
    const timeSigEvents = [{ tick: 0, numerator: 4, denominator: 4 }]; // Default 4/4
    const rawTracks = [];
    let maxSongTick = 0;

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
            const endTick = Math.max(startObj.startTick + 20, tick);
            if (endTick > maxSongTick) maxSongTick = endTick;
            trackNotes.push({
              midi: noteNum,
              startTick: startObj.startTick,
              endTick: endTick,
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
          } else if (metaType === 0x58 && metaLen >= 2 && pos + 2 <= trackEnd) {
            const num = Math.max(1, view.getUint8(pos));
            const denPow = view.getUint8(pos + 1);
            const den = Math.max(1, Math.pow(2, denPow));
            timeSigEvents.push({ tick, numerator: num, denominator: den });
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
    timeSigEvents.sort((a, b) => a.tick - b.tick);

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
    const totalTrackNotes = rawTracks.reduce((sum, tr) => sum + tr.length, 0);
    const maxSingleTrackNotes = rawTracks.reduce((max, tr) => Math.max(max, tr.length), 0);
    const useSmartSplit = (rawTracks.length === 1) ||
      (totalTrackNotes > 0 && (maxSingleTrackNotes / totalTrackNotes) > 0.80);

    if (rawTracks.length >= 2 && !useSmartSplit) {
      const avgs = rawTracks.map(tr => tr.reduce((s, n) => s + n.midi, 0) / tr.length);
      const maxAvgIdx = avgs.indexOf(Math.max.apply(null, avgs));
      rawTracks.forEach((tr, idx) => {
        const hand = (idx === maxAvgIdx || avgs[idx] >= 60) ? 'R' : 'L';
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
    } else {
      rawTracks.forEach(tr => {
        tr.forEach(n => {
          const t0 = tickToSeconds(n.startTick);
          const t1 = tickToSeconds(n.endTick);
          finalNotes.push({
            midi: n.midi,
            name: global.midiToName(n.midi),
            time: t0 + 0.8,
            duration: Math.max(0.12, t1 - t0),
            hand: 'R',
            velocity: n.velocity || 0.78
          });
        });
      });
      assignHandsSmart(finalNotes);
    }

    if (finalNotes.length === 0) throw new Error('No playable notes in MIDI');
    finalNotes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const maxEnd = finalNotes.reduce((m, n) => Math.max(m, n.time + n.duration), 0);
    const duration = maxEnd + 2.0;

    // Generate MIDI-synchronized measure & beat gridlines
    const beats = [];
    let curTick = 0;
    let barNumber = 1;
    let tsIdx = 0;
    const endGridTick = maxSongTick + ticksPerBeat * 4;

    while (curTick <= endGridTick) {
      while (tsIdx + 1 < timeSigEvents.length && timeSigEvents[tsIdx + 1].tick <= curTick) {
        tsIdx++;
      }
      const ts = timeSigEvents[tsIdx];
      const num = Math.max(1, ts.numerator || 4);
      const den = Math.max(1, ts.denominator || 4);
      const ticksPerUnitBeat = Math.max(20, Math.round(ticksPerBeat * (4 / den)));

      for (let b = 1; b <= num; b++) {
        const beatTime = tickToSeconds(curTick) + 0.8;
        if (beatTime <= duration + 0.1) {
          beats.push({
            time: Math.round(beatTime * 1000) / 1000,
            isMeasure: b === 1,
            barNumber: barNumber,
            beatNumber: b
          });
        }
        curTick += ticksPerUnitBeat;
        if (tsIdx + 1 < timeSigEvents.length && curTick >= timeSigEvents[tsIdx + 1].tick && b < num) {
          curTick = timeSigEvents[tsIdx + 1].tick;
          break;
        }
      }
      barNumber++;
      if (beats.length > 4000) break; // Safety cap for extremely long files
    }

    return {
      title: fileName.replace(/\.(mid|midi)$/i, ''),
      notes: finalNotes,
      duration: duration,
      beats: beats
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

  global.assignHandsSmart = assignHandsSmart;
  global.parseStandardMidiFile = parseStandardMidiFile;
  global.calculateAutoRange = calculateAutoRange;
})(window);
