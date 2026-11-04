/**
 * KeyFlow — Phase 5: Hybrid Sheet Music Importer
 * ===============================================
 * Provides:
 * 1. Native compressed .MXL (ZIP) extractor via browser DecompressionStream.
 * 2. Comprehensive MusicXML DOM parser supporting:
 *    - Grand staff piano scores (Staff 1 -> Right Hand 'R', Staff 2 -> Left Hand 'L').
 *    - Accurate running timeline with <backup> and <forward> voices.
 *    - Chords (<chord/>), ties merging (<tie>/<tied>), divisions, time signature, BPM.
 *    - Finger numbers (<notations><technical><fingering>) mapped to Phase 4B note.finger.
 *    - Multi-part separation (RH/LH split across Part 1 & Part 2).
 *    - Automatic lead-in grid offset (0.8s) and synchronized measure beat gridlines.
 * 3. Remote Optical Music Recognition (OMR) client for .PDF, .PNG, and .JPG sheet music
 *    via KeyFlow AI Server (POST /api/omr).
 * 4. Unified entry point: window.sheetMusicImporter.importSheetMusicFile(file, options, onProgress).
 */

(function (global) {
  'use strict';

  /********************************************************************
   * 1. NATIVE .MXL (ZIP) LOCAL FILE HEADER PARSER & DECOMPRESSOR
   ********************************************************************/

  async function decompressDeflateRaw(compressedBytes) {
    if (typeof DecompressionStream !== 'undefined') {
      try {
        const ds = new DecompressionStream('deflate-raw');
        const writer = ds.writable.getWriter();
        writer.write(compressedBytes);
        writer.close();
        const resp = new Response(ds.readable);
        const buf = await resp.arrayBuffer();
        return new TextDecoder('utf-8').decode(buf);
      } catch (err) {
        // Fallback to standard deflate stream with zlib header
        try {
          const ds2 = new DecompressionStream('deflate');
          const writer2 = ds2.writable.getWriter();
          writer2.write(compressedBytes);
          writer2.close();
          const resp2 = new Response(ds2.readable);
          const buf2 = await resp2.arrayBuffer();
          return new TextDecoder('utf-8').decode(buf2);
        } catch (e2) {}
      }
    }
    throw new Error('DecompressionStream is not supported in this browser for compressed .mxl archives.');
  }

  async function extractMusicXmlFromMxl(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    const byteLen = arrayBuffer.byteLength;
    const entries = [];

    // Attempt 1: Locate End of Central Directory Record (EOCD: 0x06054b50) to read Central Directory
    let cdOffset = -1;
    const minEocdSearch = Math.max(0, byteLen - 65557);
    for (let p = byteLen - 22; p >= minEocdSearch; p--) {
      if (view.getUint32(p, true) === 0x06054b50) {
        cdOffset = view.getUint32(p + 16, true);
        break;
      }
    }

    // Read Central Directory entries (accurate compressedSize and method even with ZIP data descriptors)
    if (cdOffset >= 0 && cdOffset < byteLen) {
      let pos = cdOffset;
      while (pos + 46 <= byteLen && view.getUint32(pos, true) === 0x02014b50) {
        const method = view.getUint16(pos + 10, true);
        const compressedSize = view.getUint32(pos + 20, true);
        const uncompressedSize = view.getUint32(pos + 24, true);
        const fileNameLen = view.getUint16(pos + 28, true);
        const extraLen = view.getUint16(pos + 30, true);
        const commentLen = view.getUint16(pos + 32, true);
        const localHeaderOffset = view.getUint32(pos + 42, true);

        const nameBytes = new Uint8Array(arrayBuffer, pos + 46, fileNameLen);
        const fileName = new TextDecoder('utf-8').decode(nameBytes);

        // Compute data offset from the entry's Local File Header
        if (localHeaderOffset + 30 <= byteLen && view.getUint32(localHeaderOffset, true) === 0x04034b50) {
          const localNameLen = view.getUint16(localHeaderOffset + 26, true);
          const localExtraLen = view.getUint16(localHeaderOffset + 28, true);
          const dataOffset = localHeaderOffset + 30 + localNameLen + localExtraLen;

          if (dataOffset + compressedSize <= byteLen) {
            entries.push({
              name: fileName,
              method: method,
              compressedSize: compressedSize,
              uncompressedSize: uncompressedSize,
              offset: dataOffset
            });
          }
        }

        pos += 46 + fileNameLen + extraLen + commentLen;
      }
    }

    // Attempt 2: If EOCD failed, scan for Central Directory headers (0x02014b50) anywhere in buffer
    if (entries.length === 0) {
      let pos = 0;
      while (pos + 46 <= byteLen) {
        if (view.getUint32(pos, true) === 0x02014b50) {
          const method = view.getUint16(pos + 10, true);
          const compressedSize = view.getUint32(pos + 20, true);
          const uncompressedSize = view.getUint32(pos + 24, true);
          const fileNameLen = view.getUint16(pos + 28, true);
          const extraLen = view.getUint16(pos + 30, true);
          const commentLen = view.getUint16(pos + 32, true);
          const localHeaderOffset = view.getUint32(pos + 42, true);

          const nameBytes = new Uint8Array(arrayBuffer, pos + 46, fileNameLen);
          const fileName = new TextDecoder('utf-8').decode(nameBytes);

          if (localHeaderOffset + 30 <= byteLen && view.getUint32(localHeaderOffset, true) === 0x04034b50) {
            const localNameLen = view.getUint16(localHeaderOffset + 26, true);
            const localExtraLen = view.getUint16(localHeaderOffset + 28, true);
            const dataOffset = localHeaderOffset + 30 + localNameLen + localExtraLen;

            if (dataOffset + compressedSize <= byteLen) {
              entries.push({
                name: fileName,
                method: method,
                compressedSize: compressedSize,
                uncompressedSize: uncompressedSize,
                offset: dataOffset
              });
            }
          }

          pos += 46 + fileNameLen + extraLen + commentLen;
        } else {
          pos++;
        }
      }
    }

    // Attempt 3: Fallback to Local File Header scan (0x04034b50) only if no Central Directory was found
    if (entries.length === 0) {
      let pos = 0;
      while (pos + 30 <= byteLen) {
        if (view.getUint32(pos, true) !== 0x04034b50) {
          pos++;
          continue;
        }

        const method = view.getUint16(pos + 8, true);
        const compressedSize = view.getUint32(pos + 18, true);
        const uncompressedSize = view.getUint32(pos + 22, true);
        const fileNameLen = view.getUint16(pos + 26, true);
        const extraLen = view.getUint16(pos + 28, true);

        const nameBytes = new Uint8Array(arrayBuffer, pos + 30, fileNameLen);
        const fileName = new TextDecoder('utf-8').decode(nameBytes);

        const dataOffset = pos + 30 + fileNameLen + extraLen;
        if (compressedSize > 0 && dataOffset + compressedSize <= byteLen) {
          entries.push({
            name: fileName,
            method: method,
            compressedSize: compressedSize,
            uncompressedSize: uncompressedSize,
            offset: dataOffset
          });
          pos = dataOffset + compressedSize;
        } else {
          pos += 30 + fileNameLen + extraLen;
        }
      }
    }

    if (entries.length === 0) {
      throw new Error('Invalid .mxl file: No ZIP file headers found.');
    }

    // 1. Try to find the rootfile specified in META-INF/container.xml
    let targetFileName = null;
    const containerEntry = entries.find(e => e.name.toLowerCase() === 'meta-inf/container.xml');
    if (containerEntry) {
      try {
        let containerXml = '';
        const rawData = new Uint8Array(arrayBuffer, containerEntry.offset, containerEntry.compressedSize);
        if (containerEntry.method === 0) {
          containerXml = new TextDecoder('utf-8').decode(rawData);
        } else if (containerEntry.method === 8) {
          containerXml = await decompressDeflateRaw(rawData);
        }
        if (containerXml) {
          const m = containerXml.match(/full-path\s*=\s*["']([^"']+)["']/i);
          if (m && m[1]) {
            targetFileName = m[1].trim();
          }
        }
      } catch (cErr) {
        console.warn('[SheetMusicImporter] Failed to parse container.xml:', cErr);
      }
    }

    // 2. Select target entry: either matching full-path or largest non-container XML entry
    let targetEntry = null;
    if (targetFileName) {
      targetEntry = entries.find(e => e.name.toLowerCase() === targetFileName.toLowerCase());
    }

    if (!targetEntry) {
      const xmlCandidates = entries.filter(e => {
        const lower = e.name.toLowerCase();
        return (lower.endsWith('.musicxml') || lower.endsWith('.xml')) &&
               !lower.startsWith('meta-inf/') &&
               !lower.endsWith('container.xml');
      });
      xmlCandidates.sort((a, b) => b.compressedSize - a.compressedSize);
      targetEntry = xmlCandidates[0];
    }

    if (!targetEntry) {
      throw new Error('No .musicxml score found inside .mxl package.');
    }

    const compressedBytes = new Uint8Array(arrayBuffer, targetEntry.offset, targetEntry.compressedSize);
    if (targetEntry.method === 0) {
      return new TextDecoder('utf-8').decode(compressedBytes);
    } else if (targetEntry.method === 8) {
      return await decompressDeflateRaw(compressedBytes);
    } else {
      throw new Error('Unsupported compression method in .mxl: ' + targetEntry.method);
    }
  }

  /********************************************************************
   * 2. COMPREHENSIVE MUSICXML DOM PARSER (GRAND STAFF, VOICES, TIES)
   ********************************************************************/

  const STEP_OFFSETS = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

  function parseMusicXmlString(xmlString, fallbackTitle = 'Imported Sheet Music') {
    if (!xmlString || typeof xmlString !== 'string') {
      throw new Error('Empty MusicXML document.');
    }

    const dom = new DOMParser().parseFromString(xmlString, 'application/xml');
    const parserError = dom.querySelector('parsererror');
    if (parserError) {
      throw new Error('MusicXML XML syntax error: ' + parserError.textContent.slice(0, 160));
    }

    // 1. Extract Score Title
    let title = '';
    const workTitle = dom.querySelector('work > work-title');
    if (workTitle && workTitle.textContent.trim()) {
      title = workTitle.textContent.trim();
    }
    if (!title) {
      const movTitle = dom.querySelector('movement-title');
      if (movTitle && movTitle.textContent.trim()) {
        title = movTitle.textContent.trim();
      }
    }
    if (!title) {
      const creditWords = dom.querySelectorAll('credit > credit-words');
      for (let i = 0; i < creditWords.length; i++) {
        const txt = creditWords[i].textContent.trim();
        if (txt.length > 1 && txt.length < 80) {
          title = txt;
          break;
        }
      }
    }
    title = (title || fallbackTitle).replace(/\.[^/.]+$/, '').slice(0, 100);

    // 2. Discover Parts
    const partElements = Array.from(dom.querySelectorAll('score-partwise > part, part'));
    if (partElements.length === 0) {
      throw new Error('MusicXML contains no <part> elements.');
    }

    // State trackers
    let currentBpm = 120;
    let beatsPerBar = 4;
    let beatType = 4;
    let divisions = 4;
    let tempoFound = false;

    // First scan for initial tempo / time signature
    const firstSound = dom.querySelector('sound[tempo]');
    if (firstSound) {
      const t = parseFloat(firstSound.getAttribute('tempo'));
      if (!isNaN(t) && t >= 30 && t <= 260) {
        currentBpm = Math.round(t);
        tempoFound = true;
      }
    }
    if (!tempoFound) {
      const metro = dom.querySelector('per-minute');
      if (metro) {
        const t = parseFloat(metro.textContent.trim());
        if (!isNaN(t) && t >= 30 && t <= 260) {
          currentBpm = Math.round(t);
        }
      }
    }

    const firstBeats = dom.querySelector('time > beats');
    const firstBeatType = dom.querySelector('time > beat-type');
    if (firstBeats) {
      const b = parseInt(firstBeats.textContent.trim(), 10);
      if (!isNaN(b) && b >= 1 && b <= 16) beatsPerBar = b;
    }
    if (firstBeatType) {
      const bt = parseInt(firstBeatType.textContent.trim(), 10);
      if (!isNaN(bt) && bt >= 1 && bt <= 16) beatType = bt;
    }

    const extractedNotes = [];
    const activeTies = new Map(); // key: "staff_voice_midi" -> Note object
    let hasExplicitStaff2 = false;

    // If score splits hands across Part 1 (Right Hand) and Part 2 (Left Hand)
    const isMultiPartPiano = partElements.length >= 2;

    for (let pIdx = 0; pIdx < partElements.length; pIdx++) {
      const partEl = partElements[pIdx];
      const measures = Array.from(partEl.querySelectorAll('measure'));
      let measureStartSec = 0;

      for (let mIdx = 0; mIdx < measures.length; mIdx++) {
        const measure = measures[mIdx];
        let localTicks = 0;
        let lastNoteStartSec = measureStartSec;
        let maxMeasureTicks = 0;

        // Check attributes for divisions, time signature, or tempo changes
        const divEl = measure.querySelector('attributes > divisions');
        if (divEl) {
          const d = parseFloat(divEl.textContent.trim());
          if (!isNaN(d) && d > 0) divisions = d;
        }

        const timeBeats = measure.querySelector('attributes > time > beats');
        const timeBeatType = measure.querySelector('attributes > time > beat-type');
        if (timeBeats) {
          const b = parseInt(timeBeats.textContent.trim(), 10);
          if (!isNaN(b) && b >= 1 && b <= 16) beatsPerBar = b;
        }
        if (timeBeatType) {
          const bt = parseInt(timeBeatType.textContent.trim(), 10);
          if (!isNaN(bt) && bt >= 1 && bt <= 16) beatType = bt;
        }

        const soundEl = measure.querySelector('sound[tempo]');
        if (soundEl) {
          const t = parseFloat(soundEl.getAttribute('tempo'));
          if (!isNaN(t) && t >= 30 && t <= 260) currentBpm = Math.round(t);
        }

        const secondsPerTick = (60 / currentBpm) / divisions;

        // Iterate through all child elements of measure in document order
        const children = Array.from(measure.children);
        for (let cIdx = 0; cIdx < children.length; cIdx++) {
          const child = children[cIdx];
          const tag = child.tagName.toLowerCase();

          if (tag === 'backup') {
            const durEl = child.querySelector('duration');
            const d = durEl ? parseFloat(durEl.textContent.trim()) || 0 : 0;
            localTicks = Math.max(0, localTicks - d);
            continue;
          }

          if (tag === 'forward') {
            const durEl = child.querySelector('duration');
            const d = durEl ? parseFloat(durEl.textContent.trim()) || 0 : 0;
            localTicks += d;
            if (localTicks > maxMeasureTicks) maxMeasureTicks = localTicks;
            continue;
          }

          if (tag !== 'note') continue;

          // Check if rest
          const isRest = child.querySelector('rest') !== null;
          const isGrace = child.querySelector('grace') !== null;
          const isChord = child.querySelector('chord') !== null;

          const durEl = child.querySelector('duration');
          const durTicks = durEl ? parseFloat(durEl.textContent.trim()) || 0 : 0;

          if (isRest) {
            if (!isChord) {
              localTicks += durTicks;
              if (localTicks > maxMeasureTicks) maxMeasureTicks = localTicks;
            }
            continue;
          }

          // Pitch extraction
          const pitchEl = child.querySelector('pitch');
          if (!pitchEl) {
            if (!isChord && !isGrace) {
              localTicks += durTicks;
              if (localTicks > maxMeasureTicks) maxMeasureTicks = localTicks;
            }
            continue;
          }

          const stepEl = pitchEl.querySelector('step');
          const octaveEl = pitchEl.querySelector('octave');
          const alterEl = pitchEl.querySelector('alter');

          const step = stepEl ? stepEl.textContent.trim().toUpperCase() : 'C';
          const octave = octaveEl ? parseInt(octaveEl.textContent.trim(), 10) : 4;
          const alter = alterEl ? parseFloat(alterEl.textContent.trim()) || 0 : 0;
          const stepOff = STEP_OFFSETS[step] !== undefined ? STEP_OFFSETS[step] : 0;
          const midi = Math.max(21, Math.min(108, Math.round((octave + 1) * 12 + stepOff + alter)));

          // Note timing
          let noteStartSec;
          if (isChord) {
            noteStartSec = lastNoteStartSec;
          } else {
            noteStartSec = measureStartSec + (localTicks * secondsPerTick);
            lastNoteStartSec = noteStartSec;
          }

          let noteDurSec = Math.max(0.06, durTicks * secondsPerTick);
          if (isGrace) {
            noteDurSec = 0.08;
            noteStartSec = Math.max(0, noteStartSec - 0.08);
          }

          // Staff / Hand Assignment:
          // Staff 1 = Treble (Right Hand 'R')
          // Staff 2 = Bass (Left Hand 'L')
          const staffEl = child.querySelector('staff');
          let staffNum = staffEl ? parseInt(staffEl.textContent.trim(), 10) : (isMultiPartPiano ? (pIdx + 1) : 1);
          if (staffNum === 2) hasExplicitStaff2 = true;

          let hand = (staffNum === 2) ? 'L' : 'R';
          if (!staffEl && !isMultiPartPiano) {
            hand = (midi < 60) ? 'L' : 'R';
          }

          // Voice
          const voiceEl = child.querySelector('voice');
          const voice = voiceEl ? voiceEl.textContent.trim() : '1';
          const tieKey = staffNum + '_' + voice + '_' + midi;

          // Ties (<tie type="start|stop"/> or <tied type="start|stop"/>)
          const tieStart = child.querySelector('tie[type="start"], tied[type="start"]') !== null;
          const tieStop = child.querySelector('tie[type="stop"], tied[type="stop"]') !== null;

          if (tieStop && activeTies.has(tieKey)) {
            // Merge duration into the previous tied note
            const priorNote = activeTies.get(tieKey);
            priorNote.duration = Math.max(priorNote.duration, (noteStartSec + noteDurSec) - priorNote.time);
            if (!tieStart) {
              activeTies.delete(tieKey);
            }
          } else {
            // New note instance
            const noteObj = {
              midi: midi,
              name: global.midiToName ? global.midiToName(midi) : 'C4',
              time: Math.round(noteStartSec * 1000) / 1000,
              duration: Math.round(noteDurSec * 1000) / 1000,
              hand: hand,
              velocity: 0.82
            };

            // Finger number badge (Phase 4B: 1..5)
            const fingeringEl = child.querySelector('notations > technical > fingering');
            if (fingeringEl) {
              const fNum = parseInt(fingeringEl.textContent.trim(), 10);
              if (!isNaN(fNum) && fNum >= 1 && fNum <= 5) {
                noteObj.finger = fNum;
              }
            }

            extractedNotes.push(noteObj);

            if (tieStart) {
              activeTies.set(tieKey, noteObj);
            }
          }

          if (!isChord && !isGrace) {
            localTicks += durTicks;
            if (localTicks > maxMeasureTicks) maxMeasureTicks = localTicks;
          }
        }

        // Measure advance: if measure duration wasn't advanced by notes, use beatsPerBar * divisions
        const expectedMeasureTicks = beatsPerBar * divisions * (4 / beatType);
        const measureDurationTicks = Math.max(maxMeasureTicks, expectedMeasureTicks);
        measureStartSec += measureDurationTicks * secondsPerTick;
      }
    }

    if (extractedNotes.length === 0) {
      throw new Error('No playable piano notes found in this sheet music file.');
    }

    // Hand balance fallback: if no explicit staff 2 existed and all notes became one hand
    if (!hasExplicitStaff2 && !isMultiPartPiano && typeof global.assignHandsSmart === 'function') {
      const allLeft = extractedNotes.every(n => n.hand === 'L');
      const allRight = extractedNotes.every(n => n.hand === 'R');
      if (allLeft || allRight) {
        global.assignHandsSmart(extractedNotes);
      }
    }

    // Apply standard KeyFlow 0.8s runway grid lead-in offset
    const GRID_OFFSET = 0.8;
    for (let i = 0; i < extractedNotes.length; i++) {
      extractedNotes[i].time = Math.round((extractedNotes[i].time + GRID_OFFSET) * 1000) / 1000;
    }

    // Sort notes strictly by start time, then MIDI pitch
    extractedNotes.sort((a, b) => (a.time - b.time) || (a.midi - b.midi));

    const lastNote = extractedNotes[extractedNotes.length - 1];
    const totalDuration = Math.max(5, Math.ceil(lastNote.time + lastNote.duration + 1.2));

    // Generate beat gridlines synchronized to time signature and tempo
    const secondsPerBeat = (60 / currentBpm) * (4 / beatType);
    let beats = [];
    if (typeof global.generateBeatGrid === 'function') {
      beats = global.generateBeatGrid(totalDuration, secondsPerBeat, beatsPerBar, GRID_OFFSET);
    } else {
      let t = GRID_OFFSET;
      let bIdx = 0;
      while (t < totalDuration) {
        beats.push({
          time: Math.round(t * 1000) / 1000,
          isBar: (bIdx % beatsPerBar) === 0,
          beatInBar: (bIdx % beatsPerBar) + 1
        });
        t += secondsPerBeat;
        bIdx++;
      }
    }

    return {
      title: title,
      notes: extractedNotes,
      duration: totalDuration,
      beats: beats,
      bpm: currentBpm,
      beatsPerBar: beatsPerBar,
      gridOffset: GRID_OFFSET,
      source: 'sheet-music'
    };
  }

  /********************************************************************
   * 3. REMOTE PDF / IMAGE OMR CLIENT (POST /api/omr)
   ********************************************************************/

  async function transcribeSheetVisualViaServer(file, serverUrl = 'http://localhost:8000', onProgress) {
    if (!file) throw new Error('No sheet music visual file provided.');

    let cleanUrl = String(serverUrl || 'http://localhost:8000').trim().replace(/\/+$/, '');
    if (!/^https?:\/\//i.test(cleanUrl)) cleanUrl = 'http://' + cleanUrl;

    if (onProgress) {
      onProgress({ stage: 'Uploading sheet music to AI OMR server...', percent: 15 });
    }

    const formData = new FormData();
    formData.append('file', file);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3-minute timeout for multi-page PDFs

    try {
      if (onProgress) {
        onProgress({ stage: 'AI Optical Music Recognition (OMR) in progress...', percent: 45 });
      }

      const res = await fetch(`${cleanUrl}/api/omr`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        let errDetail = 'Server returned HTTP ' + res.status;
        try {
          const errJson = await res.json();
          if (errJson && errJson.detail) errDetail = errJson.detail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const contentType = res.headers.get('content-type') || '';

      if (contentType.includes('application/json')) {
        const payload = await res.json();

        // 1. If server produced a MusicXML score
        if (payload.musicxml) {
          if (onProgress) onProgress({ stage: 'Parsing neural MusicXML score...', percent: 85 });
          const parsed = parseMusicXmlString(payload.musicxml, payload.title || file.name.replace(/\.[^/.]+$/, ''));
          parsed.source = 'sheet-music';
          if (onProgress) onProgress({ stage: 'Sheet music ready to play!', percent: 100 });
          return parsed;
        }

        // 2. If server produced a base64 MIDI file
        if (payload.midi_base64) {
          if (onProgress) onProgress({ stage: 'Decoding OMR MIDI notes...', percent: 85 });
          const binaryStr = atob(payload.midi_base64);
          const bytes = new Uint8Array(binaryStr.length);
          for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
          }
          if (typeof global.parseStandardMidiFile === 'function') {
            const parsedMidi = global.parseStandardMidiFile(bytes.buffer, payload.title || file.name);
            parsedMidi.source = 'sheet-music';
            if (onProgress) onProgress({ stage: 'Sheet music ready to play!', percent: 100 });
            return parsedMidi;
          }
        }
      }

      // 3. If server returned raw binary MIDI
      const arrayBuffer = await res.arrayBuffer();
      if (typeof global.parseStandardMidiFile === 'function') {
        const parsedMidi = global.parseStandardMidiFile(arrayBuffer, file.name);
        parsedMidi.source = 'sheet-music';
        if (onProgress) onProgress({ stage: 'Sheet music ready to play!', percent: 100 });
        return parsedMidi;
      }

      throw new Error('Unsupported server response format from /api/omr.');
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        throw new Error('OMR transcription timed out. Try uploading a single-page score.');
      }
      throw err;
    }
  }

  /********************************************************************
   * 4. UNIFIED SHEET MUSIC IMPORT ENTRY POINT
   ********************************************************************/

  async function importSheetMusicFile(file, options = {}, onProgress) {
    if (!file) throw new Error('No sheet music file provided.');

    const name = file.name.toLowerCase();
    const serverUrl = options.serverUrl || 'http://localhost:8000';

    // 1. Digital Uncompressed MusicXML (.musicxml or .xml)
    if (name.endsWith('.musicxml') || name.endsWith('.xml')) {
      if (onProgress) onProgress({ stage: 'Reading digital MusicXML score...', percent: 35 });
      const xmlText = await file.text();
      if (onProgress) onProgress({ stage: 'Parsing staves and fingerings...', percent: 75 });
      const song = parseMusicXmlString(xmlText, file.name);
      if (onProgress) onProgress({ stage: 'Score imported successfully!', percent: 100 });
      return song;
    }

    // 2. Digital Compressed MusicXML (.mxl archive)
    if (name.endsWith('.mxl')) {
      if (onProgress) onProgress({ stage: 'Unpacking compressed .MXL score in-browser...', percent: 30 });
      const arrayBuffer = await file.arrayBuffer();
      const xmlText = await extractMusicXmlFromMxl(arrayBuffer);
      if (onProgress) onProgress({ stage: 'Parsing Treble & Bass staves...', percent: 75 });
      const song = parseMusicXmlString(xmlText, file.name);
      if (onProgress) onProgress({ stage: 'Score imported successfully!', percent: 100 });
      return song;
    }

    // 3. Visual Scans & Documents (.pdf, .png, .jpg, .jpeg, .webp)
    if (name.endsWith('.pdf') || name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.webp')) {
      return await transcribeSheetVisualViaServer(file, serverUrl, onProgress);
    }

    throw new Error('Unsupported file extension. Please select a .musicxml, .mxl, .pdf, .png, or .jpg file.');
  }

  // Export to global scope
  global.sheetMusicImporter = {
    extractMusicXmlFromMxl: extractMusicXmlFromMxl,
    parseMusicXmlString: parseMusicXmlString,
    transcribeSheetVisualViaServer: transcribeSheetVisualViaServer,
    importSheetMusicFile: importSheetMusicFile
  };

})(typeof window !== 'undefined' ? window : globalThis);
