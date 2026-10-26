(function (global) {
  'use strict';

  /********************************************************************
   * PITCH & MUSICAL HELPERS
   ********************************************************************/
  const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const BLACK_PCS = { 1: true, 3: true, 6: true, 8: true, 10: true };

  function midiToName(midi) {
    const m = Math.max(0, Math.min(127, Math.round(midi)));
    const pc = m % 12;
    const oct = Math.floor(m / 12) - 1;
    return NOTE_NAMES[pc] + oct;
  }

  function nameToMidi(name) {
    const match = /^([A-G]#?)(-?\d+)$/i.exec(String(name).trim());
    if (!match) return 60;
    const pc = NOTE_NAMES.indexOf(match[1].toUpperCase());
    const oct = parseInt(match[2], 10);
    return (oct + 1) * 12 + (pc >= 0 ? pc : 0);
  }

  function isBlackMidi(midi) {
    return !!BLACK_PCS[midi % 12];
  }

  function midiToFreq(midi) {
    return 440 * Math.pow(2, (midi - 69) / 12);
  }

  function formatTime(sec) {
    const s = Math.max(0, Math.floor(sec || 0));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m + ':' + (r < 10 ? '0' : '') + r;
  }

  /********************************************************************
   * MUSICAL MEASURE & BEAT GRID GENERATOR
   ********************************************************************/
  function generateBeatGrid(duration, beatSec, beatsPerBar, offsetSec = 0.8) {
    const beats = [];
    const safeBeatSec = Math.max(0.08, beatSec || 0.5);
    const bpb = Math.max(1, Math.round(beatsPerBar || 4));
    let beatIdx = 0;

    for (let t = offsetSec; t <= duration + 0.05; t += safeBeatSec) {
      const barNumber = Math.floor(beatIdx / bpb) + 1;
      const beatNumber = (beatIdx % bpb) + 1;
      beats.push({
        time: Math.round(t * 1000) / 1000,
        isMeasure: beatNumber === 1,
        barNumber: barNumber,
        beatNumber: beatNumber
      });
      beatIdx++;
    }
    return beats;
  }

  /********************************************************************
   * BUILT-IN ACCURATE SONG ARRANGEMENTS
   ********************************************************************/
  function createLetItBe() {
    const notes = [];
    const beat = 0.80; // ~75 BPM
    const add = (name, startB, durB, hand, vel = 0.78) => {
      notes.push({
        midi: nameToMidi(name),
        name: name,
        time: startB * beat + 0.8,
        duration: durB * beat * 0.94,
        hand: hand,
        velocity: vel
      });
    };

    for (let c = 0; c < 2; c++) {
      const b = c * 16;
      // Bar 1: C Major -> G Major
      add('C3', b + 0, 1.9, 'L', 0.80);
      add('G3', b + 1, 0.9, 'L', 0.66);
      add('E4', b + 0, 0.92, 'R', 0.80); add('G4', b + 0, 0.92, 'R', 0.80); add('C5', b + 0, 0.92, 'R', 0.84);
      add('E4', b + 1, 0.92, 'R', 0.72); add('G4', b + 1, 0.92, 'R', 0.72); add('C5', b + 1, 0.92, 'R', 0.75);

      add('G2', b + 2, 1.9, 'L', 0.80);
      add('D3', b + 3, 0.9, 'L', 0.66);
      add('D4', b + 2, 0.92, 'R', 0.80); add('G4', b + 2, 0.92, 'R', 0.80); add('B4', b + 2, 0.92, 'R', 0.84);
      add('D4', b + 3, 0.92, 'R', 0.72); add('G4', b + 3, 0.92, 'R', 0.72); add('B4', b + 3, 0.92, 'R', 0.75);

      // Bar 2: A Minor -> Fmaj7/F6
      add('A2', b + 4, 1.9, 'L', 0.82);
      add('E3', b + 5, 0.9, 'L', 0.68);
      add('E4', b + 4, 0.92, 'R', 0.82); add('A4', b + 4, 0.92, 'R', 0.82); add('C5', b + 4, 0.92, 'R', 0.86);
      add('E4', b + 5, 0.92, 'R', 0.74); add('A4', b + 5, 0.92, 'R', 0.74); add('C5', b + 5, 0.92, 'R', 0.76);

      add('F3', b + 6, 0.95, 'L', 0.82);
      add('F3', b + 7, 0.95, 'L', 0.78);
      add('C4', b + 6, 1.9, 'L', 0.74);
      add('F4', b + 6, 1.9, 'R', 0.84);
      add('A4', b + 6, 1.9, 'R', 0.84);
      add('C5', b + 6, 1.9, 'R', 0.84);

      // Bar 3: C Major -> G Major
      add('C3', b + 8, 1.9, 'L', 0.80);
      add('G3', b + 9, 0.9, 'L', 0.68);
      add('E4', b + 8, 0.92, 'R', 0.82); add('G4', b + 8, 0.92, 'R', 0.82); add('C5', b + 8, 0.92, 'R', 0.86);
      add('E4', b + 9, 0.92, 'R', 0.74); add('G4', b + 9, 0.92, 'R', 0.74); add('C5', b + 9, 0.92, 'R', 0.76);

      add('G2', b + 10, 1.9, 'L', 0.80);
      add('D3', b + 11, 0.9, 'L', 0.68);
      add('D4', b + 10, 0.92, 'R', 0.82); add('G4', b + 10, 0.92, 'R', 0.82); add('B4', b + 10, 0.92, 'R', 0.85);
      add('G4', b + 11, 0.92, 'R', 0.75); add('B4', b + 11, 0.92, 'R', 0.75); add('D5', b + 11, 0.92, 'R', 0.78);

      // Bar 4: Iconic Walkdown (F -> C/E -> Dm7 -> C)
      add('F3', b + 12, 0.95, 'L', 0.84);
      add('A4', b + 12, 0.95, 'R', 0.84); add('C5', b + 12, 0.95, 'R', 0.84); add('F5', b + 12, 0.95, 'R', 0.88);

      add('E3', b + 13, 0.95, 'L', 0.80);
      add('G4', b + 13, 0.95, 'R', 0.80); add('C5', b + 13, 0.95, 'R', 0.80); add('E5', b + 13, 0.95, 'R', 0.84);

      add('D3', b + 14, 0.95, 'L', 0.78);
      add('F4', b + 14, 0.95, 'R', 0.78); add('A4', b + 14, 0.95, 'R', 0.78); add('D5', b + 14, 0.95, 'R', 0.82);

      add('C3', b + 15, 0.98, 'L', 0.84);
      add('E4', b + 15, 0.98, 'R', 0.82); add('G4', b + 15, 0.98, 'R', 0.82); add('C5', b + 15, 0.98, 'R', 0.86);
    }

    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const duration = 32 * beat + 2.2;
    return {
      title: 'Let It Be - The Beatles',
      notes,
      duration,
      beats: generateBeatGrid(duration, beat, 4, 0.8)
    };
  }

  function createImagine() {
    const notes = [];
    const beat = 0.80;
    const add = (name, startB, durB, hand, vel = 0.74) => {
      notes.push({
        midi: nameToMidi(name),
        name,
        time: startB * beat + 0.8,
        duration: durB * beat * 0.95,
        hand,
        velocity: vel
      });
    };

    for (let p = 0; p < 4; p++) {
      const b0 = p * 8;
      add('C3', b0 + 0, 3.8, 'L', 0.76);
      add('G2', b0 + 2.0, 1.8, 'L', 0.66);
      for (let b = 0; b < 3; b++) {
        add('E4', b0 + b, 0.48, 'R', 0.72);
        add('G4', b0 + b, 0.48, 'R', 0.72);
        add('C4', b0 + b + 0.5, 0.48, 'R', 0.62);
      }
      add('E4', b0 + 3.0, 0.48, 'R', 0.74);
      add('B4', b0 + 3.0, 0.48, 'R', 0.76);
      add('C4', b0 + 3.5, 0.48, 'R', 0.64);

      add('F2', b0 + 4.0, 3.8, 'L', 0.76);
      add('C3', b0 + 6.0, 1.8, 'L', 0.66);
      for (let b = 4; b < 7; b++) {
        add('A4', b0 + b, 0.48, 'R', 0.74);
        add('C5', b0 + b, 0.48, 'R', 0.74);
        add('F4', b0 + b + 0.5, 0.48, 'R', 0.64);
      }
      add('A4',  b0 + 7.0,  0.32, 'R', 0.76);
      add('A#4', b0 + 7.33, 0.32, 'R', 0.78);
      add('B4',  b0 + 7.66, 0.34, 'R', 0.80);
    }

    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const duration = 32 * beat + 2.2;
    return {
      title: 'Imagine - John Lennon',
      notes,
      duration,
      beats: generateBeatGrid(duration, beat, 4, 0.8)
    };
  }

  function createClairDeLune() {
    const notes = [];
    const u = 0.54;
    const add = (name, startU, durU, hand, vel = 0.70) => {
      notes.push({
        midi: nameToMidi(name),
        name,
        time: startU * u + 0.8,
        duration: durU * u * 0.96,
        hand,
        velocity: vel
      });
    };

    const seq = [
      { n: 'C#3', t: 0, d: 9, h: 'L', v: 0.66 }, { n: 'G#3', t: 0, d: 9, h: 'L', v: 0.64 },
      { n: 'C#5', t: 2, d: 4, h: 'R', v: 0.72 }, { n: 'F5',  t: 2, d: 4, h: 'R', v: 0.76 },
      { n: 'F4',  t: 6, d: 3, h: 'R', v: 0.68 }, { n: 'A#4', t: 6, d: 3, h: 'R', v: 0.72 },

      { n: 'C#3', t: 9, d: 9, h: 'L', v: 0.64 }, { n: 'F#3', t: 9, d: 9, h: 'L', v: 0.64 }, { n: 'A#3', t: 9, d: 9, h: 'L', v: 0.64 },
      { n: 'D#4', t: 9, d: 3, h: 'R', v: 0.68 }, { n: 'F#4', t: 9, d: 3, h: 'R', v: 0.72 },
      { n: 'F4',  t: 12, d: 3, h: 'R', v: 0.68 }, { n: 'G#4', t: 12, d: 3, h: 'R', v: 0.72 },
      { n: 'D#4', t: 15, d: 3, h: 'R', v: 0.66 }, { n: 'F#4', t: 15, d: 3, h: 'R', v: 0.70 },

      { n: 'C#3', t: 18, d: 9, h: 'L', v: 0.65 }, { n: 'F3', t: 18, d: 9, h: 'L', v: 0.65 }, { n: 'G#3', t: 18, d: 9, h: 'L', v: 0.65 },
      { n: 'C#4', t: 18, d: 9, h: 'R', v: 0.66 },
      { n: 'F4',  t: 19, d: 2, h: 'R', v: 0.74 }, { n: 'D#4', t: 21, d: 2, h: 'R', v: 0.70 },
      { n: 'F4',  t: 23, d: 2, h: 'R', v: 0.72 }, { n: 'C#4', t: 25, d: 4, h: 'R', v: 0.68 },

      { n: 'G#2', t: 27, d: 9, h: 'L', v: 0.64 }, { n: 'F#3', t: 27, d: 9, h: 'L', v: 0.64 }, { n: 'A#3', t: 27, d: 9, h: 'L', v: 0.64 },
      { n: 'C4',  t: 28, d: 2, h: 'R', v: 0.70 }, { n: 'D#4', t: 30, d: 2, h: 'R', v: 0.72 },
      { n: 'C4',  t: 32, d: 2, h: 'R', v: 0.68 }, { n: 'A#3', t: 34, d: 4, h: 'R', v: 0.66 }
    ];
    seq.forEach(s => add(s.n, s.t, s.d, s.h, s.v));
    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const duration = 39 * u + 2.2;
    return {
      title: 'Clair de Lune - Debussy',
      notes,
      duration,
      beats: generateBeatGrid(duration, u, 9, 0.8)
    };
  }

  function createGymnopedie() {
    // Erik Satie - Gymnopédie No. 1 (Ultimate Mellow Felt Grand piece!)
    const notes = [];
    const beat = 0.92; // Slow 3/4 meditative pulse
    const add = (name, startB, durB, hand, vel = 0.68) => {
      notes.push({
        midi: nameToMidi(name),
        name,
        time: startB * beat + 0.8,
        duration: durB * beat * 0.95,
        hand,
        velocity: vel
      });
    };

    // Iconic Gmaj7 <-> Dmaj7 alternating 3/4 accompaniment + floating right-hand melody
    const melody = [
      null, null, null, null,
      { n: 'F#5', b: 13, d: 1 }, { n: 'A5', b: 14, d: 1 }, { n: 'G5', b: 15, d: 1 },
      { n: 'F#5', b: 16, d: 1 }, { n: 'C#5', b: 17, d: 1 }, { n: 'B4', b: 18, d: 1 },
      { n: 'C#5', b: 19, d: 1 }, { n: 'D5', b: 20, d: 1 }, { n: 'A4', b: 21, d: 3 },
      { n: 'F#4', b: 24, d: 5 }
    ];

    for (let bar = 0; bar < 10; bar++) {
      const b0 = bar * 3;
      if (bar % 2 === 0) {
        // Gmaj7 bar
        add('G2', b0 + 0, 2.8, 'L', 0.66);
        add('B3', b0 + 1, 1.8, 'L', 0.62);
        add('D4', b0 + 1, 1.8, 'R', 0.64);
        add('F#4', b0 + 1, 1.8, 'R', 0.66);
      } else {
        // Dmaj7 bar
        add('D2', b0 + 0, 2.8, 'L', 0.66);
        add('A3', b0 + 1, 1.8, 'L', 0.62);
        add('C#4', b0 + 1, 1.8, 'R', 0.64);
        add('F#4', b0 + 1, 1.8, 'R', 0.66);
      }
    }

    melody.forEach(m => {
      if (m) add(m.n, m.b, m.d, 'R', 0.76);
    });

    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const duration = 30 * beat + 2.2;
    return {
      title: 'Gymnopédie No. 1 - Satie',
      notes,
      duration,
      beats: generateBeatGrid(duration, beat, 3, 0.8)
    };
  }

  function createPyramidSong() {
    // Radiohead - Pyramid Song (Iconic 3+3+4+3+3 swung pyramid chord cycle)
    const notes = [];
    const beat = 0.56;
    const add = (name, startB, durB, hand, vel = 0.74) => {
      notes.push({
        midi: nameToMidi(name),
        name,
        time: startB * beat + 0.8,
        duration: durB * beat * 0.94,
        hand,
        velocity: vel
      });
    };

    const hits = [
      { off: 0.0, dur: 1.5 },
      { off: 1.5, dur: 1.5 },
      { off: 3.0, dur: 2.0 },
      { off: 5.0, dur: 1.5 },
      { off: 6.5, dur: 1.5 }
    ];

    const progression = [
      // Phrase 1: F#m -> Gmaj7 -> A6 -> Gmaj7
      { bass: ['F#2', 'C#3'], rh: ['A3', 'C#4', 'F#4'] },
      { bass: ['G2', 'D3'],   rh: ['B3', 'D4', 'F#4'] },
      { bass: ['A2', 'E3'],   rh: ['C#4', 'E4', 'F#4', 'A4'] },
      { bass: ['A2', 'E3'],   rh: ['C#4', 'E4', 'A4'] },
      { bass: ['G2', 'D3'],   rh: ['B3', 'D4', 'F#4'] },

      // Phrase 2: F#m -> Gmaj7 -> A6 -> Gmaj7
      { bass: ['F#2', 'C#3'], rh: ['A3', 'C#4', 'F#4'] },
      { bass: ['G2', 'D3'],   rh: ['B3', 'D4', 'G4'] },
      { bass: ['A2', 'E3'],   rh: ['C#4', 'E4', 'F#4', 'A4'] },
      { bass: ['G2', 'D3'],   rh: ['B3', 'D4', 'F#4'] },
      { bass: ['G2', 'D3'],   rh: ['B3', 'D4', 'G4'] },

      // Phrase 3: Gmaj7 -> F#m -> Em7 -> Gmaj7
      { bass: ['G2', 'D3'],   rh: ['B3', 'D4', 'F#4', 'B4'] },
      { bass: ['F#2', 'C#3'], rh: ['A3', 'C#4', 'F#4', 'A4'] },
      { bass: ['E2', 'B2'],   rh: ['G3', 'B3', 'D4', 'G4'] },
      { bass: ['E2', 'B2'],   rh: ['G3', 'B3', 'E4'] },
      { bass: ['G2', 'D3'],   rh: ['B3', 'D4', 'F#4'] },

      // Phrase 4: F#m resolution
      { bass: ['F#2', 'C#3'], rh: ['A3', 'C#4', 'F#4'] },
      { bass: ['F#2', 'C#3'], rh: ['A3', 'C#4', 'E4', 'F#4'] },
      { bass: ['G2', 'D3'],   rh: ['B3', 'D4', 'F#4'] },
      { bass: ['F#2', 'C#3'], rh: ['A3', 'C#4', 'F#4'] },
      { bass: ['F#2', 'C#3'], rh: ['F#3', 'A3', 'C#4', 'F#4'] }
    ];

    for (let phrase = 0; phrase < 4; phrase++) {
      const pStart = phrase * 8.0;
      for (let h = 0; h < 5; h++) {
        const chord = progression[phrase * 5 + h];
        const hit = hits[h];
        const t = pStart + hit.off;
        chord.bass.forEach((n, idx) => add(n, t, hit.dur, 'L', idx === 0 ? 0.78 : 0.68));
        chord.rh.forEach((n, idx) => add(n, t, hit.dur, 'R', idx === chord.rh.length - 1 ? 0.80 : 0.72));
      }
    }

    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const duration = 32 * beat + 2.2;
    return {
      title: 'Pyramid Song - Radiohead',
      notes,
      duration,
      beats: generateBeatGrid(duration, beat, 8, 0.8)
    };
  }

  const SONGS = {
    let_it_be: createLetItBe(),
    imagine: createImagine(),
    clair_de_lune: createClairDeLune(),
    gymnopedie: createGymnopedie(),
    pyramid_song: createPyramidSong()
  };

  global.NOTE_NAMES = NOTE_NAMES;
  global.BLACK_PCS = BLACK_PCS;
  global.midiToName = midiToName;
  global.nameToMidi = nameToMidi;
  global.isBlackMidi = isBlackMidi;
  global.midiToFreq = midiToFreq;
  global.formatTime = formatTime;
  global.generateBeatGrid = generateBeatGrid;
  global.SONGS = SONGS;
})(window);
