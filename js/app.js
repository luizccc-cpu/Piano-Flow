(function (global) {
  'use strict';

  const SONGS = global.SONGS;
  const midiToName = global.midiToName;
  const isBlackMidi = global.isBlackMidi;
  const formatTime = global.formatTime;
  const generateBeatGrid = global.generateBeatGrid;
  const parseStandardMidiFile = global.parseStandardMidiFile;
  const calculateAutoRange = global.calculateAutoRange;
  const assignHandsSmart = global.assignHandsSmart;
  const audioEngine = global.audioEngine;
  const audioTranscriber = global.audioTranscriber;

  // Enrich built-in songs with metadata and save pristine factory copies for Reset
  const BUILTIN_IDS = ['let_it_be', 'imagine', 'clair_de_lune', 'gymnopedie', 'pyramid_song'];
  const FACTORY_BUILTIN_SONGS = {};
  BUILTIN_IDS.forEach(function (id) {
    const s = SONGS[id];
    if (!s) return;
    s.id = id;
    s.source = 'builtin';
    s.isCustom = false;
    s.isModifiedBuiltin = false;
    if (!s.bpm) {
      if (Array.isArray(s.beats) && s.beats.length >= 2) {
        const dt = s.beats[1].time - s.beats[0].time;
        s.bpm = dt > 0.05 ? Math.round(60 / dt) : 120;
      } else {
        s.bpm = 120;
      }
    }
    if (typeof s.gridOffset !== 'number') {
      s.gridOffset = (Array.isArray(s.beats) && s.beats.length > 0) ? s.beats[0].time : 0.8;
    }
    if (!s.beatsPerBar) {
      let bpb = 4;
      if (Array.isArray(s.beats)) {
        for (let i = 1; i < s.beats.length; i++) {
          if (s.beats[i].isMeasure) {
            bpb = i;
            break;
          }
        }
      }
      s.beatsPerBar = bpb;
    }
    FACTORY_BUILTIN_SONGS[id] = JSON.parse(JSON.stringify(s));
  });

  /********************************************************************
   * APPLICATION STATE
   ********************************************************************/
  const state = {
    viewMode: 'home', // 'home' | 'play' | 'edit'
    homeFilter: 'all', // 'all' | 'ai' | 'midi' | 'builtin'
    homeSearch: '',

    activeSongId: 'let_it_be',
    song: SONGS.let_it_be,
    isPlaying: false,
    currentTime: 0,
    speed: 1.0,
    horizonSec: 5.0, // Strict 5-second lookahead
    waitMode: false,
    waitingMidis: new Set(),
    clearedWaitBuckets: new Set(),
    currentWaitBucket: null,
    waitStartedAtMs: null,
    passHadLongWaitStall: false,

    handIsoMaster: false,
    leftHandOn: false,
    rightHandOn: false,
    lastIsolatedHand: 'R',

    rangeMode: 'auto', // 'auto' | 'compact' | 'full88'
    showMeasureGrid: true,
    showAllKeyNames: false,
    hudMinimized: false,

    loopEnabled: false,
    loopA: 3.0,
    loopB: 14.0,
    speedRamp: false,
    loopPassCount: 1,
    targetRampSpeed: 1.25,

    // Phase 4A: Interactive Visual Note Editor State
    editorTool: 'select', // 'select' | 'group' | 'add' | 'delete'
    editorSnapBeats: 0.25, // 0 | 0.25 | 0.125 | 0.0625
    selectedNoteIndex: -1,
    selectedNoteIndices: new Set(),
    groupDragBox: null,
    isCloningNote: false,
    undoStack: [],
    redoStack: [],

    minMidi: 48,
    maxMidi: 84,
    whiteKeys: [],
    blackKeys: [],
    keyMap: {},

    playedIndices: new Set(),
    userPressed: {}, // midi -> { hand, time }
    particles: []
  };

  function formatSpeedBadge(speed) {
    const rounded = Math.round(speed * 100) / 100;
    if (Math.abs(rounded - Math.round(rounded * 10) / 10) < 0.001) {
      return rounded.toFixed(1) + 'x';
    }
    return rounded.toFixed(2) + 'x';
  }

  /********************************************************************
   * FAULT-TOLERANT CANVAS SIZING & 60FPS RENDERER
   ********************************************************************/
  const stageEl = document.getElementById('stageContainer');
  const canvas = document.getElementById('pianoCanvas');
  const ctx = canvas.getContext('2d');

  let width = 960;
  let height = 500;
  let strikeY = 365;
  let keyboardH = 135;

  function syncViewportHeight() {
    const appShell = document.getElementById('appShell');
    if (!appShell) return;
    const isStandalone = window.navigator.standalone === true ||
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches);
    if (isStandalone) {
      appShell.style.height = '100%';
      return;
    }
    const vh = (window.visualViewport && window.visualViewport.height > 100)
      ? window.visualViewport.height
      : (window.innerHeight || 0);
    if (vh > 100) {
      appShell.style.height = Math.round(vh) + 'px';
    }
  }

  function rebuildKeyboardGeometry() {
    const rect = stageEl ? stageEl.getBoundingClientRect() : { width: 960, height: 500 };
    width = (rect.width > 100) ? rect.width : (window.innerWidth || 960);
    height = (rect.height > 80) ? rect.height : Math.max(240, (window.innerHeight || 500) - 106);

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    keyboardH = Math.round(Math.max(96, Math.min(165, height * 0.24)));
    strikeY = height - keyboardH;

    const whiteMidis = [];
    for (let m = state.minMidi; m <= state.maxMidi; m++) {
      if (!isBlackMidi(m)) whiteMidis.push(m);
    }

    const numWhite = Math.max(1, whiteMidis.length);
    const whiteW = width / numWhite;
    const blackW = whiteW * 0.62;
    const blackH = keyboardH * 0.63;

    state.whiteKeys = [];
    state.blackKeys = [];
    state.keyMap = {};

    for (let i = 0; i < whiteMidis.length; i++) {
      const m = whiteMidis[i];
      const x = i * whiteW;
      const kObj = {
        midi: m,
        name: midiToName(m),
        isBlack: false,
        x: x,
        y: strikeY,
        w: whiteW,
        h: keyboardH,
        laneX: x + whiteW * 0.08,
        laneW: whiteW * 0.84
      };
      state.whiteKeys.push(kObj);
      state.keyMap[m] = kObj;
    }

    for (let m = state.minMidi; m <= state.maxMidi; m++) {
      if (isBlackMidi(m)) {
        const prevWhite = state.keyMap[m - 1];
        if (!prevWhite) continue;
        const pc = m % 12;
        let shift = 0.5;
        if (pc === 1 || pc === 6) shift = 0.55;
        if (pc === 3 || pc === 10) shift = 0.45;

        const x = prevWhite.x + prevWhite.w - blackW * shift;
        const kObj = {
          midi: m,
          name: midiToName(m),
          isBlack: true,
          x: x,
          y: strikeY,
          w: blackW,
          h: blackH,
          laneX: x + blackW * 0.06,
          laneW: blackW * 0.88
        };
        state.blackKeys.push(kObj);
        state.keyMap[m] = kObj;
      }
    }
  }

  function drawRoundedRect(c, x, y, w, h, r) {
    const rad = Math.max(0, Math.min(r, w / 2, h / 2));
    c.beginPath();
    c.moveTo(x + rad, y);
    c.lineTo(x + w - rad, y);
    c.quadraticCurveTo(x + w, y, x + w, y + rad);
    c.lineTo(x + w, y + h - rad);
    c.quadraticCurveTo(x + w, y + h, x + w - rad, y + h);
    c.lineTo(x + rad, y + h);
    c.quadraticCurveTo(x, y + h, x, y + h - rad);
    c.lineTo(x, y + rad);
    c.quadraticCurveTo(x, y, x + rad, y);
    c.closePath();
  }

  function spawnParticles(x, w, hand) {
    if (state.particles.length > 75) return;
    const rgb = (hand === 'L') ? '56, 189, 248' : '74, 222, 128';
    for (let i = 0; i < 2; i++) {
      state.particles.push({
        x: x + Math.random() * w,
        y: strikeY + (Math.random() * 4 - 2),
        vx: (Math.random() - 0.5) * 26,
        vy: -(16 + Math.random() * 38),
        r: 1.4 + Math.random() * 2.0,
        alpha: 0.9,
        rgb: rgb
      });
    }
  }

  function renderParticles(dt) {
    for (let i = state.particles.length - 1; i >= 0; i--) {
      const p = state.particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.alpha -= dt * 1.45;
      if (p.alpha <= 0) {
        state.particles.splice(i, 1);
        continue;
      }
      ctx.save();
      ctx.beginPath();
      ctx.fillStyle = 'rgba(' + p.rgb + ',' + p.alpha + ')';
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  function renderFrame(dt) {
    ctx.clearRect(0, 0, width, height);

    // 1. Deep Runway Background
    const bgGrad = ctx.createLinearGradient(0, 0, 0, strikeY);
    bgGrad.addColorStop(0, '#070b14');
    bgGrad.addColorStop(0.65, '#0b1220');
    bgGrad.addColorStop(1, '#101a2e');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, strikeY);

    // 2. Vertical Key Lane Guides
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      const isC = (wk.midi % 12 === 0);
      ctx.beginPath();
      ctx.moveTo(wk.x, 0);
      ctx.lineTo(wk.x, strikeY);
      ctx.strokeStyle = isC ? 'rgba(148, 163, 184, 0.16)' : 'rgba(148, 163, 184, 0.045)';
      ctx.lineWidth = isC ? 1.2 : 1;
      ctx.stroke();
    }

    const horizon = state.horizonSec;
    const pxPerSec = strikeY / horizon;
    const curT = state.currentTime;

    // 3A. Scrolling Musical Measure & Beat Gridlines
    if (state.showMeasureGrid && Array.isArray(state.song.beats)) {
      const beats = state.song.beats;
      ctx.save();
      for (let i = 0; i < beats.length; i++) {
        const b = beats[i];
        if (b.time < curT) continue;
        if (b.time > curT + horizon + 0.05) break;

        const y = strikeY - (b.time - curT) * pxPerSec;
        if (y < 8 || y > strikeY - 2) continue;

        if (b.isMeasure) {
          ctx.beginPath();
          ctx.setLineDash([]);
          ctx.moveTo(0, y);
          ctx.lineTo(width - 42, y);
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          // Measure number pill on the left side
          const labelText = 'BAR ' + b.barNumber;
          ctx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
          const textW = ctx.measureText(labelText).width;
          const pillW = textW + 10;
          const pillH = 15;
          const pillX = 10;
          const pillY = Math.max(4, y - pillH - 2);

          drawRoundedRect(ctx, pillX, pillY, pillW, pillH, 4);
          ctx.fillStyle = 'rgba(15, 23, 42, 0.78)';
          ctx.fill();
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
          ctx.lineWidth = 1;
          ctx.stroke();

          ctx.fillStyle = 'rgba(125, 211, 252, 0.88)';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(labelText, pillX + 5, pillY + pillH / 2 + 0.5);
        } else {
          ctx.beginPath();
          ctx.setLineDash([4, 6]);
          ctx.moveTo(0, y);
          ctx.lineTo(width - 42, y);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
      ctx.restore();
    }

    // 3A-Edit: Subtle Sub-Beat Snap Gridlines when in Edit Mode
    if (state.viewMode === 'edit' && state.editorSnapBeats > 0 && state.editorSnapBeats < 1) {
      const bpmVal = Math.max(30, Math.min(260, state.song.bpm || 120));
      const beatSec = 60 / bpmVal;
      const stepSec = beatSec * state.editorSnapBeats;
      const gridOff = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
      if (stepSec >= 0.04) {
        const firstIdx = Math.ceil((curT - gridOff) / stepSec);
        const endT = curT + horizon + 0.05;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([2, 6]);
        for (let k = firstIdx; k < firstIdx + 160; k++) {
          const tSnap = gridOff + k * stepSec;
          if (tSnap > endT) break;
          if (tSnap < curT) continue;
          const ySnap = strikeY - (tSnap - curT) * pxPerSec;
          if (ySnap < 8 || ySnap > strikeY - 2) continue;
          ctx.moveTo(0, ySnap);
          ctx.lineTo(width - 44, ySnap);
        }
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.065)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }
    }

    // 3B. Horizontal 5-Second Horizon Markers (1s..5s on the right edge)
    ctx.save();
    ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let s = 1; s <= Math.floor(horizon); s++) {
      const y = strikeY - s * pxPerSec;
      if (y < 14) continue;
      ctx.beginPath();
      ctx.moveTo(state.showMeasureGrid ? (width - 52) : 0, y);
      ctx.lineTo(width - 42, y);
      ctx.strokeStyle = state.showMeasureGrid ? 'rgba(148, 163, 184, 0.28)' : 'rgba(255, 255, 255, 0.075)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = 'rgba(148, 163, 184, 0.58)';
      ctx.fillText(s + 's', width - 12, y);
    }
    ctx.restore();

    // 4. Falling Note Bars & Explicit Note Badges
    const activeKeys = {};
    Object.keys(state.userPressed).forEach(mStr => {
      const m = parseInt(mStr, 10);
      activeKeys[m] = { hand: state.userPressed[m].hand };
    });

    const notes = state.song.notes;

    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const nStart = n.time;
      const nEnd = n.time + n.duration;

      if (nEnd < curT - 0.04) continue;
      if (nStart > curT + horizon + 0.1) break;

      const kGeom = state.keyMap[n.midi];
      if (!kGeom) continue;

      const handActive = !state.handIsoMaster ||
        (n.hand === 'L' && state.leftHandOn) ||
        (n.hand === 'R' && state.rightHandOn);

      const rawBotY = strikeY - (nStart - curT) * pxPerSec;
      const rawTopY = strikeY - (nEnd - curT) * pxPerSec;

      const botY = Math.min(strikeY, rawBotY);
      const topY = Math.max(-16, rawTopY);
      const barH = botY - topY;
      if (barH <= 2) continue;

      const hittingNow = (nStart <= curT + 0.025 && nEnd >= curT);
      const isWaiting = state.waitingMidis.has(n.midi) && Math.abs(nStart - curT) < 0.08;

      if (hittingNow && handActive) {
        activeKeys[n.midi] = { hand: n.hand };
        if (state.isPlaying && state.waitingMidis.size === 0 && Math.random() < 0.3) {
          spawnParticles(kGeom.laneX, kGeom.laneW, n.hand);
        }
      }

      // Subtle vertical lane highlight under approaching notes
      if (handActive && rawBotY > strikeY * 0.25) {
        const alpha = Math.min(0.11, Math.max(0, 1 - (nStart - curT) / horizon) * 0.11);
        ctx.fillStyle = (n.hand === 'L')
          ? 'rgba(41, 151, 255, ' + alpha + ')'
          : 'rgba(52, 211, 153, ' + alpha + ')';
        ctx.fillRect(kGeom.laneX, botY, kGeom.laneW, strikeY - botY);
      }

      ctx.save();
      ctx.globalAlpha = handActive ? 1.0 : 0.16;

      const x = kGeom.laneX;
      const w = kGeom.laneW;
      const rad = Math.min(7, w * 0.28);

      const barGrad = ctx.createLinearGradient(x, topY, x, botY);
      if (n.hand === 'L') {
        barGrad.addColorStop(0, '#7dd3fc');
        barGrad.addColorStop(0.3, '#2997ff');
        barGrad.addColorStop(1, '#0284c7');
        ctx.shadowColor = hittingNow ? 'rgba(56, 189, 248, 0.95)' : 'rgba(41, 151, 255, 0.42)';
      } else {
        barGrad.addColorStop(0, '#86efac');
        barGrad.addColorStop(0.3, '#34d399');
        barGrad.addColorStop(1, '#16a34a');
        ctx.shadowColor = hittingNow ? 'rgba(74, 222, 128, 0.95)' : 'rgba(52, 211, 153, 0.42)';
      }
      ctx.shadowBlur = hittingNow ? 16 : 9;

      drawRoundedRect(ctx, x, topY, w, barH, rad);
      ctx.fillStyle = barGrad;
      ctx.fill();

      const isInGroup = (state.viewMode === 'edit' && state.selectedNoteIndices && state.selectedNoteIndices.has(i));
      const isSelectedInEdit = (state.viewMode === 'edit' && (i === state.selectedNoteIndex || isInGroup));
      if (isInGroup) {
        ctx.shadowColor = '#38bdf8';
        ctx.shadowBlur = 12;
      } else {
        ctx.shadowBlur = 0;
      }
      ctx.lineWidth = isSelectedInEdit ? 2.5 : (isWaiting ? 2.2 : 1.2);
      ctx.strokeStyle = isSelectedInEdit
        ? '#ffffff'
        : (isWaiting
          ? '#ffffff'
          : (n.hand === 'L' ? 'rgba(186, 230, 253, 0.85)' : 'rgba(187, 247, 208, 0.85)'));
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Group Select Checkmark Badge when multiple notes are selected or in Group Select tool mode
      if (isInGroup && (state.selectedNoteIndices.size >= 2 || state.editorTool === 'group') && w >= 12 && barH >= 14) {
        const chkR = Math.min(7, Math.max(5, w * 0.22));
        const chkX = x + w - chkR - 2;
        const chkY = Math.min(botY - chkR - 2, topY + chkR + 3);
        ctx.beginPath();
        ctx.arc(chkX, chkY, chkR, 0, Math.PI * 2);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
        ctx.font = '700 ' + Math.round(chkR * 1.35) + 'px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillStyle = '#0f172a';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('✓', chkX, chkY + 0.5);
      }

      // Top Resize Handle Bar when in Edit Mode (bright on selected note, subtle on others)
      if (state.viewMode === 'edit' && state.editorTool !== 'group' && w >= 12 && barH >= 14) {
        const handleH = isSelectedInEdit ? 8 : 5;
        const handleW = Math.max(6, w - 4);
        const handleX = x + 2;
        const handleY = topY + 1;
        drawRoundedRect(ctx, handleX, handleY, handleW, handleH, 3);
        ctx.fillStyle = isSelectedInEdit ? '#ffffff' : 'rgba(255, 255, 255, 0.35)';
        ctx.fill();
        if (isSelectedInEdit && handleW >= 12) {
          ctx.beginPath();
          ctx.moveTo(handleX + 4, handleY + handleH / 2);
          ctx.lineTo(handleX + handleW - 4, handleY + handleH / 2);
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1.3;
          ctx.stroke();
        }
      }

      // Explicit Note Name Badge inside every falling bar (e.g., F3, C4, A4, C5)
      const badgeH = Math.min(20, Math.max(14, w * 0.52));
      if (barH >= badgeH + 2 && w >= 14) {
        const badgeW = Math.max(w - 4, 14);
        const badgeX = x + (w - badgeW) / 2;
        const badgeY = Math.max(topY + 2, botY - badgeH - 4);

        drawRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, 4);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.42)';
        ctx.fill();

        const fSize = Math.min(12, Math.max(8.5, w * 0.36));
        ctx.font = '700 ' + fSize + 'px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(n.name, x + w / 2, badgeY + badgeH / 2 + 0.5);
      }

      ctx.restore();
    }

    // 4B. Group Select Marquee Box on Runway in Edit Mode
    if (state.viewMode === 'edit' && state.groupDragBox) {
      const bx = Math.min(state.groupDragBox.x0, state.groupDragBox.x1);
      const by = Math.min(state.groupDragBox.y0, state.groupDragBox.y1);
      const bw = Math.abs(state.groupDragBox.x1 - state.groupDragBox.x0);
      const bh = Math.abs(state.groupDragBox.y1 - state.groupDragBox.y0);
      if (bw > 2 || bh > 2) {
        ctx.save();
        ctx.fillStyle = 'rgba(56, 189, 248, 0.14)';
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.fillRect(bx, by, bw, bh);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.restore();
      }
    }

    // 5. Glowing "NOW" Strike Line & Contact Flares
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(0, strikeY);
    ctx.lineTo(width - 44, strikeY);
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.8)';
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.8)';
    ctx.shadowBlur = 10;
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.font = '700 11px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('NOW', width - 8, strikeY - 1);
    ctx.restore();

    Object.keys(activeKeys).forEach(mStr => {
      const m = parseInt(mStr, 10);
      const kGeom = state.keyMap[m];
      if (!kGeom) return;
      const info = activeKeys[m];
      const cx = kGeom.laneX + kGeom.laneW / 2;
      const r = kGeom.laneW * 0.95;

      const flare = ctx.createRadialGradient(cx, strikeY, 1, cx, strikeY, r);
      if (info.hand === 'L') {
        flare.addColorStop(0, 'rgba(255, 255, 255, 0.96)');
        flare.addColorStop(0.35, 'rgba(56, 189, 248, 0.82)');
        flare.addColorStop(1, 'rgba(56, 189, 248, 0)');
      } else {
        flare.addColorStop(0, 'rgba(255, 255, 255, 0.96)');
        flare.addColorStop(0.35, 'rgba(74, 222, 128, 0.82)');
        flare.addColorStop(1, 'rgba(74, 222, 128, 0)');
      }
      ctx.save();
      ctx.fillStyle = flare;
      ctx.beginPath();
      ctx.arc(cx, strikeY, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    renderParticles(dt);

    // 6. Auto-Scaled 3D Virtual Piano Keyboard
    // White Keys
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      const act = activeKeys[wk.midi];
      const kx = wk.x + 0.8;
      const ky = wk.y;
      const kw = wk.w - 1.6;
      const kh = wk.h - 4;

      ctx.save();
      drawRoundedRect(ctx, kx, ky, kw, kh, 5);
      const kGrad = ctx.createLinearGradient(kx, ky, kx, ky + kh);
      if (act) {
        if (act.hand === 'L') {
          kGrad.addColorStop(0, '#bae6fd');
          kGrad.addColorStop(0.25, '#38bdf8');
          kGrad.addColorStop(1, '#0284c7');
        } else {
          kGrad.addColorStop(0, '#bbf7d0');
          kGrad.addColorStop(0.25, '#4ade80');
          kGrad.addColorStop(1, '#16a34a');
        }
      } else {
        kGrad.addColorStop(0, '#e2e8f0');
        kGrad.addColorStop(0.12, '#f8fafc');
        kGrad.addColorStop(0.92, '#ffffff');
        kGrad.addColorStop(1, '#cbd5e1');
      }
      ctx.fillStyle = kGrad;
      ctx.fill();

      ctx.fillStyle = act ? (act.hand === 'L' ? '#0369a1' : '#15803d') : '#94a3b8';
      ctx.fillRect(kx + 1, ky + kh - 5, kw - 2, 5);

      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1;
      ctx.stroke();

      const isC = (wk.midi % 12 === 0);
      if ((isC || state.showAllKeyNames || act) && kw >= 15) {
        const fSize = Math.min(12.5, Math.max(8.5, kw * 0.34));
        ctx.font = (isC || act ? '700 ' : '600 ') + fSize + 'px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillStyle = act ? '#082f49' : (isC ? '#0f172a' : '#64748b');
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(wk.name, kx + kw / 2, ky + kh - 8);
      }
      ctx.restore();
    }

    // Black Keys
    for (let i = 0; i < state.blackKeys.length; i++) {
      const bk = state.blackKeys[i];
      const act = activeKeys[bk.midi];
      const kx = bk.x;
      const ky = bk.y;
      const kw = bk.w;
      const kh = bk.h;

      ctx.save();
      drawRoundedRect(ctx, kx, ky, kw, kh, 4);
      const bGrad = ctx.createLinearGradient(kx, ky, kx, ky + kh);
      if (act) {
        if (act.hand === 'L') {
          bGrad.addColorStop(0, '#7dd3fc');
          bGrad.addColorStop(0.4, '#0284c7');
          bGrad.addColorStop(1, '#0c4a6e');
        } else {
          bGrad.addColorStop(0, '#86efac');
          bGrad.addColorStop(0.4, '#16a34a');
          bGrad.addColorStop(1, '#14532d');
        }
      } else {
        bGrad.addColorStop(0, '#334155');
        bGrad.addColorStop(0.15, '#1e293b');
        bGrad.addColorStop(1, '#090d16');
      }
      ctx.fillStyle = bGrad;
      ctx.fill();

      if (kw >= 10) {
        drawRoundedRect(ctx, kx + 2, ky + 1, kw - 4, kh - 7, 2.5);
        ctx.strokeStyle = act ? 'rgba(255, 255, 255, 0.55)' : 'rgba(255, 255, 255, 0.13)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      if ((state.showAllKeyNames || act) && kw >= 16) {
        const fSize = Math.min(10, Math.max(7.5, kw * 0.36));
        ctx.font = '600 ' + fSize + 'px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillStyle = act ? '#ffffff' : 'rgba(226, 232, 240, 0.85)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(bk.name, kx + kw / 2, ky + kh - 7);
      }
      ctx.restore();
    }
  }

  /********************************************************************
   * PLAYBACK ENGINE, WAIT MODE & A-B LOOPING
   ********************************************************************/
  let lastTimeMs = performance.now();

  function updateWaitBanner() {
    const banner = document.getElementById('waitBanner');
    const txt = document.getElementById('waitBannerText');
    if (state.viewMode !== 'edit' && state.waitMode && state.waitingMidis.size > 0) {
      banner.classList.add('visible');
      const list = Array.from(state.waitingMidis).sort((a, b) => a - b).map(midiToName);
      txt.textContent = 'Wait Mode: Play ' + list.join(' + ');
    } else {
      banner.classList.remove('visible');
    }
  }

  function resetNoteTriggersBefore(sec) {
    state.playedIndices.clear();
    state.clearedWaitBuckets.clear();
    state.waitingMidis.clear();
    state.waitStartedAtMs = null;
    updateWaitBanner();

    const notes = state.song.notes;
    for (let i = 0; i < notes.length; i++) {
      if (notes[i].time < sec - 0.02) {
        state.playedIndices.add(i);
        state.clearedWaitBuckets.add(Math.round(notes[i].time * 50));
      }
    }
  }

  function advanceClock(dt) {
    if (!state.isPlaying) return;
    if (state.waitMode && state.waitingMidis.size > 0) {
      if (state.waitStartedAtMs !== null && (performance.now() - state.waitStartedAtMs) > 3000) {
        state.passHadLongWaitStall = true;
      }
      return;
    }

    const prevT = state.currentTime;
    let nextT = prevT + dt * state.speed;
    const notes = state.song.notes;

    if (state.waitMode) {
      for (let i = 0; i < notes.length; i++) {
        const n = notes[i];
        const handActive = !state.handIsoMaster ||
          (n.hand === 'L' && state.leftHandOn) ||
          (n.hand === 'R' && state.rightHandOn);
        if (!handActive) continue;

        const bucket = Math.round(n.time * 50);
        if (n.time >= prevT - 0.02 && n.time <= nextT && !state.clearedWaitBuckets.has(bucket)) {
          nextT = n.time;
          for (let j = i; j < notes.length; j++) {
            const nj = notes[j];
            if (Math.round(nj.time * 50) === bucket) {
              const hAct = !state.handIsoMaster ||
                (nj.hand === 'L' && state.leftHandOn) ||
                (nj.hand === 'R' && state.rightHandOn);
              if (hAct) state.waitingMidis.add(nj.midi);
            } else if (nj.time > n.time + 0.05) {
              break;
            }
          }
          state.currentWaitBucket = bucket;
          state.waitStartedAtMs = performance.now();
          updateWaitBanner();
          break;
        }
      }
    }

    state.currentTime = nextT;

    for (let i = 0; i < notes.length; i++) {
      if (state.playedIndices.has(i)) continue;
      const n = notes[i];
      if (n.time <= state.currentTime) {
        if (state.waitMode && state.waitingMidis.has(n.midi)) continue;
        state.playedIndices.add(i);
        const handActive = !state.handIsoMaster ||
          (n.hand === 'L' && state.leftHandOn) ||
          (n.hand === 'R' && state.rightHandOn);
        if (handActive) {
          audioEngine.playNote(n.midi, n.duration / state.speed, n.velocity);
        }
      } else {
        break;
      }
    }

    if (state.loopEnabled && state.currentTime >= state.loopB) {
      state.loopPassCount++;
      if (state.speedRamp && !state.passHadLongWaitStall && state.speed < state.targetRampSpeed) {
        state.speed = Math.min(state.targetRampSpeed, Math.round((state.speed + 0.05) * 100) / 100);
        const speedStr = formatSpeedBadge(state.speed);
        document.getElementById('speedBtn').textContent = speedStr;
        const miniSum = document.getElementById('hudMiniSummary');
        if (miniSum) miniSum.textContent = speedStr;
      }
      state.passHadLongWaitStall = false;
      state.waitStartedAtMs = null;
      syncLoopUI();
      seekTo(state.loopA);
      return;
    }

    if (state.currentTime >= state.song.duration) {
      state.currentTime = 0;
      resetNoteTriggersBefore(0);
      setPlaying(false);
    }

    syncTimelineUI();
  }

  function seekTo(sec) {
    state.currentTime = Math.max(0, Math.min(state.song.duration, sec));
    resetNoteTriggersBefore(state.currentTime);
    syncTimelineUI();
  }

  function setPlaying(play) {
    state.isPlaying = play;
    if (play) audioEngine.ensureContext();
    document.getElementById('playSvg').style.display = play ? 'none' : 'block';
    document.getElementById('pauseSvg').style.display = play ? 'block' : 'none';
  }

  function syncTimelineUI() {
    const dur = state.song.duration || 1;
    const pct = Math.max(0, Math.min(100, (state.currentTime / dur) * 100));
    document.getElementById('timelineFill').style.width = pct + '%';
    document.getElementById('playheadThumb').style.left = pct + '%';
    document.getElementById('topProgressFill').style.width = pct + '%';
    document.getElementById('curTimeText').textContent = formatTime(state.currentTime);
    document.getElementById('totTimeText').textContent = formatTime(dur);
  }

  function syncLoopUI() {
    const dur = state.song.duration || 1;
    const aPct = Math.max(0, Math.min(94, (state.loopA / dur) * 100));
    const bPct = Math.max(aPct + 4, Math.min(100, (state.loopB / dur) * 100));
    const box = document.getElementById('abLoopBox');
    const hA = document.getElementById('loopHandleA');
    const hB = document.getElementById('loopHandleB');

    box.style.left = aPct + '%';
    box.style.width = (bPct - aPct) + '%';
    hA.style.left = aPct + '%';
    hB.style.left = bPct + '%';

    if (state.loopEnabled && state.speedRamp) {
      box.textContent = 'LOOP • PASS ' + state.loopPassCount + ' • ' + state.speed.toFixed(2) + 'x';
    } else if (state.loopEnabled && state.loopPassCount > 1) {
      box.textContent = 'A–B LOOP • PASS ' + state.loopPassCount;
    } else {
      box.textContent = 'A–B LOOP';
    }
  }

  function tickLoop(now) {
    const dt = Math.min(0.05, Math.max(0.001, (now - lastTimeMs) / 1000));
    lastTimeMs = now;
    advanceClock(dt);
    renderFrame(dt);
    requestAnimationFrame(tickLoop);
  }

  /********************************************************************
   * USER INPUT: TOUCH/MOUSE PIANO, COMPUTER KEYBOARD & WEBMIDI
   ********************************************************************/
  function userNoteOn(midi, vel = 0.82) {
    const hand = (midi < 60) ? 'L' : 'R';
    state.userPressed[midi] = { hand: hand, time: performance.now() };
    audioEngine.playNote(midi, 0.75, vel);

    const kGeom = state.keyMap[midi];
    if (kGeom) spawnParticles(kGeom.laneX, kGeom.laneW, hand);

    if (state.waitMode && state.waitingMidis.has(midi)) {
      if (state.waitStartedAtMs !== null && (performance.now() - state.waitStartedAtMs) > 3000) {
        state.passHadLongWaitStall = true;
      }
      state.waitingMidis.delete(midi);
      if (state.waitingMidis.size === 0 && state.currentWaitBucket !== null) {
        state.clearedWaitBuckets.add(state.currentWaitBucket);
        state.waitStartedAtMs = null;
      }
      updateWaitBanner();
    }
  }

  function userNoteOff(midi) {
    delete state.userPressed[midi];
  }

  function hitTestPiano(x, y) {
    if (y < strikeY - 10) return null;
    for (let i = 0; i < state.blackKeys.length; i++) {
      const bk = state.blackKeys[i];
      if (x >= bk.x && x <= bk.x + bk.w && y >= bk.y && y <= bk.y + bk.h) return bk.midi;
    }
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      if (x >= wk.x && x <= wk.x + wk.w && y >= wk.y && y <= wk.y + wk.h) return wk.midi;
    }
    return null;
  }

  /********************************************************************
   * PHASE 4A: VISUAL NOTE EDITOR HELPERS, UNDO/REDO & CANVAS GESTURES
   ********************************************************************/
  function xToNearestMidi(x) {
    for (let i = 0; i < state.blackKeys.length; i++) {
      const bk = state.blackKeys[i];
      if (x >= bk.x && x <= bk.x + bk.w) return bk.midi;
    }
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      if (x >= wk.x && x <= wk.x + wk.w) return wk.midi;
    }
    let bestMidi = state.minMidi;
    let bestDist = Infinity;
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      const cx = wk.x + wk.w * 0.5;
      const d = Math.abs(x - cx);
      if (d < bestDist) {
        bestDist = d;
        bestMidi = wk.midi;
      }
    }
    return bestMidi;
  }

  function snapTimeToGrid(sec) {
    const clamped = Math.max(0, sec);
    if (!state.editorSnapBeats || state.editorSnapBeats <= 0) {
      return Math.round(clamped * 1000) / 1000;
    }
    const bpmVal = Math.max(30, Math.min(260, state.song.bpm || 120));
    const beatSec = 60 / bpmVal;
    const stepSec = beatSec * state.editorSnapBeats;
    const offset = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
    const snapped = offset + Math.round((clamped - offset) / stepSec) * stepSec;
    return Math.max(0, Math.round(snapped * 1000) / 1000);
  }

  function hitTestRunwayNote(x, y) {
    if (y >= strikeY || y < 0) return null;
    const horizon = state.horizonSec;
    const pxPerSec = strikeY / horizon;
    const curT = state.currentTime;
    const notes = state.song.notes;

    for (let i = notes.length - 1; i >= 0; i--) {
      const n = notes[i];
      const nStart = n.time;
      const nEnd = n.time + n.duration;
      if (nEnd < curT - 0.04 || nStart > curT + horizon + 0.1) continue;

      const kGeom = state.keyMap[n.midi];
      if (!kGeom) continue;

      const rawBotY = strikeY - (nStart - curT) * pxPerSec;
      const rawTopY = strikeY - (nEnd - curT) * pxPerSec;
      const botY = Math.min(strikeY, rawBotY);
      const topY = Math.max(-16, rawTopY);
      if (botY - topY <= 2) continue;

      if (x >= kGeom.laneX - 4 && x <= kGeom.laneX + kGeom.laneW + 4 && y >= topY - 8 && y <= botY + 4) {
        if (Math.abs(y - topY) <= 12) {
          return { index: i, zone: 'resize-top' };
        }
        return { index: i, zone: 'body' };
      }
    }
    return null;
  }

  function cloneSongSnapshot(song) {
    return {
      notes: (song.notes || []).map(function (n) {
        return {
          midi: n.midi,
          name: n.name || midiToName(n.midi),
          time: n.time,
          duration: n.duration,
          hand: n.hand || (n.midi < 60 ? 'L' : 'R'),
          velocity: n.velocity || 0.78
        };
      }),
      bpm: song.bpm || 120,
      gridOffset: typeof song.gridOffset === 'number' ? song.gridOffset : 0.8,
      beatsPerBar: song.beatsPerBar || 4,
      duration: song.duration || 15
    };
  }

  function syncUndoRedoUI() {
    const uBtn = document.getElementById('editorUndoBtn');
    const rBtn = document.getElementById('editorRedoBtn');
    if (uBtn) uBtn.disabled = state.undoStack.length === 0;
    if (rBtn) rBtn.disabled = state.redoStack.length === 0;
  }

  function syncEditorInspectorUI() {
    const labelEl = document.getElementById('inspectorNoteLabel');
    const mergeBtn = document.getElementById('inspectorMergeBtn');
    const cloneBtn = document.getElementById('inspectorCloneBtn');
    const handBtn = document.getElementById('inspectorHandBtn');
    const semiDownBtn = document.getElementById('inspectorSemiDownBtn');
    const semiUpBtn = document.getElementById('inspectorSemiUpBtn');
    const octDownBtn = document.getElementById('inspectorOctDownBtn');
    const octUpBtn = document.getElementById('inspectorOctUpBtn');
    const delBtn = document.getElementById('inspectorDeleteNoteBtn');
    const bpmInput = document.getElementById('editorBpmInput');

    if (bpmInput && document.activeElement !== bpmInput) {
      bpmInput.value = Math.round(state.song.bpm || 120);
    }

    // Prune any out-of-range indices from selectedNoteIndices
    if (state.selectedNoteIndices && state.selectedNoteIndices.size > 0) {
      Array.from(state.selectedNoteIndices).forEach(function (i) {
        if (i < 0 || i >= state.song.notes.length) {
          state.selectedNoteIndices.delete(i);
        }
      });
    }

    const groupCount = state.selectedNoteIndices ? state.selectedNoteIndices.size : 0;
    if (groupCount > 0 && (state.selectedNoteIndex < 0 || !state.selectedNoteIndices.has(state.selectedNoteIndex))) {
      const arr = Array.from(state.selectedNoteIndices);
      state.selectedNoteIndex = arr[arr.length - 1];
    }

    const idx = state.selectedNoteIndex;
    const note = (idx >= 0 && idx < state.song.notes.length) ? state.song.notes[idx] : null;

    if (!note && groupCount === 0) {
      state.isCloningNote = false;
      if (labelEl) {
        if (state.editorTool === 'group') {
          labelEl.textContent = 'Group Select: Tap notes of the same pitch or drag a box to select & merge';
        } else {
          labelEl.textContent = 'Tap a note to inspect/edit • Drag empty runway to scroll time';
        }
      }
      if (mergeBtn) {
        mergeBtn.disabled = true;
        mergeBtn.textContent = '🔗 Merge Selected';
      }
      if (cloneBtn) {
        cloneBtn.disabled = true;
        cloneBtn.classList.remove('active');
        cloneBtn.textContent = '⧉ Copy to Key';
      }
      if (handBtn) {
        handBtn.disabled = true;
        handBtn.textContent = 'Hand: R';
      }
      if (semiDownBtn) semiDownBtn.disabled = true;
      if (semiUpBtn) semiUpBtn.disabled = true;
      if (octDownBtn) octDownBtn.disabled = true;
      if (octUpBtn) octUpBtn.disabled = true;
      if (delBtn) delBtn.disabled = true;
      return;
    }

    if (groupCount >= 2) {
      const firstIdx = state.selectedNoteIndices.values().next().value;
      const firstNote = state.song.notes[firstIdx] || note;
      const pitchName = firstNote ? (firstNote.name || midiToName(firstNote.midi)) : '';
      if (mergeBtn) {
        mergeBtn.disabled = false;
        mergeBtn.textContent = '🔗 Merge Selected (' + groupCount + ')';
      }
      if (labelEl) {
        labelEl.textContent = groupCount + ' × ' + pitchName + ' notes selected — ready to merge';
      }
    } else {
      if (mergeBtn) {
        mergeBtn.disabled = true;
        mergeBtn.textContent = '🔗 Merge Selected';
      }
      if (labelEl && note) {
        if (state.isCloningNote) {
          labelEl.textContent = 'Copy Mode: Tap any piano key or runway lane to place copy';
        } else if (state.editorTool === 'group') {
          labelEl.textContent = '1 × ' + note.name + ' selected — tap more ' + note.name + ' notes or drag a box to merge';
        } else {
          labelEl.textContent = note.name + ' (MIDI ' + note.midi + ') • Start: ' + note.time.toFixed(2) + 's • Len: ' + note.duration.toFixed(2) + 's';
        }
      }
    }

    if (cloneBtn) {
      cloneBtn.disabled = !note;
      cloneBtn.classList.toggle('active', !!state.isCloningNote);
      cloneBtn.textContent = state.isCloningNote ? 'Tap Target Key...' : '⧉ Copy to Key';
    }
    if (handBtn && note) {
      handBtn.disabled = false;
      handBtn.textContent = 'Hand: ' + (note.hand === 'L' ? 'L (Blue)' : 'R (Green)');
    }
    if (semiDownBtn) semiDownBtn.disabled = !note;
    if (semiUpBtn) semiUpBtn.disabled = !note;
    if (octDownBtn) octDownBtn.disabled = !note;
    if (octUpBtn) octUpBtn.disabled = !note;
    if (delBtn) delBtn.disabled = !note && groupCount === 0;
  }
  const syncNoteInspectorUI = syncEditorInspectorUI;

  function pushUndoSnapshot() {
    state.undoStack.push(cloneSongSnapshot(state.song));
    if (state.undoStack.length > 30) {
      state.undoStack.shift();
    }
    state.redoStack = [];
    syncUndoRedoUI();
  }
  const pushEditorUndoState = pushUndoSnapshot;

  let autoSaveFlashTimer = null;
  function flashAutoSaveIndicator() {
    const bannerEl = document.getElementById('editAutoSaveBanner');
    const dotEl = document.getElementById('editAutoSaveDot');
    const textEl = document.getElementById('editAutoSaveText');
    if (!bannerEl || !textEl) return;
    if (dotEl) dotEl.className = 'autosave-dot';
    bannerEl.classList.add('saving');
    textEl.textContent = 'Saving changes...';
    if (autoSaveFlashTimer) clearTimeout(autoSaveFlashTimer);
    autoSaveFlashTimer = setTimeout(function () {
      bannerEl.classList.remove('saving');
      textEl.textContent = '✓ Changes saved automatically';
    }, 350);
  }

  function persistActiveSong() {
    if (!state.song) return;
    if (!state.song.id) state.song.id = state.activeSongId || ('custom_' + Date.now());
    if (!state.song.isCustom) {
      state.song.isModifiedBuiltin = true;
    }
    if (audioTranscriber && audioTranscriber.customSongStore) {
      audioTranscriber.customSongStore.saveSong(state.song);
    }
    flashAutoSaveIndicator();
  }

  function commitSongEdit(keepNoteRef) {
    if (!state.song || !Array.isArray(state.song.notes)) return;

    const keptGroupRefs = [];
    if (!keepNoteRef && state.selectedNoteIndices && state.selectedNoteIndices.size > 0) {
      state.selectedNoteIndices.forEach(function (i) {
        if (i >= 0 && i < state.song.notes.length) {
          keptGroupRefs.push(state.song.notes[i]);
        }
      });
    }

    state.song.notes.sort(function (a, b) {
      return (a.time - b.time) || (a.midi - b.midi);
    });

    if (keepNoteRef) {
      state.selectedNoteIndex = state.song.notes.indexOf(keepNoteRef);
      state.selectedNoteIndices.clear();
      if (state.selectedNoteIndex >= 0) {
        state.selectedNoteIndices.add(state.selectedNoteIndex);
      }
    } else if (keptGroupRefs.length > 0) {
      state.selectedNoteIndices.clear();
      keptGroupRefs.forEach(function (ref) {
        const newIdx = state.song.notes.indexOf(ref);
        if (newIdx >= 0) state.selectedNoteIndices.add(newIdx);
      });
      if (state.selectedNoteIndices.size > 0) {
        const arr = Array.from(state.selectedNoteIndices);
        state.selectedNoteIndex = arr[arr.length - 1];
      } else {
        state.selectedNoteIndex = -1;
      }
    } else if (state.selectedNoteIndex >= state.song.notes.length || state.selectedNoteIndex < 0) {
      state.selectedNoteIndex = -1;
      state.selectedNoteIndices.clear();
    }

    let maxEnd = 0;
    for (let i = 0; i < state.song.notes.length; i++) {
      const end = state.song.notes[i].time + state.song.notes[i].duration;
      if (end > maxEnd) maxEnd = end;
    }
    state.song.duration = Math.max(10, Math.round((maxEnd + 2.0) * 100) / 100);
    const bpmVal = Math.max(30, Math.min(260, state.song.bpm || 120));
    const offsetVal = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
    const bpbVal = state.song.beatsPerBar || 4;
    state.song.beats = generateBeatGrid(state.song.duration, 60 / bpmVal, bpbVal, offsetVal);

    persistActiveSong();
    resetNoteTriggersBefore(state.currentTime);
    calculateAutoRange(state);
    rebuildKeyboardGeometry();
    syncTimelineUI();
    syncLoopUI();
    syncEditorInspectorUI();
    syncUndoRedoUI();
  }

  function applySongSnapshot(snap) {
    if (!snap || !state.song) return;
    state.song.notes = snap.notes.map(function (n) {
      return {
        midi: n.midi,
        name: n.name || midiToName(n.midi),
        time: n.time,
        duration: n.duration,
        hand: n.hand,
        velocity: n.velocity
      };
    });
    state.song.bpm = snap.bpm || 120;
    state.song.gridOffset = typeof snap.gridOffset === 'number' ? snap.gridOffset : 0.8;
    state.song.beatsPerBar = snap.beatsPerBar || 4;
    state.song.duration = snap.duration || 15;
    state.selectedNoteIndex = -1;
    state.selectedNoteIndices.clear();
    state.groupDragBox = null;
    commitSongEdit(null);
  }

  function undoEdit() {
    if (state.undoStack.length === 0) return;
    state.redoStack.push(cloneSongSnapshot(state.song));
    const prev = state.undoStack.pop();
    applySongSnapshot(prev);
  }

  function redoEdit() {
    if (state.redoStack.length === 0) return;
    state.undoStack.push(cloneSongSnapshot(state.song));
    const next = state.redoStack.pop();
    applySongSnapshot(next);
  }

  const pointerMidis = {};
  let editDrag = null;
  let lastEmptyRunwayTap = { timeMs: 0, x: 0, y: 0 };

  function updateGroupDragSelection(box, baseIndices) {
    if (!box || !state.song || !Array.isArray(state.song.notes)) return;
    const minX = Math.min(box.x0, box.x1);
    const maxX = Math.max(box.x0, box.x1);
    const minY = Math.min(box.y0, box.y1);
    const maxY = Math.max(box.y0, box.y1);

    const horizon = state.horizonSec;
    const pxPerSec = strikeY / horizon;
    const curT = state.currentTime;
    const notes = state.song.notes;

    const intersecting = [];
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const nStart = n.time;
      const nEnd = n.time + n.duration;
      if (nEnd < curT - 0.04 || nStart > curT + horizon + 0.1) continue;

      const kGeom = state.keyMap[n.midi];
      if (!kGeom) continue;

      const rawBotY = strikeY - (nStart - curT) * pxPerSec;
      const rawTopY = strikeY - (nEnd - curT) * pxPerSec;
      const botY = Math.min(strikeY, rawBotY);
      const topY = Math.max(-16, rawTopY);
      if (botY - topY <= 2) continue;

      const nx0 = kGeom.laneX + 1.5;
      const nx1 = nx0 + Math.max(8, kGeom.laneW - 3);
      if (nx1 >= minX && nx0 <= maxX && botY >= minY && topY <= maxY) {
        intersecting.push({ index: i, midi: n.midi });
      }
    }

    let targetMidi = null;
    if (baseIndices && baseIndices.size > 0) {
      const firstBaseIdx = baseIndices.values().next().value;
      if (notes[firstBaseIdx]) {
        targetMidi = notes[firstBaseIdx].midi;
      }
    }
    if (targetMidi === null) {
      const startLaneMidi = xToNearestMidi(box.x0);
      const hasStartLane = intersecting.some(function (item) {
        return item.midi === startLaneMidi;
      });
      if (hasStartLane) {
        targetMidi = startLaneMidi;
      } else if (intersecting.length > 0) {
        const counts = {};
        let bestMidi = intersecting[0].midi;
        let bestCount = 0;
        for (let j = 0; j < intersecting.length; j++) {
          const m = intersecting[j].midi;
          counts[m] = (counts[m] || 0) + 1;
          if (counts[m] > bestCount) {
            bestCount = counts[m];
            bestMidi = m;
          }
        }
        targetMidi = bestMidi;
      } else {
        targetMidi = startLaneMidi;
      }
    }

    state.selectedNoteIndices = new Set(baseIndices || []);
    for (let k = 0; k < intersecting.length; k++) {
      if (intersecting[k].midi === targetMidi) {
        state.selectedNoteIndices.add(intersecting[k].index);
      }
    }

    if (state.selectedNoteIndices.size > 0) {
      const arr = Array.from(state.selectedNoteIndices);
      state.selectedNoteIndex = arr[arr.length - 1];
    } else {
      state.selectedNoteIndex = -1;
    }
    syncEditorInspectorUI();
  }

  canvas.addEventListener('pointerdown', function (e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const hitKey = hitTestPiano(x, y);

    // Edit Mode Copy-to-Key (Chord / Interval Builder): works on both virtual piano keys and runway lanes
    if (state.viewMode === 'edit' && state.isCloningNote && state.selectedNoteIndex >= 0 && state.selectedNoteIndex < state.song.notes.length) {
      if (hitKey !== null || y < strikeY) {
        const srcNote = state.song.notes[state.selectedNoteIndex];
        const targetMidi = hitKey !== null ? hitKey : xToNearestMidi(x);
        pushUndoSnapshot();

        let targetRef = null;
        for (let i = 0; i < state.song.notes.length; i++) {
          const existing = state.song.notes[i];
          if (existing.midi === targetMidi && Math.abs(existing.time - srcNote.time) < 0.01) {
            existing.duration = srcNote.duration;
            existing.hand = srcNote.hand || (targetMidi < 60 ? 'L' : 'R');
            existing.velocity = srcNote.velocity || 0.8;
            targetRef = existing;
            break;
          }
        }

        if (!targetRef) {
          targetRef = {
            midi: targetMidi,
            name: midiToName(targetMidi),
            time: srcNote.time,
            duration: srcNote.duration,
            hand: srcNote.hand || (targetMidi < 60 ? 'L' : 'R'),
            velocity: srcNote.velocity || 0.8
          };
          state.song.notes.push(targetRef);
        }

        const previewDur = Math.min(0.9, Math.max(0.25, srcNote.duration));
        audioEngine.playNote(srcNote.midi, previewDur, srcNote.velocity || 0.8);
        if (targetMidi !== srcNote.midi) {
          audioEngine.playNote(targetMidi, previewDur, srcNote.velocity || 0.8);
        }

        state.isCloningNote = false;
        commitSongEdit(targetRef);
        return;
      }
    }

    if (hitKey !== null) {
      try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
      pointerMidis[e.pointerId] = hitKey;
      userNoteOn(hitKey, 0.85);
      return;
    }

    if (y >= strikeY) return;

    // Play Mode: clicking the runway toggles Play/Pause
    if (state.viewMode !== 'edit') {
      setPlaying(!state.isPlaying);
      return;
    }

    // Edit Mode Runway Interaction
    try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
    const pxPerSec = strikeY / state.horizonSec;
    const hitNote = hitTestRunwayNote(x, y);

    if (state.editorTool === 'delete') {
      if (hitNote) {
        pushUndoSnapshot();
        state.song.notes.splice(hitNote.index, 1);
        state.selectedNoteIndex = -1;
        state.selectedNoteIndices.clear();
        commitSongEdit(null);
      }
      return;
    }

    function createNoteAtPoint(ptX, ptY) {
      const midi = xToNearestMidi(ptX);
      const rawT = state.currentTime + (strikeY - ptY) / pxPerSec;
      const snappedT = snapTimeToGrid(rawT);
      pushUndoSnapshot();
      const newNote = {
        midi: midi,
        name: midiToName(midi),
        time: snappedT,
        duration: 0.35,
        hand: midi < 60 ? 'L' : 'R',
        velocity: 0.82
      };
      state.song.notes.push(newNote);
      state.selectedNoteIndex = state.song.notes.length - 1;
      state.selectedNoteIndices.clear();
      state.selectedNoteIndices.add(state.selectedNoteIndex);
      audioEngine.playNote(midi, 0.35, 0.82);
      syncEditorInspectorUI();
      editDrag = {
        pointerId: e.pointerId,
        mode: 'resize',
        noteRef: newNote,
        origTime: newNote.time,
        origDur: newNote.duration,
        origMidi: newNote.midi,
        startX: ptX,
        startY: ptY,
        undoPushed: true,
        modified: true
      };
    }

    if (state.editorTool === 'add') {
      createNoteAtPoint(x, y);
      return;
    }

    // Group Select Mode (or Shift+Click in Select Mode)
    const isGroupAction = (state.editorTool === 'group') || (state.editorTool === 'select' && e.shiftKey);
    if (isGroupAction) {
      if (hitNote) {
        const tappedIdx = hitNote.index;
        const tappedNote = state.song.notes[tappedIdx];
        if (!tappedNote) return;

        // Seed from single selectedNoteIndex if set is empty and pitch matches
        if (state.selectedNoteIndices.size === 0 && state.selectedNoteIndex >= 0 && state.selectedNoteIndex < state.song.notes.length) {
          const prevNote = state.song.notes[state.selectedNoteIndex];
          if (prevNote && prevNote.midi === tappedNote.midi && state.selectedNoteIndex !== tappedIdx) {
            state.selectedNoteIndices.add(state.selectedNoteIndex);
          }
        }

        if (state.selectedNoteIndices.size === 0) {
          state.selectedNoteIndices.add(tappedIdx);
          state.selectedNoteIndex = tappedIdx;
          audioEngine.playNote(tappedNote.midi, 0.35, tappedNote.velocity || 0.8);
          syncEditorInspectorUI();
        } else if (state.selectedNoteIndices.has(tappedIdx)) {
          state.selectedNoteIndices.delete(tappedIdx);
          if (state.selectedNoteIndices.size > 0) {
            const rem = Array.from(state.selectedNoteIndices);
            state.selectedNoteIndex = rem[rem.length - 1];
          } else {
            state.selectedNoteIndex = -1;
          }
          syncEditorInspectorUI();
        } else {
          const firstIdx = state.selectedNoteIndices.values().next().value;
          const firstNote = state.song.notes[firstIdx];
          const targetMidi = firstNote ? firstNote.midi : tappedNote.midi;

          if (tappedNote.midi !== targetMidi) {
            syncEditorInspectorUI();
            const labelEl = document.getElementById('inspectorNoteLabel');
            if (labelEl) {
              labelEl.textContent = 'Cannot group ' + (tappedNote.name || midiToName(tappedNote.midi)) + ' with ' + midiToName(targetMidi) + ' (must be same note)';
            }
          } else {
            state.selectedNoteIndices.add(tappedIdx);
            state.selectedNoteIndex = tappedIdx;
            audioEngine.playNote(tappedNote.midi, 0.35, tappedNote.velocity || 0.8);
            syncEditorInspectorUI();
          }
        }
        return;
      }

      // Empty runway in Group Select mode: start Marquee drag box
      state.groupDragBox = { x0: x, y0: y, x1: x, y1: y };
      editDrag = {
        pointerId: e.pointerId,
        mode: 'group-box',
        startX: x,
        startY: y,
        baseIndices: new Set(state.selectedNoteIndices)
      };
      return;
    }

    // Single Select Tool Mode
    if (hitNote) {
      state.selectedNoteIndex = hitNote.index;
      state.selectedNoteIndices.clear();
      state.selectedNoteIndices.add(hitNote.index);
      const note = state.song.notes[hitNote.index];
      audioEngine.playNote(note.midi, 0.35, note.velocity || 0.8);
      syncEditorInspectorUI();
      editDrag = {
        pointerId: e.pointerId,
        mode: hitNote.zone === 'resize-top' ? 'resize' : 'move',
        noteRef: note,
        origTime: note.time,
        origDur: note.duration,
        origMidi: note.midi,
        startX: x,
        startY: y,
        undoPushed: false,
        modified: false
      };
    } else {
      const nowMs = performance.now();
      const dtTap = nowMs - lastEmptyRunwayTap.timeMs;
      const distTap = Math.hypot(x - lastEmptyRunwayTap.x, y - lastEmptyRunwayTap.y);
      if (dtTap < 320 && distTap < 20) {
        lastEmptyRunwayTap.timeMs = 0;
        createNoteAtPoint(x, y);
        return;
      }
      lastEmptyRunwayTap = { timeMs: nowMs, x: x, y: y };
      state.selectedNoteIndex = -1;
      state.selectedNoteIndices.clear();
      syncEditorInspectorUI();
      editDrag = {
        pointerId: e.pointerId,
        mode: 'scroll',
        startY: y,
        origCurrentTime: state.currentTime
      };
    }
  });

  canvas.addEventListener('pointermove', function (e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (pointerMidis[e.pointerId] !== undefined) {
      const hit = hitTestPiano(x, y);
      const prev = pointerMidis[e.pointerId];
      if (hit !== null && hit !== prev) {
        userNoteOff(prev);
        pointerMidis[e.pointerId] = hit;
        userNoteOn(hit, 0.76);
      }
      return;
    }

    if (editDrag && editDrag.pointerId === e.pointerId) {
      const pxPerSec = strikeY / state.horizonSec;

      if (editDrag.mode === 'group-box') {
        state.groupDragBox = {
          x0: editDrag.startX,
          y0: editDrag.startY,
          x1: x,
          y1: Math.min(strikeY, Math.max(0, y))
        };
        updateGroupDragSelection(state.groupDragBox, editDrag.baseIndices);
        return;
      }

      if (editDrag.mode === 'scroll') {
        const dy = y - editDrag.startY;
        const dtSec = dy / pxPerSec;
        seekTo(editDrag.origCurrentTime + dtSec);
        return;
      }

      const note = editDrag.noteRef;
      if (!note) return;

      if (editDrag.mode === 'move') {
        const dx = x - editDrag.startX;
        const dy = y - editDrag.startY;
        if (Math.abs(dx) < 3 && Math.abs(dy) < 3 && !editDrag.modified) return;

        if (!editDrag.undoPushed) {
          pushUndoSnapshot();
          editDrag.undoPushed = true;
        }
        editDrag.modified = true;

        const newMidi = xToNearestMidi(x);
        if (newMidi !== note.midi) {
          note.midi = newMidi;
          note.name = midiToName(newMidi);
          audioEngine.playNote(newMidi, 0.3, note.velocity || 0.8);
        }

        if (Math.abs(dy) >= 10) {
          const rawNewTime = editDrag.origTime - (dy / pxPerSec);
          note.time = snapTimeToGrid(rawNewTime);
        } else {
          note.time = editDrag.origTime;
        }
        syncEditorInspectorUI();
      } else if (editDrag.mode === 'resize') {
        if (!editDrag.undoPushed) {
          pushUndoSnapshot();
          editDrag.undoPushed = true;
        }
        editDrag.modified = true;

        const rawEndTime = state.currentTime + (strikeY - y) / pxPerSec;
        const snappedEnd = snapTimeToGrid(rawEndTime);
        const newDur = Math.max(0.06, Math.min(8.0, snappedEnd - note.time));
        note.duration = Math.round(newDur * 1000) / 1000;
        syncEditorInspectorUI();
      }
    }
  });

  function releasePointer(e) {
    if (pointerMidis[e.pointerId] !== undefined) {
      userNoteOff(pointerMidis[e.pointerId]);
      delete pointerMidis[e.pointerId];
    }
    if (editDrag && editDrag.pointerId === e.pointerId) {
      const finishedDrag = editDrag;
      editDrag = null;
      if (finishedDrag.mode === 'group-box') {
        if (state.groupDragBox) {
          const dist = Math.hypot(
            state.groupDragBox.x1 - state.groupDragBox.x0,
            state.groupDragBox.y1 - state.groupDragBox.y0
          );
          if (dist < 5) {
            state.selectedNoteIndices.clear();
            state.selectedNoteIndex = -1;
          } else {
            updateGroupDragSelection(state.groupDragBox, finishedDrag.baseIndices);
          }
        }
        state.groupDragBox = null;
        syncEditorInspectorUI();
        return;
      }
      if (finishedDrag.modified && finishedDrag.noteRef) {
        commitSongEdit(finishedDrag.noteRef);
      }
    }
  }
  canvas.addEventListener('pointerup', releasePointer);
  canvas.addEventListener('pointercancel', releasePointer);

  // Mouse Wheel / Trackpad Scrolling on Runway in Edit Mode
  canvas.addEventListener('wheel', function (e) {
    if (state.viewMode === 'edit') {
      e.preventDefault();
      const pxPerSec = Math.max(40, strikeY / state.horizonSec);
      seekTo(state.currentTime - (e.deltaY / pxPerSec));
    }
  }, { passive: false });

  // Computer Keyboard Support & Edit Mode Shortcuts
  const KEY_OFFSETS = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12, o: 13, l: 14 };
  const heldKeys = {};
  window.addEventListener('keydown', function (e) {
    if (e.target && (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT')) return;
    if (state.viewMode === 'home') return;

    const k = String(e.key || '').toLowerCase();

    // Edit Mode Undo / Redo & Note Deletion / Transpose Shortcuts
    if (state.viewMode === 'edit') {
      if ((e.ctrlKey || e.metaKey) && k === 'z') {
        e.preventDefault();
        if (e.shiftKey) redoEdit();
        else undoEdit();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && k === 'y') {
        e.preventDefault();
        redoEdit();
        return;
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && (state.selectedNoteIndex >= 0 || state.selectedNoteIndices.size > 0)) {
        e.preventDefault();
        pushUndoSnapshot();
        if (state.selectedNoteIndices.size > 0) {
          state.song.notes = state.song.notes.filter(function (_, i) {
            return !state.selectedNoteIndices.has(i);
          });
        } else {
          state.song.notes.splice(state.selectedNoteIndex, 1);
        }
        state.selectedNoteIndex = -1;
        state.selectedNoteIndices.clear();
        commitSongEdit(null);
        return;
      }
    }

    if (e.code === 'Space') {
      e.preventDefault();
      if (state.viewMode === 'edit') {
        setViewMode('play');
      }
      setPlaying(!state.isPlaying);
      return;
    }
    if (!e.ctrlKey && !e.metaKey && !e.altKey && (k === 'q' || k === 'm')) {
      setHudMinimized(!state.hudMinimized);
      return;
    }
    if (KEY_OFFSETS[k] !== undefined && !heldKeys[k]) {
      heldKeys[k] = true;
      userNoteOn(60 + KEY_OFFSETS[k], 0.82);
    }
  });

  window.addEventListener('keyup', function (e) {
    const k = String(e.key || '').toLowerCase();
    if (KEY_OFFSETS[k] !== undefined) {
      delete heldKeys[k];
      userNoteOff(60 + KEY_OFFSETS[k]);
    }
  });

  // WebMIDI Initialization, Sustain Pedal (CC 64), and Live Connection Badge
  const midiStatusBadge = document.getElementById('midiStatusBadge');
  const midiStatusText = document.getElementById('midiStatusText');

  function updateMidiBadge(connectedNames) {
    if (!midiStatusBadge || !midiStatusText) return;
    if (connectedNames && connectedNames.length > 0) {
      const primaryName = connectedNames[0];
      midiStatusBadge.classList.add('connected');
      midiStatusText.textContent = 'MIDI: ' + primaryName;
      midiStatusBadge.title = 'Connected MIDI: ' + connectedNames.join(', ');
    } else {
      midiStatusBadge.classList.remove('connected', 'sustain-active');
      midiStatusText.textContent = 'MIDI';
      midiStatusBadge.title = 'Connect a USB or Bluetooth MIDI keyboard';
    }
  }

  try {
    if (navigator.requestMIDIAccess) {
      navigator.requestMIDIAccess().then(function (access) {
        function bindInputs() {
          const connectedNames = [];
          access.inputs.forEach(function (input) {
            if (input.state === 'connected') {
              connectedNames.push(input.name || input.manufacturer || 'Keyboard');
            }
            input.onmidimessage = function (msg) {
              if (!msg.data || msg.data.length < 2) return;
              const cmd = msg.data[0] & 0xf0;
              const data1 = msg.data[1];
              const data2 = msg.data.length > 2 ? msg.data[2] : 0;

              if (cmd === 0x90 && data2 > 0) {
                userNoteOn(data1, data2 / 127);
              } else if (cmd === 0x80 || (cmd === 0x90 && data2 === 0)) {
                userNoteOff(data1);
              } else if (cmd === 0xb0 && data1 === 64) {
                // Control Change 64: Sustain Pedal
                const pedalDown = data2 >= 64;
                audioEngine.setSustainPedal(pedalDown);
                if (midiStatusBadge) {
                  midiStatusBadge.classList.toggle('sustain-active', pedalDown);
                }
              }
            };
          });
          updateMidiBadge(connectedNames);
        }
        bindInputs();
        access.onstatechange = bindInputs;
      }).catch(function () {});
    }
  } catch (e) {}

  /********************************************************************
   * PHASE 4A: HOME LIBRARY, VIEW SWITCHER & EDITOR TOOLBAR BINDINGS
   ********************************************************************/
  const homeView = document.getElementById('homeView');
  const homeSearchInput = document.getElementById('homeSearchInput');
  const homeSongGrid = document.getElementById('homeSongGrid');
  const homeOpenAiModalBtn = document.getElementById('homeOpenAiModalBtn');
  const backToHomeBtn = document.getElementById('backToHomeBtn');
  const toggleEditModeBtn = document.getElementById('toggleEditModeBtn');
  const editorToolbar = document.getElementById('editorToolbar');
  const hudPanel = document.getElementById('hudPanel');
  const hudToggleBtn = document.getElementById('hudToggleBtn');
  const hudMiniSummary = document.getElementById('hudMiniSummary');

  function downloadSongMidi(songObj) {
    if (!songObj || !audioTranscriber) return;
    try {
      const blob = audioTranscriber.exportSongToMidiBlob(songObj);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const safeName = String(songObj.title || 'KeyFlow_Song')
        .replace(/[^a-zA-Z0-9_\-\s]/g, '')
        .trim()
        .replace(/\s+/g, '_') || 'KeyFlow_Song';
      a.href = url;
      a.download = safeName + '.mid';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 3000);
    } catch (err) {
      console.warn('MIDI Export error:', err);
    }
  }

  function deleteOrResetSong(id) {
    const s = SONGS[id];
    if (!s) return;
    if (s.isCustom) {
      if (audioTranscriber && audioTranscriber.customSongStore) {
        audioTranscriber.customSongStore.deleteSong(id);
      }
      delete SONGS[id];
      if (state.activeSongId === id) {
        loadSongById('let_it_be');
      }
      renderHomeSongGrid();
    } else if (FACTORY_BUILTIN_SONGS[id]) {
      SONGS[id] = JSON.parse(JSON.stringify(FACTORY_BUILTIN_SONGS[id]));
      if (audioTranscriber && audioTranscriber.customSongStore) {
        audioTranscriber.customSongStore.deleteSong(id);
      }
      if (state.activeSongId === id) {
        loadSongById(id);
      }
      renderHomeSongGrid();
    }
  }

  function getSongSourceCategory(songObj) {
    if (!songObj) return 'builtin';
    const src = String(songObj.source || '').toLowerCase();
    if (src.indexOf('ai') !== -1) return 'ai';
    if (src.indexOf('midi') !== -1) return 'midi';
    if (songObj.isCustom) return 'midi';
    return 'builtin';
  }

  function renderHomeSongGrid() {
    if (!homeSongGrid) return;
    homeSongGrid.innerHTML = '';

    const allIds = Object.keys(SONGS);
    const customList = [];
    const builtinList = [];

    for (let i = 0; i < allIds.length; i++) {
      const id = allIds[i];
      const s = SONGS[id];
      if (!s || !Array.isArray(s.notes)) continue;
      if (!s.id) s.id = id;
      if (s.isCustom) {
        customList.push(s);
      } else {
        builtinList.push(s);
      }
    }

    customList.sort(function (a, b) {
      return (b.createdAt || 0) - (a.createdAt || 0);
    });

    const orderedSongs = customList.concat(builtinList);
    const query = String(state.homeSearch || '').trim().toLowerCase();
    const filter = state.homeFilter || 'all';

    const matchedSongs = orderedSongs.filter(function (s) {
      const cat = getSongSourceCategory(s);
      if (filter !== 'all' && cat !== filter) return false;
      if (query && String(s.title || '').toLowerCase().indexOf(query) === -1) return false;
      return true;
    });

    if (matchedSongs.length === 0) {
      const emptyEl = document.createElement('div');
      emptyEl.className = 'song-grid-empty';
      emptyEl.textContent = 'No songs match this filter. Upload a .MID file or convert an audio song with AI Audio!';
      homeSongGrid.appendChild(emptyEl);
      return;
    }

    matchedSongs.forEach(function (s) {
      const cat = getSongSourceCategory(s);
      const card = document.createElement('div');
      card.className = 'song-card';

      // Compact Top Row: Clickable Song Title (expands details) + Play Button
      const mainRow = document.createElement('div');
      mainRow.className = 'song-card-main';

      const titleToggle = document.createElement('div');
      titleToggle.className = 'song-card-title-toggle';
      titleToggle.title = 'Tap to expand song details & options';

      const chevronEl = document.createElement('span');
      chevronEl.className = 'song-card-chevron';
      chevronEl.textContent = '▸';

      const titleEl = document.createElement('h3');
      titleEl.className = 'song-card-title';
      titleEl.textContent = s.title;

      titleToggle.appendChild(chevronEl);
      titleToggle.appendChild(titleEl);
      titleToggle.addEventListener('click', function () {
        card.classList.toggle('expanded');
      });

      const playBtn = document.createElement('button');
      playBtn.type = 'button';
      playBtn.className = 'ui-btn ui-btn-emerald song-card-play-btn';
      playBtn.textContent = '▶ Play';
      playBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        loadSongById(s.id);
        setViewMode('play');
      });

      mainRow.appendChild(titleToggle);
      mainRow.appendChild(playBtn);

      // Collapsible Details Section: Source Badges, Metadata & Secondary Actions
      const detailsDiv = document.createElement('div');
      detailsDiv.className = 'song-card-details';

      const badgesRow = document.createElement('div');
      badgesRow.className = 'song-card-badges';

      const srcBadge = document.createElement('span');
      if (cat === 'ai') {
        srcBadge.className = 'song-badge ai';
        srcBadge.textContent = 'AI Audio';
      } else if (cat === 'midi') {
        srcBadge.className = 'song-badge midi';
        srcBadge.textContent = 'MIDI Upload';
      } else {
        srcBadge.className = 'song-badge builtin';
        srcBadge.textContent = 'Built-In Classic';
      }
      badgesRow.appendChild(srcBadge);

      if (s.isModifiedBuiltin) {
        const editBadge = document.createElement('span');
        editBadge.className = 'song-badge edited';
        editBadge.textContent = 'Edited';
        badgesRow.appendChild(editBadge);
      }

      const metaEl = document.createElement('div');
      metaEl.className = 'song-card-meta';
      const bpmNum = Math.round(s.bpm || 120);
      metaEl.textContent = formatTime(s.duration) + ' • ' + s.notes.length + ' notes • ' + bpmNum + ' BPM';

      const actionsRow = document.createElement('div');
      actionsRow.className = 'song-card-actions';

      const editBtn = document.createElement('button');
      editBtn.type = 'button';
      editBtn.className = 'ui-btn ui-btn-accent';
      editBtn.textContent = '✎ Edit Notes';
      editBtn.addEventListener('click', function () {
        loadSongById(s.id);
        setViewMode('edit');
      });

      const midiBtn = document.createElement('button');
      midiBtn.type = 'button';
      midiBtn.className = 'ui-btn';
      midiBtn.title = 'Export 2-Track Standard .MID File';
      midiBtn.textContent = '⬇ .MID';
      midiBtn.addEventListener('click', function () {
        downloadSongMidi(s);
      });

      actionsRow.appendChild(editBtn);
      actionsRow.appendChild(midiBtn);

      if (s.isCustom || s.isModifiedBuiltin) {
        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.className = 'ui-btn ui-btn-danger';
        delBtn.title = s.isCustom ? 'Delete custom song from library' : 'Reset built-in song to factory default';
        delBtn.textContent = s.isCustom ? '🗑 Delete' : '↺ Reset';
        delBtn.addEventListener('click', function () {
          deleteOrResetSong(s.id);
        });
        actionsRow.appendChild(delBtn);
      }

      detailsDiv.appendChild(badgesRow);
      detailsDiv.appendChild(metaEl);
      detailsDiv.appendChild(actionsRow);

      card.appendChild(mainRow);
      card.appendChild(detailsDiv);
      homeSongGrid.appendChild(card);
    });
  }

  function setViewMode(mode, songId) {
    if (songId && SONGS[songId]) {
      loadSongById(songId);
    }
    state.viewMode = mode;
    state.isCloningNote = false;
    state.groupDragBox = null;
    if (mode !== 'edit') {
      state.selectedNoteIndices.clear();
      state.selectedNoteIndex = -1;
    }

    if (homeView) {
      homeView.classList.toggle('hidden', mode !== 'home');
    }

    const editAutoSaveBanner = document.getElementById('editAutoSaveBanner');
    const editAutoSaveText = document.getElementById('editAutoSaveText');
    const waitBannerEl = document.getElementById('waitBanner');
    if (editAutoSaveBanner) {
      if (mode === 'edit') {
        editAutoSaveBanner.style.display = 'inline-flex';
        editAutoSaveBanner.classList.remove('saving');
        if (editAutoSaveText) editAutoSaveText.textContent = '✓ Changes saved automatically';
      } else {
        editAutoSaveBanner.style.display = 'none';
      }
    }
    if (waitBannerEl && mode === 'edit') {
      waitBannerEl.classList.remove('visible');
    }

    if (mode === 'home') {
      setPlaying(false);
      renderHomeSongGrid();
      return;
    }

    if (mode === 'edit') {
      setPlaying(false);
      if (hudPanel) hudPanel.style.display = 'none';
      if (editorToolbar) editorToolbar.style.display = 'flex';
      if (toggleEditModeBtn) {
        toggleEditModeBtn.textContent = '▶ Back to Play';
        toggleEditModeBtn.className = 'ui-btn ui-btn-emerald';
        toggleEditModeBtn.title = 'Return to Play Mode at current position';
      }
      syncEditorInspectorUI();
      syncUndoRedoUI();
      rebuildKeyboardGeometry();
    } else {
      // 'play' mode
      if (editorToolbar) editorToolbar.style.display = 'none';
      if (hudPanel) hudPanel.style.display = '';
      if (toggleEditModeBtn) {
        toggleEditModeBtn.textContent = '✎ Edit Notes';
        toggleEditModeBtn.className = 'ui-btn ui-btn-accent';
        toggleEditModeBtn.title = 'Open Visual Note Editor';
      }
      resetNoteTriggersBefore(state.currentTime);
      rebuildKeyboardGeometry();
    }
  }

  function loadSongById(id) {
    const s = SONGS[id];
    if (!s) return;
    setPlaying(false);
    state.activeSongId = id;
    state.song = s;
    if (!s.bpm) s.bpm = 120;
    if (typeof s.gridOffset !== 'number') s.gridOffset = 0.8;
    if (!s.beatsPerBar) s.beatsPerBar = 4;

    state.currentTime = 0;
    state.loopA = Math.min(2.0, s.duration * 0.12);
    state.loopB = Math.min(s.duration * 0.48, s.duration - 1.0);
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
    state.waitStartedAtMs = null;

    state.selectedNoteIndex = -1;
    state.selectedNoteIndices.clear();
    state.groupDragBox = null;
    state.isCloningNote = false;
    state.undoStack = [];
    state.redoStack = [];

    document.getElementById('songTitleHeader').textContent = s.title;

    const delBtn = document.getElementById('deleteCustomSongBtn');
    if (delBtn) {
      delBtn.style.display = (s.isCustom || s.isModifiedBuiltin) ? 'inline-flex' : 'none';
      const span = delBtn.querySelector('span') || delBtn;
      span.textContent = s.isCustom ? 'Delete Song' : 'Reset Song';
    }

    resetNoteTriggersBefore(0);
    calculateAutoRange(state);
    rebuildKeyboardGeometry();
    syncTimelineUI();
    syncLoopUI();
    syncEditorInspectorUI();
    syncUndoRedoUI();
  }

  // Home View & Top Bar Navigation Listeners
  if (backToHomeBtn) {
    backToHomeBtn.addEventListener('click', function () {
      setViewMode('home');
    });
  }

  if (toggleEditModeBtn) {
    toggleEditModeBtn.addEventListener('click', function () {
      if (state.viewMode === 'edit') {
        setViewMode('play');
      } else {
        setViewMode('edit');
      }
    });
  }

  if (homeSearchInput) {
    homeSearchInput.addEventListener('input', function (e) {
      state.homeSearch = e.target.value || '';
      renderHomeSongGrid();
    });
  }

  document.querySelectorAll('.home-filter-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      state.homeFilter = btn.getAttribute('data-filter') || 'all';
      document.querySelectorAll('.home-filter-btn').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
      renderHomeSongGrid();
    });
  });

  // Visual Note Editor Toolbar Listeners
  const toolSelectBtn = document.getElementById('toolSelectBtn');
  const toolGroupBtn = document.getElementById('toolGroupBtn');
  const toolAddBtn = document.getElementById('toolAddBtn');
  const toolDeleteBtn = document.getElementById('toolDeleteBtn');
  const editorSnapSelect = document.getElementById('editorSnapSelect');
  const editorUndoBtn = document.getElementById('editorUndoBtn');
  const editorRedoBtn = document.getElementById('editorRedoBtn');
  const editorBpmInput = document.getElementById('editorBpmInput');
  const editorGridLeftBtn = document.getElementById('editorGridLeftBtn');
  const editorGridRightBtn = document.getElementById('editorGridRightBtn');
  const inspectorMergeBtn = document.getElementById('inspectorMergeBtn');
  const inspectorCloneBtn = document.getElementById('inspectorCloneBtn');
  const inspectorHandBtn = document.getElementById('inspectorHandBtn');
  const inspectorSemiDownBtn = document.getElementById('inspectorSemiDownBtn');
  const inspectorSemiUpBtn = document.getElementById('inspectorSemiUpBtn');
  const inspectorOctDownBtn = document.getElementById('inspectorOctDownBtn');
  const inspectorOctUpBtn = document.getElementById('inspectorOctUpBtn');
  const inspectorDeleteNoteBtn = document.getElementById('inspectorDeleteNoteBtn');

  function setEditorTool(toolName) {
    state.editorTool = toolName;
    state.isCloningNote = false;
    state.groupDragBox = null;
    [toolSelectBtn, toolGroupBtn, toolAddBtn, toolDeleteBtn].forEach(function (btn) {
      if (!btn) return;
      btn.classList.toggle('active', btn.getAttribute('data-tool') === toolName);
    });
    syncEditorInspectorUI();
  }

  if (toolSelectBtn) {
    toolSelectBtn.addEventListener('click', function () { setEditorTool('select'); });
  }
  if (toolGroupBtn) {
    toolGroupBtn.addEventListener('click', function () { setEditorTool('group'); });
  }
  if (toolAddBtn) {
    toolAddBtn.addEventListener('click', function () { setEditorTool('add'); });
  }
  if (toolDeleteBtn) {
    toolDeleteBtn.addEventListener('click', function () { setEditorTool('delete'); });
  }

  if (editorSnapSelect) {
    editorSnapSelect.addEventListener('change', function (e) {
      state.editorSnapBeats = parseFloat(e.target.value) || 0;
    });
  }

  if (editorUndoBtn) {
    editorUndoBtn.addEventListener('click', function () { undoEdit(); });
  }
  if (editorRedoBtn) {
    editorRedoBtn.addEventListener('click', function () { redoEdit(); });
  }

  if (editorBpmInput) {
    editorBpmInput.addEventListener('change', function (e) {
      const val = Math.max(40, Math.min(240, parseInt(e.target.value, 10) || 120));
      e.target.value = val;
      pushUndoSnapshot();
      state.song.bpm = val;
      const keepRef = (state.selectedNoteIndex >= 0) ? state.song.notes[state.selectedNoteIndex] : null;
      commitSongEdit(keepRef);
    });
  }

  if (editorGridLeftBtn) {
    editorGridLeftBtn.addEventListener('click', function () {
      pushUndoSnapshot();
      const curOff = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
      state.song.gridOffset = Math.max(0, Math.round((curOff - 0.05) * 1000) / 1000);
      const keepRef = (state.selectedNoteIndex >= 0) ? state.song.notes[state.selectedNoteIndex] : null;
      commitSongEdit(keepRef);
    });
  }

  if (editorGridRightBtn) {
    editorGridRightBtn.addEventListener('click', function () {
      pushUndoSnapshot();
      const curOff = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
      state.song.gridOffset = Math.round((curOff + 0.05) * 1000) / 1000;
      const keepRef = (state.selectedNoteIndex >= 0) ? state.song.notes[state.selectedNoteIndex] : null;
      commitSongEdit(keepRef);
    });
  }

  function transposeSelectedNote(semitones) {
    const idx = state.selectedNoteIndex;
    if (idx < 0 || idx >= state.song.notes.length) return;
    const note = state.song.notes[idx];
    const nextMidi = Math.max(21, Math.min(108, note.midi + semitones));
    if (nextMidi === note.midi) return;
    pushUndoSnapshot();
    note.midi = nextMidi;
    note.name = midiToName(nextMidi);
    audioEngine.playNote(nextMidi, 0.35, note.velocity || 0.8);
    commitSongEdit(note);
  }

  if (inspectorMergeBtn) {
    inspectorMergeBtn.addEventListener('click', function () {
      if (!state.selectedNoteIndices || state.selectedNoteIndices.size < 2) return;

      const selectedNotes = [];
      state.selectedNoteIndices.forEach(function (idx) {
        if (idx >= 0 && idx < state.song.notes.length) {
          selectedNotes.push(state.song.notes[idx]);
        }
      });

      if (selectedNotes.length < 2) return;

      const targetMidi = selectedNotes[0].midi;
      for (let i = 1; i < selectedNotes.length; i++) {
        if (selectedNotes[i].midi !== targetMidi) {
          const labelEl = document.getElementById('inspectorNoteLabel');
          if (labelEl) {
            labelEl.textContent = 'Cannot merge ' + midiToName(selectedNotes[i].midi) + ' with ' + midiToName(targetMidi) + ' (must be same note)';
          }
          return;
        }
      }

      pushEditorUndoState();

      const startTime = Math.min.apply(null, selectedNotes.map(function (n) { return n.time; }));
      const endTime = Math.max.apply(null, selectedNotes.map(function (n) { return n.time + n.duration; }));
      const mergedDuration = Math.max(0.06, Math.round((endTime - startTime) * 1000) / 1000);
      const maxVel = Math.max.apply(null, selectedNotes.map(function (n) { return n.velocity || 0.78; }));
      const hand = selectedNotes[0].hand || (targetMidi < 60 ? 'L' : 'R');

      const mergedNote = {
        midi: targetMidi,
        name: midiToName(targetMidi),
        time: startTime,
        duration: mergedDuration,
        hand: hand,
        velocity: maxVel
      };

      state.song.notes = state.song.notes.filter(function (_, idx) {
        return !state.selectedNoteIndices.has(idx);
      });
      state.song.notes.push(mergedNote);

      state.isCloningNote = false;
      state.selectedNoteIndices.clear();
      audioEngine.playNote(mergedNote.midi, Math.min(1.2, mergedNote.duration), mergedNote.velocity);
      commitSongEdit(mergedNote);
    });
  }

  if (inspectorCloneBtn) {
    inspectorCloneBtn.addEventListener('click', function () {
      const idx = state.selectedNoteIndex;
      if (idx < 0 || idx >= state.song.notes.length) {
        state.isCloningNote = false;
        syncEditorInspectorUI();
        return;
      }
      state.isCloningNote = !state.isCloningNote;
      syncEditorInspectorUI();
    });
  }

  if (inspectorHandBtn) {
    inspectorHandBtn.addEventListener('click', function () {
      const idx = state.selectedNoteIndex;
      if (idx < 0 || idx >= state.song.notes.length) return;
      const note = state.song.notes[idx];
      pushUndoSnapshot();
      note.hand = (note.hand === 'L') ? 'R' : 'L';
      commitSongEdit(note);
    });
  }

  if (inspectorSemiDownBtn) {
    inspectorSemiDownBtn.addEventListener('click', function () { transposeSelectedNote(-1); });
  }
  if (inspectorSemiUpBtn) {
    inspectorSemiUpBtn.addEventListener('click', function () { transposeSelectedNote(1); });
  }
  if (inspectorOctDownBtn) {
    inspectorOctDownBtn.addEventListener('click', function () { transposeSelectedNote(-12); });
  }
  if (inspectorOctUpBtn) {
    inspectorOctUpBtn.addEventListener('click', function () { transposeSelectedNote(12); });
  }

  if (inspectorDeleteNoteBtn) {
    inspectorDeleteNoteBtn.addEventListener('click', function () {
      if (state.selectedNoteIndex < 0 && state.selectedNoteIndices.size === 0) return;
      pushUndoSnapshot();
      if (state.selectedNoteIndices.size > 0) {
        state.song.notes = state.song.notes.filter(function (_, idx) {
          return !state.selectedNoteIndices.has(idx);
        });
      } else {
        state.song.notes.splice(state.selectedNoteIndex, 1);
      }
      state.selectedNoteIndex = -1;
      state.selectedNoteIndices.clear();
      commitSongEdit(null);
    });
  }

  document.getElementById('rangeModeSelect').addEventListener('change', function (e) {
    state.rangeMode = e.target.value;
    calculateAutoRange(state);
    rebuildKeyboardGeometry();
  });

  document.getElementById('midiInput').addEventListener('change', function (e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (ev) {
      try {
        const parsed = parseStandardMidiFile(ev.target.result, file.name);
        const id = 'midi_' + Date.now();
        parsed.id = id;
        parsed.isCustom = true;
        parsed.source = 'midi-upload';
        parsed.createdAt = Date.now();
        SONGS[id] = parsed;
        if (audioTranscriber && audioTranscriber.customSongStore) {
          audioTranscriber.customSongStore.saveSong(parsed);
        }
        renderHomeSongGrid();
        loadSongById(id);
        setViewMode('play');
      } catch (err) {
        console.warn('MIDI parse error:', err);
      }
    };
    reader.readAsArrayBuffer(file);
    e.target.value = '';
  });

  // Play / Pause & Restart
  document.getElementById('playPauseBtn').addEventListener('click', function () {
    if (state.viewMode === 'edit') {
      setViewMode('play');
    }
    setPlaying(!state.isPlaying);
  });
  document.getElementById('restartBtn').addEventListener('click', function () {
    seekTo(state.loopEnabled ? state.loopA : 0);
  });

  // Speed Selector & Collapsible HUD Panel
  function setHudMinimized(minimized) {
    state.hudMinimized = !!minimized;
    if (hudPanel) {
      hudPanel.classList.toggle('minimized', state.hudMinimized);
    }
    if (hudMiniSummary) {
      hudMiniSummary.textContent = formatSpeedBadge(state.speed);
    }
  }

  if (hudToggleBtn) {
    hudToggleBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      setHudMinimized(!state.hudMinimized);
    });
  }
  if (hudPanel) {
    hudPanel.addEventListener('click', function () {
      if (state.hudMinimized) {
        setHudMinimized(false);
      }
    });
  }

  const SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5];
  document.getElementById('speedBtn').addEventListener('click', function (e) {
    let nextIdx = 0;
    for (let i = 0; i < SPEEDS.length; i++) {
      if (SPEEDS[i] > state.speed + 0.01) {
        nextIdx = i;
        break;
      }
    }
    state.speed = SPEEDS[nextIdx];
    const speedStr = formatSpeedBadge(state.speed);
    e.currentTarget.textContent = speedStr;
    if (hudMiniSummary) hudMiniSummary.textContent = speedStr;
    syncLoopUI();
  });

  // Wait Mode Switch
  document.getElementById('waitSwitch').addEventListener('click', function (e) {
    state.waitMode = !state.waitMode;
    e.currentTarget.classList.toggle('active-blue', state.waitMode);
    document.getElementById('waitOrb').classList.toggle('orb-dim', !state.waitMode);
    if (!state.waitMode) {
      state.waitingMidis.clear();
      state.waitStartedAtMs = null;
      updateWaitBanner();
    }
  });

  // Hand Isolation Controls
  function syncHandIsolationUI() {
    const sw = document.getElementById('handIsoSwitch');
    const orb = document.getElementById('handOrb');
    const lBtn = document.getElementById('leftHandBtn');
    const rBtn = document.getElementById('rightHandBtn');

    if (sw) {
      sw.classList.toggle('active-green', state.handIsoMaster);
      sw.setAttribute('aria-checked', state.handIsoMaster ? 'true' : 'false');
    }
    if (orb) {
      orb.classList.toggle('orb-dim', !state.handIsoMaster);
    }
    if (lBtn) {
      lBtn.className = 'hand-pill hand-pill-l ' + (state.handIsoMaster && state.leftHandOn ? 'active' : 'inactive');
    }
    if (rBtn) {
      rBtn.className = 'hand-pill hand-pill-r ' + (state.handIsoMaster && state.rightHandOn ? 'active' : 'inactive');
    }
  }

  document.getElementById('handIsoSwitch').addEventListener('click', function () {
    state.handIsoMaster = !state.handIsoMaster;
    if (state.handIsoMaster) {
      const targetHand = state.lastIsolatedHand || 'R';
      state.leftHandOn = (targetHand === 'L');
      state.rightHandOn = (targetHand === 'R');
    } else {
      state.leftHandOn = false;
      state.rightHandOn = false;
    }
    syncHandIsolationUI();
  });

  document.getElementById('leftHandBtn').addEventListener('click', function () {
    if (!state.handIsoMaster || !state.leftHandOn) {
      state.handIsoMaster = true;
      state.leftHandOn = true;
      state.rightHandOn = false;
      state.lastIsolatedHand = 'L';
    } else {
      state.handIsoMaster = false;
      state.leftHandOn = false;
      state.rightHandOn = false;
    }
    syncHandIsolationUI();
  });

  document.getElementById('rightHandBtn').addEventListener('click', function () {
    if (!state.handIsoMaster || !state.rightHandOn) {
      state.handIsoMaster = true;
      state.rightHandOn = true;
      state.leftHandOn = false;
      state.lastIsolatedHand = 'R';
    } else {
      state.handIsoMaster = false;
      state.leftHandOn = false;
      state.rightHandOn = false;
    }
    syncHandIsolationUI();
  });

  // Sound Preset Selector & Felt Softness Slider
  document.getElementById('soundCollapseBtn').addEventListener('click', function () {
    const body = document.getElementById('soundSectionBody');
    const chev = document.getElementById('soundChevron');
    const hidden = body.style.display === 'none';
    body.style.display = hidden ? 'flex' : 'none';
    chev.style.transform = hidden ? 'rotate(0deg)' : 'rotate(-90deg)';
  });

  const presetBtns = document.querySelectorAll('.sound-option-btn');
  presetBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      const preset = btn.getAttribute('data-preset');
      audioEngine.setPreset(preset);

      document.getElementById('warmthSlider').value = audioEngine.warmth;
      document.getElementById('warmthLabel').textContent = audioEngine.warmth + '%';

      presetBtns.forEach(function (b) {
        b.classList.toggle('selected', b === btn);
      });

      // Audition warm chord when selecting a piano sound
      [60, 64, 67, 72].forEach(function (m, i) {
        setTimeout(function () { audioEngine.playNote(m, 0.9, 0.72); }, i * 45);
      });
    });
  });

  document.getElementById('warmthSlider').addEventListener('input', function (e) {
    const val = parseInt(e.target.value, 10);
    document.getElementById('warmthLabel').textContent = val + '%';
    audioEngine.setPreset(audioEngine.preset, val);
  });

  // Settings Popover
  const settingsPopover = document.getElementById('settingsPopover');
  const settingsBtn = document.getElementById('settingsToggleBtn');
  settingsBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    settingsPopover.classList.toggle('open');
  });
  window.addEventListener('click', function (e) {
    if (!settingsPopover.contains(e.target) && e.target !== settingsBtn) {
      settingsPopover.classList.remove('open');
    }
  });

  document.querySelectorAll('.horizon-opt').forEach(function (btn) {
    btn.addEventListener('click', function () {
      state.horizonSec = parseFloat(btn.getAttribute('data-sec'));
      document.getElementById('horizonLabel').textContent = Math.round(state.horizonSec) + 's';
      document.querySelectorAll('.horizon-opt').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
    });
  });

  const measureGridSwitch = document.getElementById('measureGridSwitch');
  if (measureGridSwitch) {
    measureGridSwitch.addEventListener('click', function (e) {
      state.showMeasureGrid = !state.showMeasureGrid;
      e.currentTarget.classList.toggle('active-blue', state.showMeasureGrid);
    });
  }

  document.getElementById('allKeyLabelsSwitch').addEventListener('click', function (e) {
    state.showAllKeyNames = !state.showAllKeyNames;
    e.currentTarget.classList.toggle('active-blue', state.showAllKeyNames);
  });

  // Synced A-B Loop Speed Ramp Controls (Popover Switch + Bottom Bar Pill)
  const speedRampSwitch = document.getElementById('speedRampSwitch');
  const loopRampBtn = document.getElementById('loopRampBtn');

  function setSpeedRamp(enabled) {
    state.speedRamp = !!enabled;
    if (speedRampSwitch) {
      speedRampSwitch.classList.toggle('active-green', state.speedRamp);
    }
    if (loopRampBtn) {
      loopRampBtn.classList.toggle('active', state.speedRamp);
    }
    syncLoopUI();
  }

  if (speedRampSwitch) {
    speedRampSwitch.addEventListener('click', function () {
      setSpeedRamp(!state.speedRamp);
    });
  }
  if (loopRampBtn) {
    loopRampBtn.addEventListener('click', function () {
      setSpeedRamp(!state.speedRamp);
    });
  }

  // Export Active Song as Standard .MID File (Format 1 with L/R Hand Tracks)
  const exportMidiBtn = document.getElementById('exportMidiBtn');
  if (exportMidiBtn) {
    exportMidiBtn.addEventListener('click', function () {
      downloadSongMidi(state.song);
    });
  }

  // Delete Custom Song or Reset Modified Built-In Song from Settings Popover
  const deleteCustomSongBtn = document.getElementById('deleteCustomSongBtn');
  if (deleteCustomSongBtn) {
    deleteCustomSongBtn.addEventListener('click', function () {
      deleteOrResetSong(state.activeSongId);
      settingsPopover.classList.remove('open');
    });
  }

  /********************************************************************
   * PHASE 3: AI AUDIO-TO-MIDI TRANSCRIBER MODAL ("BRING YOUR OWN SONG")
   ********************************************************************/
  const transcriberModal = document.getElementById('transcriberModal');
  const closeTranscriberModalBtn = document.getElementById('closeTranscriberModalBtn');
  const cancelTranscribeBtn = document.getElementById('cancelTranscribeBtn');
  const audioDropZone = document.getElementById('audioDropZone');
  const audioFileInput = document.getElementById('audioFileInput');
  const selectedAudioFileName = document.getElementById('selectedAudioFileName');
  const selectedAudioFileMeta = document.getElementById('selectedAudioFileMeta');
  const transcribeModeSelect = document.getElementById('transcribeModeSelect');
  const transcribeSensitivitySlider = document.getElementById('transcribeSensitivitySlider');
  const transcribeSensitivityLabel = document.getElementById('transcribeSensitivityLabel');
  const serverConfigBox = document.getElementById('serverConfigBox');
  const aiServerUrlInput = document.getElementById('aiServerUrlInput');
  const testAiServerBtn = document.getElementById('testAiServerBtn');
  const aiServerStatusPill = document.getElementById('aiServerStatusPill');
  const isolatePianoSwitch = document.getElementById('isolatePianoSwitch');
  const transcribeProgressWrap = document.getElementById('transcribeProgressWrap');
  const transcribeStatusText = document.getElementById('transcribeStatusText');
  const transcribePercentText = document.getElementById('transcribePercentText');
  const transcribeProgressFill = document.getElementById('transcribeProgressFill');
  const transcribeErrorBox = document.getElementById('transcribeErrorBox');
  const startTranscribeBtn = document.getElementById('startTranscribeBtn');

  let selectedAudioFile = null;
  let isolatePianoStem = true;
  let isTranscribing = false;

  try {
    const savedServerUrl = localStorage.getItem('keyflow_ai_server_url');
    if (savedServerUrl && aiServerUrlInput) {
      aiServerUrlInput.value = savedServerUrl;
    }
  } catch (e) {}

  function setTranscriberModalOpen(open) {
    if (!transcriberModal) return;
    transcriberModal.classList.toggle('open', !!open);
    transcriberModal.setAttribute('aria-hidden', open ? 'false' : 'true');
    if (open) {
      setPlaying(false);
      if (audioFileInput) audioFileInput.value = '';
      refreshServerHealthStatus();
    }
  }

  function updateSelectedAudioFile(file) {
    selectedAudioFile = file || null;
    if (transcribeErrorBox) {
      transcribeErrorBox.style.display = 'none';
      transcribeErrorBox.textContent = '';
    }
    if (selectedAudioFile) {
      if (audioDropZone) audioDropZone.classList.add('has-file');
      const sizeMb = (selectedAudioFile.size / (1024 * 1024)).toFixed(1);
      if (selectedAudioFileName) {
        selectedAudioFileName.textContent = 'Selected: ' + selectedAudioFile.name + ' (' + sizeMb + ' MB)';
      }
      if (selectedAudioFileMeta) {
        selectedAudioFileMeta.textContent = 'Ready to convert into Left / Right hand piano notes';
      }
      if (startTranscribeBtn && !isTranscribing) {
        startTranscribeBtn.disabled = false;
      }
    } else {
      if (audioDropZone) audioDropZone.classList.remove('has-file');
      if (selectedAudioFileName) {
        selectedAudioFileName.textContent = 'Tap to choose an Audio File (.MP3, .WAV, .M4A, .OGG)';
      }
      if (selectedAudioFileMeta) {
        selectedAudioFileMeta.textContent = 'Automatically transcribes chords & melody into Left / Right hand falling notes';
      }
      if (startTranscribeBtn) {
        startTranscribeBtn.disabled = true;
      }
    }
  }

  async function refreshServerHealthStatus() {
    if (!audioTranscriber || !aiServerUrlInput || !aiServerStatusPill) return;
    const url = aiServerUrlInput.value.trim();
    try {
      localStorage.setItem('keyflow_ai_server_url', url);
    } catch (e) {}
    aiServerStatusPill.className = 'server-status-pill';
    aiServerStatusPill.textContent = 'Checking...';
    const status = await audioTranscriber.checkServerHealth(url, 1600);
    if (status.ok) {
      aiServerStatusPill.className = 'server-status-pill online';
      const info = status.data || status.info;
      const engines = (info && Array.isArray(info.engines)) ? info.engines.join(' + ') : 'Ready';
      aiServerStatusPill.textContent = 'Online (' + engines + ')';
    } else {
      aiServerStatusPill.className = 'server-status-pill offline';
      aiServerStatusPill.textContent = 'Offline (Browser Engine Ready)';
    }
  }

  if (homeOpenAiModalBtn) {
    homeOpenAiModalBtn.addEventListener('click', function () {
      setTranscriberModalOpen(true);
    });
  }
  if (closeTranscriberModalBtn) {
    closeTranscriberModalBtn.addEventListener('click', function () {
      if (!isTranscribing) setTranscriberModalOpen(false);
    });
  }
  if (cancelTranscribeBtn) {
    cancelTranscribeBtn.addEventListener('click', function () {
      if (!isTranscribing) setTranscriberModalOpen(false);
    });
  }
  if (transcriberModal) {
    transcriberModal.addEventListener('click', function (e) {
      if (e.target === transcriberModal && !isTranscribing) {
        setTranscriberModalOpen(false);
      }
    });
  }

  if (audioFileInput) {
    audioFileInput.addEventListener('change', function (e) {
      const f = e.target.files && e.target.files[0];
      if (f) updateSelectedAudioFile(f);
    });
  }

  if (audioDropZone) {
    audioDropZone.addEventListener('click', function (e) {
      if (isTranscribing) {
        e.preventDefault();
      }
    });
    audioDropZone.addEventListener('dragover', function (e) {
      e.preventDefault();
      audioDropZone.classList.add('dragover');
    });
    audioDropZone.addEventListener('dragleave', function () {
      audioDropZone.classList.remove('dragover');
    });
    audioDropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      audioDropZone.classList.remove('dragover');
      if (isTranscribing) return;
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) updateSelectedAudioFile(f);
    });
  }

  if (transcribeModeSelect && serverConfigBox) {
    transcribeModeSelect.addEventListener('change', function () {
      serverConfigBox.style.display = (transcribeModeSelect.value === 'browser') ? 'none' : 'flex';
    });
  }

  if (transcribeSensitivitySlider && transcribeSensitivityLabel) {
    transcribeSensitivitySlider.addEventListener('input', function () {
      transcribeSensitivityLabel.textContent = transcribeSensitivitySlider.value + '%';
    });
  }

  if (isolatePianoSwitch) {
    isolatePianoSwitch.addEventListener('click', function () {
      isolatePianoStem = !isolatePianoStem;
      isolatePianoSwitch.classList.toggle('active-green', isolatePianoStem);
      isolatePianoSwitch.setAttribute('aria-checked', isolatePianoStem ? 'true' : 'false');
    });
  }

  if (testAiServerBtn) {
    testAiServerBtn.addEventListener('click', function () {
      refreshServerHealthStatus();
    });
  }

  if (startTranscribeBtn) {
    startTranscribeBtn.addEventListener('click', async function () {
      if (!selectedAudioFile || isTranscribing || !audioTranscriber) return;
      isTranscribing = true;
      startTranscribeBtn.disabled = true;
      if (transcribeErrorBox) transcribeErrorBox.style.display = 'none';
      if (transcribeProgressWrap) transcribeProgressWrap.style.display = 'flex';

      const engineMode = transcribeModeSelect ? transcribeModeSelect.value : 'auto';
      const sensitivity = transcribeSensitivitySlider
        ? (parseInt(transcribeSensitivitySlider.value, 10) || 55)
        : 55;
      const serverUrl = aiServerUrlInput ? aiServerUrlInput.value.trim() : 'http://localhost:8000';

      try {
        const songObj = await audioTranscriber.transcribeAudioFile(
          selectedAudioFile,
          {
            mode: engineMode,
            engineMode: engineMode,
            serverUrl: serverUrl,
            isolatePiano: isolatePianoStem,
            sensitivity: sensitivity
          },
          function (prog) {
            if (transcribeStatusText && prog.stage) transcribeStatusText.textContent = prog.stage;
            if (typeof prog.percent === 'number') {
              const p = Math.max(0, Math.min(100, Math.round(prog.percent)));
              if (transcribePercentText) transcribePercentText.textContent = p + '%';
              if (transcribeProgressFill) transcribeProgressFill.style.width = p + '%';
            }
          }
        );

        const id = 'ai_' + Date.now();
        songObj.id = id;
        songObj.isCustom = true;
        songObj.createdAt = Date.now();
        SONGS[id] = songObj;

        if (audioTranscriber.customSongStore) {
          await audioTranscriber.customSongStore.saveSong(songObj);
        }

        isTranscribing = false;
        if (transcribeProgressWrap) transcribeProgressWrap.style.display = 'none';
        startTranscribeBtn.disabled = false;
        setTranscriberModalOpen(false);
        renderHomeSongGrid();
        loadSongById(id);
        setViewMode('play');
      } catch (err) {
        isTranscribing = false;
        startTranscribeBtn.disabled = false;
        if (transcribeErrorBox) {
          transcribeErrorBox.style.display = 'block';
          transcribeErrorBox.textContent = (err && err.message) ? err.message : 'Could not transcribe audio file.';
        }
      }
    });
  }

  // Audio Mute Button
  document.getElementById('muteBtn').addEventListener('click', function () {
    audioEngine.muted = !audioEngine.muted;
    document.getElementById('volOnSvg').style.display = audioEngine.muted ? 'none' : 'block';
    document.getElementById('volOffSvg').style.display = audioEngine.muted ? 'block' : 'none';
  });

  // A-B Loop & Interactive Scrubbing
  const abLoopBtn = document.getElementById('abLoopBtn');
  const abLoopBox = document.getElementById('abLoopBox');
  const handleA = document.getElementById('loopHandleA');
  const handleB = document.getElementById('loopHandleB');
  const timelineTrack = document.getElementById('timelineTrack');

  abLoopBtn.addEventListener('click', function () {
    state.loopEnabled = !state.loopEnabled;
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
    state.waitStartedAtMs = null;
    abLoopBtn.classList.toggle('active', state.loopEnabled);
    abLoopBox.classList.toggle('visible', state.loopEnabled);
    handleA.classList.toggle('visible', state.loopEnabled);
    handleB.classList.toggle('visible', state.loopEnabled);
    syncLoopUI();
  });

  let dragTarget = null;
  function eventToSongSec(e) {
    const rect = timelineTrack.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
    return ratio * state.song.duration;
  }

  handleA.addEventListener('pointerdown', function (e) {
    e.stopPropagation();
    dragTarget = 'A';
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
  });
  handleB.addEventListener('pointerdown', function (e) {
    e.stopPropagation();
    dragTarget = 'B';
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
  });
  timelineTrack.addEventListener('pointerdown', function (e) {
    if (e.target === handleA || e.target === handleB) return;
    dragTarget = 'playhead';
    seekTo(eventToSongSec(e));
  });

  window.addEventListener('pointermove', function (e) {
    if (!dragTarget) return;
    const sec = eventToSongSec(e);
    if (dragTarget === 'playhead') {
      seekTo(sec);
    } else if (dragTarget === 'A') {
      state.loopA = Math.max(0, Math.min(state.loopB - 1.5, sec));
      syncLoopUI();
    } else if (dragTarget === 'B') {
      state.loopB = Math.min(state.song.duration, Math.max(state.loopA + 1.5, sec));
      syncLoopUI();
    }
  });

  window.addEventListener('pointerup', function () {
    dragTarget = null;
  });

  // Responsive Canvas Resize Listeners (with visualViewport & ResizeObserver for iPad Safari/Chrome)
  function handleViewportResize() {
    syncViewportHeight();
    rebuildKeyboardGeometry();
  }

  window.addEventListener('resize', handleViewportResize);
  window.addEventListener('orientationchange', function () {
    setTimeout(handleViewportResize, 80);
  });
  try {
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', handleViewportResize);
    }
  } catch (e) {}
  try {
    if (window.ResizeObserver && stageEl) {
      const ro = new ResizeObserver(function () {
        rebuildKeyboardGeometry();
      });
      ro.observe(stageEl);
    }
  } catch (e) {}

  // Startup Initialization
  syncViewportHeight();
  syncHandIsolationUI();
  loadSongById('let_it_be');
  setViewMode('home');
  if (audioTranscriber && audioTranscriber.customSongStore) {
    audioTranscriber.customSongStore.getAllSongs().then(function (savedList) {
      if (Array.isArray(savedList)) {
        savedList.forEach(function (item) {
          if (item && item.id && Array.isArray(item.notes) && item.notes.length > 0) {
            if (BUILTIN_IDS.indexOf(item.id) !== -1) {
              item.isCustom = false;
              item.isModifiedBuiltin = true;
              item.source = 'builtin';
            } else {
              item.isCustom = true;
            }
            SONGS[item.id] = item;
          }
        });
        if (SONGS[state.activeSongId]) {
          loadSongById(state.activeSongId);
        }
        renderHomeSongGrid();
      }
    }).catch(function () {});
  }
  setTimeout(handleViewportResize, 120);
  requestAnimationFrame(tickLoop);
})(window);
