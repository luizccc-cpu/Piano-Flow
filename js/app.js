(function (global) {
  'use strict';

  const SONGS = global.SONGS;
  const midiToName = global.midiToName;
  const isBlackMidi = global.isBlackMidi;
  const formatTime = global.formatTime;
  const parseStandardMidiFile = global.parseStandardMidiFile;
  const calculateAutoRange = global.calculateAutoRange;
  const assignHandsSmart = global.assignHandsSmart;
  const audioEngine = global.audioEngine;
  const audioTranscriber = global.audioTranscriber;

  /********************************************************************
   * APPLICATION STATE
   ********************************************************************/
  const state = {
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

      ctx.shadowBlur = 0;
      ctx.lineWidth = isWaiting ? 2.2 : 1.2;
      ctx.strokeStyle = isWaiting
        ? '#ffffff'
        : (n.hand === 'L' ? 'rgba(186, 230, 253, 0.85)' : 'rgba(187, 247, 208, 0.85)');
      ctx.stroke();

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
    if (state.waitMode && state.waitingMidis.size > 0) {
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

  const pointerMidis = {};
  canvas.addEventListener('pointerdown', function (e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const hit = hitTestPiano(x, y);
    if (hit !== null) {
      try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
      pointerMidis[e.pointerId] = hit;
      userNoteOn(hit, 0.85);
    } else if (y < strikeY) {
      setPlaying(!state.isPlaying);
    }
  });

  canvas.addEventListener('pointermove', function (e) {
    if (pointerMidis[e.pointerId] === undefined) return;
    const rect = canvas.getBoundingClientRect();
    const hit = hitTestPiano(e.clientX - rect.left, e.clientY - rect.top);
    const prev = pointerMidis[e.pointerId];
    if (hit !== null && hit !== prev) {
      userNoteOff(prev);
      pointerMidis[e.pointerId] = hit;
      userNoteOn(hit, 0.76);
    }
  });

  function releasePointer(e) {
    if (pointerMidis[e.pointerId] !== undefined) {
      userNoteOff(pointerMidis[e.pointerId]);
      delete pointerMidis[e.pointerId];
    }
  }
  canvas.addEventListener('pointerup', releasePointer);
  canvas.addEventListener('pointercancel', releasePointer);

  // Computer Keyboard Support
  const KEY_OFFSETS = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12, o: 13, l: 14 };
  const heldKeys = {};
  window.addEventListener('keydown', function (e) {
    if (e.target && (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT')) return;
    if (e.code === 'Space') {
      e.preventDefault();
      setPlaying(!state.isPlaying);
      return;
    }
    const k = String(e.key || '').toLowerCase();
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
   * UI BINDINGS & INTERACTIVE CONTROLS
   ********************************************************************/
  function registerCustomSongInSelect(songObj) {
    if (!songObj || !songObj.id) return;
    SONGS[songObj.id] = songObj;
    const sel = document.getElementById('songSelect');
    if (!sel) return;
    let existingOpt = null;
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === songObj.id) {
        existingOpt = sel.options[i];
        break;
      }
    }
    const labelPrefix = (songObj.source && songObj.source.indexOf('ai') === 0) ? '★ [AI] ' : '★ ';
    const labelText = songObj.title.indexOf('★') === 0 ? songObj.title : (labelPrefix + songObj.title);
    if (existingOpt) {
      existingOpt.textContent = labelText;
    } else {
      const opt = document.createElement('option');
      opt.value = songObj.id;
      opt.textContent = labelText;
      sel.appendChild(opt);
    }
  }

  function loadSongById(id) {
    const s = SONGS[id];
    if (!s) return;
    setPlaying(false);
    state.activeSongId = id;
    state.song = s;
    state.currentTime = 0;
    state.loopA = Math.min(2.0, s.duration * 0.12);
    state.loopB = Math.min(s.duration * 0.48, s.duration - 1.0);
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
    state.waitStartedAtMs = null;
    document.getElementById('songTitleHeader').textContent = s.title;

    const sel = document.getElementById('songSelect');
    if (sel && sel.value !== id) {
      sel.value = id;
    }

    const delBtn = document.getElementById('deleteCustomSongBtn');
    if (delBtn) {
      delBtn.style.display = s.isCustom ? 'inline-flex' : 'none';
    }

    resetNoteTriggersBefore(0);
    calculateAutoRange(state);
    rebuildKeyboardGeometry();
    syncTimelineUI();
    syncLoopUI();
  }

  document.getElementById('songSelect').addEventListener('change', function (e) {
    loadSongById(e.target.value);
  });

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
        registerCustomSongInSelect(parsed);
        loadSongById(id);
        if (audioTranscriber && audioTranscriber.customSongStore) {
          audioTranscriber.customSongStore.saveSong(parsed);
        }
      } catch (err) {
        console.warn('MIDI parse error:', err);
      }
    };
    reader.readAsArrayBuffer(file);
  });

  // Play / Pause & Restart
  document.getElementById('playPauseBtn').addEventListener('click', function () {
    setPlaying(!state.isPlaying);
  });
  document.getElementById('restartBtn').addEventListener('click', function () {
    seekTo(state.loopEnabled ? state.loopA : 0);
  });

  // Speed Selector & Collapsible HUD Panel
  const hudPanel = document.getElementById('hudPanel');
  const hudToggleBtn = document.getElementById('hudToggleBtn');
  const hudMiniSummary = document.getElementById('hudMiniSummary');

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
      document.getElementById('horizonLabel').textContent = state.horizonSec.toFixed(1) + 's';
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

  // Manual Smart Hand Re-split Button inside Settings Popover
  const resplitHandsBtn = document.getElementById('resplitHandsBtn');
  if (resplitHandsBtn) {
    resplitHandsBtn.addEventListener('click', function () {
      if (state.song && Array.isArray(state.song.notes)) {
        assignHandsSmart(state.song.notes);
        if (state.song.isCustom && audioTranscriber && audioTranscriber.customSongStore) {
          audioTranscriber.customSongStore.saveSong(state.song);
        }
        const span = resplitHandsBtn.querySelector('span') || resplitHandsBtn;
        const origText = 'Re-split Hands (Smart AI)';
        span.textContent = '✓ Hands Re-split (Smart AI)';
        setTimeout(function () {
          span.textContent = origText;
        }, 1400);
      }
    });
  }

  // Export Active Song as Standard .MID File (Format 1 with L/R Hand Tracks)
  const exportMidiBtn = document.getElementById('exportMidiBtn');
  if (exportMidiBtn) {
    exportMidiBtn.addEventListener('click', function () {
      if (!state.song || !audioTranscriber) return;
      try {
        const blob = audioTranscriber.exportSongToMidiBlob(state.song);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const safeName = String(state.song.title || 'KeyFlow_Song')
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
    });
  }

  // Delete Custom / Transcribed Song from Persistent Library
  const deleteCustomSongBtn = document.getElementById('deleteCustomSongBtn');
  if (deleteCustomSongBtn) {
    deleteCustomSongBtn.addEventListener('click', function () {
      const curId = state.activeSongId;
      const curSong = SONGS[curId];
      if (!curSong || !curSong.isCustom) return;
      if (audioTranscriber && audioTranscriber.customSongStore) {
        audioTranscriber.customSongStore.deleteSong(curId);
      }
      delete SONGS[curId];
      const sel = document.getElementById('songSelect');
      if (sel) {
        for (let i = sel.options.length - 1; i >= 0; i--) {
          if (sel.options[i].value === curId) {
            sel.remove(i);
          }
        }
      }
      loadSongById('let_it_be');
      settingsPopover.classList.remove('open');
    });
  }

  /********************************************************************
   * PHASE 3: AI AUDIO-TO-MIDI TRANSCRIBER MODAL ("BRING YOUR OWN SONG")
   ********************************************************************/
  const transcriberModal = document.getElementById('transcriberModal');
  const openTranscriberModalBtn = document.getElementById('openTranscriberModalBtn');
  const closeTranscriberModalBtn = document.getElementById('closeTranscriberModalBtn');
  const cancelTranscribeBtn = document.getElementById('cancelTranscribeBtn');
  const audioDropzone = document.getElementById('audioDropzone');
  const audioFileInput = document.getElementById('audioFileInput');
  const dropzonePrimaryText = document.getElementById('dropzonePrimaryText');
  const transcribeEngineSelect = document.getElementById('transcribeEngineSelect');
  const transcribeSensitivitySlider = document.getElementById('transcribeSensitivitySlider');
  const sensitivityValueLabel = document.getElementById('sensitivityValueLabel');
  const serverConfigBox = document.getElementById('serverConfigBox');
  const aiServerUrlInput = document.getElementById('aiServerUrlInput');
  const testAiServerBtn = document.getElementById('testAiServerBtn');
  const serverStatusPill = document.getElementById('serverStatusPill');
  const isolatePianoSwitch = document.getElementById('isolatePianoSwitch');
  const transcribeProgressWrap = document.getElementById('transcribeProgressWrap');
  const transcribeStageText = document.getElementById('transcribeStageText');
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
      if (audioDropzone) audioDropzone.classList.add('has-file');
      const sizeMb = (selectedAudioFile.size / (1024 * 1024)).toFixed(1);
      if (dropzonePrimaryText) {
        dropzonePrimaryText.innerHTML = 'Selected: <strong>' + selectedAudioFile.name + '</strong> (' + sizeMb + ' MB)';
      }
      if (startTranscribeBtn && !isTranscribing) {
        startTranscribeBtn.disabled = false;
      }
    } else {
      if (audioDropzone) audioDropzone.classList.remove('has-file');
      if (dropzonePrimaryText) {
        dropzonePrimaryText.innerHTML = 'Tap to choose an <strong>.MP3, .WAV, .M4A, or .OGG</strong> song';
      }
      if (startTranscribeBtn) {
        startTranscribeBtn.disabled = true;
      }
    }
  }

  async function refreshServerHealthStatus() {
    if (!audioTranscriber || !aiServerUrlInput || !serverStatusPill) return;
    const url = aiServerUrlInput.value.trim();
    try {
      localStorage.setItem('keyflow_ai_server_url', url);
    } catch (e) {}
    serverStatusPill.className = 'server-status-pill';
    serverStatusPill.textContent = 'Checking...';
    const status = await audioTranscriber.checkServerHealth(url, 1600);
    if (status.ok) {
      serverStatusPill.className = 'server-status-pill online';
      const engines = (status.info && Array.isArray(status.info.engines)) ? status.info.engines.join(' + ') : 'Ready';
      serverStatusPill.textContent = 'Online (' + engines + ')';
    } else {
      serverStatusPill.className = 'server-status-pill offline';
      serverStatusPill.textContent = 'Offline (Browser Engine Ready)';
    }
  }

  if (openTranscriberModalBtn) {
    openTranscriberModalBtn.addEventListener('click', function () {
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

  if (audioDropzone && audioFileInput) {
    audioDropzone.addEventListener('click', function () {
      if (!isTranscribing) audioFileInput.click();
    });
    audioDropzone.addEventListener('dragover', function (e) {
      e.preventDefault();
      audioDropzone.classList.add('dragover');
    });
    audioDropzone.addEventListener('dragleave', function () {
      audioDropzone.classList.remove('dragover');
    });
    audioDropzone.addEventListener('drop', function (e) {
      e.preventDefault();
      audioDropzone.classList.remove('dragover');
      if (isTranscribing) return;
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) updateSelectedAudioFile(f);
    });
    audioFileInput.addEventListener('change', function (e) {
      const f = e.target.files && e.target.files[0];
      if (f) updateSelectedAudioFile(f);
    });
  }

  if (transcribeEngineSelect && serverConfigBox) {
    transcribeEngineSelect.addEventListener('change', function () {
      serverConfigBox.style.display = (transcribeEngineSelect.value === 'browser') ? 'none' : 'flex';
    });
  }

  if (transcribeSensitivitySlider && sensitivityValueLabel) {
    transcribeSensitivitySlider.addEventListener('input', function () {
      sensitivityValueLabel.textContent = transcribeSensitivitySlider.value + '%';
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

      const engineMode = transcribeEngineSelect ? transcribeEngineSelect.value : 'auto';
      const sensitivity = transcribeSensitivitySlider
        ? (parseInt(transcribeSensitivitySlider.value, 10) / 100)
        : 0.55;
      const serverUrl = aiServerUrlInput ? aiServerUrlInput.value.trim() : 'http://localhost:8000';

      try {
        const songObj = await audioTranscriber.transcribeAudioFile(
          selectedAudioFile,
          {
            engineMode: engineMode,
            serverUrl: serverUrl,
            isolatePiano: isolatePianoStem,
            sensitivity: sensitivity
          },
          function (prog) {
            if (transcribeStageText && prog.stage) transcribeStageText.textContent = prog.stage;
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

        registerCustomSongInSelect(songObj);
        loadSongById(id);

        if (audioTranscriber.customSongStore) {
          await audioTranscriber.customSongStore.saveSong(songObj);
        }

        isTranscribing = false;
        if (transcribeProgressWrap) transcribeProgressWrap.style.display = 'none';
        startTranscribeBtn.disabled = false;
        setTranscriberModalOpen(false);
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
  if (audioTranscriber && audioTranscriber.customSongStore) {
    audioTranscriber.customSongStore.getAllSongs().then(function (savedList) {
      if (Array.isArray(savedList)) {
        savedList.forEach(function (item) {
          if (item && item.id && Array.isArray(item.notes) && item.notes.length > 0) {
            item.isCustom = true;
            registerCustomSongInSelect(item);
          }
        });
      }
    }).catch(function () {});
  }
  setTimeout(handleViewportResize, 120);
  requestAnimationFrame(tickLoop);
})(window);
