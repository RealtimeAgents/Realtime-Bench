/* A presentation/input shell: game state, timers, random seeds and scoring stay in the game. */
(async () => {
  'use strict';
  const frame = document.getElementById('game');
  const viewport = document.getElementById('viewport');
  const stage = document.getElementById('stage');
  const controls = document.getElementById('controls');
  const params = new URLSearchParams(location.search);
  // Each task keeps its controls in environment/mobile-profile.json, next to this shell's folder.
  const profile = await fetch('../mobile-profile.json').then(response => response.ok ? response.json() : null);
  const game = params.get('game');
  if (!profile) {
    document.getElementById('title').textContent = 'Unknown game';
    return;
  }
  document.getElementById('title').textContent = game.replace(/^\d+_/, '').replaceAll('-', ' ');
  let zoomed = false;
  let secondaryClick = false;
  const panButton = document.getElementById('pan');
  const panSurface = document.getElementById('pan-surface');
  let view = {left: 0, top: 0, width: 1920, height: 1080};
  function resize() {
    // Keep the benchmark's 1920 x 1080 logical viewport, including inside nested frames.
    const fit = Math.min(viewport.clientWidth / view.width, innerHeight * 0.65 / view.height);
    const scale = zoomed ? Math.max(fit, 0.85) : fit;
    frame.style.transform = `scale(${scale}) translate(${-view.left}px, ${-view.top}px)`;
    stage.style.width = `${view.width * scale}px`;
    stage.style.height = `${view.height * scale}px`;
  }
  document.getElementById('zoom').onclick = event => {
    zoomed = !zoomed;
    panButton.disabled = !zoomed;
    panButton.setAttribute('aria-pressed', 'false');
    panSurface.hidden = true;
    event.currentTarget.textContent = zoomed ? 'Fit screen' : 'Zoom in';
    resize();
    viewport.scrollLeft = (stage.clientWidth - viewport.clientWidth) / 2;
    viewport.scrollTop = (stage.clientHeight - viewport.clientHeight) / 2;
  };
  panButton.onclick = () => {
    panSurface.hidden = !panSurface.hidden;
    panButton.setAttribute('aria-pressed', String(!panSurface.hidden));
  };
  let panStart = null;
  panSurface.addEventListener('pointerdown', event => {
    event.preventDefault(); panSurface.setPointerCapture(event.pointerId);
    panStart = {x: event.clientX, y: event.clientY, left: viewport.scrollLeft, top: viewport.scrollTop};
  });
  panSurface.addEventListener('pointermove', event => {
    if (!panStart) return;
    viewport.scrollLeft = panStart.left + panStart.x - event.clientX;
    viewport.scrollTop = panStart.top + panStart.y - event.clientY;
  });
  for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    panSurface.addEventListener(type, () => { panStart = null; });
  }
  new ResizeObserver(resize).observe(viewport);

  function activeDocument() {
    let doc = frame.contentDocument;
    while (doc) {
      const nested = doc.activeElement?.tagName === 'IFRAME' ? doc.activeElement : doc.querySelector('iframe');
      if (!nested?.contentDocument) break;
      doc = nested.contentDocument;
    }
    return doc;
  }
  const held = new Map();
  const inputObservers = new Set();
  // observeInput() listeners see the trusted physical press behind a virtual key, including
  // repeats while that pointer remains held. Synthetic page input is never reported.
  window.mobileControls = {
    observeInput(listener) {
      inputObservers.add(listener);
      return () => inputObservers.delete(listener);
    },
  };
  function noteInput(event) {
    if (event.isTrusted) inputObservers.forEach(listener => listener(event));
  }
  function keyInfo(name) {
    const special = {
      Space: [' ', 'Space', 32], Enter: ['Enter', 'Enter', 13], Escape: ['Escape', 'Escape', 27],
      Backspace: ['Backspace', 'Backspace', 8], Delete: ['Delete', 'Delete', 46],
      Shift: ['Shift', 'ShiftLeft', 16], Ctrl: ['Control', 'ControlLeft', 17],
      ArrowLeft: ['ArrowLeft', 'ArrowLeft', 37], ArrowUp: ['ArrowUp', 'ArrowUp', 38],
      ArrowRight: ['ArrowRight', 'ArrowRight', 39], ArrowDown: ['ArrowDown', 'ArrowDown', 40],
    };
    return special[name] || [name.toLowerCase(), /^\d$/.test(name) ? `Digit${name}` : `Key${name.toUpperCase()}`, name.toUpperCase().charCodeAt(0)];
  }
  function sendKey(doc, name, type, repeat = false) {
    if (!doc) return;
    const [key, code, keyCode] = keyInfo(name);
    const target = doc.activeElement || doc.body;
    const options = {key, code, keyCode, which: keyCode, bubbles: true, cancelable: true,
      repeat, shiftKey: held.has('Shift'), ctrlKey: held.has('Ctrl')};
    const accepted = target.dispatchEvent(new doc.defaultView.KeyboardEvent(type, options));
    if (accepted && type === 'keydown' && (key.length === 1 || key === 'Enter')) {
      const character = key === 'Enter' ? 13 : key.charCodeAt(0);
      target.dispatchEvent(new doc.defaultView.KeyboardEvent('keypress', {
        ...options, keyCode: character, which: character, charCode: character,
      }));
    }
  }
  function release(name) {
    const press = held.get(name);
    if (!press) return;
    clearTimeout(press.delay); clearInterval(press.repeat);
    held.delete(name);
    sendKey(press.doc, name, 'keyup');
    press.button.classList.remove('held');
  }
  function releaseAll() { [...held.keys()].forEach(release); }
  const labels = {ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓', Space: 'Space'};
  for (const name of [...new Set(profile.keys)]) {
    const button = document.createElement('button');
    button.type = 'button'; button.textContent = labels[name] || name;
    button.setAttribute('aria-label', `Hold ${name}`);
    button.addEventListener('pointerdown', event => {
      if (held.has(name)) return;
      event.preventDefault(); button.setPointerCapture(event.pointerId);
      const press = {doc: activeDocument(), button};
      held.set(name, press); button.classList.add('held');
      sendKey(press.doc, name, 'keydown');
      noteInput(event);
      press.delay = setTimeout(() => {
        press.repeat = setInterval(() => {
          sendKey(press.doc, name, 'keydown', true);
          noteInput(event);
        }, 60);
      }, 350);
    });
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      button.addEventListener(type, () => release(name));
    }
    controls.append(button);
  }
  if (profile.secondaryClick) {
    const button = document.createElement('button');
    button.type = 'button'; button.textContent = 'Right click';
    button.setAttribute('aria-pressed', 'false');
    button.onclick = () => {
      secondaryClick = !secondaryClick;
      button.setAttribute('aria-pressed', String(secondaryClick));
      button.classList.toggle('held', secondaryClick);
    };
    controls.append(button);
  }
  window.addEventListener('blur', releaseAll);
  document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });
  window.addEventListener('pagehide', releaseAll);

  function fitGame() {
    const selector = profile.view;
    if (!selector) return;
    const doc = activeDocument();
    const rects = [...(doc?.querySelectorAll(selector) || [])].map(node => node.getBoundingClientRect())
      .filter(rect => rect.width > 0 && rect.height > 0);
    if (!rects.length) return;
    const left = Math.max(0, Math.min(...rects.map(rect => rect.left)) - 8);
    const top = Math.max(0, Math.min(...rects.map(rect => rect.top)) - 8);
    view = {left, top,
      width: Math.min(1920, Math.max(...rects.map(rect => rect.right)) + 8) - left,
      height: Math.min(1080, Math.max(...rects.map(rect => rect.bottom)) + 8) - top};
    resize();
  }
  const installed = new WeakSet();
  // Touch-to-mouse conversion is restricted to canvas and draggable game surfaces. Form fields
  // and ordinary links keep the browser's native tap, focus, keyboard and scrolling behavior.
  function bindDocument(doc) {
    if (!doc || installed.has(doc)) return;
    installed.add(doc);
    const win = doc.defaultView;
    let gesture = null;
    function mouse(target, type, touch, buttons, button = 0) {
      target.dispatchEvent(new win.MouseEvent(type, {
        bubbles: true, cancelable: true, view: win, clientX: touch.clientX, clientY: touch.clientY,
        screenX: touch.screenX, screenY: touch.screenY, button, buttons,
      }));
    }
    for (const type of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel']) {
      doc.addEventListener(type, event => {
        if (event.pointerType === 'mouse' || !(profile.mouseSurface || secondaryClick)) return;
        if (!event.target.closest('canvas')) return;
        // Avoid also delivering a primary native pointer action before a translated mouse action.
        event.preventDefault(); event.stopImmediatePropagation();
      }, {capture: true, passive: false});
    }
    doc.addEventListener('touchstart', event => {
      if (gesture || event.touches.length !== 1) return;
      const selector = (profile.mouseSurface || secondaryClick) ? 'canvas, [draggable="true"]' : '[draggable="true"]';
      const target = event.target.closest(selector);
      if (!target) return;
      event.preventDefault(); event.stopImmediatePropagation();
      const touch = event.changedTouches[0];
      gesture = {target, id: touch.identifier, x: touch.clientX, y: touch.clientY, last: touch, button: secondaryClick ? 2 : 0};
      if (target.draggable) {
        gesture.transfer = new win.DataTransfer();
        target.dispatchEvent(new win.DragEvent('dragstart', {bubbles: true, cancelable: true, dataTransfer: gesture.transfer}));
      } else {
        mouse(target, 'mousemove', touch, 0, gesture.button);
        mouse(target, 'mousedown', touch, gesture.button === 2 ? 2 : 1, gesture.button);
      }
    }, {capture: true, passive: false});
    doc.addEventListener('touchmove', event => {
      if (!gesture) return;
      const touch = [...event.changedTouches].find(t => t.identifier === gesture.id);
      if (!touch) return;
      event.preventDefault(); event.stopImmediatePropagation(); gesture.last = touch;
      if (gesture.transfer) {
        const target = doc.elementFromPoint(touch.clientX, touch.clientY);
        target?.dispatchEvent(new win.DragEvent('dragover', {bubbles: true, cancelable: true,
          clientX: touch.clientX, clientY: touch.clientY, dataTransfer: gesture.transfer}));
      } else mouse(gesture.target, 'mousemove', touch, gesture.button === 2 ? 2 : 1, gesture.button);
    }, {capture: true, passive: false});
    function end(event) {
      if (!gesture) return;
      const touch = [...event.changedTouches].find(t => t.identifier === gesture.id);
      if (!touch) return;
      event.preventDefault(); event.stopImmediatePropagation();
      const {target, transfer, x, y, button} = gesture;
      gesture = null;
      if (transfer) {
        if (event.type !== 'touchcancel') doc.elementFromPoint(touch.clientX, touch.clientY)?.dispatchEvent(
          new win.DragEvent('drop', {bubbles: true, cancelable: true, clientX: touch.clientX,
            clientY: touch.clientY, dataTransfer: transfer}));
        target.dispatchEvent(new win.DragEvent('dragend', {bubbles: true, dataTransfer: transfer}));
      } else {
        mouse(target, 'mouseup', touch, 0, button);
        if (event.type !== 'touchcancel' && Math.hypot(touch.clientX - x, touch.clientY - y) < 12) mouse(target, button === 2 ? 'contextmenu' : 'click', touch, 0, button);
      }
    }
    doc.addEventListener('touchend', end, {capture: true, passive: false});
    doc.addEventListener('touchcancel', end, {capture: true, passive: false});
    const frames = new WeakSet();
    function scan() {
      doc.querySelectorAll('iframe').forEach(child => {
        if (!frames.has(child)) {
          frames.add(child);
          child.addEventListener('load', () => { releaseAll(); bindDocument(child.contentDocument); });
        }
        bindDocument(child.contentDocument);
      });
    }
    new win.MutationObserver(scan).observe(doc, {childList: true, subtree: true});
    scan();
    fitGame();
    win.setTimeout(fitGame, 2000);
  }
  frame.addEventListener('load', () => { releaseAll(); bindDocument(frame.contentDocument); });
  // Expose the embedded game's gameAPI through the shell so state reads still reach it.
  window.gameAPI = new Proxy({}, {get(_target, property) {
    const api = frame.contentWindow?.gameAPI;
    const value = api?.[property];
    return typeof value === 'function' ? value.bind(api) : value;
  }});
  const url = new URL('../index.html', location.href);
  url.search = params.get('query') || '';
  url.hash = location.hash;
  frame.src = url;
  resize();
})();
