/* Loaded before each game's scripts. Desktop and embedded game documents are untouched. */
(() => {
  if (window !== window.top) {
    try { window.__mobileTaskShell = window.parent.mobileControls || null; } catch (_) {}
    return;
  }
  const phone = matchMedia('(max-width: 760px)').matches ||
    (matchMedia('(pointer: coarse)').matches && screen.width <= 1024);
  if (!phone) return;
  const entry = new URL(document.currentScript.src);
  const shell = new URL('index.html', entry);
  shell.searchParams.set('game', document.currentScript.dataset.game);
  shell.searchParams.set('query', location.search);
  shell.hash = location.hash;
  window.stop();
  location.replace(shell);
})();
