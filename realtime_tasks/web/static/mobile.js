/* Keep fixed dialogs inside the visible area when the keyboard or browser chrome moves. */
(() => {
  'use strict';
  const mobileLayout = matchMedia('(max-width: 900px)');
  const viewport = window.visualViewport;
  function update() {
    const style = document.documentElement.style;
    if (!mobileLayout.matches) {
      style.removeProperty('--mobile-viewport-height');
      style.removeProperty('--mobile-viewport-top');
      return;
    }
    style.setProperty('--mobile-viewport-height', `${viewport?.height || innerHeight}px`);
    style.setProperty('--mobile-viewport-top', `${viewport?.offsetTop || 0}px`);
  }
  mobileLayout.addEventListener('change', update);
  window.addEventListener('resize', update);
  viewport?.addEventListener('resize', update);
  viewport?.addEventListener('scroll', update);
  update();
})();
