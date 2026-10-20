window.OPEN_SOURCE_GAME_CONFIG = (function () {
  // Player returns, derived read-only from the live ball. The ball's horizontal velocity flips from
  // negative to positive only when it rebounds off the left paddle (pong.js:525-534); a goal resets
  // the ball without a sign flip (pong.js:476-480). Sampled by the wrapper's 250 ms poll (signal)
  // and by every getState (state/terminal); a court crossing takes 8 s, so no flip is missed.
  // A reloaded iframe produces a new pong object, which restarts the count.
  let tracked = { pong: null, lastDx: null, returns: 0 };

  function trackReturns(pong) {
    if (!pong || !pong.ball) return 0;
    if (tracked.pong !== pong) tracked = { pong: pong, lastDx: null, returns: 0 };
    const dx = pong.ball.dx;
    if (typeof dx === "number") {
      if (tracked.lastDx !== null && tracked.lastDx < 0 && dx > 0) tracked.returns += 1;
      tracked.lastDx = dx;
    }
    return tracked.returns;
  }

  return {
    title: "JavaScript Pong",
    entry: "play.html",
    repository: "https://github.com/jakesgordon/javascript-pong",
    commit: "ca3240536e4f79ab7144388e56ed19de715b6662",
    license: "MIT",
    deferStart: true,

    start(c) {
      const pong = c.window()?.pong;
      if (pong && !pong.playing) {
        pong.startSinglePlayer();
        pong.runner.loop();
      }
    },

    signal(c) {
      const pong = c.window()?.pong;
      if (!pong) return null;
      return {
        playing: pong.playing,
        scores: pong.scores,
        returns: trackReturns(pong),
        ball: pong.ball ? [pong.ball.x, pong.ball.y, pong.ball.dx, pong.ball.dy] : null,
        paddle: pong.leftPaddle ? pong.leftPaddle.y : null
      };
    },

    state(c) {
      const pong = c.window()?.pong;
      if (!pong) return {};
      return {
        player_returns: trackReturns(pong),
        player_score: Number(pong.scores?.[0] || 0),
        opponent_score: Number(pong.scores?.[1] || 0),
        match_active: Boolean(pong.playing)
      };
    },

    terminal(c) {
      const pong = c.window()?.pong;
      if (!pong) return null;
      trackReturns(pong);
      if ((pong.scores?.[1] || 0) >= 3) return "opponent_reached_three";
      return null;
    }
  };
}());
