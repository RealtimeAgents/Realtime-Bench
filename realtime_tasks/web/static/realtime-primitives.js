(() => {
  "use strict";

  function setup(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, rect.width * dpr);
    canvas.height = Math.max(1, rect.height * dpr);
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {context, width: rect.width, height: rect.height};
  }

  function line(canvas, values, {color = "#3478f6", fill = "rgba(52,120,246,.12)", min = 0, max = 100, threshold = null} = {}) {
    const {context: ctx, width, height} = setup(canvas);
    const pad = 12;
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(120,135,160,.2)";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = Math.round(height * i / 4) + .5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }
    if (threshold !== null) {
      const y = height - pad - ((threshold - min) / (max - min)) * (height - pad * 2);
      ctx.setLineDash([5, 4]); ctx.strokeStyle = "#d64a4a";
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); ctx.setLineDash([]);
    }
    if (!values || values.length < 2) return;
    const points = values.map((value, index) => ({
      x: pad + index * (width - pad * 2) / (values.length - 1),
      y: height - pad - ((Math.max(min, Math.min(max, value)) - min) / (max - min)) * (height - pad * 2),
    }));
    ctx.beginPath(); ctx.moveTo(points[0].x, height - pad);
    points.forEach(point => ctx.lineTo(point.x, point.y));
    ctx.lineTo(points.at(-1).x, height - pad); ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    ctx.beginPath(); points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
    const last = points.at(-1); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(last.x, last.y, 4, 0, Math.PI * 2); ctx.fill();
  }

  function scatter(canvas, groups, {min = -1, max = 1, limit = .25} = {}) {
    const {context: ctx, width, height} = setup(canvas);
    const pad = 15;
    ctx.clearRect(0, 0, width, height);
    const toY = value => height - pad - ((value - min) / (max - min)) * (height - pad * 2);
    ctx.fillStyle = "rgba(34,197,94,.08)"; ctx.fillRect(0, toY(limit), width, toY(-limit) - toY(limit));
    for (const value of [-limit, 0, limit]) {
      ctx.setLineDash(value ? [5, 4] : []); ctx.strokeStyle = value ? "#d67b35" : "#9aa5b5";
      ctx.beginPath(); ctx.moveTo(0, toY(value)); ctx.lineTo(width, toY(value)); ctx.stroke();
    }
    ctx.setLineDash([]);
    groups.forEach((group, groupIndex) => group.values.forEach((value, index) => {
      const x = pad + (index + groupIndex * .18) * (width - pad * 2) / Math.max(1, group.values.length - 1);
      ctx.fillStyle = group.color; ctx.beginPath(); ctx.arc(x, toY(value), 4, 0, Math.PI * 2); ctx.fill();
    }));
  }

  function heat(value, min = 20, max = 42) {
    const ratio = Math.max(0, Math.min(1, (value - min) / (max - min)));
    const hue = 205 - ratio * 205;
    return `hsl(${hue} 76% ${62 - ratio * 12}%)`;
  }

  const escape = value => String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;",
  })[character]);

  function facts(record) {
    return `<div class="rt-facts">${record.facts.map(fact => `<div class="rt-fact"><span>${escape(fact.label)}</span><b>${escape(fact.value)}</b></div>`).join("")}</div>`;
  }

  function action(record, label, extraClass = "") {
    return `<button class="rt-action ${extraClass}" data-act="${escape(record.ref)}">${escape(label)}</button>`;
  }

  window.RTCharts = {line, scatter, heat};
  window.RTUI = {escape, facts, action};
})();
