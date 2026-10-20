from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import posixpath
import re
import signal
import subprocess
import sys
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, parse, request

from realtime_framework.entity.config import DEFAULT_TASK_SEED, task_seed

ROOT = Path(os.environ.get("TASK_PAGES_ROOT", Path(__file__).resolve().parents[1])).resolve()
HOST = os.environ.get("TASK_PAGES_HOST", "127.0.0.1")
PORT = int(os.environ.get("TASK_PAGES_PORT", "18080"))
RT_PORT = int(os.environ.get("RT_SUITE_PORT", "5091"))
RT_PUBLIC_BASE = os.environ.get("RT_PUBLIC_BASE", "").rstrip("/")
SEED = DEFAULT_TASK_SEED

GAME_ROOT = (ROOT / "realtime_tasks/game").resolve()
WEB_CATALOG = ROOT / "realtime_tasks/web"
GAME_CATALOG = ROOT / "realtime_tasks/game"


def scalar(path: Path, key: str) -> str | None:
    value = tomllib.loads(path.read_text(encoding="utf-8")).get("metadata", {}).get(key)
    return None if value is None or value == "" else str(value)


def load_games() -> list[dict[str, str]]:
    rows = []
    for path in sorted(GAME_CATALOG.glob("*/task.toml")):
        task_id = scalar(path, "task_id")
        game_key = scalar(path, "game_key")
        suffix = scalar(path, "url_suffix") or ""
        if not task_id:
            continue
        if game_key:
            rel = f"/realtime_tasks/game/{task_id}/environment/index.html"
        else:
            continue
        if (ROOT / rel.lstrip("/")).is_file():
            rows.append(
                {
                    "task_id": task_id,
                    "path": rel + suffix,
                    "score_field": scalar(path, "score_field") or "",
                    "target": scalar(path, "target") or "",
                }
            )
    return rows


def load_web() -> list[dict[str, str]]:
    rows = []
    for path in sorted(WEB_CATALOG.glob("*/task.toml")):
        if path.name.startswith("_"):
            continue
        task_id = scalar(path, "task_id")
        scenario = scalar(path, "scenario")
        if task_id and scenario:
            rows.append({"task_id": task_id, "scenario": scenario})
    return rows


GAMES = load_games()
WEB = load_web()
WEB_BY_ID = {row["task_id"]: row for row in WEB}
ALLOWED_ROOTS = tuple(path.resolve() for path in GAME_ROOT.glob("*/environment")) + (
    (ROOT / "realtime_tasks/game/assets").resolve(),
)


def realtime_is_ready() -> bool:
    try:
        with request.urlopen(f"http://127.0.0.1:{RT_PORT}/health", timeout=2) as response:
            payload = json.load(response)
        return payload.get("service") == "realtime-suite" and payload.get("scenarios") == len(WEB)
    except (OSError, ValueError):
        return False


def stop_local_realtime(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def start_local_realtime() -> subprocess.Popen | None:
    if realtime_is_ready():
        print(f"Reusing realtime task backend on port {RT_PORT}", flush=True)
        return None
    log_path = ROOT / "temp" / "task_pages_realtime" / f"realtime-suite-{RT_PORT}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "RT_SUITE_HOST": "127.0.0.1", "RT_SUITE_PORT": str(RT_PORT)}
    print(f"Starting realtime task backend locally on port {RT_PORT}", flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "realtime_tasks.web.app"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            if realtime_is_ready():
                return process
            time.sleep(0.1)
        raise RuntimeError(f"Realtime backend did not become ready. See {log_path}")
    except BaseException:
        stop_local_realtime(process)
        raise


def item_list(rows: list[dict[str, str]], category: str) -> str:
    items = []
    for row in rows:
        task_id = row["task_id"]
        if category == "web":
            href = "/open/web/" + parse.quote(task_id, safe="")
            detail = f"scenario={row['scenario']}"
        else:
            href = row["path"]
            detail = ""
        extra = ""
        if category == "game":
            board = "/game-score-monitor?" + parse.urlencode(
                {
                    "url": row["path"],
                    "task": task_id,
                    "score_field": row.get("score_field", ""),
                    "target": row.get("target", ""),
                }
            )
            extra = (
                f' <a class="sub" href="{html.escape(board)}" target="_blank" rel="noopener" '
                f'title="Open a monitor window that polls this game\'s gameAPI.getState()">score monitor</a>'
            )
        items.append(
            f'<li><a href="{html.escape(href)}" target="_blank" rel="noopener">'
            f"{html.escape(task_id)}</a>"
            + (f'<span class="sub">{html.escape(detail)}</span>' if detail else "")
            + extra
            + "</li>"
        )
    return "".join(items)


def hub_html() -> bytes:
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task Pages Hub</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1120px;margin:0 auto;padding:24px;line-height:1.45}}
h1{{margin:0 0 8px}}p{{margin:0 0 18px;color:#444}}
section{{border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:16px 0}}
h2{{margin:0 0 10px;font-size:18px}}ul{{margin:0;padding-left:18px;columns:2;column-gap:36px}}
li{{break-inside:avoid;margin:0 0 7px}}.sub{{margin-left:8px;color:#666;font-size:12px}}
@media(max-width:900px){{ul{{columns:1}}}}
@media(max-width:760px){{body{{padding:16px;overflow-wrap:anywhere}}li a{{display:inline-block;min-height:44px;padding:10px 0;box-sizing:border-box;touch-action:manipulation}}.sub{{display:block;margin-left:0}}}}
</style></head><body>
<h1>Realtime-Bench Task Pages Hub</h1>
<p>点击 task 打开页面。Game 直接读取本地页面；Web 每次点击都会创建新的实时 episode。</p>
<section><h2>Game ({len(GAMES)})</h2><ul>{item_list(GAMES, "game")}</ul></section>
<section><h2>Web ({len(WEB)})</h2><ul>{item_list(WEB, "web")}</ul></section>
</body></html>"""
    return content.encode("utf-8")


def seeded_page(body: bytes) -> bytes:
    """Inline the runtime's seeded-random script ahead of every script in a game document."""
    from realtime_tasks.game.runtime import seeded_random_source

    tag = f"<script>{seeded_random_source(SEED)}</script>".encode()
    for opening in (rb"<head(?:\s[^>]*)?>", rb"<html(?:\s[^>]*)?>", rb"<!doctype[^>]*>"):
        match = re.search(opening, body, re.IGNORECASE)
        if match:
            return body[: match.end()] + tag + body[match.end() :]
    return tag + body


def safe_static(path: str) -> Path | None:
    decoded = parse.unquote(path)
    normalized = posixpath.normpath(decoded)
    candidate = (ROOT / normalized.lstrip("/")).resolve()
    if candidate.is_dir():
        candidate = (candidate / "index.html").resolve()
    if not candidate.is_file():
        return None
    for allowed in ALLOWED_ROOTS:
        try:
            candidate.relative_to(allowed)
            return candidate
        except ValueError:
            pass
    return None


SCORE_MONITOR_HTML = "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n<title>Score monitor</title>\n<style>\n  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:860px;margin:0 auto;padding:20px 24px;line-height:1.45;color:#222}\n  h1{margin:0 0 4px;font-size:20px} h1 code{font-size:16px;color:#555}\n  p.hint{margin:0 0 14px;color:#666;font-size:13px}\n  .bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:14px}\n  button{font:inherit;padding:6px 12px;border:1px solid #bbb;border-radius:6px;background:#f7f7f7;cursor:pointer}\n  button.primary{background:#1a73e8;color:#fff;border-color:#1a73e8}\n  #conn{margin-left:auto;font-size:13px;color:#666}\n  .score{border:1px solid #ddd;border-radius:10px;padding:14px 16px;margin-bottom:12px}\n  .stats{display:flex;gap:12px;align-items:stretch}\n  .stat{flex:1;padding:8px 12px;border-radius:8px;background:#f7f8fa}\n  .stat .label{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#777}\n  .stat .value{font-size:34px;font-weight:700;line-height:1.15;font-variant-numeric:tabular-nums;color:#222}\n  .stat.current{flex:1.6}\n  .stat.current .value{font-size:48px}\n  .stat .value .target{font-size:20px;font-weight:400;color:#777}\n  #badge{display:inline-block;margin-left:10px;padding:2px 8px;border-radius:999px;background:#1e8e3e;color:#fff;font-size:12px;font-weight:600;vertical-align:middle}\n  #badge:empty{display:none}\n  .field{color:#777;font-size:12px;margin-top:8px}\n  .progress{height:8px;background:#eee;border-radius:4px;margin-top:12px;overflow:hidden}\n  #fill{height:100%;width:0;background:#1a73e8;transition:width .2s}\n  .reached #fill{background:#1e8e3e} .reached .stat.current .value{color:#1e8e3e}\n  .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 16px;font-size:14px;margin-bottom:12px}\n  .grid div{border-bottom:1px dashed #e5e5e5;padding:4px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap} .grid b{float:right;font-variant-numeric:tabular-nums;margin-left:8px}\n  .gridnote{color:#777;font-size:12px;margin:2px 0 6px}\n  .warn{color:#b00020}\n  details{margin:8px 0} pre{background:#f6f6f6;border-radius:6px;padding:10px;font-size:12px;max-height:280px;overflow:auto;margin:6px 0 0}\n  h2{font-size:15px;margin:14px 0 6px} #log{margin:0;padding-left:18px;font-size:13px;max-height:220px;overflow:auto}\n  #log li{margin:0 0 3px;font-variant-numeric:tabular-nums}\n  @media(max-width:760px){body{padding:16px;overflow-wrap:anywhere}.stats{flex-wrap:wrap}.stat{min-width:110px}.grid{grid-template-columns:1fr}button{min-height:44px;touch-action:manipulation}}\n</style>\n</head>\n<body>\n<h1>Score monitor <code id=\"task\"></code></h1>\n<p class=\"hint\">This page does not modify the game page. It opens the game with window.open and polls that window's <code>window.gameAPI.getState()</code> every 250 ms, the same interface the harness reads. Keep the game window and this page side by side. <b>Open game</b> opens the game once and afterwards just brings it to the front. <b>Reset</b> clears this monitor and, if the game window is open, restarts the game via gameAPI.reset().</p>\n<div class=\"bar\">\n  <button id=\"open\" class=\"primary\">Open game</button>\n  <button id=\"reset\">Reset</button>\n  <span id=\"conn\">Not connected</span>\n</div>\n<div class=\"score\" id=\"scorebox\">\n  <div class=\"stats\">\n    <div class=\"stat current\"><div class=\"label\">current</div><div class=\"value\"><span id=\"score\">–</span> <span class=\"target\">/ <span id=\"target\">?</span></span><span id=\"badge\"></span></div></div>\n    <div class=\"stat\"><div class=\"label\">peak</div><div class=\"value\" id=\"peak\">–</div></div>\n    <div class=\"stat\"><div class=\"label\">baseline</div><div class=\"value\" id=\"baseline\">–</div></div>\n  </div>\n  <div class=\"progress\"><div id=\"fill\"></div></div>\n  <div class=\"field\"><span id=\"basewarn\" class=\"warn\"></span> score_field = <code id=\"field\">?</code>. The harness marks the task solved when the <em>peak</em> reaches the target; progress = (peak − baseline) / (target − baseline); baseline = first value after reset. Note: the shared shim may count 1–2 free seconds right after the page loads; the harness resets before scoring, so press Reset if you want the same clean start.</div>\n</div>\n<div class=\"grid\">\n  <div>status <b id=\"status\">–</b></div>\n  <div>terminal <b id=\"terminal\">–</b></div>\n</div>\n<div class=\"gridnote\">Other fields reported by this game's getState() (game_state.* and metrics.*), listed automatically:</div>\n<div class=\"grid\" id=\"grid\"></div>\n<details><summary>Full getState() response</summary><pre id=\"all\">–</pre></details>\n<h2>Events <span class=\"gridnote\" style=\"font-weight:400\">(newest first; +seconds since this episode started; keys and clicks are logged too)</span></h2>\n<ul id=\"log\"></ul>\n<script>\n(function () {\n  const q = new URLSearchParams(location.search);\n  const gameUrl = q.get('url') || '';\n  const task = q.get('task') || '';\n  const scoreField = q.get('score_field') || 'metrics.primary_score';\n  const target = Number(q.get('target'));\n  const hasTarget = Number.isFinite(target) && q.get('target') !== null && q.get('target') !== '';\n  const $ = (id) => document.getElementById(id);\n  $('task').textContent = task || gameUrl;\n  $('field').textContent = scoreField;\n  $('target').textContent = hasTarget ? String(target) : 'unknown';\n  document.title = 'Score monitor · ' + (task || gameUrl);\n\n  let win = null, lastStatus = null, lastTerminal = null, reached = false, wasReadable = false;\n  let baseline = null, peak = null, closedLogged = false, gridKeys = '';\n  let hookedTopDoc = null, lastInput = null;\n  // Read-only observers on the game window so Events can show what was pressed. The shim only\n  // counts inputs that reach the inner game frame, so keys landing on the wrapper page are flagged.\n  function noteInput(kind, label, delivered) {\n    const now = Date.now();\n    if (lastInput && lastInput.kind === kind && lastInput.label === label && lastInput.delivered === delivered && now - lastInput.at < 1000 && lastInput.li.isConnected) {\n      lastInput.count += 1; lastInput.at = now;\n      lastInput.li.textContent = lastInput.prefix + `${kind} ${label} ×${lastInput.count}` + (delivered ? '' : '  (NOT delivered to the game frame: focus is outside it, the shim does not count this)');\n      return;\n    }\n    const prefix = `[${new Date().toLocaleTimeString('en-GB', { hour12: false })} +${((now - t0) / 1000).toFixed(1)}s] `;\n    log(`${kind} ${label}` + (delivered ? '' : '  (NOT delivered to the game frame: focus is outside it, the shim does not count this)'), delivered ? '' : 'warn');\n    lastInput = { kind, label, delivered, count: 1, at: now, li: $('log').firstChild, prefix };\n  }\n  function hookDoc(doc, delivered) {\n    if (!doc || doc.__monitorHooked) return;\n    doc.__monitorHooked = true;\n    doc.addEventListener('keydown', (e) => { if (e.isTrusted) noteInput('key', e.key === ' ' ? 'Space' : e.key, delivered); }, true);\n    doc.addEventListener('pointerdown', (e) => { if (e.isTrusted) noteInput('click', `(${Math.round(e.clientX)},${Math.round(e.clientY)})`, delivered); }, true);\n  }\n  function hookInputs() {\n    let topDoc; try { topDoc = win.document; } catch (e) { return; }\n    if (!topDoc) return;\n    const frame = topDoc.getElementById('game-frame');\n    if (hookedTopDoc !== topDoc) {\n      hookedTopDoc = topDoc;\n      hookDoc(topDoc, !frame);                                   // no inner frame → the top doc is the game\n      if (frame) frame.addEventListener('load', () => { try { hookDoc(frame.contentDocument, true); } catch (e) {} });\n    }\n    if (frame) { try { hookDoc(frame.contentDocument, true); } catch (e) {} }   // (re)attach after each inner reload\n  }\n  // Same call sequence the harness uses right before the agent loop (game/runtime.py reset_episode).\n  function harnessStyleReset(api) {\n    try { if (typeof api.init === 'function') api.init({ seed: 42 }); } catch (e) { log('init failed: ' + e, 'warn'); }\n    api.reset({ seed: 42 });\n  }\n  function resetTracking() { baseline = null; peak = null; reached = false; }\n  function clearPanel() {\n    for (const id of ['score', 'peak', 'baseline', 'status', 'terminal', 'all']) $(id).textContent = '–';\n    $('grid').textContent = ''; gridKeys = '';\n    $('fill').style.width = '0%'; $('scorebox').classList.remove('reached'); $('badge').textContent = ''; $('basewarn').textContent = '';\n  }\n  function clearEvents() { $('log').textContent = ''; }\n  function windowIsOpen() { return !!(win && !win.closed); }\n  let t0 = Date.now();                      // episode clock: restarted by Open game and Reset\n\n  function log(msg, cls) {\n    const li = document.createElement('li');\n    const wall = new Date().toLocaleTimeString('en-GB', { hour12: false });\n    li.textContent = `[${wall} +${((Date.now() - t0) / 1000).toFixed(1)}s] ${msg}`;\n    if (cls) li.className = cls;\n    const ul = $('log'); ul.insertBefore(li, ul.firstChild);\n  }\n  function dig(obj, path) {\n    return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);\n  }\n  function fmt(v) {\n    if (v === undefined || v === null) return '–';\n    if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(parseFloat(v.toFixed(4)));  // up to 4 decimals, no false rounding to the target\n    return String(v);\n  }\n\n  $('open').onclick = () => {\n    if (!gameUrl) { log('Missing url parameter; cannot open the game', 'warn'); return; }\n    if (windowIsOpen()) { win.focus(); log('Game window already open; brought it to the front'); return; }\n    const opened = window.open(gameUrl, 'game:' + gameUrl);\n    if (!opened) { log('The browser blocked the popup; allow popups for this site and retry', 'warn'); return; }\n    win = opened; closedLogged = false; t0 = Date.now(); hookedTopDoc = null; lastInput = null;\n    resetTracking(); wasReadable = false; lastStatus = null; lastTerminal = null;\n    log('Opened game window ' + gameUrl);\n  };\n  $('reset').onclick = () => {\n    clearEvents(); clearPanel(); resetTracking(); t0 = Date.now(); lastInput = null;\n    if (!windowIsOpen()) { win = null; lastStatus = null; lastTerminal = null; $('conn').textContent = 'Not connected: click \"Open game\"'; log('Monitor reset (no game window; click Open game to start one)'); return; }\n    try {\n      if (win.gameAPI && win.gameAPI.reset) { harnessStyleReset(win.gameAPI); log('Reset: gameAPI.init({seed:42}) + reset({seed:42}), new episode'); }\n      else log('Reset: game window is open but gameAPI is not ready yet', 'warn');\n    } catch (e) { log('reset failed: ' + e, 'warn'); }\n  };\n\n  function scalarEntries(st) {\n    const out = [];\n    for (const section of ['game_state', 'metrics']) {\n      const obj = st[section];\n      if (!obj || typeof obj !== 'object') continue;\n      for (const key of Object.keys(obj).sort()) {\n        const path = section + '.' + key;\n        if (path === scoreField) continue;                       // already shown as \"current\"\n        const v = obj[key];\n        const ok = typeof v === 'number' || typeof v === 'boolean' || (typeof v === 'string' && v.length <= 40);\n        if (ok) out.push([path, v]);\n      }\n    }\n    return out;\n  }\n  function renderGrid(st) {\n    const entries = scalarEntries(st);\n    const keys = entries.map(e => e[0]).join('|');\n    const grid = $('grid');\n    if (keys !== gridKeys) {                                      // rebuild only when the field set changes\n      gridKeys = keys; grid.textContent = '';\n      for (const [path] of entries) {\n        const cell = document.createElement('div'); cell.dataset.path = path;\n        cell.title = path; cell.appendChild(document.createTextNode(path.replace(/^game_state\\./, '') + ' '));\n        const b = document.createElement('b'); cell.appendChild(b); grid.appendChild(cell);\n      }\n    }\n    for (const [path, v] of entries) {\n      const cell = grid.querySelector(`div[data-path=\"${path}\"] b`);\n      if (cell) cell.textContent = fmt(v);\n    }\n  }\n  function render(st) {\n    const val = dig(st, scoreField);\n    $('score').textContent = fmt(val);\n    $('status').textContent = fmt(st.status);\n    $('terminal').textContent = st.terminal ? `${st.terminal.isTerminal}${st.terminal.reason ? ' (' + st.terminal.reason + ')' : ''}` : '–';\n    renderGrid(st);\n    $('all').textContent = JSON.stringify(st, null, 1);\n    if (typeof val === 'number') {\n      if (baseline === null) { baseline = val; peak = val; }\n      if (val > peak) peak = val;\n    }\n    $('peak').textContent = fmt(peak);\n    $('baseline').textContent = fmt(baseline);\n    if (hasTarget && peak !== null) {\n      // Same rule as the harness (game/verifier.py score_target): success = peak >= target with baseline < target.\n      const baselineSuspect = baseline >= target;\n      $('basewarn').textContent = baselineSuspect ? '(baseline already ≥ target: the harness would mark this run invalid)' : '';\n      const denom = target - baseline;\n      const pct = denom > 0 ? Math.max(0, Math.min(1, (peak - baseline) / denom)) : 0;\n      $('fill').style.width = (pct * 100).toFixed(1) + '%';\n      const ok = !baselineSuspect && peak >= target;\n      $('scorebox').classList.toggle('reached', ok);\n      $('badge').textContent = ok ? 'target reached' : '';\n      if (ok && !reached) { reached = true; log(`peak reached the target: ${fmt(peak)} >= ${target} (the harness would mark the task solved and end the episode here)`); }\n    } else {\n      $('fill').style.width = '0%'; $('scorebox').classList.remove('reached'); $('badge').textContent = ''; $('basewarn').textContent = '';\n    }\n    if (st.status !== lastStatus) { if (lastStatus !== null) log(`status: ${lastStatus} → ${st.status}`); lastStatus = st.status; }\n    const term = st.terminal && st.terminal.isTerminal;\n    if (term !== lastTerminal) { if (lastTerminal !== null) log(`terminal: ${term}${term && st.terminal.reason ? ' (' + st.terminal.reason + ')' : ''}`); lastTerminal = term; }\n  }\n\n  setInterval(() => {\n    if (!win) { $('conn').textContent = 'Not connected: click \"Open game\"'; return; }\n    if (win.closed) { $('conn').textContent = 'Game window closed'; if (!closedLogged) { closedLogged = true; log('Game window closed', 'warn'); } return; }\n    let api;\n    try { api = win.gameAPI; } catch (e) { $('conn').textContent = 'Cannot read the game window (different origin?)'; return; }\n    if (!api || typeof api.getState !== 'function') { $('conn').textContent = 'Game loading…'; return; }\n    let st;\n    try { st = api.getState(); } catch (e) { $('conn').textContent = 'getState() threw: ' + e; return; }\n    if (!st) { $('conn').textContent = 'getState() returned nothing'; return; }\n    if (!wasReadable) { wasReadable = true; log('Connected to the game window\\'s gameAPI'); }\n    hookInputs();\n    $('conn').textContent = 'Connected, refreshing every 250 ms';\n    render(st);\n  }, 250);\n})();\n</script>\n</body>\n</html>\n"


class Handler(BaseHTTPRequestHandler):
    server_version = "TaskPagesHub/1.0"

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def proxy_realtime(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        if content_length > 1024 * 1024:
            self.send_error(413, "Request body too large")
            return
        body = self.rfile.read(content_length) if content_length else None
        headers = {
            name: value
            for name in ("Accept", "Content-Type")
            if (value := self.headers.get(name)) is not None
        }
        endpoint = f"http://127.0.0.1:{RT_PORT}{self.path}"
        proxy_request = request.Request(
            endpoint,
            data=body,
            method=self.command,
            headers=headers,
        )

        def forward(response) -> None:
            response_body = response.read()
            self.send_response(response.status)
            for name in (
                "Content-Type",
                "Content-Disposition",
                "Cache-Control",
                "ETag",
                "Last-Modified",
            ):
                value = response.headers.get(name)
                if value is not None:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(response_body)

        try:
            with request.urlopen(proxy_request, timeout=10) as response:
                forward(response)
        except error.HTTPError as response:
            forward(response)
        except error.URLError as exc:
            self.send_error(502, f"Realtime backend unavailable: {exc}")

    def open_web(self, task_id: str) -> None:
        row = WEB_BY_ID.get(task_id)
        if row is None:
            self.send_error(404, "Unknown Web task")
            return
        scenario = row["scenario"]
        endpoint = f"http://127.0.0.1:{RT_PORT}/api/reset/{parse.quote(scenario, safe='')}"
        reset_request = request.Request(
            endpoint,
            data=json.dumps({"seed": SEED}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(reset_request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            episode = payload["episode"]
        except (error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.send_error(502, f"Could not reset realtime task: {exc}")
            return
        browser_base = RT_PUBLIC_BASE if RT_PUBLIC_BASE else ""
        location = (
            f"{browser_base}/{parse.quote(scenario, safe='')}"
            f"?episode={parse.quote(episode, safe='')}"
        )
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = parse.urlsplit(self.path).path
        if path == "/game-score-monitor":
            self.send_bytes(200, SCORE_MONITOR_HTML.encode(), "text/html; charset=utf-8")
            return
        if path in {"/", "/task_pages_hub.html"}:
            self.send_bytes(200, hub_html(), "text/html; charset=utf-8")
            return
        if path == "/hub-health":
            self.send_bytes(200, b'{"service":"task-pages-hub"}\n', "application/json")
            return
        web_prefix = "/open/web/"
        if path.startswith(web_prefix):
            self.open_web(parse.unquote(path[len(web_prefix) :]))
            return

        target = safe_static(path)
        if target is None:
            self.proxy_realtime()
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type == "text/html" and target.is_relative_to(GAME_ROOT):
            body = seeded_page(body)
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
        }:
            content_type += "; charset=utf-8"
        self.send_bytes(200, body, content_type)

    def do_POST(self) -> None:
        self.proxy_realtime()


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main() -> None:
    global HOST, PORT, RT_PORT, RT_PUBLIC_BASE, SEED
    parser = argparse.ArgumentParser(description="Preview local Game and Web task pages.")
    parser.add_argument("--host", default=HOST, help="hub bind address (default: %(default)s)")
    parser.add_argument(
        "--port", type=port_number, default=str(PORT), help="hub port (default: %(default)s)"
    )
    parser.add_argument(
        "--rt-port", type=port_number, default=str(RT_PORT),
        help="Web backend port (default: %(default)s)",
    )
    parser.add_argument(
        "--rt-public-base", default=RT_PUBLIC_BASE,
        help="browser-visible Web backend URL (default: proxy through the hub)",
    )
    parser.add_argument(
        "--seed", type=task_seed, default=os.environ.get("SEED", str(DEFAULT_TASK_SEED)),
        help="task random seed for Game and Web pages (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.port == args.rt_port:
        parser.error("--port and --rt-port must be different")
    HOST, PORT, RT_PORT, SEED = args.host, args.port, args.rt_port, args.seed
    RT_PUBLIC_BASE = args.rt_public_base.rstrip("/")
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = "127.0.0.1,localhost,::1," + os.environ.get(name, "")

    def stop_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    process = None
    server = None
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
        process = start_local_realtime()
        print(
            f"Task pages: http://{HOST}:{PORT}/\n"
            "Each click on a Web task creates a fresh realtime episode.\n"
            "Press Ctrl-C to stop; a local backend started here is stopped too.",
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        stop_local_realtime(process)
        if server is not None:
            server.server_close()


if __name__ == "__main__":
    main()
