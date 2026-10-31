// Benchmark result views layered over Harbor's job page. Native trial routes stay intact.
const base = new URL('.', import.meta.url).pathname.replace(/\/$/, '');
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));
const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const elapsed = value => {
  if (value == null) return '—';
  const tenths = Math.round(value * 10);
  return `${Math.floor(tenths / 600)}m ${(tenths % 600 / 10).toFixed(1)}s`;
};
const href = trial => escape(base + trial.href);
const css = `
:host{display:block;color:var(--foreground,inherit);font-family:inherit;font-size:14px;line-height:1.5;margin-top:20px}
*{box-sizing:border-box}button,input,select{font:inherit;color:inherit}button,a,input,select{outline-offset:4px}
button,select,input{background:transparent;border:1px solid #8884;border-radius:5px;padding:7px 12px}
button,a{cursor:pointer}a{color:inherit;text-decoration:none}a:hover{text-decoration:underline}
.tabs{display:flex;gap:24px;border-bottom:1px solid #8884;margin-bottom:24px}
.tabs button{border:0;border-radius:0;border-bottom:2px solid transparent;padding:12px 0;color:inherit;opacity:.6}
.tabs [aria-selected=true]{border-bottom-color:currentColor;opacity:1;font-weight:600}
h2{font-size:16px;margin:0 0 18px;font-weight:600}p{margin:8px 0}.muted{opacity:.65;font-size:12px}
.overview{border:1px solid #8884;padding:24px;max-width:460px;margin-bottom:24px}.metric{font-size:42px;line-height:1.2;font-weight:600;letter-spacing:-1px}
.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:20px}.toolbar input{min-width:220px;flex:1}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12px;align-items:center;margin:12px 0 24px}
.legend span{display:inline-flex;align-items:center;gap:7px}.swatch{display:inline-block;width:12px;height:12px;border:1px solid #8885;background:#ddd}
.waffle-wrap{overflow:auto;padding:20px 12px;border:1px solid #8883}
.waffle{width:auto;max-width:100%;margin:0 auto;border-collapse:separate;border-spacing:0;font-size:11px}
.waffle th,.waffle td{border:0;padding:1px 12px;line-height:16px;white-space:nowrap}
.waffle thead th{padding-bottom:14px;text-align:center;opacity:1;font-weight:400}
.waffle tbody th{text-align:right;font-weight:400;opacity:.65;max-width:260px;overflow:hidden;text-overflow:ellipsis}
.waffle td{text-align:center;vertical-align:middle}
.waffle-attempts{display:flex;justify-content:center;gap:4px}
.waffle-cell{display:block;flex:none;width:14px;height:14px;padding:0;border:1px solid #8884;border-radius:0}
.waffle-cell:hover,.waffle-cell:focus-visible{outline:2px solid currentColor;outline-offset:2px}
.waffle-cell.pass,.swatch.pass{background:#fff}
.waffle-cell.fail,.swatch.fail{background:#000}
.waffle-cell.timeout,.swatch.timeout{background:#dc2626;border-color:#dc2626}
.waffle-cell.error,.swatch.error{background:repeating-linear-gradient(45deg,#ddd,#ddd 3px,#999 3px,#999 5px)}
.waffle-cell.unscored{border-style:dashed}.timeout-text{color:#dc2626;font-weight:600}
.swatch.unscored{background:transparent;border-style:dashed}
.waffle-tooltip{position:fixed;z-index:1000;width:310px;max-width:calc(100vw - 16px);padding:12px;background:var(--background,#fff);color:inherit;border:1px solid #8885;box-shadow:0 4px 16px #0002;border-radius:4px;pointer-events:none;font-size:12px}
.waffle-tooltip strong{display:block;overflow-wrap:anywhere;margin-bottom:8px}.tooltip-trial{display:flex;justify-content:space-between;gap:12px;margin-top:5px}
.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left;font-size:13px}th{font-weight:500;opacity:.65;white-space:nowrap}
th,td{padding:12px 14px;border-bottom:1px solid #8883}td:first-child,th:first-child{padding-left:0}td{white-space:nowrap}
.badge{display:inline-block;border:1px solid #8884;border-radius:4px;padding:2px 8px;font-size:12px}
.badge.pass{background:#fff;color:#000}.badge.fail{background:#000;color:#fff;border-color:#000}.badge.timeout{background:#fef2f2;color:#b91c1c;border-color:#fecaca}
.pager{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:20px}.pager button:disabled{opacity:.35;cursor:default}
.empty{padding:35px 0;opacity:.65}[hidden]{display:none!important}
@media(max-width:600px){.tabs{gap:20px}.overview{max-width:none}.waffle-wrap{padding:16px 4px}.waffle th,.waffle td{padding-right:6px;padding-left:6px}.waffle tbody th{max-width:170px}.waffle thead th{max-width:130px;white-space:normal;overflow-wrap:anywhere}.pager{flex-wrap:wrap}}
`;

class ResultsView extends HTMLElement {
  connectedCallback() {
    if (this.shadowRoot) return;
    this.attachShadow({mode: 'open'});
    // Keep Harbor's page-level navigation shortcuts out of these controls.
    this.shadowRoot.addEventListener('keydown', event => event.stopPropagation());
    this.tab = 'Overview'; this.query = ''; this.filter = ''; this.page = 0;
    this.abort = new AbortController();
    this.load();
    this.timer = setInterval(() => this.load(), 15000);
  }
  disconnectedCallback() {
    clearInterval(this.timer); this.abort.abort();
    this.nativeRoot.style.display = this.originalDisplay;
    this.nativeTabs.style.display = this.originalTabsDisplay;
    for (const [node, display] of this.nativeMetrics || []) node.style.display = display;
  }
  async load() {
    if (this.loading) return;
    this.loading = true;
    try {
      const response = await fetch(`${base}/api/rtbench/jobs/${encodeURIComponent(this.job)}`, {
        signal: this.abort.signal, cache: 'no-store',
      });
      if (!response.ok) throw new Error('Results unavailable');
      const data = await response.json();
      labels[this.job] = data.display_name;
      renameJobs();
      const signature = JSON.stringify(data);
      if (signature !== this.signature) {
        this.data = data; this.signature = signature; this.render();
      }
    } catch (error) {
      if (!this.data && error.name !== 'AbortError') {
        this.shadowRoot.innerHTML = '<p>Benchmark results unavailable. The original Harbor results remain below.</p>';
      }
    } finally { this.loading = false; }
  }
  render() {
    const data = this.data;
    this.nativeMetrics ||= new Map();
    for (const node of document.querySelectorAll('main [data-slot="tooltip-trigger"]')) {
      if (!/^\d+(?:\.\d+)? mean\s/.test(node.textContent.trim())) continue;
      if (!this.nativeMetrics.has(node)) this.nativeMetrics.set(node, node.style.display);
      node.style.display = 'none';
    }
    this.nativeRoot.style.display = this.tab === 'Config' ? this.originalDisplay : 'none';
    this.nativeTabs.style.display = 'none';
    this.shadowRoot.innerHTML = `<style>${css}</style>
      <div class="tabs" role="tablist" aria-label="Job results">${['Overview','Trials','Config'].map(tab =>
        `<button id="tab-${tab}" role="tab" aria-selected="${tab === this.tab}" tabindex="${tab === this.tab ? 0 : -1}" aria-controls="panel" data-tab="${tab}">${tab}</button>`).join('')}</div>
      <section id="panel" role="tabpanel" aria-labelledby="tab-${this.tab}"></section>`;
    for (const button of this.shadowRoot.querySelectorAll('[data-tab]')) {
      button.onclick = () => {
        this.tab = button.dataset.tab; this.page = 0; this.render();
        if (this.tab === 'Config') {
          const config = [...this.nativeTabs.querySelectorAll('[role=tab]')].find(t => t.textContent.trim() === 'Config');
          config?.dispatchEvent(new MouseEvent('mousedown', {bubbles:true, button:0}));
        }
        this.shadowRoot.querySelector(`[data-tab="${this.tab}"]`).focus();
      };
      button.onkeydown = event => {
        if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
        event.preventDefault();
        const buttons = [...this.shadowRoot.querySelectorAll('[data-tab]')];
        const i = buttons.indexOf(button);
        buttons[event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (i + (event.key === 'ArrowRight' ? 1 : buttons.length - 1)) % buttons.length].click();
      };
    }
    const panel = this.shadowRoot.querySelector('#panel');
    if (this.tab === 'Config') return;
    panel.innerHTML = `${this.tab === 'Overview' ? `<div class="overview"><h2>Average pass rate</h2><div class="metric">${percent(data.average_pass_rate)}</div>
        <p class="muted">${data.passes} / ${data.scored} scored trials · ${data.trials.length - data.scored} running or pending</p></div>` : ''}
      <div class="toolbar"><input type="search" aria-label="Search tasks" placeholder="Search tasks…" value="${escape(this.query)}">
      ${this.tab === 'Trials' ? `<select aria-label="Filter results"><option value="">All results</option>${['Pass','Fail','Timeout','Error','Running','Pending'].map(status => `<option${this.filter === status ? ' selected' : ''}>${status}</option>`).join('')}</select>` : ''}</div>
      <div class="content"></div>`;
    panel.querySelector('input').oninput = event => { this.query = event.target.value; this.page = 0; this.renderContent(); };
    if (panel.querySelector('select')) panel.querySelector('select').onchange = event => {
      this.filter = event.target.value; this.page = 0; this.renderContent();
    };
    this.renderContent();
  }
  renderContent() {
    const isTrials = this.tab === 'Trials';
    const items = (isTrials ? this.data.trials : this.data.tasks).filter(item =>
      (isTrials ? item.task : item.name).toLowerCase().includes(this.query.toLowerCase()) &&
      (!isTrials || !this.filter || item.status === this.filter));
    const size = 50, pages = Math.max(1, Math.ceil(items.length / size));
    this.page = Math.min(this.page, pages - 1);
    const visible = isTrials ? items.slice(this.page * size, (this.page + 1) * size) : items;
    const content = this.shadowRoot.querySelector('.content');
    if (isTrials) {
      content.innerHTML = `<div class="table-wrap"><table><thead><tr><th>Task</th><th>Trial</th><th>Result</th><th>Reward</th><th>Duration</th></tr></thead><tbody>${visible.map(trial =>
        `<tr><td><a href="${href(trial)}">${escape(trial.task)}</a></td><td><a href="${href(trial)}">${escape(trial.name)}</a></td>
        <td><span class="badge ${trial.status.toLowerCase()}">${escape(trial.status)}</span>${trial.error_reason ? ` <span class="muted">${escape(trial.error_reason)}</span>` : ''}</td><td>${trial.reward ?? '—'}</td>
        <td class="${trial.status === 'Timeout' ? 'timeout-text' : ''}">${elapsed(trial.duration_s)}</td></tr>`).join('')}</tbody></table></div>`;
    } else {
      const groups = [...new Map(this.data.trials.map(trial => [
        JSON.stringify([trial.model, trial.effort]), {model:trial.model, effort:trial.effort},
      ])).entries()];
      const attempts = (task, key) => task.trials.filter(trial => JSON.stringify([trial.model, trial.effort]) === key)
        .sort((a, b) => (a.run ?? Infinity) - (b.run ?? Infinity) || a.name.localeCompare(b.name, undefined, {numeric:true}));
      content.innerHTML = `<div class="legend"><span><i class="swatch pass"></i>Pass</span><span><i class="swatch fail"></i>Fail</span>
        <span><i class="swatch timeout"></i>Timeout</span><span><i class="swatch error"></i>Error</span><span><i class="swatch unscored"></i>Running / pending</span></div>
        <p class="muted">Each square is one run. Runs are ordered left to right. Hover for its result; select to open the trial.</p>
        <div class="waffle-wrap"><table class="waffle" aria-label="Task results waffle"><thead><tr><th scope="col">Task</th>
          ${groups.map(([,group]) => `<th scope="col">${escape(group.model)}<br><span class="muted">${escape(group.effort ?? 'default')}</span></th>`).join('')}</tr></thead>
          <tbody>${visible.map((task, taskIndex) => `<tr><th scope="row" title="${escape(task.name)}">${escape(task.name)}</th>${groups.map(([key], groupIndex) => {
            const trials = attempts(task, key);
            if (!trials.length) return '<td>—</td>';
            return `<td><div class="waffle-attempts">${trials.map((trial, trialIndex) => {
              const description = `${task.name} · Run ${trial.run ?? trialIndex + 1} · ${trial.status}${trial.error_reason ? ` (${trial.error_reason})` : ''} · ${elapsed(trial.duration_s)}`;
              return `<a class="waffle-cell ${trial.status.toLowerCase()}${trial.passed == null ? ' unscored' : ''}" href="${href(trial)}"
                data-task="${taskIndex}" data-group="${groupIndex}" data-trial="${trialIndex}" aria-label="${escape(description)}"></a>`;
            }).join('')}</div></td>`;
          }).join('')}</tr>`).join('')}</tbody></table></div>
        <div class="waffle-tooltip" id="waffle-tooltip" role="tooltip" hidden></div>`;
      const tooltip = content.querySelector('.waffle-tooltip');
      for (const cell of content.querySelectorAll('.waffle-cell')) {
        const task = visible[Number(cell.dataset.task)], [key] = groups[Number(cell.dataset.group)];
        const trialIndex = Number(cell.dataset.trial), trial = attempts(task, key)[trialIndex];
        const show = () => {
          tooltip.innerHTML = `<strong>${escape(task.name)}</strong><div class="tooltip-trial"><span>Run ${escape(trial.run ?? trialIndex + 1)}</span>
            <span class="${trial.status === 'Timeout' ? 'timeout-text' : ''}">${escape(trial.status)}${trial.error_reason ? ` (${escape(trial.error_reason)})` : ''} · ${elapsed(trial.duration_s)}</span></div>`;
          tooltip.hidden = false; cell.setAttribute('aria-describedby', 'waffle-tooltip');
          const box = cell.getBoundingClientRect();
          tooltip.style.left = `${Math.max(8, Math.min(box.left, innerWidth - tooltip.offsetWidth - 8))}px`;
          tooltip.style.top = `${Math.max(8, box.bottom + tooltip.offsetHeight + 8 > innerHeight ? box.top - tooltip.offsetHeight - 8 : box.bottom + 8)}px`;
        };
        const hide = () => { tooltip.hidden = true; cell.removeAttribute('aria-describedby'); };
        cell.onmouseenter = cell.onfocus = show;
        cell.onmouseleave = cell.onblur = hide;
        cell.onkeydown = event => { if (event.key === 'Escape') hide(); };
      }
    }
    if (!items.length) content.innerHTML += '<p class="empty">No results match these filters.</p>';
    if (!isTrials) return;
    content.insertAdjacentHTML('beforeend', `<div class="pager"><span>${items.length ? this.page * size + 1 : 0}–${Math.min((this.page + 1) * size, items.length)} of ${items.length} ${isTrials ? 'trials' : 'tasks'}</span>
      <span><button data-page="-1" ${this.page === 0 ? 'disabled' : ''}>Previous</button> &nbsp; Page ${this.page + 1} of ${pages} &nbsp;
      <button data-page="1" ${this.page + 1 === pages ? 'disabled' : ''}>Next</button></span></div>`);
    for (const button of content.querySelectorAll('[data-page]')) button.onclick = () => {
      this.page += Number(button.dataset.page); this.renderContent();
    };
  }
}
customElements.define('rtbench-results', ResultsView);

let labels = {}, labelsLoading = false, labelsLoadedAt = 0, current;
function renameJobs() {
  const main = document.querySelector('main');
  if (!main) return;
  // Only replace complete job labels, never identifiers inside URLs or configuration text.
  for (const element of main.querySelectorAll('h1, a, td, nav span')) {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      const key = node.nodeValue.trim();
      const label = Object.hasOwn(labels, key) ? labels[key] : null;
      if (label && label !== node.nodeValue.trim()) node.nodeValue = label;
    }
  }
}
export function enhanceResults() {
  if (!labelsLoading && Date.now() - labelsLoadedAt > 30000) {
    labelsLoading = true; labelsLoadedAt = Date.now();
    fetch(`${base}/api/rtbench/jobs`, {cache:'no-store'}).then(response => {
      if (!response.ok) throw new Error('Labels unavailable');
      return response.json();
    }).then(value => { labels = value; renameJobs(); }).catch(() => {}).finally(() => { labelsLoading = false; });
  }
  renameJobs();
  const path = location.pathname.slice(base.length);
  const match = path.match(/^\/jobs\/([^/]+)\/?$/);
  const job = match ? decodeURIComponent(match[1]) : null;
  if (current && (current.job !== job || !current.isConnected || !current.nativeRoot.isConnected)) {
    current.remove(); current = null;
  }
  if (!job || current) return;
  const tabs = [...document.querySelectorAll('main [role=tablist]')].find(list =>
    [...list.querySelectorAll('[role=tab]')].some(tab => tab.textContent.trim() === 'Config'));
  if (!tabs) return;
  const panel = document.getElementById(tabs.querySelector('[role=tab][aria-selected=true]')?.getAttribute('aria-controls'));
  if (!panel) return;
  let root = tabs.parentElement;
  while (root && !root.contains(panel)) root = root.parentElement;
  if (!root || root.tagName === 'MAIN') return;
  current = document.createElement('rtbench-results');
  Object.assign(current, {job, nativeRoot:root, nativeTabs:tabs,
    originalDisplay:root.style.display, originalTabsDisplay:tabs.style.display});
  root.before(current);
}
