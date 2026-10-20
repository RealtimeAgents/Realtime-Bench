// Realtime-Bench additions to Harbor's viewer; the React bundle and source MP4 stay unchanged.
// Recordings get an active-loop player with action overlays. Harbor displays that do not
// describe benchmark results (token usage, cost, execution environment, local run paths) are omitted.
import { enhanceResults } from './rtbench-results.js';
const styles = `
:host{display:block;color:inherit;font:inherit;--panel:#17191d;--ink:#f4f4f5;--accent:#f7cd54}
*{box-sizing:border-box}.stage{position:relative;background:#000;line-height:0}.stage video{display:block;width:100%;max-height:70vh;object-fit:contain}.canvas{position:absolute;pointer-events:none;overflow:hidden}.canvas svg{width:100%;height:100%;position:absolute;inset:0}.caption,.thinking{position:absolute;left:12px;color:var(--ink);background:#111e;border-radius:4px;padding:7px 10px;font:13px/1.4 system-ui;white-space:pre-wrap;max-width:calc(100% - 24px);overflow:hidden;text-shadow:0 1px #000}.caption{top:12px;max-height:38%;overflow:auto;overflow-wrap:anywhere;pointer-events:auto;user-select:text;scrollbar-width:thin}.thinking{bottom:12px}.caption:empty,.thinking:empty{display:none}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 0;font:13px/1.4 system-ui}.controls button,.controls select{font:inherit;color:inherit;background:transparent;border:1px solid #8886;border-radius:5px;padding:5px 9px;cursor:pointer}.controls input[type=range]{flex:1;min-width:110px;accent-color:var(--accent)}.controls label{display:flex;gap:5px;align-items:center;cursor:pointer}.controls output{font-variant-numeric:tabular-nums}.note{font:12px/1.5 system-ui;opacity:.72;margin:3px 0 8px}:host(:fullscreen){display:flex;flex-direction:column;justify-content:center;background:#111;color:#fff;padding:16px}:host(:fullscreen) .stage video{max-height:85vh}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
`;
const finite = Number.isFinite;
const stamp = t => `${Math.floor(Math.max(0,t)/60)}:${String(Math.floor(Math.max(0,t)%60)).padStart(2,'0')}`;

class RecordingPlayer extends HTMLElement {
  connectedCallback() {
    if (this.shadowRoot) return;
    this.attachShadow({mode:'open'});
    this.shadowRoot.innerHTML = `<style>${styles}</style>
      <div class="stage"><video playsinline preload="metadata" aria-label="Episode recording"></video><div class="canvas"><svg aria-hidden="true" preserveAspectRatio="none" viewBox="0 0 1000 1000"></svg><div class="caption" aria-live="off"></div><div class="thinking"></div></div></div>
      <div class="controls"><button class="play" aria-label="Play">Play</button><output>0:00 / 0:00</output><input class="seek" type="range" min="0" max="0" value="0" step="0.01" aria-label="Playback position"><select aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><label><input class="annotations" type="checkbox" checked>Annotations</label><label><input class="full" type="checkbox">Full recording</label><button class="fullscreen" aria-label="Fullscreen">Fullscreen</button></div><p class="note">Loading playback timing…</p>`;
    const get = q => this.shadowRoot.querySelector(q);
    this.video=get('video');this.stage=get('.stage');this.canvas=get('.canvas');this.svg=get('svg');this.caption=get('.caption');this.thinking=get('.thinking');this.play=get('.play');this.seek=get('.seek');this.output=get('output');this.annotations=get('.annotations');this.full=get('.full');this.note=get('.note');this.lastOverlay='';
    this.metadata=null;this.frame=null;this.disposed=false;this.abort=new AbortController();
    this.play.onclick=()=>{if(this.video.paused){const [start,end]=this.bounds();if(this.video.currentTime>=end-.02)this.video.currentTime=start;this.video.play().catch(()=>{});}else this.video.pause();};
    this.seek.oninput=()=>{this.video.currentTime=this.bounds()[0]+Number(this.seek.value);this.update();};
    get('select').onchange=e=>{this.video.playbackRate=Number(e.target.value);};
    this.annotations.onchange=()=>this.update();
    this.full.onchange=()=>{this.clamp();this.update();};
    get('.fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else this.requestFullscreen?.();};
    this.video.onloadedmetadata=()=>{this.initializePosition();this.layout();this.update();};
    this.video.onseeking=()=>{this.clamp();this.update();};
    this.video.ontimeupdate=()=>this.update();
    this.video.onplay=()=>{this.clamp();this.update();};
    this.video.onpause=()=>this.update();
    this.video.onended=()=>this.update();
    this.video.onerror=()=>{this.note.textContent='This recording could not be played in this browser. The original download remains available.';};
    this.resize=new ResizeObserver(()=>this.layout());this.resize.observe(this.video);
    const loop=(_,info)=>{if(this.disposed)return;this.update(info?.mediaTime);this.frame=this.video.requestVideoFrameCallback?this.video.requestVideoFrameCallback(loop):requestAnimationFrame(loop);};
    this.frame=this.video.requestVideoFrameCallback?this.video.requestVideoFrameCallback(loop):requestAnimationFrame(loop);
    const src=this.getAttribute('src');
    fetch(src.replace(/\/file(?:\?.*)?$/, '/timeline'),{signal:this.abort.signal,cache:'no-store'})
      .then(r=>{if(!r.ok)throw new Error('Timing unavailable');return r.json();})
      .then(meta=>{
        if(this.disposed)return;
        this.metadata=meta;
        this.annotations.disabled=!!meta.baked_annotations || meta.state==='unavailable';
        this.full.disabled=!meta.clip || !!meta.baked_annotations;
        if(meta.baked_annotations){this.annotations.checked=false;this.note.textContent='Historical recording with embedded annotations; additional overlays are disabled.';}
        else if(meta.state==='unavailable'){this.annotations.checked=false;this.full.checked=true;this.note.textContent='Active-loop timing is unavailable. Showing the full original recording.';}
        else{this.note.textContent='Active loop · browser overlays · original video unchanged';if(meta.alignment?.basis==='legacy_recorder_launch_log')this.note.textContent+=' · approximate historical timing';}
        this.video.src=src;
      })
      .catch(error=>{if(error.name==='AbortError')return;this.metadata={state:'unavailable',events:[],inferences:[]};this.annotations.checked=false;this.annotations.disabled=true;this.full.checked=true;this.full.disabled=true;this.note.textContent='Timing unavailable. Showing the full original recording.';this.video.src=src;});
  }
  disconnectedCallback(){this.disposed=true;this.abort?.abort();this.resize?.disconnect();this.video?.pause();if(this.video?.cancelVideoFrameCallback)this.video.cancelVideoFrameCallback(this.frame);else cancelAnimationFrame(this.frame);}
  bounds(){
    const duration=finite(this.video.duration)?this.video.duration:0,clip=this.metadata?.clip;
    if(this.full.checked||!clip)return[0,duration];
    const start=Math.max(0,Math.min(duration,finite(clip.start_s)?clip.start_s:0));
    const end=Math.max(start,Math.min(duration,finite(clip.end_s)?clip.end_s:duration));
    return[start,end];
  }
  initializePosition(){if(finite(this.video.duration))this.video.currentTime=this.bounds()[0];}
  clamp(){
    if(!finite(this.video.duration))return;
    const[start,end]=this.bounds();
    if(this.video.currentTime<start-.005)this.video.currentTime=start;
    else if(this.video.currentTime>end+.005){this.video.pause();this.video.currentTime=end;}
  }
  layout(){
    const v=this.video,box=v.getBoundingClientRect();if(!v.videoWidth||!box.width)return;
    const scale=Math.min(box.width/v.videoWidth,box.height/v.videoHeight),w=v.videoWidth*scale,h=v.videoHeight*scale;
    Object.assign(this.canvas.style,{width:`${w}px`,height:`${h}px`,left:`${v.offsetLeft+(box.width-w)/2}px`,top:`${v.offsetTop+(box.height-h)/2}px`});
    this.svg.setAttribute('viewBox',`0 0 ${v.videoWidth} ${v.videoHeight}`);
  }
  update(frameTime){
    if(!this.video)return;
    const[start,end]=this.bounds(),time=finite(frameTime)&&!this.video.seeking?frameTime:this.video.currentTime;
    if(!this.video.paused&&time>=end&&end>start){this.video.pause();if(this.video.currentTime>end)this.video.currentTime=end;}
    this.seek.max=String(Math.max(0,end-start));this.seek.value=String(Math.max(0,Math.min(end-start,time-start)));
    this.output.textContent=`${stamp(time-start)} / ${stamp(end-start)}`;
    this.play.textContent=this.video.paused?'Play':'Pause';this.play.setAttribute('aria-label',this.play.textContent);
    const active=time-(this.metadata?.timeline_offset_s||0),enabled=this.annotations.checked&&!this.annotations.disabled;
    const events=this.metadata?.events||[];
    let selected=null,phase='';
    if(enabled)for(let i=events.length-1;i>=0;i--){
      const e=events[i];if(active<e.time)continue;
      const resolved=Math.max(e.time,finite(e.resolved_time)?e.resolved_time:e.time),next=events[i+1]?.time??Infinity;
      if(active<Math.min(resolved,next)){selected=e;phase='Dispatching';}
      else if(active<Math.min(resolved+1,next)){selected=e;phase=`Result: ${e.status}`;}
      break;
    }
    const inference=enabled?(this.metadata?.inferences||[]).find(s=>active>=s.start&&active<s.end):null;
    const key=JSON.stringify([selected?.decision,phase,inference?.decision,enabled]);if(key===this.lastOverlay)return;this.lastOverlay=key;
    this.caption.textContent=selected?`#${selected.decision} · ${phase}\n${String(selected.text||'').slice(0,650)}`:'';
    if(selected?.thinking&&phase.startsWith('Result'))this.caption.textContent+=`\nRecorded thought: ${String(selected.thinking)}`;
    this.caption.scrollTop=0;
    this.thinking.textContent=inference?`Thinking / awaiting model · #${inference.decision}`:'';
    this.svg.replaceChildren();
    if(!selected)return;
    const width=this.video.videoWidth||1000,height=this.video.videoHeight||1000;
    const add=(tag,attrs,text)=>{const node=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))node.setAttribute(k,String(v));if(text)node.textContent=text;this.svg.append(node);return node;};
    for(const target of selected.targets||[]){
      const box=target.bbox,mark=target.mark;
      if(!Array.isArray(box)||box.length!==4||!box.every(v=>finite(v)&&v>=0&&v<=1)||box[2]<=box[0]||box[3]<=box[1]||!/^([PR]\d+)$/.test(mark))continue;
      const [x1,y1,x2,y2]=[box[0]*width,box[1]*height,box[2]*width,box[3]*height];
      const rect=add('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,fill:'none',stroke:'#ff3030','stroke-width':3,'vector-effect':'non-scaling-stroke','data-target':mark});
      const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=`${mark}${target.label?`: ${target.label}`:''} · target in the action observation`;rect.append(title);
      const font=Math.max(18,width*.014),pad=font*.25,labelWidth=Math.min(width,mark.length*font*.7+2*pad),labelHeight=font+2*pad;
      const x=Math.min(x1,width-labelWidth),y=Math.min(height-labelHeight,Math.max(0,y1>=labelHeight?y1-labelHeight:y1));
      add('rect',{x,y,width:labelWidth,height:labelHeight,rx:3,fill:'#d50000'});
      add('text',{x:x+pad,y:y+pad+font*.8,fill:'#fff','font-size':font,'font-family':'system-ui, sans-serif','font-weight':700},mark);
    }
    if(phase!=='Dispatching')return;
    const points=(selected.points||[]).filter(p=>p.length===2&&p.every(v=>finite(v)&&v>=0&&v<=1));
    const line=(x1,y1,x2,y2)=>add('line',{x1,y1,x2,y2,stroke:'#f7cd54','stroke-width':3,'vector-effect':'non-scaling-stroke'});
    for(const[x,y]of points){line((x-.012)*width,y*height,(x+.012)*width,y*height);line(x*width,(y-.018)*height,x*width,(y+.018)*height);}
    if(points.length===2)line(points[0][0]*width,points[0][1]*height,points[1][0]*width,points[1][1]*height);
  }
}
customElements.define('rtbench-recording-player',RecordingPlayer);

const enhanced=new Map();
function scan(){
  for(const[video,item]of enhanced){
    if(!video.isConnected||video.getAttribute('src')!==item.src){item.player.remove();enhanced.delete(video);video.style.display=item.display;video.controls=true;}
  }
  for(const video of document.querySelectorAll('video[src]')){
    if(enhanced.has(video))continue;
    const src=video.getAttribute('src');
    if(!/\/api\/jobs\/[^/]+\/trials\/[^/]+\/recording\/file(?:\?|$)/.test(src))continue;
    const player=document.createElement('rtbench-recording-player');player.setAttribute('src',src);
    enhanced.set(video,{player,src,display:video.style.display});
    video.pause();video.controls=false;video.preload='none';video.style.display='none';video.after(player);
  }
}
// Benchmark trials record no Harbor token usage or phase timing. Match Harbor's own empty-state
// text rather than its markup, and show a panel again once it has data.
const emptyPanels={Tokens:'No token data',Timing:'No timing data'};
const omitted=new Map();
function omitEmptyPanels(){
  for(const[panel,text]of omitted){
    if(!panel.isConnected||!panel.textContent.includes(text)){panel.style.display='';omitted.delete(panel);}
  }
  for(const node of document.querySelectorAll('div')){
    if(node.childElementCount)continue;
    const text=node.textContent.trim(),title=Object.keys(emptyPanels).find(key=>emptyPanels[key]===text);
    if(!title)continue;
    let panel=node.parentElement;
    while(panel&&!panel.textContent.trim().startsWith(title))panel=panel.parentElement;
    if(panel&&panel!==document.body&&!omitted.has(panel)){panel.style.display='none';omitted.set(panel,text);}
  }
}

// Scores do not depend on token use, so results carry no Harbor usage or cost, and every run uses
// the benchmark's own sandbox rather than a Harbor environment type. Omit those table columns,
// and their column-picker entries, by Harbor's own labels.
const omittedColumns=new Set(['Uncached Input Tokens','Output Tokens','Cached Input Tokens','Cost USD','Environment']);
function omitColumns(){
  for(const table of document.querySelectorAll('table')){
    const header=table.tHead?.rows[0];
    if(!header)continue;
    const omit=new Set([...header.cells].flatMap((cell,i)=>omittedColumns.has(cell.textContent.trim())?[i]:[]));
    for(const row of table.rows){
      if(row.cells.length!==header.cells.length)continue;  // Loading and empty-state rows span the table.
      for(const[i,cell]of[...row.cells].entries()){
        const hide=omit.has(i);
        if(hide!==(cell.dataset.rtbenchOmitted==='')){cell.style.display=hide?'none':'';hide?cell.dataset.rtbenchOmitted='':delete cell.dataset.rtbenchOmitted;}
      }
    }
  }
  for(const option of document.querySelectorAll('[role="option"]'))
    if(omittedColumns.has(option.textContent.trim()))option.style.display='none';
  // The native count includes the omitted fields; keep the picker label accurate.
  for(const button of document.querySelectorAll('button')){
    if(!/^\d+ columns$/.test(button.textContent.trim()))continue;
    const walker=document.createTreeWalker(button,NodeFilter.SHOW_TEXT);
    let node;
    while(node=walker.nextNode()){
      if(/\d+ columns/.test(node.nodeValue))node.nodeValue=node.nodeValue.replace(/\d+ columns/,'Columns');
    }
  }
}

const absoluteLocalPath = value => /^(?:file:\/\/|\/(?!\/)|[A-Za-z]:[\\/])/.test(value.trim());
const omittedPaths = new Map();
// Job and trial headers use different markup for their local directory. Hide the
// directory and its copy control, while retaining hosted links and artifact names.
function omitLocalRunPaths(){
  for(const [node,display] of omittedPaths){
    if(!node.isConnected||!absoluteLocalPath(node.textContent)){
      node.style.display=display;omittedPaths.delete(node);
    }
  }
  for(const label of document.querySelectorAll('main span')){
    if(label.childElementCount||label.closest('pre,code'))continue;
    const row=label.parentElement;
    if(label.textContent.trim()==='Local'&&label.nextElementSibling?.tagName==='SPAN'&&row.querySelector('button[aria-label="Copy to clipboard"]'))
      row.style.display='none';
    if(absoluteLocalPath(label.textContent)){
      const node=row.tagName==='DIV'&&row.textContent.trim()===label.textContent.trim()?row:label;
      if(!omittedPaths.has(node))omittedPaths.set(node,node.style.display);
      node.style.display='none';
    }
  }
  // Redact configuration path values in both plain and syntax-highlighted JSON.
  // Keep the field and all other configuration values visible; do not edit source data.
  for(const line of document.querySelectorAll('main pre code .line')){
    if(!/^\s*"(?:path|[\w]*_path|[\w]*_dir|benchmark_root|runner_python)"\s*:/.test(line.textContent))continue;
    const walker=document.createTreeWalker(line,NodeFilter.SHOW_TEXT);
    let node;
    while(node=walker.nextNode()){
      const text=node.nodeValue.replace(/"(?:[^"\\]|\\.)*"/g,value=>{
        try{return absoluteLocalPath(JSON.parse(value))?'"[local path]"':value;}catch{return value;}
      });
      if(text!==node.nodeValue)node.nodeValue=text;
    }
  }
}

let pending=false;
const update=()=>{pending=false;scan();omitEmptyPanels();omitColumns();omitLocalRunPaths();enhanceResults();};
new MutationObserver(()=>{if(!pending){pending=true;queueMicrotask(update);}}).observe(document.documentElement,{childList:true,subtree:true,characterData:true,attributes:true,attributeFilter:['src']});
update();
