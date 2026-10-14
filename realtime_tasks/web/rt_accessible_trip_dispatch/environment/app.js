
'use strict';
const TASK='rt_accessible_trip_dispatch', EPISODE=window.__taskBootstrap.v0;
let world=null,picker=null,page='assignments',vehicleId='',plannerId='',view='trip',draft={},showFinal=false,filter='All',query='',lastData='',searchDraft='',filterDraft='All';
const $=s=>document.querySelector(s),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clock=n=>{if(n===null)return '—';let total=14*3600+Math.floor(n);return `${String(Math.floor(total/3600)).padStart(2,'0')}:${String(Math.floor(total%3600/60)).padStart(2,'0')}:${String(total%60).padStart(2,'0')}`};
const needs=t=>Object.entries(t.needs).filter(([k,n])=>n).map(([k,n])=>`${n} ${k==='wheelchair'?'Wheelchair':'Ambulatory'}`).join(' · ');
const assignment=id=>world.assignments.find(a=>a.id===id),trip=id=>world.trips.find(t=>t.id===id);
const aid=t=>Object.hasOwn(draft,t.id)?draft[t.id]:t.assignment;
const finalized=t=>['Completed','No Show','Cancelled'].includes(t.status);
const badge=t=>`<span class="badge ${['Cancelled','No Show'].includes(t.status)?'cancel':['Dropoff Enroute','Completed'].includes(t.status)?'moving':''}">${esc(t.status)}</span>`;
const options=(value,empty=true)=>`${empty?'<option value="">Unassigned</option>':''}${world.assignments.map(a=>`<option value="${a.id}" ${value===a.id?'selected':''}>${esc(a.vehicle)} · ${esc(a.driver)}</option>`).join('')}`;
// Dispatch rows use an inline picker instead of a native <select>: the browser's option popup is
// 24 px per row and absent from the interactables, so picking a vehicle by coordinates can land
// one row off. Each option is a normal button with its own label.
const STATUS_OPTIONS=['Scheduled','Will Call','On Hold','Pickup Enroute','Pickup Arrived','Dropoff Enroute','Dropoff Arrived','Completed','No Show','Cancelled'];
const pickerChoices=(t,kind)=>kind==='assign'?[{value:'',label:'Unassigned'},...world.assignments.map(a=>({value:a.id,label:`${a.vehicle} · ${a.driver}`}))]:STATUS_OPTIONS.map(x=>({value:x,label:x}));
const pickerLabel=(t,kind)=>kind==='assign'?(assignment(t.assignment)?`${assignment(t.assignment).vehicle} · ${assignment(t.assignment).driver}`:'Unassigned'):t.status;
function pickerMarkup(t,kind){
 const open=!!picker&&picker.trip===t.id&&picker.kind===kind,field=kind==='assign'?'Assignment':'Status',current=pickerLabel(t,kind),selected=kind==='assign'?t.assignment:t.status;
 return `<div class="picker"><button class="pick" data-pick="${kind}" data-trip="${t.id}" aria-haspopup="listbox" aria-expanded="${open}" aria-label="${field} for ${esc(t.client)} leg ${t.leg}: ${esc(current)}"><span>${esc(current)}</span><span aria-hidden="true">▾</span></button>${open?`<div class="pick-list" role="listbox" aria-label="${field} options for ${esc(t.client)} leg ${t.leg}">${pickerChoices(t,kind).map(o=>`<button class="pick-option ${o.value===selected?'current':''}" data-choose="${kind}" data-trip="${t.id}" data-value="${esc(o.value)}" aria-label="${field} for ${esc(t.client)} leg ${t.leg}: ${esc(o.label)}">${esc(o.label)}</button>`).join('')}</div>`:''}</div>`;
}
function render(){
 if(!world)return;
 $('#updated').textContent='Updated '+clock(world.time);
 $('#crumb').textContent=page==='vehicle'?'Vehicle':page[0].toUpperCase()+page.slice(1);
 document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.page===page||(page==='vehicle'&&b.dataset.page==='vehicles')));
 if(page==='assignments')renderAssignments();else if(page==='dispatch')renderDispatch();else if(page==='vehicles')renderVehicles();else renderVehicle();
 if(plannerId)renderPlanner();
}
function renderAssignments(){
 $('#content').innerHTML=`<div class="heading"><h1>Assignments</h1><span class="small">3 driver / vehicle pairs</span></div><div class="toolbar"><span>Service date</span><input aria-label="Service date" type="text" value="18/09/2026" readonly><span class="small">Show all scheduled work</span></div><div class="sheet"><table class="assignment-table"><thead><tr><th>Planner</th><th>Driver</th><th>Vehicle</th><th>Time</th><th>Activity</th></tr></thead><tbody>${world.assignments.map(a=>`<tr><td><button class="square" data-plan="${a.id}" aria-label="Open planner for ${esc(a.driver)}">⌁</button></td><td>${esc(a.driver)}</td><td><button class="link" data-vehicle="${a.id}">${esc(a.vehicle)}</button></td><td>${a.shift}</td><td><div class="activity">${world.trips.filter(t=>t.assignment===a.id).map(t=>`<i class="${['No Show','Cancelled'].includes(t.status)?'cancelled':t.status==='Completed'?'final':''}" title="${esc(t.client+' · '+t.status)}"></i>`).join('')}</div></td></tr>`).join('')}</tbody></table><div class="legend"><span><i class="dot"></i>Pending</span><span><i class="dot green"></i>Completed</span><span><i class="dot red"></i>Cancelled / No Show</span></div></div>`;
}
function renderVehicles(){
 $('#content').innerHTML=`<div class="heading"><h1>Vehicles</h1><span class="small">Active fleet</span></div><div class="sheet"><table><thead><tr><th>Name</th><th>Class</th><th>Plate</th><th>Capacity configuration</th><th>Base</th></tr></thead><tbody>${world.assignments.map(a=>`<tr><td><button class="link" data-vehicle="${a.id}">${esc(a.vehicle)}</button></td><td>Van</td><td>${esc(a.plate)}</td><td>${esc(a.configuration)}</td><td>Riverside</td></tr>`).join('')}</tbody></table></div>`;
}
function renderVehicle(){
 const a=assignment(vehicleId);if(!a)return;
 $('#content').innerHTML=`<div class="heading"><h1>Vehicle · ${esc(a.vehicle)}</h1><button id="back-vehicle">Back to assignments</button></div><div class="profile"><div class="profile-grid"><section><h2>Name and Description</h2><div class="field"><label>Name</label>${esc(a.vehicle)}</div><div class="field"><label>Class</label>Van</div><div class="field"><label>Make</label>Ford</div><div class="field"><label>Color</label>White</div></section><section><h2>License Plate and Class</h2><div class="field"><label>Plate number</label>${esc(a.plate)}</div><div class="field"><label>License class</label>D</div><h2 style="margin-top:24px">Capacity</h2><div class="field"><label>Configuration</label>${esc(a.configuration)}</div><table><thead><tr><th>Wheelchair</th><th>Ambulatory</th></tr></thead><tbody><tr><td>${a.capacity.wheelchair}</td><td>${a.capacity.ambulatory}</td></tr></tbody></table></section><section><h2>Location</h2><div class="field"><label>Base location</label>Riverside depot</div><div class="field"><label>Zone</label>Central</div><h2 style="margin-top:24px">Service Information</h2><div class="field"><label>Active status</label>Active</div></section></div><div class="small" style="margin-top:25px">Vehicle record · Fleet administration</div></div>`;
}
function renderDispatch(){
 let rows=world.trips.filter(t=>(filter==='All'||(filter==='Active'?!finalized(t):t.status===filter))&&(`${t.client} ${t.id} ${t.start} ${t.end}`.toLowerCase().includes(query.toLowerCase()))).sort((a,b)=>a.id.localeCompare(b.id));
 $('#content').innerHTML=`<div class="heading"><h1>Dispatch</h1><span class="small">${world.trips.length} trips · Riverside</span></div><div class="toolbar"><span>Filter trips</span><input value="18/09/2026" readonly aria-label="Dispatch date" style="width:128px"><label>Status <select id="status-filter">${['All','Active','Scheduled','Will Call','No Show','Cancelled','Completed'].map(x=>`<option ${filter===x?'selected':''}>${x}</option>`).join('')}</select></label><input id="trip-search" type="search" aria-label="Search client or address" placeholder="Search client or address" value="${esc(query)}"><button id="filter-trips">Filter</button></div><div class="dispatch">${rows.map(t=>`<div class="block-label">${t.pu===null?'Will Call':clock(t.pu)} &nbsp; · &nbsp; Trip ${t.id} — Leg ${t.leg}${t.pair?' of 2':''}</div><article class="trip-row ${t.status==='Cancelled'?'cancelled':''}" data-trip-row="${t.id}"><div>${pickerMarkup(t,'assign')}${pickerMarkup(t,'status')}</div><div><div class="client">${esc(t.client)}</div><div class="needs">${needs(t)}</div>${t.pair?`<span class="small">Other leg: ${t.pair}</span>`:''}</div><div class="addresses"><div>PU: <b>${clock(t.pu)}</b> &nbsp; ${esc(t.start)}</div><div>DO: <b>${clock(t.drop)}</b> &nbsp; ${esc(t.end)}</div></div><div class="note">${esc(t.note||'No dispatch note')}<br><span class="small">${esc(t.status_by)} · ${clock(t.updated)}</span></div></article>`).join('')||'<div class="empty">No trips match these filters.</div>'}</div>`;
}
const coords={'6 River Walk':[115,90],'40 Cedar Centre':[385,90],'12 Quay Street':[170,215],'21 Orchard Clinic':[380,250],'18 Quay Street':[200,315],'8 Quay Street':[170,170],'3 Linden Court':[95,340],'16 River Centre':[300,350]};
function mapMarkup(rows){
 const pts=rows.flatMap(t=>[{p:coords[t.start],at:t.pu},{p:coords[t.end],at:t.drop}]).sort((a,b)=>(a.at??9999)-(b.at??9999)).map(s=>s.p).filter(Boolean);
 return `<svg viewBox="0 0 500 420" aria-label="Local route map" role="img"><rect width="500" height="420" fill="#eef1e7"/><path d="M0 145 Q150 105 260 180 T500 205 L500 234 Q370 240 248 206 T0 177Z" fill="#bedce6"/><g stroke="#fff" stroke-width="19"><path d="M65 0V420M170 0V420M300 0V420M400 0V420M0 90H500M0 250H500M0 350H500"/></g><g stroke="#d0d7cf" stroke-width="2"><path d="M65 0V420M170 0V420M300 0V420M400 0V420M0 90H500M0 250H500M0 350H500"/></g><g fill="#96af86"><rect x="318" y="112" width="64" height="38" rx="6"/><rect x="82" y="269" width="65" height="52" rx="7"/></g><g font-size="11" fill="#718072"><text x="178" y="42">QUAY STREET</text><text x="313" y="79">CEDAR ROAD</text><text x="85" y="380">LINDEN COURT</text><text x="303" y="335">RIVER CENTRE</text></g>${rows.map(t=>{const p=coords[t.start],q=coords[t.end];return `<polyline points="${p[0]},${p[1]} ${q[0]},${p[1]} ${q[0]},${q[1]}" fill="none" stroke="${Object.hasOwn(draft,t.id)?'#378aad':'#929c9e'}" stroke-width="4" opacity=".75"/>`}).join('')}${pts.map((p,i)=>`<circle cx="${p[0]}" cy="${p[1]}" r="12" fill="#357fa5"/><text x="${p[0]}" y="${p[1]+4}" text-anchor="middle" font-size="11" fill="white">${i+1}</text>`).join('')}</svg><div class="map-key">Route key<br><span></span>Current route<br><span class="new"></span>New route<br>● Pickup / drop-off</div>`;
}
function renderPlanner(){
 const a=assignment(plannerId), rows=world.trips.filter(t=>aid(t)===a.id&&(showFinal||!finalized(t))).sort((a,b)=>(a.pu??9999)-(b.pu??9999));
 const unassigned=world.trips.filter(t=>!aid(t)&&!finalized(t));
 const stops=rows.flatMap(t=>[{t,at:t.pu,label:'Pickup',address:t.start},{t,at:t.drop,label:'Drop-off',address:t.end}]).sort((a,b)=>(a.at??9999)-(b.at??9999));
 $('#planner').innerHTML=`<section class="planner" aria-label="Assignment Planner"><div class="planner-head"><strong>Assignment Planner · Fri 18 September</strong><button id="previous-assignment" aria-label="Previous assignment">‹</button><select id="planner-assignment" aria-label="Planner assignment">${options(a.id,false)}</select><button id="next-assignment" aria-label="Next assignment">›</button></div><div class="planner-main"><div class="plan-left"><div class="plan-tabs"><button data-plan-view="trip" class="${view==='trip'?'active':''}">Trip View</button><button data-plan-view="stop" class="${view==='stop'?'active':''}">Stop View</button><label><input id="show-final" type="checkbox" ${showFinal?'checked':''}> Show finalized trips</label></div><div class="plan-title">${esc(a.vehicle)} · ${esc(a.driver)} &nbsp; ${a.shift} &nbsp; (${rows.length})</div>${view==='trip'?rows.map((t,i)=>`<div class="plan-trip"><span class="number">${i+1}</span>${badge(t)}<div><b>${esc(t.client)}</b> <span class="small">${t.id} · Leg ${t.leg}</span><div class="needs">${needs(t)}</div><div class="addresses"><div>PU: ${clock(t.pu)} ${esc(t.start)}</div><div>DO: ${clock(t.drop)} ${esc(t.end)}</div></div></div><button class="square" data-remove="${t.id}" aria-label="Remove ${esc(t.client)} leg ${t.leg} from assignment" ${finalized(t)?'disabled':''}>−</button></div>`).join(''):stops.map((s,i)=>`<div class="stop-line"><span class="number">${i+1}</span><b>${clock(s.at)}</b><div>${s.label}: <strong>${esc(s.t.client)}</strong><div class="small">${esc(s.address)} · ${needs(s.t)}</div></div>${badge(s.t)}</div>`).join('')}${rows.length?'':'<div class="empty">No trips on this assignment.</div>'}</div><div class="map">${mapMarkup(rows)}</div></div><div class="unassigned"><table><thead><tr><th>Add</th><th>Trip ID</th><th>Client</th><th>Pickup time</th><th>Pickup address</th><th>Drop-off address</th><th>Seating needs</th><th>Mode</th></tr></thead><tbody>${unassigned.map(t=>`<tr><td><button class="square" data-add="${t.id}" aria-label="Add ${esc(t.client)} leg ${t.leg}">+</button></td><td>${t.id}</td><td>${esc(t.client)} <span class="small">(Leg ${t.leg})</span></td><td>${clock(t.pu)}</td><td>${esc(t.start)}</td><td>${esc(t.end)}</td><td>${needs(t)}</td><td>Ambulette</td></tr>`).join('')}</tbody></table>${unassigned.length?'':'<div class="empty">No unassigned trips.</div>'}</div><div class="planner-footer"><span class="badge">Unassigned Trips (${unassigned.length})</span><span class="save-note">${Object.keys(draft).length?'Unsaved assignment changes':''}</span><div class="right"><button id="cancel-planner">Cancel</button> <button id="save-planner" class="primary">▣ &nbsp; Close</button></div></div></section>`;
}
async function command(c){const r=await runtime.command(c);if(r?.ok)runtime.toast('Dispatch',r.message||'Saved.');return r;}
function openPlanner(id){plannerId=id;draft={};view='trip';showFinal=false;$('#shell').inert=true;renderPlanner();}
function closePlanner(){plannerId='';draft={};$('#planner').innerHTML='';$('#shell').inert=false;render();}
document.addEventListener('click',async e=>{
 const b=e.target.closest('button');if(!b)return;
 if(b.dataset.pick){picker=(picker&&picker.trip===b.dataset.trip&&picker.kind===b.dataset.pick)?null:{trip:b.dataset.trip,kind:b.dataset.pick};render();return;}
 if(b.dataset.choose){const kind=b.dataset.choose,id=b.dataset.trip,value=b.dataset.value;picker=null;await command(kind==='assign'?{op:'assign',trip:id,assignment:value}:{op:'status',trip:id,status:value});render();return;}
 if(b.dataset.page){page=b.dataset.page;render();}
 else if(b.dataset.plan)openPlanner(b.dataset.plan);
 else if(b.dataset.vehicle){vehicleId=b.dataset.vehicle;page='vehicle';render();}
 else if(b.id==='back-vehicle'){page='assignments';render();}
 else if(b.dataset.add){draft[b.dataset.add]=plannerId;renderPlanner();}
 else if(b.dataset.remove){draft[b.dataset.remove]='';renderPlanner();}
 else if(b.dataset.planView){view=b.dataset.planView;renderPlanner();}
 else if(b.id==='cancel-planner')closePlanner();
 else if(b.id==='save-planner'){const changes=Object.entries(draft).filter(([id,a])=>trip(id).assignment!==a).map(([id,a])=>({trip:id,assignment:a}));if(!changes.length||(await command({op:'plan',changes}))?.ok)closePlanner();}
 else if(['previous-assignment','next-assignment'].includes(b.id)){let i=world.assignments.findIndex(a=>a.id===plannerId);plannerId=world.assignments[(i+(b.id==='next-assignment'?1:2))%3].id;renderPlanner();}
 else if(b.id==='filter-trips'){filter=filterDraft=$('#status-filter').value;query=searchDraft=$('#trip-search').value;renderDispatch();}
});
document.addEventListener('change',async e=>{
 const el=e.target;
 if(el.dataset.assign){await command({op:'assign',trip:el.dataset.assign,assignment:el.value});render();}
 else if(el.dataset.status){await command({op:'status',trip:el.dataset.status,status:el.value});render();}
 else if(el.id==='planner-assignment'){plannerId=el.value;renderPlanner();}
 else if(el.id==='show-final'){showFinal=el.checked;renderPlanner();}
 else if(el.id==='status-filter')filterDraft=el.value;
});
document.addEventListener('input',e=>{if(e.target.id==='trip-search')searchDraft=e.target.value;});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&picker){picker=null;render();}if(e.key==='Escape'&&plannerId)closePlanner();if(e.key==='Enter'&&e.target.id==='trip-search')$('#filter-trips').click();});
const runtime=new RealtimeRuntime({scenario:TASK,episode:EPISODE,onState:s=>{world=s.world;$('#updated').textContent='Updated '+clock(world.time);const data=JSON.stringify([world.assignments,world.trips]);if(data!==lastData){const focus=document.activeElement?.id;lastData=data;render();if(page==='dispatch'){if($('#trip-search'))$('#trip-search').value=searchDraft;if($('#status-filter'))$('#status-filter').value=filterDraft;if(focus==='trip-search')$('#trip-search')?.focus();}}}}).start();
