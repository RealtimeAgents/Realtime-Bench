
  const $ = id => document.getElementById(id);
  let current = null, lastTimeline = '', seenCommentAt = 0;  // the newest comment opened from the strip
  function textElement(tag, text, className='') { const el=document.createElement(tag);el.textContent=text;if(className)el.className=className;return el; }
  const clock = seconds => '12:'+String(Math.floor(seconds/60)).padStart(2,'0')+':'+String(seconds%60).padStart(2,'0');
  const shipTo = a => a.street+', Apt '+a.unit+', '+a.city+' OR '+a.postal;
  const firstSentence = text => (text.match(/^.*?[.!?](?=\s|$)/) || [text])[0];
  function paint(state) {
    current = state.world;
    // Lea's newest comment since the last address edit, unless it was already opened from the strip.
    const fresh = current.new_comment && current.new_comment.at > seenCommentAt ? current.new_comment : null;
    $('announce-alert').hidden = !fresh;
    $('announce-title').textContent = fresh ? 'New comment from '+fresh.author+' · '+clock(Math.floor(fresh.at)) : '';
    $('announce-text').textContent = fresh ? firstSentence(fresh.text) : '';
    $('timeline-unread').hidden = !fresh;
    $('order-status').textContent = current.fulfillment;
    $('fulfillment-status').textContent = current.fulfillment;
    $('clock-now').textContent = clock(current.clock_seconds);
    const next = current.next_collection;
    $('collection-label').textContent = next ? (next.last ? 'Last collection ' : 'Next collection ') + next.time + ' in' : 'Today\'s collections have finished.';
    $('collection-countdown').textContent = next ? String(next.in_seconds) : '';
    $('collection-unit').hidden = !next;
    $('last-collection').textContent = current.last_collection;
    $('last-collection-note').hidden = !next || next.last;  // the countdown itself names the last collection then
    $('hold-details').hidden = !current.held;
    $('hold-reason-text').textContent = current.hold_reason;
    $('release-actions').hidden = !current.held || !!current.tracking;
    $('fulfillment-menu-button').hidden = current.held || !!current.tracking;
    $('customer-menu-button').hidden = !!current.tracking;
    const request = current.request;
    $('request-line').hidden = !!current.tracking;
    $('request-line').replaceChildren(...(request
      ? [textElement('strong','Warehouse request'), textElement('span','sent '+clock(request.at)+' · label prints for '+shipTo(request.shipping))]
      : [textElement('strong','On hold'), textElement('span','no request with the warehouse; releasing the hold sends the current shipping address')]));
    $('request-warning').hidden = !current.request_stale || !!current.tracking;
    $('request-warning').textContent = current.request_stale
      ? 'The shipping address differs from the request at the warehouse, which still reads '+shipTo(request.shipping)+'. Hold fulfillment to withdraw that request, then release the hold to send the current address.'
      : '';
    const a = current.shipping;
    $('shipping-address').replaceChildren(...['Mara Chen',a.street,'Apt '+a.unit,a.city+' OR '+a.postal,'United States'].map(text=>textElement('div',text)));
    $('tracking').hidden = !current.tracking;
    $('package-note').hidden = !!current.tracking;
    $('tracking').replaceChildren(...(current.tracking?[textElement('strong','Handed to carrier'),textElement('span','Tracking '+current.tracking)]:[]));
    $('feedback').textContent = current.message;
    $('post-comment').disabled = !!state.resolved || !!state.terminal_reason;
    const serialized = JSON.stringify([current.history, fresh && fresh.at]);
    if(serialized !== lastTimeline) {
      lastTimeline = serialized;
      $('timeline').replaceChildren(...current.history.map(item => {
        const li=document.createElement('li'), who=textElement('strong',item.author), at=document.createElement('time'), p=textElement('p',item.text);
        at.textContent=clock(Math.floor(item.at));
        const staffComment=item.author.includes(' · ') || (item.author==='Warehouse' && item.at===0);
        const isNew=!!fresh && item.at===fresh.at && item.author===fresh.author;
        li.className=(staffComment?'staff-comment':'system-event')+(isNew?' is-new':'');
        if(staffComment){
          const initials=item.author.startsWith('Alex')?'AM':item.author[0];
          const avatar=textElement('span',initials,'avatar timeline-avatar');avatar.setAttribute('aria-hidden','true');
          const body=document.createElement('div');body.className='comment-body';body.append(who,...(isNew?[textElement('span','New','new-tag')]:[]),at,p);li.append(avatar,body);
        }else{
          const dot=document.createElement('span');dot.className='event-dot';dot.setAttribute('aria-hidden','true');li.append(dot,who,at,p);
        }
        return li;
      }));
    }
  }
  const runtime = new RealtimeRuntime({scenario:window.__taskBootstrap.v0,episode:window.__taskBootstrap.v1,onState:paint}).start();
  function closeMenus(){for(const name of ['fulfillment','customer']){$(name+'-menu').hidden=true;$(name+'-menu-button').setAttribute('aria-expanded','false');}}
  for(const name of ['fulfillment','customer']) $(name+'-menu-button').onclick=event=>{event.stopPropagation();const opening=$(name+'-menu').hidden;closeMenus();$(name+'-menu').hidden=!opening;$(name+'-menu-button').setAttribute('aria-expanded',String(opening));};
  document.addEventListener('click',event=>{if(!event.target.closest('.menu-wrap'))closeMenus();});
  document.addEventListener('keydown',event=>{if(event.key==='Escape')closeMenus();});
  document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
  $('open-hold').onclick=()=>{closeMenus();$('hold-form').reset();$('hold-dialog').showModal();};
  $('open-address').onclick=()=>{closeMenus();for(const key of ['street','unit','city','postal'])$(key).value=current.shipping[key];$('address-dialog').showModal();};
  $('hold-form').onsubmit=async event=>{event.preventDefault();const reason=$('hold-form').querySelector('input[name=hold-reason]:checked');const result=await runtime.command({operation:'hold',items:$('hold-items').checked?['linen-cover']:[],reason:reason?reason.value:''});if(result?.ok)$('hold-dialog').close();};
  $('address-form').onsubmit=async event=>{event.preventDefault();const shipping=Object.fromEntries(['street','unit','city','postal'].map(key=>[key,$(key).value]));const result=await runtime.command({operation:'save_shipping',shipping});if(result?.ok)$('address-dialog').close();};
  $('release-hold').onclick=()=>runtime.command({operation:'release'});
  // Opening the comment from the strip clears the strip and the badge in this browser; the server clears them on the next address edit.
  $('announce-link').onclick=event=>{event.preventDefault();if(!current||!current.new_comment)return;seenCommentAt=current.new_comment.at;const entry=$('timeline').querySelector('.is-new');if(entry)entry.scrollIntoView({block:'nearest'});paint(runtime.state);};
  $('comment-form').onsubmit=async event=>{event.preventDefault();const text=$('comment-text').value.trim();if(!text)return;const result=await runtime.command({operation:'comment',text});if(result?.ok)$('comment-text').value='';};
