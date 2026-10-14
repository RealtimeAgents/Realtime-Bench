"""Read-only DOM interactable extractor evaluated over CDP by the host.

The expression evaluates geometry and accessibility attributes but never calls application
functions or dispatches input.
"""

import json

_EXTRACT = r"""
(spec => {
  const clean = value => String(value ?? "").replace(/\s+/g, " ").trim();
  const cssEscape = value => (window.CSS && CSS.escape) ? CSS.escape(value) : String(value).replace(/["\\]/g, "\\$&");
  const rules = Array.isArray(spec.elements) ? spec.elements : [];
  const canvasRules = Array.isArray(spec.canvases) ? spec.canvases : [];
  const actionableRoles = new Set(["button","link","checkbox","radio","switch","slider","spinbutton","textbox","combobox","listbox","menuitem","menuitemcheckbox","menuitemradio","option","tab","treeitem"]);
  const candidateSelector = [
    "button", "a[href]", "input:not([type=hidden])", "textarea", "select", "summary",
    "[contenteditable=true]", "[contenteditable='']", "[role]", "[tabindex]", "[data-tool-interactive]",
    "[draggable=true]", "[data-tool-drop-target]", "[onclick]", "[onmousedown]",
    "[onpointerdown]", "[ontouchstart]"
  ].join(",");
  const semanticContainerSelector = "article,tr,li,[role=row],[role=group]";

  function allRoots(root, output=[]) {
    output.push(root);
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) allRoots(el.shadowRoot, output);
    }
    return output;
  }

  function parentElementAcrossRoots(el) {
    if (el?.parentElement) return el.parentElement;
    const root = el?.getRootNode?.();
    return root?.host && root.host.nodeType === 1 ? root.host : null;
  }

  function hiddenByTree(el) {
    const view = el.ownerDocument?.defaultView || window;
    for (let node=el; node && node.nodeType===1; node=parentElementAcrossRoots(node)) {
      const style = view.getComputedStyle(node);
      if (node.hidden || node.inert || style.display === "none" ||
          style.visibility === "hidden" || style.visibility === "collapse" ||
          Number(style.opacity || 1) <= 0) return true;
      if (node.tagName === "DETAILS" && !node.open && el !== node.querySelector("summary") && !node.querySelector("summary")?.contains(el)) return true;
    }
    return false;
  }

  function visibleRect(el) {
    if (!el.isConnected || hiddenByTree(el)) return null;
    const rects = [...el.getClientRects()].filter(r => r.width > .5 && r.height > .5);
    if (!rects.length) return null;
    const view = el.ownerDocument?.defaultView || window;
    const vv = view.visualViewport;
    const left = vv ? vv.offsetLeft : 0, top = vv ? vv.offsetTop : 0;
    const right = left + (vv ? vv.width : view.innerWidth), bottom = top + (vv ? vv.height : view.innerHeight);
    const rect = rects.find(r => r.right > left && r.bottom > top && r.left < right && r.top < bottom);
    if (!rect) return null;
    return {
      left: Math.max(left, rect.left), top: Math.max(top, rect.top),
      right: Math.min(right, rect.right), bottom: Math.min(bottom, rect.bottom)
    };
  }

  function labelledbyText(el) {
    const labelled = clean(el.getAttribute("aria-labelledby"));
    if (!labelled) return "";
    const root = el.getRootNode?.();
    return labelled.split(/\s+/).map(id => clean(
      root?.getElementById?.(id)?.textContent || el.ownerDocument?.getElementById(id)?.textContent
    )).filter(Boolean).join(" ");
  }

  function ariaName(el) {
    const labelled = labelledbyText(el);
    if (labelled) return labelled;
    const direct = clean(el.getAttribute("aria-label"));
    if (direct) return direct;
    if (el.labels && el.labels.length) {
      const text = [...el.labels].map(label => clean(label.textContent)).filter(Boolean).join(" ");
      if (text) return text;
    }
    return clean(el.getAttribute("title") || el.getAttribute("alt") || el.getAttribute("placeholder") ||
                 ((el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT") ? el.value : "") ||
                 el.innerText || el.textContent);
  }

  function semanticContainer(el) {
    for (let node=parentElementAcrossRoots(el); node; node=parentElementAcrossRoots(node)) {
      try { if (node.matches(semanticContainerSelector)) return node; } catch (_) {}
    }
    return null;
  }

  function semanticContext(el) {
    const container = semanticContainer(el);
    if (!container) return "";
    const parts = [];
    const doc = el.ownerDocument;
    const showText = doc.defaultView?.NodeFilter?.SHOW_TEXT ?? 4;
    const walker = doc.createTreeWalker(container, showText);
    for (let textNode=walker.nextNode(); textNode; textNode=walker.nextNode()) {
      const value = clean(textNode.nodeValue);
      const owner = textNode.parentElement;
      if (!value || !owner || el.contains(owner) || hiddenByTree(owner) || !visibleRect(owner)) continue;
      let belongsToOtherControl = false;
      for (let node=owner; node && node !== container; node=parentElementAcrossRoots(node)) {
        if (node === el || el.contains(node)) {
          belongsToOtherControl = true;
          break;
        }
        try {
          if (node.matches(candidateSelector)) {
            belongsToOtherControl = true;
            break;
          }
        } catch (_) {}
      }
      if (!belongsToOtherControl && parts[parts.length - 1] !== value) parts.push(value);
    }
    return clean(parts.join(" ")).slice(0, 240);
  }

  function semanticLabel(label, context) {
    if (!context) return label;
    const comparable = value => clean(value).toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
    const labelKey = comparable(label), contextKey = comparable(context);
    if (!labelKey) return context;
    if (!contextKey || labelKey.includes(contextKey)) return label;
    const labelTokens = new Set(labelKey.split(/\s+/).filter(Boolean));
    const contextTokens = [...new Set(contextKey.split(/\s+/).filter(Boolean))];
    const covered = contextTokens.filter(token => labelTokens.has(token)).length;
    // A page-authored name such as "Reserve — 7:30 PM — Dining room — Capacity 4" is already
    // contextual. Do not duplicate nearly all of it merely because the card also says Available.
    if (contextTokens.length && covered / contextTokens.length >= .65) return label;
    return `${label} — ${context}`.slice(0, 320);
  }

  function roleOf(el) {
    const explicit = clean(el.getAttribute("role")).toLowerCase();
    if (explicit) return explicit;
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button" || tag === "summary") return "button";
    if (tag === "textarea") return "textbox";
    if (tag === "select") return el.multiple ? "listbox" : "combobox";
    if (tag === "input") {
      const type = (el.type || "text").toLowerCase();
      if (["button","submit","reset","image","file"].includes(type)) return "button";
      if (["checkbox","radio","range"].includes(type)) return type === "range" ? "slider" : type;
      return "textbox";
    }
    return "control";
  }

  function isDragSource(el) {
    // Only an explicit draggable="true" (also set by `el.draggable = true`) marks a drag source.
    // Chrome reports links and images as draggable by default; reading the property would list
    // every link as drag-only and hide its click.
    return el.getAttribute("draggable") === "true";
  }

  function isDropTarget(el) {
    const value = el.getAttribute("data-tool-drop-target");
    return value !== null && value !== "false";
  }

  function actionsFor(el, role) {
    const tag = el.tagName.toLowerCase(), type = (el.type || "").toLowerCase();
    const actions = [];
    if (isDragSource(el)) actions.push("drag");
    if (isDropTarget(el)) actions.push("drop");
    if (!actions.length) {
      if (role === "textbox" || el.isContentEditable) actions.push("click", "type");
      else if (tag === "select" || role === "combobox" || role === "listbox") actions.push("click", "select");
      else if (role === "slider") actions.push("click", "key");
      else actions.push("click");
      if (tag === "input" && ["text","search","email","url","tel","password","number","date","time"].includes(type) && !actions.includes("type")) actions.push("type");
    }
    return [...new Set(actions)].sort();
  }

  function disabledReasons(el, role, modal) {
    const reasons = [];
    if (el.disabled || el.closest("fieldset[disabled]")) reasons.push("disabled");
    if (el.getAttribute("aria-disabled") === "true") reasons.push("aria_disabled");
    if (el.closest("[inert]")) reasons.push("inert");
    if ((el.ownerDocument?.defaultView || window).getComputedStyle(el).pointerEvents === "none") reasons.push("pointer_events_none");
    if (modal && !modal.contains(el)) reasons.push("blocked_by_modal");
    if ((role === "textbox" || el.isContentEditable) && (el.readOnly || el.getAttribute("aria-readonly") === "true")) reasons.push("readonly");
    return reasons;
  }

  function hitPoint(el, rect, preferSelf=false) {
    const doc = el.ownerDocument;
    const view = doc.defaultView || window;
    const insetX = Math.min(8, rect.right - rect.left), insetY = Math.min(8, rect.bottom - rect.top);
    const midX = (rect.left + rect.right) / 2, midY = (rect.top + rect.bottom) / 2;
    const points = [
      [midX, midY],
      [rect.left + insetX, rect.top + insetY], [rect.right - insetX, rect.top + insetY],
      [rect.left + insetX, rect.bottom - insetY], [rect.right - insetX, rect.bottom - insetY],
      [rect.left + insetX, midY], [rect.right - insetX, midY],
      [midX, rect.top + insetY], [midX, rect.bottom - insetY]
    ];
    if (preferSelf) {
      for (const [x,y] of points) {
        const hit = doc.elementsFromPoint(x, y)
          .find(node => view.getComputedStyle(node).pointerEvents !== "none");
        if (hit === el) return [x,y];
      }
      return null;
    }
    for (const [x,y] of points) {
      const stack = doc.elementsFromPoint(x, y);
      const hit = stack.find(node => view.getComputedStyle(node).pointerEvents !== "none");
      if (hit && (hit === el || el.contains(hit) || hit.contains(el) || hit.closest("label")?.control === el)) return [x,y];
    }
    return null;
  }

  const desktopSize = Array.isArray(spec.__desktop_size) ? spec.__desktop_size : [screen.width, screen.height];
  const screenScaleX = screen.width ? Number(desktopSize[0]) / screen.width : 1;
  const screenScaleY = screen.height ? Number(desktopSize[1]) / screen.height : 1;
  const contentLeft = window.screenX + Math.max(0, (window.outerWidth - window.innerWidth) / 2);
  const contentTop = window.screenY + Math.max(0, window.outerHeight - window.innerHeight);
  const topRectToDesktop = rect => [
    Math.round((contentLeft + rect.left) * screenScaleX),
    Math.round((contentTop + rect.top) * screenScaleY),
    Math.round((contentLeft + rect.right) * screenScaleX),
    Math.round((contentTop + rect.bottom) * screenScaleY)
  ];
  const topPointToDesktop = point => point ? [
    Math.round((contentLeft + point[0]) * screenScaleX),
    Math.round((contentTop + point[1]) * screenScaleY)
  ] : null;

  const contexts = [];
  let inaccessibleFrames = 0;
  function addContext(doc, path="top", offsetX=0, offsetY=0, scaleX=1, scaleY=1) {
    const view = doc?.defaultView;
    if (!doc || !view) return;
    const context = {doc, view, path, offsetX, offsetY, scaleX, scaleY};
    contexts.push(context);
    const seenFrames = new Set();
    for (const root of allRoots(doc)) {
      for (const frame of root.querySelectorAll("iframe,frame")) {
        if (seenFrames.has(frame) || hiddenByTree(frame)) continue;
        seenFrames.add(frame);
        if (!visibleRect(frame)) continue;
        const frameRect = frame.getBoundingClientRect();
        let childDoc = null, childView = null;
        try {
          childDoc = frame.contentDocument;
          childView = frame.contentWindow;
          if (!childDoc || !childView || childDoc.location.origin !== doc.location.origin) {
            inaccessibleFrames += 1;
            continue;
          }
        } catch (_) {
          inaccessibleFrames += 1;
          continue;
        }
        const boxWidth = Number(frame.offsetWidth || frameRect.right - frameRect.left) || 1;
        const boxHeight = Number(frame.offsetHeight || frameRect.bottom - frameRect.top) || 1;
        const elementScaleX = (frameRect.right - frameRect.left) / boxWidth;
        const elementScaleY = (frameRect.bottom - frameRect.top) / boxHeight;
        const viewportWidth = Number(childView.innerWidth || frame.clientWidth) || 1;
        const viewportHeight = Number(childView.innerHeight || frame.clientHeight) || 1;
        const childScaleX = scaleX * elementScaleX * Number(frame.clientWidth || viewportWidth) / viewportWidth;
        const childScaleY = scaleY * elementScaleY * Number(frame.clientHeight || viewportHeight) / viewportHeight;
        const childOffsetX = offsetX + scaleX * (frameRect.left + Number(frame.clientLeft || 0) * elementScaleX);
        const childOffsetY = offsetY + scaleY * (frameRect.top + Number(frame.clientTop || 0) * elementScaleY);
        addContext(childDoc, `${path}/frame-${seenFrames.size}`, childOffsetX, childOffsetY, childScaleX, childScaleY);
      }
    }
  }
  addContext(document);

  const toTopRect = (context, rect) => ({
    left: context.offsetX + rect.left * context.scaleX,
    top: context.offsetY + rect.top * context.scaleY,
    right: context.offsetX + rect.right * context.scaleX,
    bottom: context.offsetY + rect.bottom * context.scaleY
  });
  const toTopPoint = (context, point) => point ? [
    context.offsetX + point[0] * context.scaleX,
    context.offsetY + point[1] * context.scaleY
  ] : null;
  const toDesktopRect = (context, rect) => topRectToDesktop(toTopRect(context, rect));
  const toDesktopPoint = (context, point) => topPointToDesktop(toTopPoint(context, point));

  function matchingRule(el) {
    for (const rule of rules) {
      try { if (el.matches(rule.selector)) return rule; } catch (_) {}
    }
    return null;
  }

  function matchingCanvasRule(el) {
    for (const rule of canvasRules) {
      try { if (el.matches(rule.selector)) return rule; } catch (_) {}
    }
    return null;
  }

  function expandTemplate(template, el) {
    return clean(String(template || "").replace(/\{([^{}]+)\}/g, (_, name) =>
      el.getAttribute(name) ?? ""
    ));
  }

  function customIdentity(el, rule) {
    if (rule) {
      if (rule.id_template) {
        const value = expandTemplate(rule.id_template, el);
        if (value) return value;
      }
      const base = rule.id_attribute ? el.getAttribute(rule.id_attribute) : el.getAttribute("data-tool-id");
      const entity = rule.entity_attribute ? el.getAttribute(rule.entity_attribute) : el.getAttribute("data-entity-id");
      if (base && entity) return `${base}:${entity}`;
      if (entity) return entity;
      if (base) return base;
    }
    const base = el.getAttribute("data-tool-id"), entity = el.getAttribute("data-entity-id");
    if (base && entity) return `${base}:${entity}`;
    if (entity) return entity;
    if (base) return base;
    if (el.id) return el.id;
    return null;
  }

  function scopedIdentity(context, value) {
    return context.path === "top" ? value : `${context.path}:${value}`;
  }

  const candidates = [];
  for (const context of contexts) {
    const seen = new Set();
    for (const root of allRoots(context.doc)) {
      const add = el => {
        if (!seen.has(el)) {
          seen.add(el);
          candidates.push({el, context, rule: matchingRule(el)});
        }
      };
      for (const el of root.querySelectorAll(candidateSelector)) add(el);
      // Task rules are authoritative declarations of interaction. A selector must be able to
      // introduce an element, rather than merely decorating generic HTML controls.
      for (const rule of rules) {
        try { for (const el of root.querySelectorAll(rule.selector)) add(el); } catch (_) {}
      }
    }
  }

  const actionable = [], unavailable = [], canvases = [];
  let order = 0;
  for (const candidate of candidates) {
    const {el, context} = candidate;
    const rule = candidate.rule || matchingRule(el);
    const role = clean(rule?.role || roleOf(el)).toLowerCase();
    const tag = el.tagName.toLowerCase();
    const tabindexAttribute = el.getAttribute("tabindex");
    const tabindex = tabindexAttribute === null ? Number.NaN : Number(tabindexAttribute);
    const dragDrop = isDragSource(el) || isDropTarget(el);
    const eventBound = ["onclick","onmousedown","onpointerdown","ontouchstart"]
      .some(name => el.hasAttribute(name));
    if ((tag === "div" || tag === "span") && !rule && !actionableRoles.has(role)
        && !(Number.isFinite(tabindex) && tabindex >= 0)
        && !el.hasAttribute("data-tool-interactive") && !dragDrop && !eventBound) continue;
    if (el.hasAttribute("role") && !rule && !actionableRoles.has(role)
        && !(Number.isFinite(tabindex) && tabindex >= 0) && !dragDrop && !eventBound) continue;
    const rect = visibleRect(el);
    if (!rect) continue;
    let actions = Array.isArray(rule?.actions) ? [...rule.actions] : actionsFor(el, role);
    const modalCandidates = [...context.doc.querySelectorAll("dialog[open], [role=dialog][aria-modal=true], [aria-modal=true]")]
      .filter(node => visibleRect(node));
    const activeModal = modalCandidates.length ? modalCandidates[modalCandidates.length - 1] : null;
    const reasons = disabledReasons(el, role, activeModal);
    if (reasons.includes("readonly")) actions = actions.filter(action => action !== "type");
    const point = hitPoint(el, rect, el.getAttribute("data-tool-drop-target") === "append");
    if (!point) reasons.push("occluded");
    const label = clean(rule?.label || (rule?.label_template && expandTemplate(rule.label_template, el)) ||
      (rule?.label_attribute && el.getAttribute(rule.label_attribute)) || ariaName(el));
    const contextText = semanticContext(el);
    // Text-observation models (no screenshot) see rendered content, CSS appearance and field
    // state instead of pixels, never hidden application state. Screenshot models omit these fields.
    let textObservation = {};
    if (spec.__text_observation) {
      const style = context.doc.defaultView.getComputedStyle(el);
      const editableField = role === "textbox" || el.isContentEditable;
      const nativeOptions = tag === "select" ? Array.from(el.options).map((option, index) => ({
        index, label: clean(option.label || option.textContent), selected: option.selected,
        disabled: !!option.disabled || !!option.closest("optgroup[disabled]"),
        hidden: option.hidden || context.doc.defaultView.getComputedStyle(option).display === "none"
      })) : [];
      textObservation = {
        visible_content: {source: "rendered_dom_and_accessibility",
          text: clean(el.innerText || "").slice(0, 240),
          image_labels: [...el.querySelectorAll("img")].filter(node => visibleRect(node))
            .map(node => clean(node.alt || node.title)).filter(Boolean),
          background_color: style.backgroundColor, foreground_color: style.color,
          border: style.border},
        select_options: nativeOptions, select_multiple: tag === "select" && el.multiple,
        input_constraints: editableField ? {
          placeholder: el.getAttribute("placeholder") || "",
          pattern: el.getAttribute("pattern") || "",
          min: el.getAttribute("min"), max: el.getAttribute("max"),
          step: el.getAttribute("step"), maxlength: el.getAttribute("maxlength"),
          options: el.list ? Array.from(el.list.options).filter(o => !o.disabled)
            .map(o => ({value: o.value, label: o.label})) : []
        } : {},
        focused: el === context.doc.activeElement,
        value: (editableField || tag === "select") && (el.type || "").toLowerCase() !== "password"
          ? clean(tag === "select" ? [...el.selectedOptions].map(o => o.label).join(", ") :
            (el.isContentEditable ? el.innerText : el.value)).slice(0, 160) : "",
        text_selected: typeof el.selectionStart === "number" &&
          typeof el.selectionEnd === "number" &&
          el.selectionStart === 0 && el.selectionEnd >= String(el.value || "").length &&
          String(el.value || "").length > 0
      };
    }
    const item = {
      id: scopedIdentity(context, customIdentity(el, rule) || `dom-${String(order + 1).padStart(4,"0")}`),
      dom_order: order++, source: "dom", frame_path: context.path, role,
      label, context: contextText,
      semantic_label: semanticLabel(label, contextText),
      tag: el.tagName.toLowerCase(), input_type: (el.type || "").toLowerCase(),
      ...textObservation,
      bbox: toDesktopRect(context, rect), click_point: toDesktopPoint(context, point), actions,
      enabled: !reasons.some(reason => ["disabled","aria_disabled","inert","pointer_events_none","blocked_by_modal","occluded"].includes(reason)),
      readonly: !!el.readOnly || el.getAttribute("aria-readonly") === "true",
      editable: role === "textbox" || el.isContentEditable,
      checked: typeof el.checked === "boolean" ? !!el.checked : (el.getAttribute("aria-checked") === null ? null : el.getAttribute("aria-checked") === "true"),
      selected: el.getAttribute("aria-selected") === null ? (el.getAttribute("aria-pressed") === null ? null : el.getAttribute("aria-pressed") === "true") : el.getAttribute("aria-selected") === "true",
      expanded: el.getAttribute("aria-expanded") === null ? null : el.getAttribute("aria-expanded") === "true",
      modal_scope: !!activeModal && activeModal.contains(el), occluded: reasons.includes("occluded"),
      unavailable_reasons: [...new Set(reasons)].sort()
    };
    (item.enabled ? actionable : unavailable).push(item);
  }

  function readPath(value, path) {
    if (!path) return undefined;
    for (const part of String(path).split(".")) {
      if (value == null) return undefined;
      value = value[part];
    }
    return value;
  }

  let canvasIndex = 0;
  for (const frameContext of contexts) {
    const seen = new Set();
    for (const root of allRoots(frameContext.doc)) {
      for (const canvas of root.querySelectorAll("canvas")) {
        if (seen.has(canvas)) continue;
        seen.add(canvas);
        const rect = visibleRect(canvas);
        if (!rect) continue;
        canvasIndex += 1;
        const point = hitPoint(canvas, rect);
        const rule = matchingCanvasRule(canvas);
        const label = clean(rule?.label || ariaName(canvas) || "Canvas");
        const semanticContextText = semanticContext(canvas);
        const canvasId = scopedIdentity(frameContext,
          canvas.getAttribute("data-tool-id") || canvas.id || `canvas-${canvasIndex}`);
        const coverage = clean(rule?.coverage || spec.canvas_default_coverage || "unsupported");
        const item = {
          id: canvasId,
          dom_order: order++, source: "dom", frame_path: frameContext.path,
          role: "canvas", label, context: semanticContextText,
          semantic_label: semanticLabel(label, semanticContextText),
          tag: "canvas", bbox: toDesktopRect(frameContext, rect),
          click_point: toDesktopPoint(frameContext, point),
          css_size: [Math.round(rect.right - rect.left), Math.round(rect.bottom - rect.top)],
          backing_size: [Number(canvas.width || 0), Number(canvas.height || 0)],
          device_pixel_ratio: Number(frameContext.view.devicePixelRatio || 1),
          actions: Array.isArray(rule?.actions) ? [...rule.actions] : (point ? ["click"] : []),
          enabled: !!point, occluded: !point, unavailable_reasons: point ? [] : ["occluded"],
          internal_target_coverage: coverage,
          interaction: clean(rule?.interaction || "")
        };

        if (rule?.grid) {
          const columns = rule.grid.column_labels || [], rows = rule.grid.row_labels || [];
          const insetX = Number(rule.grid.inset_x || 0), insetY = Number(rule.grid.inset_y || 0);
          const x0 = rect.left + (rect.right - rect.left) * insetX;
          const y0 = rect.top + (rect.bottom - rect.top) * insetY;
          const x1 = rect.right - (rect.right - rect.left) * insetX;
          const y1 = rect.bottom - (rect.bottom - rect.top) * insetY;
          const origin = toDesktopPoint(frameContext, [x0, y0]);
          const endX = toDesktopPoint(frameContext, [x1, y0]);
          const endY = toDesktopPoint(frameContext, [x0, y1]);
          item.internal_target_map = {
            kind: "grid", column_labels: columns, row_labels: rows, origin,
            step: [columns.length > 1 ? (endX[0] - origin[0]) / (columns.length - 1) : 0,
                   rows.length > 1 ? (endY[1] - origin[1]) / (rows.length - 1) : 0],
            target_count: columns.length * rows.length
          };
          item.internal_target_coverage = "complete";
        }

        let internalTargetCount = 0;
        if (rule?.targets) {
          const provider = rule.targets;
          const values = readPath(frameContext.view, provider.root_path);
          const logicalWidth = Number(provider.logical_size?.[0] || canvas.width || 1);
          const logicalHeight = Number(provider.logical_size?.[1] || canvas.height || 1);
          const offsetX = Number(readPath(frameContext.view, provider.offset_x_path) || 0);
          const offsetY = Number(readPath(frameContext.view, provider.offset_y_path) || 0);
          if (values && typeof values[Symbol.iterator] === "function") {
            let targetIndex = 0;
            for (const value of Array.from(values).slice(0, 120)) {
              targetIndex += 1;
              if (provider.visible_path && !readPath(value, provider.visible_path)) continue;
              const x = Number(readPath(value, provider.x_path)) + offsetX;
              const y = Number(readPath(value, provider.y_path)) + offsetY;
              if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
              const localX = rect.left + x / logicalWidth * (rect.right - rect.left);
              const localY = rect.top + y / logicalHeight * (rect.bottom - rect.top);
              if (localX < rect.left || localX > rect.right || localY < rect.top || localY > rect.bottom) continue;
              const radius = Number(readPath(value, provider.radius_path) || provider.radius || 12);
              const radiusX = Math.max(2, radius / logicalWidth * (rect.right - rect.left));
              const radiusY = Math.max(2, radius / logicalHeight * (rect.bottom - rect.top));
              const enabled = provider.enabled_path ? !!readPath(value, provider.enabled_path) : true;
              const identity = clean(readPath(value, provider.id_path)) || String(targetIndex);
              const targetLabel = clean(provider.label || "Canvas target");
              const target = {
                id: scopedIdentity(frameContext, `${provider.id_prefix || "target"}-${identity}`),
                dom_order: order++, source: "canvas_internal", frame_path: frameContext.path,
                canvas_id: canvasId, role: "target", label: targetLabel, context: label,
                semantic_label: `${targetLabel} — ${label}`,
                tag: "canvas-target",
                bbox: toDesktopRect(frameContext, {left: localX-radiusX, top: localY-radiusY,
                  right: localX+radiusX, bottom: localY+radiusY}),
                click_point: toDesktopPoint(frameContext, [localX, localY]), actions: ["click"],
                enabled, readonly: false, editable: false, checked: null, selected: null,
                expanded: null, modal_scope: false, occluded: !point,
                unavailable_reasons: point ? (enabled ? [] : ["inactive"]) : ["occluded"]
              };
              (enabled && point ? actionable : unavailable).push(target);
              internalTargetCount += 1;
            }
          }
          item.internal_target_coverage = "complete";
        }
        item.internal_target_count = internalTargetCount || item.internal_target_map?.target_count || 0;
        canvases.push(item);
      }
    }
  }

  const readinessRule = spec.readiness && typeof spec.readiness === "object" ? spec.readiness : {};
  let readinessElement = null;
  if (readinessRule.selector) {
    for (const context of contexts) {
      try { readinessElement = context.doc.querySelector(readinessRule.selector); } catch (_) {}
      if (readinessElement) break;
    }
  }
  const limitations = ["browser_chrome_not_inspected"];
  if (inaccessibleFrames) limitations.push("cross_origin_frames_not_inspected");
  if (canvases.some(canvas => canvas.internal_target_coverage === "unsupported")) {
    limitations.push("canvas_internal_targets_unsupported");
  }

  return {
    scene: {status:"ok", kind:"browser_dom", application:"google-chrome", url:location.href,
            title:document.title, active:document.hasFocus(), visible:document.visibilityState === "visible"},
    binding: {stable:true, revision:clean(document.body?.dataset?.toolRevision || document.body?.dataset?.revision || "")},
    actionable_now: actionable,
    visible_but_unavailable: unavailable,
    canvas_surfaces: canvases,
    limitations,
    readiness: {document_ready_state: document.readyState,
                selector_present: !!readinessElement,
                selector_text: clean(readinessElement?.textContent || "")},
    metrics: {screenX, screenY, outerWidth, outerHeight, innerWidth, innerHeight, devicePixelRatio,
              screenWidth:screen.width, screenHeight:screen.height, contentLeft, contentTop,
              inspectedFrameDocuments:contexts.length, inaccessibleFrames}
  };
})
"""


def build_expression(spec: dict, desktop_size=(1920, 1080)) -> str:
    """Build the read-only extractor expression for an existing CDP connection."""
    resolved = dict(spec or {})
    resolved["__desktop_size"] = list(desktop_size)
    return f"({_EXTRACT})({json.dumps(resolved, separators=(',', ':'))})"
