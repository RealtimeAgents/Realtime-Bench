(() => {
  "use strict";

  // Keep page-level accessible names aligned with the semantic context in the interactables
  // bound to each observation. Realtime pages replace whole rows/cards as state changes, so this
  // must be a live enhancement rather than a one-shot load handler.
  const actionableSelector = [
    "button", "a[href]", "input:not([type=hidden])", "textarea", "select", "summary",
    "[contenteditable=true]", "[contenteditable='']", "[role=button]", "[role=link]",
    "[role=checkbox]", "[role=radio]", "[role=switch]", "[role=slider]",
    "[role=spinbutton]", "[role=textbox]", "[role=combobox]", "[role=listbox]",
    "[role=menuitem]", "[role=menuitemcheckbox]", "[role=menuitemradio]", "[role=option]",
    "[role=tab]", "[role=treeitem]", "[tabindex]", "[data-tool-interactive]",
    "[draggable=true]", "[data-tool-drop-target]",
  ].join(",");
  const semanticContainerSelector = [
    "article", "tr", "li", "fieldset", "[role=row]", "[role=group]", "[role=listitem]",
  ].join(",");
  const generatedAttribute = "data-semantic-generated-name";
  const baseAttribute = "data-semantic-base-name";
  const originalAriaAttribute = "data-semantic-original-aria-label";
  const hadAriaAttribute = "data-semantic-had-aria-label";
  let contextSequence = 0;

  const clean = value => String(value ?? "").replace(/\s+/g, " ").trim();
  const key = value => clean(value).toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

  function isVisible(element) {
    if (!(element instanceof Element) || !element.isConnected || element.hidden || element.inert) return false;
    const style = getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || style.visibility === "collapse") return false;
    return element.getClientRects().length > 0;
  }

  function labelledbyText(element) {
    const ids = clean(element.getAttribute("aria-labelledby"));
    if (!ids) return "";
    const root = element.getRootNode?.();
    return clean(ids.split(/\s+/).map(id =>
      root?.getElementById?.(id)?.textContent || document.getElementById(id)?.textContent || ""
    ).join(" "));
  }

  function accessibleName(element) {
    // aria-labelledby has precedence in the accessible-name computation.
    const labelled = labelledbyText(element);
    if (labelled) return labelled;
    const direct = clean(element.getAttribute("aria-label"));
    if (direct) return direct;
    if (element.labels?.length) {
      const labels = clean([...element.labels].map(label => label.textContent).join(" "));
      if (labels) return labels;
    }
    return clean(
      element.getAttribute("title") || element.getAttribute("alt") ||
      element.getAttribute("placeholder") ||
      ((element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT")
        ? element.value : "") || element.innerText || element.textContent
    );
  }

  function nameWithoutAriaLabel(element) {
    const labelled = labelledbyText(element);
    if (labelled) return labelled;
    if (element.labels?.length) {
      const labels = clean([...element.labels].map(label => label.textContent).join(" "));
      if (labels) return labels;
    }
    return clean(
      element.getAttribute("title") || element.getAttribute("alt") ||
      element.getAttribute("placeholder") ||
      ((element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.tagName === "SELECT")
        ? element.value : "") || element.innerText || element.textContent
    );
  }

  function parentAcrossRoots(element) {
    if (element?.parentElement) return element.parentElement;
    const root = element?.getRootNode?.();
    return root?.host instanceof Element ? root.host : null;
  }

  function semanticContainer(element) {
    for (let node = parentAcrossRoots(element); node; node = parentAcrossRoots(node)) {
      try {
        if (node.matches(semanticContainerSelector)) return node;
      } catch (_) {}
    }
    return null;
  }

  function semanticContext(element) {
    const container = semanticContainer(element);
    if (!container) return "";
    const parts = [];
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    for (let textNode = walker.nextNode(); textNode; textNode = walker.nextNode()) {
      const value = clean(textNode.nodeValue);
      const owner = textNode.parentElement;
      if (!value || !owner || element.contains(owner) ||
          owner.closest("[data-semantic-context-node]") || !isVisible(owner)) continue;
      let insideAnotherControl = false;
      for (let node = owner; node && node !== container; node = parentAcrossRoots(node)) {
        if (node === element || element.contains(node)) {
          insideAnotherControl = true;
          break;
        }
        try {
          if (node.matches(actionableSelector)) {
            insideAnotherControl = true;
            break;
          }
        } catch (_) {}
      }
      if (!insideAnotherControl && parts.at(-1) !== value) parts.push(value);
    }
    return clean(parts.join(" ")).slice(0, 240);
  }

  function mergeName(base, context) {
    if (!base) return context;
    if (!context) return base;
    const baseKey = key(base), contextKey = key(context);
    if (baseKey.includes(contextKey) || contextKey.includes(baseKey) && contextKey === baseKey) return base;
    return `${base} — ${context}`.slice(0, 320);
  }

  function stableIdentity(element, index) {
    return clean(
      element.getAttribute("data-entity-id") || element.getAttribute("data-tool-id") ||
      element.id || element.getAttribute("name") || `item ${index + 1}`
    );
  }

  function authorBaseName(element) {
    const generated = element.getAttribute(generatedAttribute);
    const currentAria = clean(element.getAttribute("aria-label"));
    if (generated && currentAria === generated) {
      return element.getAttribute(hadAriaAttribute) === "true"
        ? clean(element.getAttribute(originalAriaAttribute))
        : nameWithoutAriaLabel(element);
    }
    if (generated && currentAria !== generated) {
      // Application code replaced our label; treat the new value as author-owned.
      element.removeAttribute(generatedAttribute);
      element.removeAttribute(baseAttribute);
      element.removeAttribute(originalAriaAttribute);
      element.removeAttribute(hadAriaAttribute);
    }
    return accessibleName(element);
  }

  function clearGeneratedAriaLabel(element) {
    if (!element.hasAttribute(generatedAttribute)) return;
    if (element.getAttribute(hadAriaAttribute) === "true") {
      element.setAttribute("aria-label", element.getAttribute(originalAriaAttribute) || "");
    } else {
      element.removeAttribute("aria-label");
    }
    element.removeAttribute(generatedAttribute);
    element.removeAttribute(baseAttribute);
    element.removeAttribute(originalAriaAttribute);
    element.removeAttribute(hadAriaAttribute);
  }

  function applyAriaLabel(element, base, desired) {
    // Do not break author-provided aria-labelledby relationships. Those already carry a semantic
    // name; duplicate labelledby controls get an extra off-screen context node instead.
    const labelledIds = clean(element.getAttribute("aria-labelledby"));
    if (labelledIds && !element.hasAttribute(generatedAttribute)) {
      if (desired === base) return;
      const contextId = element.dataset.semanticContextId || `semantic-context-${++contextSequence}`;
      element.dataset.semanticContextId = contextId;
      let contextNode = document.getElementById(contextId);
      if (!contextNode) {
        contextNode = document.createElement("span");
        contextNode.id = contextId;
        contextNode.dataset.semanticContextNode = "true";
        contextNode.style.cssText = "position:absolute!important;width:1px!important;height:1px!important;padding:0!important;margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;white-space:nowrap!important;border:0!important";
        element.insertAdjacentElement("afterend", contextNode);
      }
      contextNode.textContent = desired.slice(base.length).replace(/^\s*—\s*/, "");
      if (!labelledIds.split(/\s+/).includes(contextId)) {
        element.setAttribute("aria-labelledby", `${labelledIds} ${contextId}`);
      }
      return;
    }
    if (!element.hasAttribute(generatedAttribute)) {
      element.setAttribute(hadAriaAttribute, String(element.hasAttribute("aria-label")));
      element.setAttribute(originalAriaAttribute, element.getAttribute("aria-label") || "");
    }
    if (element.getAttribute("aria-label") !== desired) element.setAttribute("aria-label", desired);
    element.setAttribute(baseAttribute, base);
    element.setAttribute(generatedAttribute, desired);
  }

  function refresh() {
    const controls = [...document.querySelectorAll(actionableSelector)].filter(isVisible);
    const records = controls.map((element, index) => ({
      element,
      index,
      base: authorBaseName(element),
      context: semanticContext(element),
    }));
    const counts = new Map();
    records.forEach(record => counts.set(key(record.base), (counts.get(key(record.base)) || 0) + 1));

    const desiredCounts = new Map();
    records.forEach(record => {
      const duplicate = !record.base || (counts.get(key(record.base)) || 0) > 1;
      if (!duplicate) {
        clearGeneratedAriaLabel(record.element);
        return;
      }
      let desired = duplicate ? mergeName(record.base, record.context) : record.base;
      if (!desired) desired = `Control — ${stableIdentity(record.element, record.index)}`;
      const desiredKey = key(desired);
      const occurrence = (desiredCounts.get(desiredKey) || 0) + 1;
      desiredCounts.set(desiredKey, occurrence);
      if (occurrence > 1) desired = `${desired} — ${stableIdentity(record.element, record.index)}`.slice(0, 320);
      applyAriaLabel(record.element, record.base, desired);
    });
  }

  let queued = false;
  function scheduleRefresh() {
    if (queued) return;
    queued = true;
    queueMicrotask(() => {
      queued = false;
      refresh();
    });
  }

  function install() {
    if (window.__semanticAccessibility?.installed) {
      window.__semanticAccessibility.refresh();
      return;
    }
    const observer = new MutationObserver(scheduleRefresh);
    observer.observe(document.documentElement, {
      subtree: true, childList: true, characterData: true,
      attributes: true,
      attributeFilter: ["aria-label", "aria-labelledby", "disabled", "hidden", "class", "style"],
    });
    window.__semanticAccessibility = {installed: true, refresh, observer};
    refresh();
  }

  window.installSemanticAccessibility = install;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install, {once: true});
  else install();
})();
