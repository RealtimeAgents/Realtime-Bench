(() => {
  "use strict";

  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

  class RealtimeRuntime {
    constructor({scenario, episode, primaryAction, onState, pollMs = 500}) {
      this.scenario = scenario;
      this.episode = episode;
      this.primaryAction = primaryAction;
      this.onState = onState;
      this.pollMs = pollMs;
      this.state = null;
      this.offset = 0;
      this.pending = false;
      this.stopped = false;
      this.latestGeneration = -1;
      this.latestVersion = 0;
      this.commandCounter = 0;
      // Harness-only action completion barrier. GUI input is still delivered through the desktop;
      // these counters merely let the harness wait until any business request caused by that input
      // has completed while the simulated clock remains frozen.
      this.inputEventSequence = 0;
      this.businessActionStarted = 0;
      this.businessActionCompleted = 0;
      this.lastInputEvent = null;
      this.lastBusinessOutcome = null;
      this.externalBusinessActions = new Map();
      this.actionBarrierChannel = new BroadcastChannel("realtime-action-barrier-v1");
      this.actionBarrierChannel.addEventListener("message", async event => {
        const barrierId = event.data && event.data.barrier_id;
        const sequence = this.externalBusinessActions.get(barrierId);
        if (!sequence) return;
        this.externalBusinessActions.delete(barrierId);
        try {
          await this.poll();
          this.lastBusinessOutcome = {sequence, ok: true, transport: "navigation"};
        } catch (error) {
          this.lastBusinessOutcome = {
            sequence, ok: false, transport: "navigation", error: error.message || String(error),
          };
        } finally {
          this.businessActionCompleted = Math.max(this.businessActionCompleted, sequence);
        }
      });
      for (const type of [
        "click", "contextmenu", "keydown", "keyup", "beforeinput", "input", "change",
        "focusout", "submit", "pointerdown", "pointerup", "pointermove", "mousedown",
        "mouseup", "mousemove", "wheel", "drop",
      ]) {
        document.addEventListener(type, event => {
          if (!event.isTrusted) return;
          this.inputEventSequence += 1;
          this.lastInputEvent = type;
        }, true);
      }
      // Harness-visible render barrier. This exposes no evaluator-only information: `state` is the
      // same public payload already rendered by the page. The harness calls poll() after advancing
      // logical time and waits until state.benchmark_frame matches before taking the next screenshot.
      globalThis.__realtimeRuntime = this;
    }

    serverNow() { return Date.now() + this.offset; }

    phaseCountdown() {
      if (!this.state) return {label: "Connecting", ms: 0};
      const state = this.state;
      if (state.phase === "confirming") return {label: "Verifying live records", ms: 0};
      if (state.phase === "active") return {label: "Live updates", ms: 0};
      if (state.phase === "resolved") return {label: "Decision accepted", ms: 0};
      if (state.phase === "expired") return {label: "Session closed", ms: 0};
      return {label: "Live updates", ms: 0};
    }

    formatDuration() {
      if (!this.state) return "Connecting";
      if (this.state.phase === "resolved") return "Accepted";
      if (this.state.phase === "expired") return "Closed";
      return "Updated now";
    }

    accept(state) {
      if (!state || !state.ok) return false;
      // The harness-owned benchmark frame is stronger than the ordinary monotonic-version guard.
      // Clock activation can intentionally rebase a just-restarted episode from a wall-advanced
      // version back to logical frame 0 within the same generation.
      if (Number.isFinite(state.benchmark_frame)) {
        if (state.generation < this.latestGeneration) return false;
        this.latestGeneration = state.generation;
        this.latestVersion = state.version;
        return true;
      }
      if (state.generation < this.latestGeneration) return false;
      if (state.generation === this.latestGeneration && state.version < this.latestVersion) return false;
      if (state.generation > this.latestGeneration) this.latestVersion = 0;
      this.latestGeneration = state.generation;
      this.latestVersion = state.version;
      return true;
    }

    async request(url, options = {}) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 3000);
      try {
        const response = await fetch(url, {...options, signal: controller.signal, cache: "no-store"});
        const data = await response.json();
        return {response, data};
      } finally {
        clearTimeout(timeout);
      }
    }

    async poll() {
      const sent = Date.now();
      const {response, data} = await this.request(
        `/api/${encodeURIComponent(this.scenario)}/state?episode=${encodeURIComponent(this.episode)}`,
      );
      const received = Date.now();
      if (!response.ok || !data.ok) throw new Error(data.error || "Live state is unavailable.");
      data.client_offset_ms = data.server_now_ms - (sent + received) / 2;
      this.offset = data.client_offset_ms;
      if (this.accept(data)) {
        this.state = data;
        this.onState(data, this);
      }
    }

    async loop() {
      while (!this.stopped) {
        try {
          await this.poll();
        } catch (error) {
          this.toast("Live connection interrupted", error.message, "error");
        }
        const terminal = this.state && ["resolved", "expired"].includes(this.state.phase);
        await sleep(terminal ? 1200 : this.pollMs);
      }
    }

    start() { this.loop(); return this; }
    stop() { this.stopped = true; }

    actionBarrierSnapshot() {
      return {
        input_event_sequence: this.inputEventSequence,
        business_action_started: this.businessActionStarted,
        business_action_completed: this.businessActionCompleted,
      };
    }

    trackExternalNavigation(anchor) {
      if (!(anchor instanceof HTMLAnchorElement)) {
        throw new TypeError("external business navigation requires an anchor");
      }
      const barrierId = globalThis.crypto?.randomUUID?.() || this.newCommandId();
      const sequence = ++this.businessActionStarted;
      const target = new URL(anchor.href, location.href);
      target.searchParams.set("__rt_action_barrier", barrierId);
      anchor.href = target.href;
      this.externalBusinessActions.set(barrierId, sequence);
      return sequence;
    }

    async waitForActionBarrier(baseline, dispatchWaitMs = 750, completionWaitMs = 4500) {
      const sleepShort = () => sleep(10);
      const dispatchDeadline = performance.now() + dispatchWaitMs;
      while (
        this.inputEventSequence <= baseline.input_event_sequence
        && performance.now() < dispatchDeadline
      ) await sleepShort();

      const inputObserved = this.inputEventSequence > baseline.input_event_sequence;
      if (inputObserved) {
        // Let a burst of key/pointer events and their synchronous handlers drain before deciding
        // whether this GUI action started a server mutation.
        let lastSequence = this.inputEventSequence;
        let quietSince = performance.now();
        while (performance.now() - quietSince < 50 && performance.now() < dispatchDeadline) {
          await sleepShort();
          if (this.inputEventSequence !== lastSequence) {
            lastSequence = this.inputEventSequence;
            quietSince = performance.now();
          }
        }
      }

      let target = this.businessActionStarted;
      const businessStarted = target > baseline.business_action_started;
      if (businessStarted) {
        const completionDeadline = performance.now() + completionWaitMs;
        let quietSince = performance.now();
        while (performance.now() < completionDeadline) {
          if (this.businessActionStarted !== target) {
            target = this.businessActionStarted;
            quietSince = performance.now();
          }
          if (this.businessActionCompleted >= target && performance.now() - quietSince >= 100) break;
          await sleepShort();
        }
        if (this.businessActionCompleted < target) {
          throw new Error(`business action ${target} did not complete before the barrier timeout`);
        }
      }
      return {
        ok: true,
        input_observed: inputObserved,
        input_event_sequence: this.inputEventSequence,
        last_input_event: this.lastInputEvent,
        business_started: businessStarted,
        business_action_started: this.businessActionStarted,
        business_action_completed: this.businessActionCompleted,
        last_business_outcome: this.lastBusinessOutcome,
      };
    }

    async act(record, action = this.primaryAction) {
      if (this.pending || !this.state || !record) return null;
      this.pending = true;
      const businessSequence = ++this.businessActionStarted;
      try {
        const {response, data} = await this.request(`/api/${encodeURIComponent(this.scenario)}/action`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            episode: this.episode,
            generation: this.state.generation,
            record: record.ref,
            action,
            version: this.state.version,
            view_token: this.state.view_token,
          }),
        });
        if (!response.ok || !data.ok) {
          this.lastBusinessOutcome = {sequence: businessSequence, ok: false, status: response.status};
          this.toast("Decision not completed", data.error || "Review the current live state and try again.", "error");
          return data;
        }
        await this.poll();
        this.lastBusinessOutcome = {sequence: businessSequence, ok: true, status: response.status};
        return data;
      } catch (error) {
        this.lastBusinessOutcome = {sequence: businessSequence, ok: false, error: error.message || String(error)};
        this.toast("Control request failed", error.message || "The operation could not be confirmed.", "error");
        return null;
      } finally {
        this.businessActionCompleted = Math.max(this.businessActionCompleted, businessSequence);
        this.pending = false;
      }
    }

    newCommandId() {
      if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
        return globalThis.crypto.randomUUID();
      }
      this.commandCounter += 1;
      return `${this.episode}-${Date.now()}-${this.commandCounter}`;
    }

    async command(command, commandId = null) {
      if (this.pending || !this.state || !command) return null;
      this.pending = true;
      const businessSequence = ++this.businessActionStarted;
      const intentId = commandId || this.newCommandId();
      try {
        const {response, data} = await this.request(`/api/${encodeURIComponent(this.scenario)}/action`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            episode: this.episode,
            generation: this.state.generation,
            version: this.state.version,
            view_token: this.state.view_token,
            command_id: intentId,
            command,
          }),
        });
        if (!response.ok || !data.ok) {
          this.lastBusinessOutcome = {sequence: businessSequence, ok: false, status: response.status};
          if (data.terminal_reason) {
            await this.poll();
            return data;
          }
          this.toast("Change not recorded", data.error || "The command was not accepted.", "error");
          return data;
        }
        await this.poll();
        this.lastBusinessOutcome = {sequence: businessSequence, ok: true, status: response.status};
        return data;
      } catch (error) {
        this.lastBusinessOutcome = {sequence: businessSequence, ok: false, error: error.message || String(error)};
        this.toast("Control request failed", error.message || "The command could not be confirmed.", "error");
        return null;
      } finally {
        this.businessActionCompleted = Math.max(this.businessActionCompleted, businessSequence);
        this.pending = false;
      }
    }

    toast(title, message, kind = "") {
      let stack = document.getElementById("rt-toasts");
      if (!stack) {
        stack = document.createElement("div");
        stack.id = "rt-toasts";
        stack.className = "rt-toast-stack";
        document.body.appendChild(stack);
      }
      const item = document.createElement("div");
      item.className = `rt-toast ${kind}`;
      const heading = document.createElement("strong");
      const copy = document.createElement("span");
      heading.textContent = title;
      copy.textContent = message;
      item.append(heading, copy);
      stack.appendChild(item);
      setTimeout(() => item.remove(), 5000);
    }
  }

  window.RealtimeRuntime = RealtimeRuntime;
})();
