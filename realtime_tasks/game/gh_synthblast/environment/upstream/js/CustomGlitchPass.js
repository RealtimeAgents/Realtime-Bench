import { GlitchPass as ThreeGlitchPass } from "three/examples/jsm/postprocessing/GlitchPass.js";

// Preserve the game's event-driven trigger on top of Three.js's maintained pass.
class GlitchPass extends ThreeGlitchPass {
    constructor(dtSize = 64) {
        super(dtSize);
        this.duration = 30;
        this.framesRemaining = this.duration;
    }

    trigger() {
        this._curF = 0;
        this.framesRemaining = this.duration;
        this.enabled = true;
    }

    render(renderer, writeBuffer, readBuffer, deltaTime, maskActive) {
        super.render(renderer, writeBuffer, readBuffer, deltaTime, maskActive);
        this.framesRemaining--;
        if (this.framesRemaining <= 0) {
            this.enabled = false;
        }
    }
}

export { GlitchPass };
