// Modified by Realtime-Bench contributors: build each level from the fixed layout
// stream, then hand the remaining randomness to the task-seed stream. The modified game
// remains Apache-2.0; see LICENSE at the upstream root and ../../../REPAIR.md for the adaptation record.
import Level from "./Level.js";
import Hero from "../../js/classes/Hero.js";

class Game {
    constructor(rendererThree, sounds, width, height) {
        this.rendererThree = rendererThree;
        this.sounds = sounds;
        this.updateWidthHeight(width, height);
        this.levelNumber = 1;
        this.levelChangeTime = (new Date()).getTime();
        this.hero = new Hero(-75, 0);
        if (window.__synthblastResetLevelSeed) window.__synthblastResetLevelSeed(this.levelNumber);
        this.level = new Level(this, this.rendererThree, this.sounds);
        if (window.__synthblastLevelBuilt) window.__synthblastLevelBuilt(this.levelNumber);
    }

    render(fpsAdjustment) {
        if (this.level.hasWon()) {
            this.levelNumber++;
            this.levelChangeTime = (new Date()).getTime();
            if (window.__synthblastResetLevelSeed) window.__synthblastResetLevelSeed(this.levelNumber);
            this.level = new Level(this, this.rendererThree, this.sounds);
            if (window.__synthblastLevelBuilt) window.__synthblastLevelBuilt(this.levelNumber);
            this.hero.stopShooting();
            this.hero.stopTurning();
            this.hero.stop();
        }
            this.level.render(fpsAdjustment);

    }

    isOver() {
        return this.hero.hitPoints <= 0
    }

    updateWidthHeight(width, height) {
        this.width = width;
        this.height = height; // - 50; // possible ad space
    }
}

export default Game;
