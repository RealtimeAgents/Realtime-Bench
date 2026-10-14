import * as PIXI from "pixi.js";

class ProgressBar {
    constructor(x, y, width, height, stage, color) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.stage = stage;
        this.color = color;
        this.percent = 0;

        let bkgnd = new PIXI.Graphics();
        bkgnd.rect(x, y, width, height).fill(0x000000);
        bkgnd.alpha = 1;
        stage.addChild(bkgnd);

        this.bar = new PIXI.Graphics();
        stage.addChild(this.bar);

    }

    setProgress(percent) {
        this.percent = percent;
        let newWidth =  this.width * percent;
        let delta = this.width - newWidth;
        this.bar.clear();
        this.bar.rect(this.x + delta, this.y, newWidth, this.height).fill(this.color);
    }


}

export default ProgressBar;
