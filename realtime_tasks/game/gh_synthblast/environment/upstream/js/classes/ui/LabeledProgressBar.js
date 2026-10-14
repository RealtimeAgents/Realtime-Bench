import * as PIXI from "pixi.js";
import ProgressBar from "./ProgressBar.js";

class LabeledProgressBar {
    constructor(x, y, width, height, stage, label, color) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.stage = stage;
        this.label = label;
        this.color = color;
        this.percent = 0;

        let labelStyle = new PIXI.TextStyle({
            fontFamily: "\"Courier New\", Courier, monospace",
            fontSize: 10,
            fill: "#99FFFF",
        });

        this.labelText = new PIXI.Text({text: label, style: labelStyle});

        let bkgnd = new PIXI.Graphics();
        let padding = 3;
        bkgnd.alpha = 1;
        bkgnd.rect(x - (padding + this.labelText.width + 5), y, width + (padding + this.labelText.width + 5), height).fill(0x000000);
        stage.addChild(bkgnd);

        this.progressBar = new ProgressBar(
            x ,
            y,
            width,
            height,
            stage,
            color);


        this.labelText.position.set(x - this.labelText.width - 5, y);
        stage.addChild(this.labelText);


    }

    setProgress(percent) {
        this.progressBar.setProgress(percent);
        // this.stage.removeChild(this.bar);
        // this.percent = percent;
        // let bar = new PIXI.Graphics();
        // let newWidth =  this.width * percent;
        // let delta = this.width - newWidth;
        // bar.beginFill(this.color);
        // bar.drawRect(this.x + delta, this.y, newWidth, this.height);
        // this.bar = bar;
        // this.stage.addChild(bar);
    }


}

export default LabeledProgressBar;
