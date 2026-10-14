import * as PIXI from "pixi.js";

class Button {

    constructor(x, y, width, height, stage, label, callback) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.stage = stage;
        this.label = label;
        this.callback = callback;

        // FILL
        let fill = new PIXI.Graphics();
        fill.rect(x, y, width, height).fill(0xFF00FF);
        fill.alpha = 0.35;
        this.fill = fill;
        stage.addChild(fill);

        let fontSize = Math.round(height * 0.8);
        let labelStyle = new PIXI.TextStyle({
            fontFamily: "\"Courier New\", Courier, monospace",
            fontSize: fontSize,
            fill: "#99FFFF",
        });

        let glow = new PIXI.BlurFilter({strength: 8});
        this.glow = glow;

        // nice, neon glow (blur)
        this.blurText = new PIXI.Text({text: label, style: labelStyle});
        while (this.blurText.width > width * .9 && labelStyle.fontSize > 8) {
            labelStyle.fontSize -= 1;
        }
        this.blurText.position.set(x + (width - this.blurText.width) / 2, y + (height - labelStyle.fontSize) / 2);
        this.blurText.filters = [glow];
        stage.addChild(this.blurText);

        // the text
        this.labelText = new PIXI.Text({text: label, style: labelStyle.clone()});
        this.labelText.position.set(x + (width - this.labelText.width) / 2, y + (height - labelStyle.fontSize) / 2);
        stage.addChild(this.labelText);

        // button outline glow
        let outlineBlur = new PIXI.Graphics();
        outlineBlur.rect(x, y, width, height).stroke({width: 1, color: 0x99FFFF});
        outlineBlur.filters = [glow];
        this.outlineBlur = outlineBlur;
        stage.addChild(outlineBlur);

        // button outline
        let outline = new PIXI.Graphics();
        outline.rect(x, y, width, height).stroke({width: 2, color: 0x99FFFF});
        outline.alpha = 0.5;
        this.outline = outline;
        stage.addChild(outline);

    }

    isIn(x, y) {
        return x >= this.x && x <= this.x + this.width && y >= this.y && y <= this.y + this.height;
    }

    setLabel(label) {
        this.label = label;
        this.blurText.text = label;
        this.labelText.text = label;
        this.blurText.x = this.x + (this.width - this.blurText.width) / 2;
        this.labelText.x = this.x + (this.width - this.labelText.width) / 2;
    }

    down(event) {
        if (this.isIn(event.clientX, event.clientY)) {
            this.fill.alpha = .7;
        } else {
            this.fill.alpha = .35;
        }

    }

    up(event) {
        this.fill.alpha = 0.35;
        if (this.isIn(event.clientX, event.clientY)) {
            this.callback();
        }
    }


}

export default Button;
