import * as THREE from "three";
import Particle from "./Particle.js";

class Drone {

    constructor(x, y, scene, level) {
        let self = this;
        this.level = level;
        this.scene = scene;
        this.alive = true;
        this.hitPoints = 2;
        this.loaded = false;

        let loader = new THREE.ObjectLoader();
        loader.load(
            // resource URL
            "js/models/Drone.json",

            // onLoad callback
            // Here the loaded data is assumed to be an object
            function ( obj ) {
                // Add the loaded object to the scene
                self.object = obj;
                self.loaded = true;
                self.finishConstruction(x,y);
            },

            // onProgress callback
            function ( xhr ) {
                console.log( (xhr.loaded / xhr.total * 100) + '% loaded' );
            },

            // onError callback
            function ( err ) {
                console.error("error loading");
                console.error( err );
            }
        );

    }

    finishConstruction(x, y) {
        this.object.position.x = x;
        this.object.position.y = y;
        this.object.position.z = 1.5;
        this.object.rotation.x = 90 * Math.PI / 180;
        this.object.rotateOnWorldAxis(new THREE.Vector3(0, 0, 1), Math.PI / 2);

        this.maxSpeed = 0.05;
        this.currentSpeed = 0;
        this.acceleration = 0.0025;
        this.deceleration = 0.0035;
        this.hoverHeight = 1.5;
        this.hoverAmplitude = 0.2;
        this.hoverPhase = 0;
        this.idleHoverRate = Math.PI * 2 / 180;
        this.movingHoverRate = Math.PI * 2 / 90;
        this.followDistance = 5;
        this.arrivalRadius = 0.5;
        this.lookAtRadius = 1.25;
        this.lookAtSpeed = 0.012;
        this.slowdownRadius = 3;
        this.maxTurnSpeed = Math.PI / 60;
        this.currentTurnSpeed = 0;
        this.turnAcceleration = Math.PI / 600;
        this.turnDeceleration = Math.PI / 450;
        this.turnSlowdownAngle = Math.PI / 3;
        this.rotationAxis = new THREE.Vector3(0, 0, 1);
        let rotationAngle = 45 * Math.PI / 180;
        this.direction = new THREE.Vector3(0, 1, 0);
        // this.direction.applyAxisAngle(this.rotationAxis, rotationAngle);
        // this.object.rotateOnAxis(this.rotationAxis, rotationAngle);
        this.moveInc = this.direction.clone();
        this.moveInc.multiplyScalar(this.maxSpeed);
        this.newWander();
        this.particles = [];
        this.scene.add(this.object)
    }

    hit(impact, fpsAdjustment) {
        this.hitPoints -= impact;
        if (this.hitPoints <= 0) {
            for (let i = 0; i < 100; i++) {
                this.particles.push(new Particle(
                    this.object.position.x,
                    this.object.position.y,
                    this.object.position.z,
                    this.scene,
                    fpsAdjustment
                ));
            }
            this.alive = false;
            this.object.visible = false;
            this.deathTime = (new Date()).getTime();
        }
    }

    getHitPoints() {
        return Math.max(0, this.hitPoints);
    }


    getX() {
        return this.object.position.x;
    }

    getY() {
        return this.object.position.y;
    }

    getZ() {
        return this.object.position.z;
    }

    newWander() {
        this.wanderAngle = Math.random() * Math.PI;
    }


    update(hero, enemies, fpsAdjustment) {
        if (!this.loaded) return;
        this.updateHover(fpsAdjustment);
        if (!this.level.inPlay) return;

        // go to enemy closest to hero
        // if no enemy close to hero, go to hero
        if (this.alive) {
            if (!hero.isAlive()) return;

            const heroPosition = hero.getPosition();
            const formationPoint = heroPosition.clone().add(
                hero.direction.clone().multiplyScalar(this.followDistance)
            );
            formationPoint.z = this.object.position.z;

            const toFormationPoint = formationPoint.clone().sub(this.object.position);
            toFormationPoint.z = 0;

            const distance = toFormationPoint.length();
            const isSettling = distance <= this.arrivalRadius
                || (distance <= this.lookAtRadius && this.currentSpeed <= this.lookAtSpeed);
            if (!isSettling) {
                this.turnToward(toFormationPoint, fpsAdjustment);
                const speedRatio = THREE.MathUtils.smoothstep(
                    distance,
                    this.arrivalRadius,
                    this.slowdownRadius
                );
                const targetSpeed = this.maxSpeed * speedRatio;
                const rate = targetSpeed > this.currentSpeed ? this.acceleration : this.deceleration;
                this.currentSpeed = this.moveToward(this.currentSpeed, targetSpeed, rate * fpsAdjustment);
                this.moveInc = this.direction.clone().multiplyScalar(this.currentSpeed * fpsAdjustment);
                this.move();
            } else {
                this.currentSpeed = this.moveToward(
                    this.currentSpeed,
                    0,
                    this.deceleration * fpsAdjustment
                );
                if (this.currentSpeed <= this.lookAtSpeed) {
                    const toHero = heroPosition.clone().sub(this.object.position);
                    toHero.z = 0;
                    this.turnToward(toHero, fpsAdjustment);
                }
                if (this.currentSpeed > 0.001) {
                    this.moveInc = this.direction.clone().multiplyScalar(this.currentSpeed * fpsAdjustment);
                    this.move();
                } else {
                    this.currentSpeed = 0;
                }
            }
        } else {
            let currentTime = (new Date()).getTime();
            if (currentTime - this.deathTime < 200) {
                this.particles.forEach(particle => {
                    particle.update();
                });
            } else {
                this.particles.forEach(particle => {
                    particle.remove();

                });
                this.particles = [];
            }

        }

    }

    updateHover(fpsAdjustment) {
        const speedRatio = THREE.MathUtils.clamp(this.currentSpeed / this.maxSpeed, 0, 1);
        const hoverRate = THREE.MathUtils.lerp(
            this.idleHoverRate,
            this.movingHoverRate,
            speedRatio
        );
        this.hoverPhase += hoverRate * fpsAdjustment;
        this.object.position.z = this.hoverHeight + Math.sin(this.hoverPhase) * this.hoverAmplitude;
    }

    moveToward(current, target, amount) {
        if (current < target) return Math.min(current + amount, target);
        return Math.max(current - amount, target);
    }

    turnToward(targetDirection, fpsAdjustment) {
        if (targetDirection.lengthSq() === 0) return;

        targetDirection.normalize();
        const cross = this.direction.x * targetDirection.y - this.direction.y * targetDirection.x;
        const dot = THREE.MathUtils.clamp(this.direction.dot(targetDirection), -1, 1);
        const angle = Math.atan2(cross, dot);
        const speedRatio = THREE.MathUtils.smoothstep(
            Math.abs(angle),
            0,
            this.turnSlowdownAngle
        );
        const targetTurnSpeed = Math.sign(angle) * this.maxTurnSpeed * speedRatio;
        const rate = Math.abs(targetTurnSpeed) > Math.abs(this.currentTurnSpeed)
            ? this.turnAcceleration
            : this.turnDeceleration;
        this.currentTurnSpeed = this.moveToward(
            this.currentTurnSpeed,
            targetTurnSpeed,
            rate * fpsAdjustment
        );

        let turn = this.currentTurnSpeed * fpsAdjustment;
        if (Math.abs(turn) >= Math.abs(angle)) {
            turn = angle;
            this.currentTurnSpeed = 0;
        }

        this.direction.applyAxisAngle(this.rotationAxis, turn).normalize();
        this.object.rotateOnWorldAxis(this.rotationAxis, turn);
    }

    move() {
        if (this.loaded) {
            this.object.position.add(this.moveInc);
        }
    }

    unMove() {
        this.object.position.add(this.moveInc.negate());
        this.newWander();
        this.moveInc.negate();
    }

    isAlive() {
        return this.alive;
    }

}

export default Drone;
