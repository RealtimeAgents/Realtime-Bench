import * as THREE from "three";
import {LineMaterial} from "three/examples/jsm/lines/LineMaterial.js";
import {LineSegments2} from "three/examples/jsm/lines/LineSegments2.js";
import {LineSegmentsGeometry} from "three/examples/jsm/lines/LineSegmentsGeometry.js";

function addGlowEdges(mesh, geometry, color) {
    const edges = new THREE.EdgesGeometry(geometry);
    const edgeGeometry = new LineSegmentsGeometry().fromEdgesGeometry(edges);
    edges.dispose();

    [
        {width: 7, opacity: 0.04},
        {width: 4.5, opacity: 0.08},
        {width: 2.5, opacity: 0.16},
        {width: 1.5, opacity: 0.9}
    ].forEach(layer => {
        const material = new LineMaterial({
            color: color,
            linewidth: layer.width,
            blending: THREE.AdditiveBlending,
            transparent: true,
            opacity: layer.opacity,
            depthWrite: false,
            toneMapped: false
        });
        mesh.add(new LineSegments2(edgeGeometry, material));
    });
}

class Coin {
    constructor(x, y, scene) {
        this.scene = scene;
        this.isEnhanced = Math.random() < 1 / 3;
        this.ammoValue = this.isEnhanced ? 5 : 2;
        const edgeColor = this.isEnhanced ? 0xFF00FF : 0x00FFFF;

        let sizeFactor = 1;

        let size = 0.25 * sizeFactor;
        let bottomSize = 0.25 * sizeFactor;
        let bottomHeight = bottomSize * .6;

        // let cubeGeometry = new THREE.BoxBufferGeometry(size / 2, size, size);
        let cubeGeometry = new THREE.ConeGeometry(.6 * size, size, 4);
        let cubeMaterial = new THREE.MeshLambertMaterial(
            {
                color: 0xFFFFFF,
                emissive: 0x550055
            });
        let cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
        addGlowEdges(cube, cubeGeometry, edgeColor);
        //cube.rotation.x = Math.PI / 4;
        cube.rotation.x = Math.PI / 2;
        // cube.position.x = x;
        // cube.position.y = y;
        cube.position.z = 0.25 * sizeFactor;


        let bottomGeometry = new THREE.ConeGeometry(.6 * size, bottomHeight, 4);
        let bottomMaterial = new THREE.MeshLambertMaterial(
            {
                color: 0x00FFFF,
                emissive: 0x330033
            });
        let bottom = new THREE.Mesh(bottomGeometry, bottomMaterial);
        addGlowEdges(bottom, bottomGeometry, edgeColor);
        bottom.rotation.x = Math.PI / 2 * 3;
        bottom.position.z = .03;



        let group = new THREE.Group();
        group.add( cube );
        group.add( bottom );
        group.position.x = x;
        group.position.y = y;
        group.position.z = bottomSize;

        scene.add(group);
        this.object = group;
        this.isTaken = false;
        this.isAlive = true;

        this.rotationAxis = new THREE.Vector3(0, 0, 1);

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

    getPosition() {
        return this.object.position;
    }

    moveTo(v, fpsAdjustment) {
        let delta = new THREE.Vector3(v.x - this.getX(), v.y - this.getY(), 0);
        delta.normalize();
        delta.multiplyScalar(.1 * fpsAdjustment);
        this.object.position.add(delta);
    }

    update() {
        if (this.isAlive && this.isTaken) {
            this.object.visible = false;
            this.isAlive = false;
        } else {
            this.object.rotateOnAxis(this.rotationAxis,  Math.PI / 60);
        }
    }
}

export default Coin;
