import { defineConfig } from "vite";

export default defineConfig({
    build: {
        chunkSizeWarningLimit: 700,
        rolldownOptions: {
            output: {
                manualChunks(id) {
                    if (id.includes("node_modules/three/")) return "three";
                    if (id.includes("node_modules/pixi.js/")) return "pixi";
                },
            },
        },
    },
});
