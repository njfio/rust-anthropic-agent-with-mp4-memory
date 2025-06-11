// Scene Manager
// Handles Three.js scene initialization and management

import { gameState } from './GameState.js';

export class SceneManager {
    constructor() {
        this.initialized = false;
    }

    init() {
        if (this.initialized) return;

        // Scene setup
        gameState.scene = new THREE.Scene();
        gameState.scene.fog = new THREE.Fog(0x000000, 10, 50);
        
        // Camera setup
        gameState.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        
        // Renderer setup
        gameState.renderer = new THREE.WebGLRenderer({ 
            canvas: document.getElementById('gameCanvas'),
            antialias: true 
        });
        gameState.renderer.setSize(window.innerWidth, window.innerHeight);
        gameState.renderer.shadowMap.enabled = true;
        gameState.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.setupLighting();
        this.setupEventListeners();
        
        this.initialized = true;
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        gameState.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        directionalLight.shadow.camera.left = -50;
        directionalLight.shadow.camera.right = 50;
        directionalLight.shadow.camera.top = 50;
        directionalLight.shadow.camera.bottom = -50;
        gameState.scene.add(directionalLight);
    }

    setupEventListeners() {
        window.addEventListener('resize', this.onWindowResize.bind(this));
    }

    onWindowResize() {
        gameState.camera.aspect = window.innerWidth / window.innerHeight;
        gameState.camera.updateProjectionMatrix();
        gameState.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    render() {
        if (gameState.renderer && gameState.scene && gameState.camera) {
            gameState.renderer.render(gameState.scene, gameState.camera);
        }
    }
}

// Singleton instance
export const sceneManager = new SceneManager();