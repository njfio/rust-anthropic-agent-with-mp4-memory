import { CONFIG, COLORS } from '../config.js';

export class RenderSystem {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.canvas = null;
    }
    
    init() {
        // Get canvas element
        this.canvas = document.getElementById('gameCanvas');
        if (!this.canvas) {
            throw new Error('Game canvas not found');
        }
        
        // Initialize Three.js components
        this.initScene();
        this.initCamera();
        this.initRenderer();
        this.initLighting();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    initScene() {
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.Fog(0x000000, CONFIG.RENDER.FOG_NEAR, CONFIG.RENDER.FOG_FAR);
    }
    
    initCamera() {
        this.camera = new THREE.PerspectiveCamera(
            CONFIG.RENDER.CAMERA_FOV,
            window.innerWidth / window.innerHeight,
            CONFIG.RENDER.CAMERA_NEAR,
            CONFIG.RENDER.CAMERA_FAR
        );
    }
    
    initRenderer() {
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: this.canvas,
            antialias: true 
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }
    
    initLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(
            COLORS.AMBIENT_LIGHT, 
            0.5
        );
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(
            COLORS.DIRECTIONAL_LIGHT, 
            0.5
        );
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        
        // Shadow settings
        directionalLight.shadow.camera.left = -50;
        directionalLight.shadow.camera.right = 50;
        directionalLight.shadow.camera.top = 50;
        directionalLight.shadow.camera.bottom = -50;
        directionalLight.shadow.mapSize.width = CONFIG.RENDER.SHADOW_MAP_SIZE;
        directionalLight.shadow.mapSize.height = CONFIG.RENDER.SHADOW_MAP_SIZE;
        
        this.scene.add(directionalLight);
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    update(player) {
        if (player && player.mesh) {
            this.updateCamera(player);
        }
    }
    
    updateCamera(player) {
        // Third person camera following player
        const distance = CONFIG.RENDER.CAMERA_DISTANCE;
        const height = CONFIG.RENDER.CAMERA_HEIGHT;
        
        this.camera.position.x = player.mesh.position.x - 
            distance * Math.sin(player.mesh.rotation.y);
        this.camera.position.y = player.mesh.position.y + height;
        this.camera.position.z = player.mesh.position.z - 
            distance * Math.cos(player.mesh.rotation.y);
        
        this.camera.lookAt(player.mesh.position);
    }
    
    render() {
        this.renderer.render(this.scene, this.camera);
    }
    
    clearScene() {
        // Remove all meshes except lights
        const toRemove = [];
        
        this.scene.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                toRemove.push(child);
            }
        });
        
        toRemove.forEach(child => {
            this.scene.remove(child);
            
            // Dispose of resources
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => material.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
    }
}