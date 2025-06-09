import { Entity } from './Entity.js';
import { CONFIG } from '../config.js';

export class Item extends Entity {
    constructor(x, z, type) {
        super(x, 1, z);
        this.type = type;
        
        // Load item config
        const config = CONFIG.ITEM.TYPES[type];
        this.color = config.color;
        this.effect = config.effect;
        this.value = config.value;
        
        // Animation
        this.baseY = 1;
        this.animationTime = 0;
        
        // Create mesh
        this.createMesh();
    }
    
    createMesh() {
        const geometry = new THREE.OctahedronGeometry(0.5);
        const material = new THREE.MeshLambertMaterial({ 
            color: this.color,
            emissive: this.color,
            emissiveIntensity: 0.3
        });
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.position.copy(this.position);
    }
    
    update(time) {
        // Rotate and bob animation
        this.mesh.rotation.y = time;
        this.mesh.position.y = this.baseY + Math.sin(time * 2) * 0.2;
    }
    
    getEffect() {
        return {
            type: this.type,
            effect: this.effect,
            value: this.value
        };
    }
}