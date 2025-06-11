import { Entity } from './Entity.js';
import { CONFIG } from '../config.js';

export class Enemy extends Entity {
    constructor(x, z, type, events) {
        super(x, 0.75, z);
        this.events = events;
        this.type = type;
        
        // Load enemy config
        const config = CONFIG.ENEMY.TYPES[type];
        this.maxHealth = config.hp;
        this.health = config.hp;
        this.attack = config.attack;
        this.exp = config.exp;
        this.speed = config.speed;
        this.color = config.color;
        
        // AI state
        this.lastAttackTime = 0;
        this.target = null;
        
        // Create mesh
        this.createMesh();
    }
    
    createMesh() {
        const geometry = new THREE.BoxGeometry(1, 1.5, 1);
        const material = new THREE.MeshLambertMaterial({ color: this.color });
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.position.copy(this.position);
        this.mesh.castShadow = true;
    }
    
    update(player, walls, currentTime) {
        if (!player || player.isDead()) return;
        
        const distance = this.mesh.position.distanceTo(player.mesh.position);
        
        // AI behavior
        if (distance < CONFIG.ENEMY.AGGRO_RANGE) {
            // Move towards player
            const direction = player.mesh.position.clone()
                .sub(this.mesh.position)
                .normalize();
            const moveVector = direction.multiplyScalar(this.speed);
            
            // Check collision before moving
            const newPosition = this.mesh.position.clone().add(moveVector);
            if (!this.checkCollision(newPosition, walls)) {
                this.mesh.position.add(moveVector);
                this.position.copy(this.mesh.position);
            }
            
            // Face player
            this.mesh.lookAt(player.mesh.position);
            
            // Attack if close enough
            if (distance < CONFIG.ENEMY.ATTACK_RANGE && 
                currentTime - this.lastAttackTime > CONFIG.ENEMY.ATTACK_COOLDOWN) {
                this.attackPlayer(player);
                this.lastAttackTime = currentTime;
            }
        }
    }
    
    attackPlayer(player) {
        const damage = player.takeDamage(this.attack);
        this.events.emit('enemy:attack', { damage, enemy: this });
    }
    
    takeDamage(amount) {
        super.takeDamage(amount);
        
        // Visual feedback
        this.flashDamage();
    }
    
    flashDamage() {
        const originalColor = this.mesh.material.color.getHex();
        this.mesh.material.color.setHex(0xffffff);
        
        setTimeout(() => {
            if (this.mesh && this.mesh.material) {
                this.mesh.material.color.setHex(originalColor);
            }
        }, 100);
    }
    
    getRewards() {
        return {
            exp: this.exp,
            gold: Math.floor(Math.random() * 20) + 10,
            souls: Math.floor(this.exp / 10)
        };
    }
}