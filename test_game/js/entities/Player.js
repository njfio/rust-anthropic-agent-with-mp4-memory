import { Entity } from './Entity.js';
import { CONFIG, COLORS } from '../config.js';

export class Player extends Entity {
    constructor(x, z, events) {
        super(x, 1.5, z);
        this.events = events;
        
        // Stats
        this.level = 1;
        this.exp = 0;
        this.expNeeded = 100;
        this.attack = CONFIG.PLAYER.BASE_ATTACK;
        this.defense = CONFIG.PLAYER.BASE_DEFENSE;
        this.gold = 0;
        this.souls = 0;
        this.inventory = [];
        
        // Movement
        this.speed = CONFIG.PLAYER.BASE_SPEED;
        this.rotationSpeed = CONFIG.PLAYER.ROTATION_SPEED;
        
        // Combat
        this.lastAttackTime = 0;
        this.attackCooldown = 500;
        
        // Create mesh
        this.createMesh();
    }
    
    createMesh() {
        // Player body
        const geometry = new THREE.CylinderGeometry(0.5, 0.5, 2, 8);
        const material = new THREE.MeshLambertMaterial({ color: COLORS.PLAYER });
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.position.copy(this.position);
        this.mesh.castShadow = true;
        
        // Add weapon (sword)
        const swordGeometry = new THREE.BoxGeometry(0.1, 1.5, 0.1);
        const swordMaterial = new THREE.MeshLambertMaterial({ color: COLORS.SWORD });
        const sword = new THREE.Mesh(swordGeometry, swordMaterial);
        sword.position.set(0.7, 0, 0);
        this.mesh.add(sword);
    }
    
    applyMetaUpgrades(upgrades) {
        this.maxHealth = CONFIG.PLAYER.BASE_HP + (upgrades.startingHealth * 20);
        this.health = this.maxHealth;
        this.attack = CONFIG.PLAYER.BASE_ATTACK + (upgrades.startingAttack * 2);
        this.defense = CONFIG.PLAYER.BASE_DEFENSE + (upgrades.startingDefense * 2);
    }
    
    move(inputData, walls) {
        // Rotation based on mouse
        this.mesh.rotation.y += inputData.mouseX * this.rotationSpeed;
        
        // Movement
        const moveVector = new THREE.Vector3();
        
        if (inputData.forward) moveVector.z -= this.speed;
        if (inputData.backward) moveVector.z += this.speed;
        if (inputData.left) moveVector.x -= this.speed;
        if (inputData.right) moveVector.x += this.speed;
        
        // Apply rotation to movement
        moveVector.applyQuaternion(this.mesh.quaternion);
        
        // Check collision before moving
        const newPosition = this.mesh.position.clone().add(moveVector);
        if (!this.checkCollision(newPosition, walls)) {
            this.mesh.position.add(moveVector);
            this.position.copy(this.mesh.position);
        }
    }
    
    attack() {
        const currentTime = Date.now();
        if (currentTime - this.lastAttackTime < this.attackCooldown) {
            return null;
        }
        
        this.lastAttackTime = currentTime;
        
        return {
            range: CONFIG.PLAYER.ATTACK_RANGE,
            angle: CONFIG.PLAYER.ATTACK_ANGLE,
            damage: this.calculateDamage()
        };
    }
    
    calculateDamage() {
        return this.attack + Math.floor(Math.random() * 5);
    }
    
    takeDamage(amount) {
        const damage = Math.max(1, amount - this.defense);
        super.takeDamage(damage);
        
        if (this.isDead()) {
            this.events.emit('player:death');
        }
        
        return damage;
    }
    
    addExperience(amount) {
        this.exp += amount;
        
        while (this.exp >= this.expNeeded) {
            this.levelUp();
        }
    }
    
    levelUp() {
        this.level++;
        this.exp -= this.expNeeded;
        this.expNeeded = Math.floor(this.expNeeded * 1.5);
        this.maxHealth += CONFIG.PLAYER.LEVEL_HP_BONUS;
        this.health = this.maxHealth;
        this.attack += CONFIG.PLAYER.LEVEL_ATTACK_BONUS;
        this.defense += CONFIG.PLAYER.LEVEL_DEFENSE_BONUS;
        
        this.events.emit('player:levelup', { level: this.level });
    }
    
    addGold(amount) {
        this.gold += amount;
    }
    
    addSouls(amount) {
        this.souls += amount;
    }
    
    buffAttack(amount) {
        this.attack += amount;
    }
    
    buffDefense(amount) {
        this.defense += amount;
    }
    
    upgradeMaxHealth(amount) {
        this.maxHealth += amount;
        this.health += amount;
    }
    
    healFull() {
        this.health = this.maxHealth;
    }
    
    useItem() {
        if (this.inventory.length > 0) {
            const item = this.inventory.shift();
            // Apply item effect
            this.events.emit('item:used', item);
        }
    }
    
    getStats() {
        return {
            level: this.level,
            exp: this.exp,
            expNeeded: this.expNeeded,
            hp: this.health,
            maxHp: this.maxHealth,
            attack: this.attack,
            defense: this.defense,
            gold: this.gold,
            souls: this.souls,
            inventory: this.inventory
        };
    }
}