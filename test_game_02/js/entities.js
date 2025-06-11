/**
 * Entity System for Shadowfall Depths
 * Base classes for game entities (Player, Enemies, Items)
 */

class Entity {
    constructor(x, y, width, height) {
        this.id = Utils.generateUUID();
        this.x = x || 0;
        this.y = y || 0;
        this.width = width || 32;
        this.height = height || 32;
        this.vx = 0;
        this.vy = 0;
        this.health = 100;
        this.maxHealth = 100;
        this.alive = true;
        this.visible = true;
        this.solid = true;
        this.tags = new Set();
        
        // Animation properties
        this.sprite = null;
        this.frame = 0;
        this.animationSpeed = 0.1;
        this.lastFrameTime = 0;
        
        // Combat properties
        this.damage = 10;
        this.defense = 0;
        this.attackCooldown = 0;
        this.maxAttackCooldown = 500;
    }
    
    update(deltaTime) {
        // Update position
        this.x += this.vx * deltaTime / 1000;
        this.y += this.vy * deltaTime / 1000;
        
        // Update attack cooldown
        if (this.attackCooldown > 0) {
            this.attackCooldown -= deltaTime;
        }
        
        // Update animation
        this.updateAnimation(deltaTime);
    }
    
    updateAnimation(deltaTime) {
        this.lastFrameTime += deltaTime;
        if (this.lastFrameTime >= this.animationSpeed * 1000) {
            this.frame = (this.frame + 1) % 4; // Assuming 4-frame animation
            this.lastFrameTime = 0;
        }
    }
    
    render(ctx) {
        if (!this.visible) return;
        
        // Simple colored rectangle for now
        ctx.fillStyle = this.getColor();
        ctx.fillRect(this.x, this.y, this.width, this.height);
        
        // Health bar for entities with less than max health
        if (this.health < this.maxHealth) {
            this.renderHealthBar(ctx);
        }
    }
    
    renderHealthBar(ctx) {
        const barWidth = this.width;
        const barHeight = 4;
        const barY = this.y - 8;
        
        // Background
        ctx.fillStyle = '#333';
        ctx.fillRect(this.x, barY, barWidth, barHeight);
        
        // Health
        const healthPercent = this.health / this.maxHealth;
        ctx.fillStyle = healthPercent > 0.5 ? '#4CAF50' : healthPercent > 0.25 ? '#FF9800' : '#F44336';
        ctx.fillRect(this.x, barY, barWidth * healthPercent, barHeight);
    }
    
    getColor() {
        return '#FFFFFF';
    }
    
    getBounds() {
        return {
            x: this.x,
            y: this.y,
            width: this.width,
            height: this.height
        };
    }
    
    getCenter() {
        return {
            x: this.x + this.width / 2,
            y: this.y + this.height / 2
        };
    }
    
    collidesWith(other) {
        return Utils.rectOverlap(
            this.x, this.y, this.width, this.height,
            other.x, other.y, other.width, other.height
        );
    }
    
    takeDamage(amount, source = null) {
        const actualDamage = Math.max(0, amount - this.defense);
        this.health -= actualDamage;
        
        if (this.health <= 0) {
            this.health = 0;
            this.die();
        }
        
        // Emit damage event
        gameEngine.emit('entity:damaged', {
            entity: this,
            damage: actualDamage,
            source: source
        });
        
        return actualDamage;
    }
    
    heal(amount) {
        const oldHealth = this.health;
        this.health = Math.min(this.maxHealth, this.health + amount);
        const actualHealing = this.health - oldHealth;
        
        gameEngine.emit('entity:healed', {
            entity: this,
            healing: actualHealing
        });
        
        return actualHealing;
    }
    
    die() {
        this.alive = false;
        gameEngine.emit('entity:died', { entity: this });
    }
    
    canAttack() {
        return this.attackCooldown <= 0;
    }
    
    attack(target) {
        if (!this.canAttack() || !target) return false;
        
        const damage = target.takeDamage(this.damage, this);
        this.attackCooldown = this.maxAttackCooldown;
        
        gameEngine.emit('entity:attacked', {
            attacker: this,
            target: target,
            damage: damage
        });
        
        return true;
    }
    
    addTag(tag) {
        this.tags.add(tag);
    }
    
    removeTag(tag) {
        this.tags.delete(tag);
    }
    
    hasTag(tag) {
        return this.tags.has(tag);
    }
}

class Player extends Entity {
    constructor(x, y) {
        super(x, y, 24, 24);
        this.maxHealth = 100;
        this.health = this.maxHealth;
        this.damage = 25;
        this.speed = 150;
        this.experience = 0;
        this.level = 1;
        this.souls = 0;
        
        // Player-specific stats
        this.luck = 0;
        this.critChance = 0.05;
        this.critMultiplier = 2.0;
        
        // Inventory
        this.inventory = [];
        this.maxInventorySize = 20;
        
        // Movement
        this.moveDirection = { x: 0, y: 0 };
        this.isMoving = false;
        
        this.addTag('player');
    }
    
    update(deltaTime) {
        super.update(deltaTime);
        
        // Apply movement
        if (this.moveDirection.x !== 0 || this.moveDirection.y !== 0) {
            // Normalize diagonal movement
            const magnitude = Math.sqrt(this.moveDirection.x ** 2 + this.moveDirection.y ** 2);
            const normalizedX = this.moveDirection.x / magnitude;
            const normalizedY = this.moveDirection.y / magnitude;
            
            this.vx = normalizedX * this.speed;
            this.vy = normalizedY * this.speed;
            this.isMoving = true;
        } else {
            this.vx = 0;
            this.vy = 0;
            this.isMoving = false;
        }
        
        // Update camera to follow player
        gameEngine.setCameraTarget(this.x + this.width / 2, this.y + this.height / 2);
    }
    
    getColor() {
        return '#4CAF50'; // Green for player
    }
    
    setMoveDirection(x, y) {
        this.moveDirection.x = Utils.clamp(x, -1, 1);
        this.moveDirection.y = Utils.clamp(y, -1, 1);
    }
    
    addItem(item) {
        if (this.inventory.length >= this.maxInventorySize) {
            return false;
        }
        
        this.inventory.push(item);
        gameEngine.emit('player:itemAdded', { player: this, item: item });
        return true;
    }
    
    removeItem(item) {
        const index = this.inventory.indexOf(item);
        if (index > -1) {
            this.inventory.splice(index, 1);
            gameEngine.emit('player:itemRemoved', { player: this, item: item });
            return true;
        }
        return false;
    }
    
    gainExperience(amount) {
        this.experience += amount;
        const expNeeded = this.getExpNeededForLevel(this.level + 1);
        
        if (this.experience >= expNeeded) {
            this.levelUp();
        }
        
        gameEngine.emit('player:expGained', { player: this, amount: amount });
    }
    
    getExpNeededForLevel(level) {
        return level * 100; // Simple linear progression
    }
    
    levelUp() {
        this.level++;
        const healthIncrease = 20;
        this.maxHealth += healthIncrease;
        this.health += healthIncrease;
        this.damage += 5;
        
        gameEngine.emit('player:levelUp', { player: this, level: this.level });
    }
    
    gainSouls(amount) {
        this.souls += amount;
        gameEngine.emit('player:soulsGained', { player: this, amount: amount });
    }
    
    spendSouls(amount) {
        if (this.souls >= amount) {
            this.souls -= amount;
            gameEngine.emit('player:soulsSpent', { player: this, amount: amount });
            return true;
        }
        return false;
    }
}

class Enemy extends Entity {
    constructor(x, y, type = 'basic') {
        super(x, y, 20, 20);
        this.type = type;
        this.speed = 50;
        this.detectionRange = 100;
        this.attackRange = 30;
        this.target = null;
        this.ai = new BasicAI(this);
        this.expValue = 10;
        this.soulValue = 1;
        
        this.addTag('enemy');
        this.setupByType(type);
    }
    
    setupByType(type) {
        switch (type) {
            case 'goblin':
                this.health = this.maxHealth = 30;
                this.damage = 15;
                this.speed = 60;
                this.expValue = 15;
                this.soulValue = 2;
                break;
            case 'orc':
                this.health = this.maxHealth = 60;
                this.damage = 25;
                this.speed = 40;
                this.expValue = 25;
                this.soulValue = 3;
                break;
            case 'skeleton':
                this.health = this.maxHealth = 40;
                this.damage = 20;
                this.speed = 45;
                this.expValue = 20;
                this.soulValue = 2;
                break;
            default: // basic
                this.health = this.maxHealth = 25;
                this.damage = 10;
                this.speed = 50;
                break;
        }
    }
    
    update(deltaTime) {
        super.update(deltaTime);
        
        if (this.ai) {
            this.ai.update(deltaTime);
        }
    }
    
    getColor() {
        const colors = {
            basic: '#FF5722',
            goblin: '#8BC34A',
            orc: '#795548',
            skeleton: '#9E9E9E'
        };
        return colors[this.type] || colors.basic;
    }
    
    die() {
        super.die();
        
        // Drop experience and souls
        if (this.target && this.target.hasTag('player')) {
            this.target.gainExperience(this.expValue);
            this.target.gainSouls(this.soulValue);
        }
    }
}

class BasicAI {
    constructor(entity) {
        this.entity = entity;
        this.state = 'idle'; // idle, chasing, attacking
        this.lastStateChange = 0;
        this.pathfindingCooldown = 0;
    }
    
    update(deltaTime) {
        this.pathfindingCooldown -= deltaTime;
        
        // Find player target
        if (!this.entity.target) {
            this.findTarget();
        }
        
        if (this.entity.target && this.entity.target.alive) {
            const distance = Utils.distance(
                this.entity.x, this.entity.y,
                this.entity.target.x, this.entity.target.y
            );
            
            if (distance <= this.entity.attackRange) {
                this.setState('attacking');
            } else if (distance <= this.entity.detectionRange) {
                this.setState('chasing');
            } else {
                this.setState('idle');
                this.entity.target = null;
            }
        } else {
            this.setState('idle');
        }
        
        this.executeState(deltaTime);
    }
    
    setState(newState) {
        if (this.state !== newState) {
            this.state = newState;
            this.lastStateChange = Date.now();
        }
    }
    
    executeState(deltaTime) {
        switch (this.state) {
            case 'idle':
                this.entity.vx = 0;
                this.entity.vy = 0;
                break;
                
            case 'chasing':
                this.chaseTarget(deltaTime);
                break;
                
            case 'attacking':
                this.attackTarget(deltaTime);
                break;
        }
    }
    
    findTarget() {
        // Simple target finding - look for player
        // In a real game, you'd iterate through entities
        if (window.player && window.player.alive) {
            const distance = Utils.distance(
                this.entity.x, this.entity.y,
                window.player.x, window.player.y
            );
            
            if (distance <= this.entity.detectionRange) {
                this.entity.target = window.player;
            }
        }
    }
    
    chaseTarget(deltaTime) {
        if (!this.entity.target) return;
        
        const dx = this.entity.target.x - this.entity.x;
        const dy = this.entity.target.y - this.entity.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > 0) {
            this.entity.vx = (dx / distance) * this.entity.speed;
            this.entity.vy = (dy / distance) * this.entity.speed;
        }
    }
    
    attackTarget(deltaTime) {
        this.entity.vx = 0;
        this.entity.vy = 0;
        
        if (this.entity.canAttack() && this.entity.target) {
            this.entity.attack(this.entity.target);
        }
    }
}

// Export classes
window.Entity = Entity;
window.Player = Player;
window.Enemy = Enemy;
window.BasicAI = BasicAI;