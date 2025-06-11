import { CONFIG } from './config.js';
import { Utils } from './utils.js';
import { Player } from './entities/Player.js';
import { Enemy } from './entities/Enemy.js';
import { Item } from './entities/Item.js';

export class EntityManager {
    constructor(events) {
        this.events = events;
        this.player = null;
        this.enemies = [];
        this.items = [];
    }
    
    createPlayer(x, z) {
        this.player = new Player(x, z, this.events);
        return this.player;
    }
    
    createEnemy(x, z, type) {
        const enemy = new Enemy(x, z, type, this.events);
        this.enemies.push(enemy);
        return enemy;
    }
    
    createItem(x, z, type) {
        const item = new Item(x, z, type);
        this.items.push(item);
        return item;
    }
    
    getPlayer() {
        return this.player;
    }
    
    getPlayerStats() {
        return this.player ? this.player.getStats() : null;
    }
    
    applyMetaUpgrades(upgrades) {
        if (this.player) {
            this.player.applyMetaUpgrades(upgrades);
        }
    }
    
    update(walls) {
        if (!this.player) return;
        
        // Update player
        this.player.update();
        
        // Update enemies
        this.updateEnemies(walls);
        
        // Update items
        this.updateItems();
    }
    
    updateEnemies(walls) {
        const currentTime = Date.now();
        
        for (let i = this.enemies.length - 1; i >= 0; i--) {
            const enemy = this.enemies[i];
            
            if (enemy.isDead()) {
                this.handleEnemyDeath(enemy, i);
                continue;
            }
            
            enemy.update(this.player, walls, currentTime);
        }
    }
    
    updateItems() {
        const time = Date.now() * 0.001;
        this.items.forEach(item => item.update(time));
    }
    
    handleEnemyDeath(enemy, index) {
        // Grant rewards
        const rewards = enemy.getRewards();
        this.player.addExperience(rewards.exp);
        this.player.addGold(rewards.gold);
        this.player.addSouls(rewards.souls);
        
        // Emit event
        this.events.emit('enemy:defeated', rewards);
        
        // Drop item chance
        if (Math.random() < CONFIG.ITEM.DROP_CHANCE) {
            const types = Object.keys(CONFIG.ITEM.TYPES);
            const type = Utils.randomChoice(types);
            const item = this.createItem(
                enemy.mesh.position.x,
                enemy.mesh.position.z,
                type
            );
            item.mesh.position.y = 1; // Reset Y position
        }
        
        // Remove from scene and array
        enemy.destroy();
        this.enemies.splice(index, 1);
    }
    
    movePlayer(inputData, walls) {
        if (this.player) {
            this.player.move(inputData, walls);
        }
    }
    
    playerAttack() {
        if (!this.player) return;
        
        const attackData = this.player.attack();
        if (!attackData) return;
        
        // Check hit enemies
        this.enemies.forEach(enemy => {
            const distance = this.player.mesh.position.distanceTo(enemy.mesh.position);
            
            if (distance < attackData.range) {
                // Check if enemy is in front of player
                const toEnemy = enemy.mesh.position.clone()
                    .sub(this.player.mesh.position)
                    .normalize();
                const forward = new THREE.Vector3(0, 0, -1)
                    .applyQuaternion(this.player.mesh.quaternion);
                const angle = forward.angleTo(toEnemy);
                
                if (angle < attackData.angle) {
                    const damage = this.player.calculateDamage();
                    enemy.takeDamage(damage);
                    
                    // Knockback
                    const knockback = toEnemy.multiplyScalar(0.5);
                    enemy.mesh.position.add(knockback);
                    
                    this.events.emit('player:attack', { damage });
                }
            }
        });
    }
    
    useItem() {
        if (this.player) {
            this.player.useItem();
        }
    }
    
    checkItemPickups(player) {
        for (let i = this.items.length - 1; i >= 0; i--) {
            const item = this.items[i];
            const distance = player.mesh.position.distanceTo(item.mesh.position);
            
            if (distance < CONFIG.ITEM.PICKUP_RANGE) {
                // Apply item effect
                const effect = item.getEffect();
                this.applyItemEffect(effect);
                
                // Remove item
                item.destroy();
                this.items.splice(i, 1);
                
                // Emit event
                this.events.emit('item:pickup', effect);
            }
        }
    }
    
    applyItemEffect(effect) {
        if (!this.player) return;
        
        switch (effect.effect) {
            case 'heal':
                this.player.heal(effect.value);
                break;
            case 'buff_attack':
                this.player.buffAttack(effect.value);
                break;
            case 'buff_defense':
                this.player.buffDefense(effect.value);
                break;
            case 'gold':
                this.player.addGold(effect.value);
                break;
        }
    }
    
    upgradePlayerHealth(amount) {
        if (this.player) {
            this.player.upgradeMaxHealth(amount);
        }
    }
    
    upgradePlayerAttack(amount) {
        if (this.player) {
            this.player.buffAttack(amount);
        }
    }
    
    upgradePlayerDefense(amount) {
        if (this.player) {
            this.player.buffDefense(amount);
        }
    }
    
    healPlayerFull() {
        if (this.player) {
            this.player.healFull();
        }
    }
    
    clearAll() {
        // Clear player
        if (this.player) {
            this.player.destroy();
            this.player = null;
        }
        
        // Clear enemies
        this.enemies.forEach(enemy => enemy.destroy());
        this.enemies = [];
        
        // Clear items
        this.items.forEach(item => item.destroy());
        this.items = [];
    }
}