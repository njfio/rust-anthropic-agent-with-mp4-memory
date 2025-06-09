// Item Manager
// Handles item creation, placement, and effects

import { gameState } from '../core/GameState.js';
import { playerStats } from '../core/PlayerStats.js';

export class ItemManager {
    constructor() {
        this.itemTypes = {
            health: { color: 0x00ff00, effect: 'heal', value: 30 },
            attack: { color: 0xff0000, effect: 'buff_attack', value: 5 },
            defense: { color: 0x0000ff, effect: 'buff_defense', value: 3 },
            gold: { color: 0xffff00, effect: 'gold', value: 50 }
        };
    }

    createItem(x, z, type) {
        const itemData = this.itemTypes[type];
        const itemGeometry = new THREE.OctahedronGeometry(0.5);
        const itemMaterial = new THREE.MeshLambertMaterial({ 
            color: itemData.color,
            emissive: itemData.color,
            emissiveIntensity: 0.3
        });
        const item = new THREE.Mesh(itemGeometry, itemMaterial);
        
        item.position.set(x, 1, z);
        item.userData = {
            type: type,
            effect: itemData.effect,
            value: itemData.value
        };
        
        gameState.items.push(item);
        gameState.scene.add(item);
        
        return item;
    }

    placeItems(rooms) {
        const placedItems = [];
        
        rooms.forEach((room, index) => {
            const itemCount = Math.floor(Math.random() * 2) + 1;
            for (let i = 0; i < itemCount; i++) {
                const x = (room.x1 + Math.random() * (room.x2 - room.x1)) * 4;
                const z = (room.y1 + Math.random() * (room.y2 - room.y1)) * 4;
                
                const types = ['health', 'attack', 'defense', 'gold'];
                const type = types[Math.floor(Math.random() * types.length)];
                
                const item = this.createItem(
                    x - (20 * 4) / 2,
                    z - (20 * 4) / 2,
                    type
                );
                placedItems.push(item);
            }
        });
        
        return placedItems;
    }

    applyItemEffect(item) {
        const effect = item.userData.effect;
        const value = item.userData.value;
        let message = '';
        
        switch(effect) {
            case 'heal':
                const healAmount = playerStats.heal(value);
                message = `Healed for ${healAmount} HP!`;
                break;
            case 'buff_attack':
                playerStats.buffAttack(value);
                message = `Attack increased by ${value}!`;
                break;
            case 'buff_defense':
                playerStats.buffDefense(value);
                message = `Defense increased by ${value}!`;
                break;
            case 'gold':
                playerStats.addGold(value);
                message = `Found ${value} gold!`;
                break;
        }
        
        return message;
    }

    removeItem(item) {
        const index = gameState.items.indexOf(item);
        if (index > -1) {
            gameState.items.splice(index, 1);
        }
        gameState.scene.remove(item);
    }

    animateItems() {
        const time = Date.now() * 0.001;
        
        gameState.items.forEach(item => {
            item.rotation.y = time;
            item.position.y = 1 + Math.sin(time * 2) * 0.2;
        });
        
        // Animate exit portal
        gameState.scene.traverse((child) => {
            if (child.userData && child.userData.type === 'exit') {
                child.rotation.y = time;
                child.children.forEach(glow => {
                    if (glow.material) {
                        glow.material.opacity = 0.3 + Math.sin(time * 3) * 0.2;
                    }
                });
            }
        });
    }

    useInventoryItem() {
        if (playerStats.inventory.length > 0) {
            const item = playerStats.inventory.shift();
            return `Used ${item.name}!`;
        }
        return null;
    }
}

// Singleton instance
export const itemManager = new ItemManager();