// Enemy Manager
// Handles enemy creation, AI, and combat

import { gameState } from '../core/GameState.js';
import { playerStats } from '../core/PlayerStats.js';

export class EnemyManager {
    constructor() {
        this.enemyTypes = {
            basic: { color: 0xff0000, hp: 30, attack: 5, exp: 20, speed: 0.02 },
            elite: { color: 0xff00ff, hp: 60, attack: 10, exp: 50, speed: 0.03 },
            boss: { color: 0xff8800, hp: 100, attack: 15, exp: 100, speed: 0.025 }
        };
    }

    createEnemy(x, z, type = 'basic') {
        const enemyData = this.enemyTypes[type];
        const enemyGeometry = new THREE.BoxGeometry(1, 1.5, 1);
        const enemyMaterial = new THREE.MeshLambertMaterial({ color: enemyData.color });
        const enemy = new THREE.Mesh(enemyGeometry, enemyMaterial);
        
        enemy.position.set(x, 0.75, z);
        enemy.castShadow = true;
        enemy.userData = {
            type: type,
            hp: enemyData.hp,
            maxHp: enemyData.hp,
            attack: enemyData.attack,
            exp: enemyData.exp,
            speed: enemyData.speed,
            lastAttack: 0
        };
        
        gameState.enemies.push(enemy);
        gameState.scene.add(enemy);
        
        return enemy;
    }

    placeEnemies(rooms) {
        const placedEnemies = [];
        
        rooms.forEach((room, index) => {
            if (index === 0) return; // No enemies in starting room
            
            const enemyCount = Math.floor(Math.random() * 3) + 1;
            for (let i = 0; i < enemyCount; i++) {
                const x = (room.x1 + Math.random() * (room.x2 - room.x1)) * 4;
                const z = (room.y1 + Math.random() * (room.y2 - room.y1)) * 4;
                
                const type = Math.random() < 0.8 ? 'basic' : 'elite';
                const enemy = this.createEnemy(
                    x - (20 * 4) / 2,
                    z - (20 * 4) / 2,
                    type
                );
                placedEnemies.push(enemy);
            }
        });
        
        return placedEnemies;
    }

    updateEnemies() {
        const currentTime = Date.now();
        const attackedPlayer = [];
        
        gameState.enemies.forEach(enemy => {
            // Simple AI - move towards player
            const distance = enemy.position.distanceTo(gameState.player.position);
            
            if (distance < 15) {
                const direction = gameState.player.position.clone().sub(enemy.position).normalize();
                const moveVector = direction.multiplyScalar(enemy.userData.speed);
                
                // Check collision before moving
                const newPosition = enemy.position.clone().add(moveVector);
                if (!this.checkWallCollision(newPosition)) {
                    enemy.position.add(moveVector);
                }
                
                // Face player
                enemy.lookAt(gameState.player.position);
                
                // Attack if close enough
                if (distance < 2 && currentTime - enemy.userData.lastAttack > 1000) {
                    const attackResult = this.enemyAttack(enemy);
                    enemy.userData.lastAttack = currentTime;
                    attackedPlayer.push(attackResult);
                }
            }
        });
        
        return attackedPlayer;
    }

    enemyAttack(enemy) {
        const damage = Math.max(1, enemy.userData.attack - playerStats.defense);
        const result = playerStats.takeDamage(damage);
        
        return {
            damage: result.damage,
            isDead: result.isDead
        };
    }

    defeatedEnemy(enemy) {
        const index = gameState.enemies.indexOf(enemy);
        if (index > -1) {
            gameState.enemies.splice(index, 1);
        }
        
        // Grant exp and gold
        const expGained = enemy.userData.exp;
        const goldGained = Math.floor(Math.random() * 20) + 10;
        const soulsGained = Math.floor(enemy.userData.exp / 10);
        
        playerStats.gainExp(expGained);
        playerStats.addGold(goldGained);
        playerStats.addSouls(soulsGained);
        
        gameState.scene.remove(enemy);
        
        return {
            exp: expGained,
            gold: goldGained,
            souls: soulsGained,
            shouldDropItem: Math.random() < 0.3,
            position: { x: enemy.position.x, z: enemy.position.z }
        };
    }

    checkWallCollision(position) {
        const enemyRadius = 0.5;
        
        for (const wall of gameState.walls) {
            const wallBox = new THREE.Box3().setFromObject(wall);
            const enemySphere = new THREE.Sphere(position, enemyRadius);
            
            if (wallBox.intersectsSphere(enemySphere)) {
                return true;
            }
        }
        return false;
    }
}

// Singleton instance
export const enemyManager = new EnemyManager();