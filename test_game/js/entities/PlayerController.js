// Player Controller
// Handles player creation, movement, combat, and camera

import { gameState } from '../core/GameState.js';
import { playerStats } from '../core/PlayerStats.js';

export class PlayerController {
    constructor() {
        this.speed = 0.1;
        this.rotSpeed = 0.02; // Reduced from 0.05 for smoother camera
        this.attackRange = 3;
        this.attackAngle = Math.PI / 4;
        this.cameraYaw = 0; // Track camera rotation separately
    }

    createPlayer(x, z) {
        // Use CylinderGeometry instead of CapsuleGeometry for compatibility
        const playerGeometry = new THREE.CylinderGeometry(0.5, 0.5, 2, 8);
        const playerMaterial = new THREE.MeshLambertMaterial({ color: 0x00ff00 });
        gameState.player = new THREE.Mesh(playerGeometry, playerMaterial);
        gameState.player.position.set(x, 1.5, z);
        gameState.player.castShadow = true;
        
        // Add weapon (sword)
        const swordGeometry = new THREE.BoxGeometry(0.1, 1.5, 0.1);
        const swordMaterial = new THREE.MeshLambertMaterial({ color: 0xcccccc });
        const sword = new THREE.Mesh(swordGeometry, swordMaterial);
        sword.position.set(0.7, 0, 0);
        gameState.player.add(sword);
        
        gameState.scene.add(gameState.player);
        
        // Camera follows player
        this.updateCamera();
    }

    updatePlayer() {
        if (!gameState.player || !gameState.gameRunning) return;
        
        // Smooth camera rotation with dampening
        const targetYaw = this.cameraYaw + gameState.mouseX * this.rotSpeed;
        this.cameraYaw += (targetYaw - this.cameraYaw) * 0.1; // Smooth interpolation
        gameState.player.rotation.y = this.cameraYaw;
        
        // Gradually reduce mouse input for smoother feel
        gameState.mouseX *= 0.9;
        
        // Movement
        const moveVector = new THREE.Vector3();
        
        if (gameState.controls.forward) {
            moveVector.z -= this.speed;
        }
        if (gameState.controls.backward) {
            moveVector.z += this.speed;
        }
        if (gameState.controls.left) {
            moveVector.x -= this.speed;
        }
        if (gameState.controls.right) {
            moveVector.x += this.speed;
        }
        
        // Apply rotation to movement
        moveVector.applyQuaternion(gameState.player.quaternion);
        
        // Check collision before moving
        const newPosition = gameState.player.position.clone().add(moveVector);
        if (!this.checkWallCollision(newPosition)) {
            gameState.player.position.add(moveVector);
        }
        
        // Attack
        if (gameState.controls.attack) {
            this.playerAttack();
            gameState.controls.attack = false;
        }
    }

    updateCamera() {
        if (!gameState.player) return;
        
        // Third person camera
        const distance = 10;
        const height = 8;
        
        gameState.camera.position.x = gameState.player.position.x - distance * Math.sin(gameState.player.rotation.y);
        gameState.camera.position.y = gameState.player.position.y + height;
        gameState.camera.position.z = gameState.player.position.z - distance * Math.cos(gameState.player.rotation.y);
        
        gameState.camera.lookAt(gameState.player.position);
    }

    centerCamera() {
        // Reset camera to behind player
        this.cameraYaw = 0;
        gameState.player.rotation.y = 0;
        gameState.mouseX = 0; // Reset mouse accumulation
    }

    checkWallCollision(position) {
        const playerRadius = 0.5;
        
        for (const wall of gameState.walls) {
            const wallBox = new THREE.Box3().setFromObject(wall);
            const playerSphere = new THREE.Sphere(position, playerRadius);
            
            if (wallBox.intersectsSphere(playerSphere)) {
                return true;
            }
        }
        return false;
    }

    playerAttack() {
        const hitEnemies = [];
        
        gameState.enemies.forEach(enemy => {
            const distance = gameState.player.position.distanceTo(enemy.position);
            
            if (distance < this.attackRange) {
                // Check if enemy is in front of player
                const toEnemy = enemy.position.clone().sub(gameState.player.position).normalize();
                const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(gameState.player.quaternion);
                const angle = forward.angleTo(toEnemy);
                
                if (angle < this.attackAngle) {
                    // Deal damage
                    const damage = playerStats.attack + Math.floor(Math.random() * 5);
                    enemy.userData.hp -= damage;
                    
                    hitEnemies.push({ enemy, damage });
                    
                    // Knockback
                    const knockback = toEnemy.multiplyScalar(0.5);
                    enemy.position.add(knockback);
                }
            }
        });
        
        return hitEnemies;
    }

    checkItemPickup() {
        const pickupRange = 2;
        const pickedUpItems = [];
        
        gameState.items.forEach((item, index) => {
            const distance = gameState.player.position.distanceTo(item.position);
            
            if (distance < pickupRange) {
                pickedUpItems.push({ item, index });
            }
        });
        
        return pickedUpItems;
    }

    checkExit() {
        let nearExit = false;
        
        gameState.scene.traverse((child) => {
            if (child.userData && child.userData.type === 'exit') {
                const distance = gameState.player.position.distanceTo(child.position);
                if (distance < 3) {
                    nearExit = true;
                }
            }
        });
        
        return nearExit;
    }
}

// Singleton instance
export const playerController = new PlayerController();