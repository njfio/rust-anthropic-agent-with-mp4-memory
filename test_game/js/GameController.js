// Game Controller
// Main game controller that orchestrates all systems

import { gameState } from './core/GameState.js';
import { playerStats, metaProgression } from './core/PlayerStats.js';
import { sceneManager } from './core/SceneManager.js';
import { dungeonGenerator } from './world/DungeonGenerator.js';
import { playerController } from './entities/PlayerController.js';
import { enemyManager } from './entities/EnemyManager.js';
import { itemManager } from './entities/ItemManager.js';
import { inputManager } from './input/InputManager.js';
import { uiManager } from './ui/UIManager.js';

export class GameController {
    constructor() {
        this.initialized = false;
        this.setupEventListeners();
    }

    init() {
        if (this.initialized) return;
        
        sceneManager.init();
        // inputManager is initialized automatically
        this.initialized = true;
    }

    setupEventListeners() {
        document.addEventListener('useItem', this.onUseItem.bind(this));
        document.addEventListener('centerCamera', this.onCenterCamera.bind(this));
    }

    startNewRun() {
        // Reset player stats and apply meta upgrades
        playerStats.reset();
        this.applyMetaUpgrades();
        
        gameState.dungeonFloor = 1;
        gameState.gameRunning = true;
        
        uiManager.hideMainMenu();
        
        this.generateNewFloor();
        uiManager.updateUI();
        uiManager.addMessage('Welcome to the dungeon!', 'level');
        
        this.gameLoop();
    }

    applyMetaUpgrades() {
        playerStats.maxHp = 100 + (metaProgression.upgrades.startingHealth * 20);
        playerStats.hp = playerStats.maxHp;
        playerStats.attack = 10 + (metaProgression.upgrades.startingAttack * 2);
        playerStats.defense = 5 + (metaProgression.upgrades.startingDefense * 2);
    }

    generateNewFloor() {
        const rooms = dungeonGenerator.generateDungeon();
        
        // Place player in first room
        const startRoom = rooms[0];
        const startPos = dungeonGenerator.getRoomCenter(startRoom);
        playerController.createPlayer(startPos.x, startPos.z);
        
        // Place enemies and items
        enemyManager.placeEnemies(rooms);
        itemManager.placeItems(rooms);
        
        // Place exit in last room
        const exitRoom = rooms[rooms.length - 1];
        const exitPos = dungeonGenerator.getRoomCenter(exitRoom);
        dungeonGenerator.createExit(exitPos.x, exitPos.z);
    }

    nextFloor() {
        gameState.dungeonFloor++;
        
        uiManager.showUpgradeChoices(() => {
            this.generateNewFloor();
            uiManager.updateUI();
            uiManager.addMessage(`Floor ${gameState.dungeonFloor} - Enemies grow stronger!`, 'level');
        });
    }

    gameOver() {
        gameState.gameRunning = false;
        
        // Save souls to meta progression
        metaProgression.addSouls(playerStats.souls);
        
        uiManager.showGameOver();
    }

    returnToMenu() {
        uiManager.showMainMenu();
    }

    onUseItem() {
        const message = itemManager.useInventoryItem();
        if (message) {
            uiManager.addMessage(message, 'item');
            uiManager.updateUI();
        }
    }

    onCenterCamera() {
        playerController.centerCamera();
        uiManager.addMessage('Camera centered', 'item');
    }

    gameLoop() {
        if (!gameState.gameRunning) return;
        
        requestAnimationFrame(this.gameLoop.bind(this));
        
        // Update player
        playerController.updatePlayer();
        
        // Handle player attacks
        if (gameState.controls.attack) {
            const hitEnemies = playerController.playerAttack();
            hitEnemies.forEach(({ enemy, damage }) => {
                uiManager.addMessage(`You hit the enemy for ${damage} damage!`, 'damage');
                
                if (enemy.userData.hp <= 0) {
                    const result = enemyManager.defeatedEnemy(enemy);
                    uiManager.addMessage(`Enemy defeated! +${result.exp} EXP`, 'level');
                    
                    // Check level up
                    if (playerStats.exp >= playerStats.expNeeded) {
                        const levelResult = playerStats.levelUp();
                        uiManager.addMessage(`LEVEL UP! You are now level ${levelResult.level}!`, 'level');
                    }
                    
                    // Drop item chance
                    if (result.shouldDropItem) {
                        const types = ['health', 'attack', 'defense', 'gold'];
                        const type = types[Math.floor(Math.random() * types.length)];
                        itemManager.createItem(result.position.x, result.position.z, type);
                    }
                    
                    uiManager.updateUI();
                }
            });
        }
        
        // Update enemies
        const enemyAttacks = enemyManager.updateEnemies();
        enemyAttacks.forEach(attack => {
            uiManager.addMessage(`Enemy hits you for ${attack.damage} damage!`, 'damage');
            
            if (attack.isDead) {
                this.gameOver();
                return;
            }
            
            uiManager.updateUI();
        });
        
        // Check item pickup
        const pickedUpItems = playerController.checkItemPickup();
        pickedUpItems.forEach(({ item, index }) => {
            const message = itemManager.applyItemEffect(item);
            itemManager.removeItem(item);
            uiManager.addMessage(message, item.userData.effect === 'heal' ? 'heal' : 'item');
            uiManager.updateUI();
        });
        
        // Check exit
        if (playerController.checkExit()) {
            this.nextFloor();
        }
        
        // Animate items
        itemManager.animateItems();
        
        // Update camera
        playerController.updateCamera();
        
        // Render
        sceneManager.render();
    }
}

// Singleton instance
export const gameController = new GameController();

// Make functions globally accessible for HTML buttons
window.startNewRun = () => gameController.startNewRun();
window.returnToMenu = () => gameController.returnToMenu();