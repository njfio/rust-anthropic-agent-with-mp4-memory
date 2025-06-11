import { CONFIG, GAME_STATES } from './config.js';
import { EventEmitter, Utils } from './utils.js';
import { RenderSystem } from './systems/RenderSystem.js';
import { DungeonSystem } from './systems/DungeonSystem.js';
import { EntityManager } from './EntityManager.js';
import { UISystem } from './systems/UISystem.js';
import { InputSystem } from './systems/InputSystem.js';
import { MetaProgressionSystem } from './systems/MetaProgressionSystem.js';

export class GameEngine {
    constructor() {
        this.state = GAME_STATES.MENU;
        this.dungeonFloor = 1;
        this.events = new EventEmitter();
        
        // Initialize systems
        this.renderSystem = new RenderSystem();
        this.dungeonSystem = new DungeonSystem(this.events);
        this.entityManager = new EntityManager(this.events);
        this.uiSystem = new UISystem(this.events);
        this.inputSystem = new InputSystem(this.events);
        this.metaProgression = new MetaProgressionSystem();
        
        // Bind methods
        this.gameLoop = this.gameLoop.bind(this);
        this.startNewRun = this.startNewRun.bind(this);
        this.returnToMenu = this.returnToMenu.bind(this);
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    init() {
        try {
            // Initialize all systems
            this.renderSystem.init();
            this.inputSystem.init();
            this.uiSystem.init();
            this.metaProgression.init();
            
            // Show main menu
            this.uiSystem.showMainMenu();
            this.uiSystem.updateMetaUpgrades(this.metaProgression.getUpgrades());
            
            // Make methods globally accessible for HTML onclick
            window.startNewRun = this.startNewRun;
            window.returnToMenu = this.returnToMenu;
            
            console.log('Game engine initialized successfully');
        } catch (error) {
            console.error('Failed to initialize game engine:', error);
        }
    }
    
    setupEventListeners() {
        // Player events
        this.events.on('player:death', () => this.gameOver());
        this.events.on('player:levelup', (data) => this.handleLevelUp(data));
        this.events.on('player:attack', (data) => this.handlePlayerAttack(data));
        
        // Enemy events
        this.events.on('enemy:defeated', (data) => this.handleEnemyDefeated(data));
        this.events.on('enemy:attack', (data) => this.handleEnemyAttack(data));
        
        // Item events
        this.events.on('item:pickup', (data) => this.handleItemPickup(data));
        
        // Level events
        this.events.on('level:exit', () => this.nextFloor());
        this.events.on('level:upgrade', (upgrade) => this.applyUpgrade(upgrade));
        
        // Input events
        this.events.on('input:move', (data) => this.handlePlayerMove(data));
        this.events.on('input:attack', () => this.handleAttackInput());
        this.events.on('input:use', () => this.handleUseItem());
    }
    
    startNewRun() {
        try {
            // Apply meta upgrades
            const upgrades = this.metaProgression.getUpgrades();
            this.entityManager.applyMetaUpgrades(upgrades);
            
            // Reset game state
            this.dungeonFloor = 1;
            this.state = GAME_STATES.PLAYING;
            
            // Hide menus
            this.uiSystem.hideAllMenus();
            
            // Generate first floor
            this.generateDungeon();
            
            // Update UI
            this.uiSystem.updateStats(this.entityManager.getPlayerStats());
            this.uiSystem.addMessage('Welcome to the dungeon!', 'level');
            
            // Start game loop
            this.gameLoop();
        } catch (error) {
            console.error('Failed to start new run:', error);
            this.returnToMenu();
        }
    }
    
    generateDungeon() {
        try {
            // Clear previous dungeon
            this.clearDungeon();
            
            // Generate new dungeon layout
            const dungeonData = this.dungeonSystem.generateDungeon();
            
            // Create 3D representation
            const { walls, floor } = this.dungeonSystem.createDungeon3D(
                dungeonData.grid,
                this.renderSystem.scene
            );
            
            // Place player
            const startPos = dungeonData.rooms[0].center;
            const player = this.entityManager.createPlayer(
                startPos.x * CONFIG.DUNGEON.CELL_SIZE,
                startPos.z * CONFIG.DUNGEON.CELL_SIZE
            );
            this.renderSystem.scene.add(player.mesh);
            
            // Place enemies
            this.placeEnemies(dungeonData.rooms);
            
            // Place items
            this.placeItems(dungeonData.rooms);
            
            // Place exit
            const exitPos = dungeonData.rooms[dungeonData.rooms.length - 1].center;
            const exit = this.dungeonSystem.createExit(
                exitPos.x * CONFIG.DUNGEON.CELL_SIZE,
                exitPos.z * CONFIG.DUNGEON.CELL_SIZE
            );
            this.renderSystem.scene.add(exit);
            
        } catch (error) {
            console.error('Failed to generate dungeon:', error);
        }
    }
    
    placeEnemies(rooms) {
        rooms.forEach((room, index) => {
            if (index === 0) return; // No enemies in starting room
            
            const enemyCount = Utils.randomInt(1, CONFIG.ENEMY.MAX_PER_ROOM);
            for (let i = 0; i < enemyCount; i++) {
                const x = Utils.randomRange(room.x1, room.x2) * CONFIG.DUNGEON.CELL_SIZE;
                const z = Utils.randomRange(room.y1, room.y2) * CONFIG.DUNGEON.CELL_SIZE;
                
                const type = Math.random() < 0.8 ? 'basic' : 'elite';
                const enemy = this.entityManager.createEnemy(
                    x - (CONFIG.DUNGEON.SIZE * CONFIG.DUNGEON.CELL_SIZE) / 2,
                    z - (CONFIG.DUNGEON.SIZE * CONFIG.DUNGEON.CELL_SIZE) / 2,
                    type
                );
                this.renderSystem.scene.add(enemy.mesh);
            }
        });
    }
    
    placeItems(rooms) {
        rooms.forEach((room) => {
            const itemCount = Utils.randomInt(0, CONFIG.ITEM.MAX_PER_ROOM);
            for (let i = 0; i < itemCount; i++) {
                const x = Utils.randomRange(room.x1, room.x2) * CONFIG.DUNGEON.CELL_SIZE;
                const z = Utils.randomRange(room.y1, room.y2) * CONFIG.DUNGEON.CELL_SIZE;
                
                const types = Object.keys(CONFIG.ITEM.TYPES);
                const type = Utils.randomChoice(types);
                
                const item = this.entityManager.createItem(
                    x - (CONFIG.DUNGEON.SIZE * CONFIG.DUNGEON.CELL_SIZE) / 2,
                    z - (CONFIG.DUNGEON.SIZE * CONFIG.DUNGEON.CELL_SIZE) / 2,
                    type
                );
                this.renderSystem.scene.add(item.mesh);
            }
        });
    }
    
    clearDungeon() {
        // Clear entities
        this.entityManager.clearAll();
        
        // Clear dungeon system
        this.dungeonSystem.clear();
        
        // Clear scene objects
        this.renderSystem.clearScene();
    }
    
    gameLoop() {
        if (this.state === GAME_STATES.PLAYING) {
            requestAnimationFrame(this.gameLoop);
            
            // Update systems
            this.entityManager.update(this.dungeonSystem.getWalls());
            this.renderSystem.update(this.entityManager.getPlayer());
            
            // Check collisions
            this.checkCollisions();
            
            // Render
            this.renderSystem.render();
        }
    }
    
    checkCollisions() {
        const player = this.entityManager.getPlayer();
        if (!player) return;
        
        // Check item pickups
        this.entityManager.checkItemPickups(player);
        
        // Check exit
        const exit = this.dungeonSystem.getExit();
        if (exit && player.mesh.position.distanceTo(exit.position) < 3) {
            this.events.emit('level:exit');
        }
    }
    
    handlePlayerMove(data) {
        if (this.state === GAME_STATES.PLAYING) {
            this.entityManager.movePlayer(data, this.dungeonSystem.getWalls());
        }
    }
    
    handleAttackInput() {
        if (this.state === GAME_STATES.PLAYING) {
            this.entityManager.playerAttack();
        }
    }
    
    handleUseItem() {
        if (this.state === GAME_STATES.PLAYING) {
            this.entityManager.useItem();
        }
    }
    
    handlePlayerAttack(data) {
        this.uiSystem.addMessage(`You hit the enemy for ${data.damage} damage!`, 'damage');
    }
    
    handleEnemyAttack(data) {
        this.uiSystem.addMessage(`Enemy hits you for ${data.damage} damage!`, 'damage');
        this.uiSystem.updateStats(this.entityManager.getPlayerStats());
    }
    
    handleEnemyDefeated(data) {
        this.uiSystem.addMessage(`Enemy defeated! +${data.exp} EXP`, 'level');
        this.uiSystem.updateStats(this.entityManager.getPlayerStats());
    }
    
    handleItemPickup(data) {
        const messages = {
            heal: `Healed for ${data.value} HP!`,
            buff_attack: `Attack increased by ${data.value}!`,
            buff_defense: `Defense increased by ${data.value}!`,
            gold: `Found ${data.value} gold!`
        };
        
        this.uiSystem.addMessage(messages[data.effect] || 'Item picked up!', 'item');
        this.uiSystem.updateStats(this.entityManager.getPlayerStats());
    }
    
    handleLevelUp(data) {
        this.uiSystem.addMessage(`LEVEL UP! You are now level ${data.level}!`, 'level');
        this.uiSystem.updateStats(this.entityManager.getPlayerStats());
    }
    
    nextFloor() {
        this.dungeonFloor++;
        this.state = GAME_STATES.LEVEL_COMPLETE;
        
        // Show upgrade choices
        const upgrades = [
            { name: 'Health Boost', effect: () => this.entityManager.upgradePlayerHealth(30) },
            { name: 'Attack Power', effect: () => this.entityManager.upgradePlayerAttack(5) },
            { name: 'Defense Up', effect: () => this.entityManager.upgradePlayerDefense(3) },
            { name: 'Full Heal', effect: () => this.entityManager.healPlayerFull() }
        ];
        
        // Random 3 choices
        const choices = [];
        while (choices.length < 3) {
            const upgrade = Utils.randomChoice(upgrades);
            if (!choices.includes(upgrade)) {
                choices.push(upgrade);
            }
        }
        
        this.uiSystem.showUpgradeChoices(choices, (upgrade) => {
            upgrade.effect();
            this.state = GAME_STATES.PLAYING;
            this.generateDungeon();
            this.uiSystem.updateStats(this.entityManager.getPlayerStats());
            this.uiSystem.addMessage(`Floor ${this.dungeonFloor} - Enemies grow stronger!`, 'level');
        });
    }
    
    gameOver() {
        this.state = GAME_STATES.GAME_OVER;
        
        // Save souls
        const stats = this.entityManager.getPlayerStats();
        this.metaProgression.addSouls(stats.souls);
        
        // Show game over screen
        this.uiSystem.showGameOver(this.dungeonFloor, stats.souls);
    }
    
    returnToMenu() {
        this.state = GAME_STATES.MENU;
        this.clearDungeon();
        this.uiSystem.showMainMenu();
        this.uiSystem.updateMetaUpgrades(this.metaProgression.getUpgrades());
    }
}