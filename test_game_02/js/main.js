/**
 * Main Game Initialization
 * Entry point for Shadowfall Depths
 */

// Game state
let gameState = {
    currentFloor: 1,
    score: 0,
    isPlaying: false,
    isPaused: false
};

// Game entities
let player = null;
let entities = [];
let dungeon = null;

// UI elements
const UI = {
    startScreen: null,
    gameOverScreen: null,
    upgradesScreen: null,
    pauseScreen: null,
    startButton: null,
    restartButton: null,
    upgradesButton: null,
    
    init() {
        this.startScreen = document.getElementById('startScreen');
        this.gameOverScreen = document.getElementById('gameOverScreen');
        this.upgradesScreen = document.getElementById('upgradesScreen');
        this.pauseScreen = document.getElementById('pauseScreen');
        
        this.startButton = document.getElementById('startGame');
        this.restartButton = document.getElementById('restartGame');
        this.upgradesButton = document.getElementById('upgradesBtn');
        
        this.bindEvents();
    },
    
    bindEvents() {
        this.startButton?.addEventListener('click', () => startGame());
        this.restartButton?.addEventListener('click', () => startGame());
        this.upgradesButton?.addEventListener('click', () => showUpgrades());
        
        document.getElementById('closeUpgrades')?.addEventListener('click', () => hideUpgrades());
        document.getElementById('resumeGame')?.addEventListener('click', () => resumeGame());
        document.getElementById('quitGame')?.addEventListener('click', () => quitToMenu());
    },
    
    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.add('hidden');
        });
        document.getElementById(screenId)?.classList.remove('hidden');
    },
    
    updateStats() {
        if (player) {
            document.getElementById('currentFloor').textContent = gameState.currentFloor;
            document.getElementById('currentScore').textContent = gameState.score;
            document.getElementById('healthText').textContent = `${player.health}/${player.maxHealth}`;
            document.getElementById('playerDamage').textContent = player.damage;
            document.getElementById('playerSpeed').textContent = '100%';
            document.getElementById('playerLuck').textContent = `${player.luck}%`;
            
            // Update health bar
            const healthBar = document.getElementById('healthBar');
            const healthPercent = (player.health / player.maxHealth) * 100;
            healthBar.style.width = `${healthPercent}%`;
        }
    }
};

function initGame() {
    console.log('Initializing Shadowfall Depths...');
    
    // Initialize UI
    UI.init();
    
    // Initialize game engine event listeners
    gameEngine.on('engine:update', updateGame);
    gameEngine.on('engine:render', renderGame);
    
    // Initialize input system
    if (window.InputManager) {
        window.InputManager.init();
        window.InputManager.bindPlayerControls();
    }
    
    // Show start screen
    UI.showScreen('startScreen');
    
    console.log('Game initialized successfully');
}

function startGame() {
    console.log('Starting new game...');
    
    // Reset game state
    gameState.currentFloor = 1;
    gameState.score = 0;
    gameState.isPlaying = true;
    gameState.isPaused = false;
    
    // Create player
    player = new Player(400, 300);
    window.player = player; // Make globally accessible for AI
    
    // Create simple dungeon room
    entities = [player];
    
    // Add some enemies for testing
    for (let i = 0; i < 3; i++) {
        const enemy = new Enemy(
            Utils.random(100, 700),
            Utils.random(100, 500),
            Utils.randomChoice(['basic', 'goblin', 'skeleton'])
        );
        entities.push(enemy);
    }
    
    // Hide start screen and start game engine
    UI.showScreen('');
    gameEngine.setState('playing');
    gameEngine.start();
    
    console.log('Game started');
}

function updateGame(deltaTime) {
    if (!gameState.isPlaying || gameState.isPaused) return;
    
    // Update all entities
    entities.forEach(entity => {
        if (entity.alive) {
            entity.update(deltaTime);
        }
    });
    
    // Remove dead entities (except player)
    entities = entities.filter(entity => entity.alive || entity === player);
    
    // Check for collisions
    checkCollisions();
    
    // Update UI
    UI.updateStats();
    
    // Check win/lose conditions
    if (!player.alive) {
        gameOver();
    }
    
    // Check if all enemies are dead
    const aliveEnemies = entities.filter(e => e.hasTag('enemy') && e.alive);
    if (aliveEnemies.length === 0) {
        // Spawn more enemies or advance floor
        spawnEnemies();
    }
}

function renderGame(ctx) {
    if (!gameState.isPlaying) return;
    
    // Render simple background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(-1000, -1000, 2000, 2000);
    
    // Render grid for reference
    renderGrid(ctx);
    
    // Render all entities
    entities.forEach(entity => {
        if (entity.visible && gameEngine.isInViewport(entity.x, entity.y, entity.width, entity.height)) {
            entity.render(ctx);
        }
    });
}

function renderGrid(ctx) {
    const gridSize = 50;
    const startX = Math.floor(gameEngine.camera.x / gridSize) * gridSize;
    const startY = Math.floor(gameEngine.camera.y / gridSize) * gridSize;
    const endX = startX + gameEngine.viewport.width + gridSize;
    const endY = startY + gameEngine.viewport.height + gridSize;
    
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    
    // Vertical lines
    for (let x = startX; x <= endX; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, startY);
        ctx.lineTo(x, endY);
        ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = startY; y <= endY; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(startX, y);
        ctx.lineTo(endX, y);
        ctx.stroke();
    }
}

function checkCollisions() {
    for (let i = 0; i < entities.length; i++) {
        for (let j = i + 1; j < entities.length; j++) {
            const entityA = entities[i];
            const entityB = entities[j];
            
            if (entityA.alive && entityB.alive && entityA.collidesWith(entityB)) {
                handleCollision(entityA, entityB);
            }
        }
    }
}

function handleCollision(entityA, entityB) {
    // Simple collision handling - separate entities
    const dx = entityB.x - entityA.x;
    const dy = entityB.y - entityA.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    
    if (distance > 0) {
        const pushDistance = 2;
        const pushX = (dx / distance) * pushDistance;
        const pushY = (dy / distance) * pushDistance;
        
        entityA.x -= pushX;
        entityA.y -= pushY;
        entityB.x += pushX;
        entityB.y += pushY;
    }
}

function spawnEnemies() {
    const enemyCount = Math.min(5, gameState.currentFloor + 2);
    
    for (let i = 0; i < enemyCount; i++) {
        const enemy = new Enemy(
            Utils.random(100, 700),
            Utils.random(100, 500),
            Utils.randomChoice(['basic', 'goblin', 'skeleton', 'orc'])
        );
        entities.push(enemy);
    }
}

function gameOver() {
    gameState.isPlaying = false;
    gameEngine.setState('gameOver');
    
    // Update game over screen
    document.getElementById('gameOverStats').textContent = 
        `You reached floor ${gameState.currentFloor} with a score of ${gameState.score}`;
    
    UI.showScreen('gameOverScreen');
    
    console.log('Game Over');
}

function pauseGame() {
    if (gameState.isPlaying && !gameState.isPaused) {
        gameState.isPaused = true;
        gameEngine.pause();
        UI.showScreen('pauseScreen');
    }
}

function resumeGame() {
    if (gameState.isPaused) {
        gameState.isPaused = false;
        gameEngine.resume();
        UI.showScreen('');
    }
}

function quitToMenu() {
    gameState.isPlaying = false;
    gameState.isPaused = false;
    gameEngine.stop();
    gameEngine.setState('menu');
    UI.showScreen('startScreen');
}

function showUpgrades() {
    UI.showScreen('upgradesScreen');
}

function hideUpgrades() {
    if (gameState.isPlaying) {
        UI.showScreen('');
    } else {
        UI.showScreen('startScreen');
    }
}

// Global keyboard handler
document.addEventListener('keydown', (event) => {
    switch (event.code) {
        case 'Escape':
            if (gameState.isPlaying) {
                if (gameState.isPaused) {
                    resumeGame();
                } else {
                    pauseGame();
                }
            }
            break;
    }
});

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    initGame();
});

// Enable debug mode
window.DEBUG = true;

console.log('Main game script loaded');