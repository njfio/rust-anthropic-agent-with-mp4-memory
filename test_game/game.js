// 3D Roguelike Dungeon Crawler
// Complete game implementation with Three.js

// Game state and configuration
const gameState = {
    scene: null,
    camera: null,
    renderer: null,
    player: null,
    enemies: [],
    items: [],
    walls: [],
    floor: null,
    dungeonFloor: 1,
    gameRunning: false,
    controls: {
        forward: false,
        backward: false,
        left: false,
        right: false,
        attack: false
    },
    mouseX: 0,
    mouseY: 0
};

// Player stats and meta-progression
const playerStats = {
    level: 1,
    exp: 0,
    expNeeded: 100,
    hp: 100,
    maxHp: 100,
    attack: 10,
    defense: 5,
    gold: 0,
    inventory: [],
    souls: 0
};

// Meta upgrades (persist between runs)
const metaUpgrades = {
    totalSouls: parseInt(localStorage.getItem('totalSouls') || '0'),
    upgrades: {
        startingHealth: parseInt(localStorage.getItem('startingHealth') || '0'),
        startingAttack: parseInt(localStorage.getItem('startingAttack') || '0'),
        startingDefense: parseInt(localStorage.getItem('startingDefense') || '0'),
        expBonus: parseInt(localStorage.getItem('expBonus') || '0'),
        goldBonus: parseInt(localStorage.getItem('goldBonus') || '0')
    }
};

// Dungeon configuration
const DUNGEON_SIZE = 20;
const CELL_SIZE = 4;
const WALL_HEIGHT = 5;

// Initialize Three.js
function initThreeJS() {
    // Scene setup
    gameState.scene = new THREE.Scene();
    gameState.scene.fog = new THREE.Fog(0x000000, 10, 50);
    
    // Camera setup
    gameState.camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    
    // Renderer setup
    gameState.renderer = new THREE.WebGLRenderer({ 
        canvas: document.getElementById('gameCanvas'),
        antialias: true 
    });
    gameState.renderer.setSize(window.innerWidth, window.innerHeight);
    gameState.renderer.shadowMap.enabled = true;
    gameState.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
    gameState.scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    directionalLight.shadow.camera.left = -50;
    directionalLight.shadow.camera.right = 50;
    directionalLight.shadow.camera.top = 50;
    directionalLight.shadow.camera.bottom = -50;
    gameState.scene.add(directionalLight);
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
    gameState.camera.aspect = window.innerWidth / window.innerHeight;
    gameState.camera.updateProjectionMatrix();
    gameState.renderer.setSize(window.innerWidth, window.innerHeight);
}

// Dungeon generation
function generateDungeon() {
    // Clear existing dungeon
    clearDungeon();
    
    // Create dungeon grid
    const dungeon = Array(DUNGEON_SIZE).fill().map(() => Array(DUNGEON_SIZE).fill(1));
    
    // Generate rooms and corridors using BSP
    const rooms = [];
    generateRooms(dungeon, rooms, 0, 0, DUNGEON_SIZE - 1, DUNGEON_SIZE - 1, 3);
    connectRooms(dungeon, rooms);
    
    // Create 3D representation
    createDungeon3D(dungeon);
    
    // Place player in first room
    const startRoom = rooms[0];
    const startX = Math.floor((startRoom.x1 + startRoom.x2) / 2) * CELL_SIZE;
    const startZ = Math.floor((startRoom.y1 + startRoom.y2) / 2) * CELL_SIZE;
    createPlayer(startX, startZ);
    
    // Place enemies and items
    placeEnemies(rooms);
    placeItems(rooms);
    
    // Place exit in last room
    const exitRoom = rooms[rooms.length - 1];
    const exitX = Math.floor((exitRoom.x1 + exitRoom.x2) / 2) * CELL_SIZE;
    const exitZ = Math.floor((exitRoom.y1 + exitRoom.y2) / 2) * CELL_SIZE;
    createExit(exitX, exitZ);
}

function generateRooms(dungeon, rooms, x1, y1, x2, y2, depth) {
    if (depth <= 0 || (x2 - x1) < 6 || (y2 - y1) < 6) {
        // Create room
        const roomX1 = x1 + 1;
        const roomY1 = y1 + 1;
        const roomX2 = x2 - 1;
        const roomY2 = y2 - 1;
        
        rooms.push({ x1: roomX1, y1: roomY1, x2: roomX2, y2: roomY2 });
        
        for (let x = roomX1; x <= roomX2; x++) {
            for (let y = roomY1; y <= roomY2; y++) {
                dungeon[y][x] = 0;
            }
        }
        return;
    }
    
    // Split horizontally or vertically
    if (Math.random() > 0.5) {
        // Vertical split
        const splitX = x1 + Math.floor((x2 - x1) / 2);
        generateRooms(dungeon, rooms, x1, y1, splitX, y2, depth - 1);
        generateRooms(dungeon, rooms, splitX + 1, y1, x2, y2, depth - 1);
    } else {
        // Horizontal split
        const splitY = y1 + Math.floor((y2 - y1) / 2);
        generateRooms(dungeon, rooms, x1, y1, x2, splitY, depth - 1);
        generateRooms(dungeon, rooms, x1, splitY + 1, x2, y2, depth - 1);
    }
}

function connectRooms(dungeon, rooms) {
    for (let i = 0; i < rooms.length - 1; i++) {
        const room1 = rooms[i];
        const room2 = rooms[i + 1];
        
        const x1 = Math.floor((room1.x1 + room1.x2) / 2);
        const y1 = Math.floor((room1.y1 + room1.y2) / 2);
        const x2 = Math.floor((room2.x1 + room2.x2) / 2);
        const y2 = Math.floor((room2.y1 + room2.y2) / 2);
        
        // Create L-shaped corridor
        for (let x = Math.min(x1, x2); x <= Math.max(x1, x2); x++) {
            dungeon[y1][x] = 0;
        }
        for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y++) {
            dungeon[y][x2] = 0;
        }
    }
}

function createDungeon3D(dungeon) {
    // Floor
    const floorGeometry = new THREE.PlaneGeometry(
        DUNGEON_SIZE * CELL_SIZE,
        DUNGEON_SIZE * CELL_SIZE
    );
    const floorMaterial = new THREE.MeshLambertMaterial({ color: 0x444444 });
    gameState.floor = new THREE.Mesh(floorGeometry, floorMaterial);
    gameState.floor.rotation.x = -Math.PI / 2;
    gameState.floor.receiveShadow = true;
    gameState.scene.add(gameState.floor);
    
    // Walls
    const wallGeometry = new THREE.BoxGeometry(CELL_SIZE, WALL_HEIGHT, CELL_SIZE);
    const wallMaterial = new THREE.MeshLambertMaterial({ color: 0x666666 });
    
    for (let y = 0; y < DUNGEON_SIZE; y++) {
        for (let x = 0; x < DUNGEON_SIZE; x++) {
            if (dungeon[y][x] === 1) {
                const wall = new THREE.Mesh(wallGeometry, wallMaterial);
                wall.position.set(
                    x * CELL_SIZE - (DUNGEON_SIZE * CELL_SIZE) / 2 + CELL_SIZE / 2,
                    WALL_HEIGHT / 2,
                    y * CELL_SIZE - (DUNGEON_SIZE * CELL_SIZE) / 2 + CELL_SIZE / 2
                );
                wall.castShadow = true;
                wall.receiveShadow = true;
                gameState.walls.push(wall);
                gameState.scene.add(wall);
            }
        }
    }
}

// Player creation and management
function createPlayer(x, z) {
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
    updateCamera();
}

function updateCamera() {
    if (!gameState.player) return;
    
    // Third person camera
    const distance = 10;
    const height = 8;
    
    gameState.camera.position.x = gameState.player.position.x - distance * Math.sin(gameState.player.rotation.y);
    gameState.camera.position.y = gameState.player.position.y + height;
    gameState.camera.position.z = gameState.player.position.z - distance * Math.cos(gameState.player.rotation.y);
    
    gameState.camera.lookAt(gameState.player.position);
}

// Enemy creation and AI
function createEnemy(x, z, type = 'basic') {
    const enemyTypes = {
        basic: { color: 0xff0000, hp: 30, attack: 5, exp: 20, speed: 0.02 },
        elite: { color: 0xff00ff, hp: 60, attack: 10, exp: 50, speed: 0.03 },
        boss: { color: 0xff8800, hp: 100, attack: 15, exp: 100, speed: 0.025 }
    };
    
    const enemyData = enemyTypes[type];
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
}

function placeEnemies(rooms) {
    rooms.forEach((room, index) => {
        if (index === 0) return; // No enemies in starting room
        
        const enemyCount = Math.floor(Math.random() * 3) + 1;
        for (let i = 0; i < enemyCount; i++) {
            const x = (room.x1 + Math.random() * (room.x2 - room.x1)) * CELL_SIZE;
            const z = (room.y1 + Math.random() * (room.y2 - room.y1)) * CELL_SIZE;
            
            const type = Math.random() < 0.8 ? 'basic' : 'elite';
            createEnemy(
                x - (DUNGEON_SIZE * CELL_SIZE) / 2,
                z - (DUNGEON_SIZE * CELL_SIZE) / 2,
                type
            );
        }
    });
}

// Item system
function createItem(x, z, type) {
    const itemTypes = {
        health: { color: 0x00ff00, effect: 'heal', value: 30 },
        attack: { color: 0xff0000, effect: 'buff_attack', value: 5 },
        defense: { color: 0x0000ff, effect: 'buff_defense', value: 3 },
        gold: { color: 0xffff00, effect: 'gold', value: 50 }
    };
    
    const itemData = itemTypes[type];
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
}

function placeItems(rooms) {
    rooms.forEach((room, index) => {
        const itemCount = Math.floor(Math.random() * 2) + 1;
        for (let i = 0; i < itemCount; i++) {
            const x = (room.x1 + Math.random() * (room.x2 - room.x1)) * CELL_SIZE;
            const z = (room.y1 + Math.random() * (room.y2 - room.y1)) * CELL_SIZE;
            
            const types = ['health', 'attack', 'defense', 'gold'];
            const type = types[Math.floor(Math.random() * types.length)];
            
            createItem(
                x - (DUNGEON_SIZE * CELL_SIZE) / 2,
                z - (DUNGEON_SIZE * CELL_SIZE) / 2,
                type
            );
        }
    });
}

// Exit portal
function createExit(x, z) {
    const exitGeometry = new THREE.CylinderGeometry(1.5, 1.5, 0.2, 8);
    const exitMaterial = new THREE.MeshLambertMaterial({ 
        color: 0x00ffff,
        emissive: 0x00ffff,
        emissiveIntensity: 0.5
    });
    const exit = new THREE.Mesh(exitGeometry, exitMaterial);
    
    exit.position.set(x, 0.1, z);
    exit.userData = { type: 'exit' };
    gameState.scene.add(exit);
    
    // Add glow effect
    const glowGeometry = new THREE.RingGeometry(1.5, 2, 8);
    const glowMaterial = new THREE.MeshBasicMaterial({ 
        color: 0x00ffff,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.5
    });
    const glow = new THREE.Mesh(glowGeometry, glowMaterial);
    glow.rotation.x = -Math.PI / 2;
    glow.position.y = 0.2;
    exit.add(glow);
}

// Input handling
function setupControls() {
    document.addEventListener('keydown', (e) => {
        switch(e.key.toLowerCase()) {
            case 'w':
            case 'arrowup':
                gameState.controls.forward = true;
                break;
            case 's':
            case 'arrowdown':
                gameState.controls.backward = true;
                break;
            case 'a':
            case 'arrowleft':
                gameState.controls.left = true;
                break;
            case 'd':
            case 'arrowright':
                gameState.controls.right = true;
                break;
            case ' ':
                gameState.controls.attack = true;
                e.preventDefault();
                break;
            case 'e':
                useItem();
                break;
        }
    });
    
    document.addEventListener('keyup', (e) => {
        switch(e.key.toLowerCase()) {
            case 'w':
            case 'arrowup':
                gameState.controls.forward = false;
                break;
            case 's':
            case 'arrowdown':
                gameState.controls.backward = false;
                break;
            case 'a':
            case 'arrowleft':
                gameState.controls.left = false;
                break;
            case 'd':
            case 'arrowright':
                gameState.controls.right = false;
                break;
            case ' ':
                gameState.controls.attack = false;
                break;
        }
    });
    
    document.addEventListener('mousemove', (e) => {
        gameState.mouseX = (e.clientX / window.innerWidth) * 2 - 1;
        gameState.mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
    });
}

// Game loop and updates
function updatePlayer() {
    if (!gameState.player || !gameState.gameRunning) return;
    
    const speed = 0.1;
    const rotSpeed = 0.05;
    
    // Rotation based on mouse
    gameState.player.rotation.y += gameState.mouseX * rotSpeed;
    
    // Movement
    const moveVector = new THREE.Vector3();
    
    if (gameState.controls.forward) {
        moveVector.z -= speed;
    }
    if (gameState.controls.backward) {
        moveVector.z += speed;
    }
    if (gameState.controls.left) {
        moveVector.x -= speed;
    }
    if (gameState.controls.right) {
        moveVector.x += speed;
    }
    
    // Apply rotation to movement
    moveVector.applyQuaternion(gameState.player.quaternion);
    
    // Check collision before moving
    const newPosition = gameState.player.position.clone().add(moveVector);
    if (!checkWallCollision(newPosition)) {
        gameState.player.position.add(moveVector);
    }
    
    // Attack
    if (gameState.controls.attack) {
        playerAttack();
        gameState.controls.attack = false;
    }
    
    // Check item pickup
    checkItemPickup();
    
    // Check exit
    checkExit();
}

function checkWallCollision(position) {
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

function playerAttack() {
    const attackRange = 3;
    const attackAngle = Math.PI / 4;
    
    gameState.enemies.forEach(enemy => {
        const distance = gameState.player.position.distanceTo(enemy.position);
        
        if (distance < attackRange) {
            // Check if enemy is in front of player
            const toEnemy = enemy.position.clone().sub(gameState.player.position).normalize();
            const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(gameState.player.quaternion);
            const angle = forward.angleTo(toEnemy);
            
            if (angle < attackAngle) {
                // Deal damage
                const damage = playerStats.attack + Math.floor(Math.random() * 5);
                enemy.userData.hp -= damage;
                
                addMessage(`You hit the enemy for ${damage} damage!`, 'damage');
                
                // Knockback
                const knockback = toEnemy.multiplyScalar(0.5);
                enemy.position.add(knockback);
                
                // Check if enemy is dead
                if (enemy.userData.hp <= 0) {
                    defeatedEnemy(enemy);
                }
            }
        }
    });
}

function defeatedEnemy(enemy) {
    const index = gameState.enemies.indexOf(enemy);
    if (index > -1) {
        gameState.enemies.splice(index, 1);
    }
    
    // Grant exp and gold
    playerStats.exp += enemy.userData.exp;
    playerStats.gold += Math.floor(Math.random() * 20) + 10;
    playerStats.souls += Math.floor(enemy.userData.exp / 10);
    
    addMessage(`Enemy defeated! +${enemy.userData.exp} EXP`, 'level');
    
    // Check level up
    if (playerStats.exp >= playerStats.expNeeded) {
        levelUp();
    }
    
    // Drop item chance
    if (Math.random() < 0.3) {
        const types = ['health', 'attack', 'defense', 'gold'];
        const type = types[Math.floor(Math.random() * types.length)];
        createItem(enemy.position.x, enemy.position.z, type);
    }
    
    gameState.scene.remove(enemy);
    updateUI();
}

function levelUp() {
    playerStats.level++;
    playerStats.exp -= playerStats.expNeeded;
    playerStats.expNeeded = Math.floor(playerStats.expNeeded * 1.5);
    playerStats.maxHp += 20;
    playerStats.hp = playerStats.maxHp;
    playerStats.attack += 3;
    playerStats.defense += 2;
    
    addMessage(`LEVEL UP! You are now level ${playerStats.level}!`, 'level');
    updateUI();
}

function updateEnemies() {
    const currentTime = Date.now();
    
    gameState.enemies.forEach(enemy => {
        // Simple AI - move towards player
        const distance = enemy.position.distanceTo(gameState.player.position);
        
        if (distance < 15) {
            const direction = gameState.player.position.clone().sub(enemy.position).normalize();
            const moveVector = direction.multiplyScalar(enemy.userData.speed);
            
            // Check collision before moving
            const newPosition = enemy.position.clone().add(moveVector);
            if (!checkWallCollision(newPosition)) {
                enemy.position.add(moveVector);
            }
            
            // Face player
            enemy.lookAt(gameState.player.position);
            
            // Attack if close enough
            if (distance < 2 && currentTime - enemy.userData.lastAttack > 1000) {
                enemyAttack(enemy);
                enemy.userData.lastAttack = currentTime;
            }
        }
    });
}

function enemyAttack(enemy) {
    const damage = Math.max(1, enemy.userData.attack - playerStats.defense);
    playerStats.hp -= damage;
    
    addMessage(`Enemy hits you for ${damage} damage!`, 'damage');
    
    if (playerStats.hp <= 0) {
        gameOver();
    }
    
    updateUI();
}

function checkItemPickup() {
    const pickupRange = 2;
    
    gameState.items.forEach((item, index) => {
        const distance = gameState.player.position.distanceTo(item.position);
        
        if (distance < pickupRange) {
            // Apply item effect
            switch(item.userData.effect) {
                case 'heal':
                    playerStats.hp = Math.min(playerStats.hp + item.userData.value, playerStats.maxHp);
                    addMessage(`Healed for ${item.userData.value} HP!`, 'heal');
                    break;
                case 'buff_attack':
                    playerStats.attack += item.userData.value;
                    addMessage(`Attack increased by ${item.userData.value}!`, 'item');
                    break;
                case 'buff_defense':
                    playerStats.defense += item.userData.value;
                    addMessage(`Defense increased by ${item.userData.value}!`, 'item');
                    break;
                case 'gold':
                    playerStats.gold += item.userData.value;
                    addMessage(`Found ${item.userData.value} gold!`, 'item');
                    break;
            }
            
            // Remove item
            gameState.scene.remove(item);
            gameState.items.splice(index, 1);
            updateUI();
        }
    });
}

function checkExit() {
    gameState.scene.traverse((child) => {
        if (child.userData && child.userData.type === 'exit') {
            const distance = gameState.player.position.distanceTo(child.position);
            if (distance < 3) {
                nextFloor();
            }
        }
    });
}

function useItem() {
    if (playerStats.inventory.length > 0) {
        const item = playerStats.inventory.shift();
        // Apply item effect
        addMessage(`Used ${item.name}!`, 'item');
        updateUI();
    }
}

// Animation
function animateItems() {
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

// UI updates
function updateUI() {
    document.getElementById('playerLevel').textContent = playerStats.level;
    document.getElementById('dungeonFloor').textContent = gameState.dungeonFloor;
    document.getElementById('playerHP').textContent = playerStats.hp;
    document.getElementById('playerMaxHP').textContent = playerStats.maxHp;
    document.getElementById('playerExp').textContent = playerStats.exp;
    document.getElementById('expNeeded').textContent = playerStats.expNeeded;
    document.getElementById('playerAtk').textContent = playerStats.attack;
    document.getElementById('playerDef').textContent = playerStats.defense;
    document.getElementById('playerGold').textContent = playerStats.gold;
    
    // Update health bar
    const healthPercent = (playerStats.hp / playerStats.maxHp) * 100;
    document.getElementById('healthBar').style.width = healthPercent + '%';
    
    // Update exp bar
    const expPercent = (playerStats.exp / playerStats.expNeeded) * 100;
    document.getElementById('expBar').style.width = expPercent + '%';
    
    // Update inventory
    const inventoryList = document.getElementById('inventoryList');
    inventoryList.innerHTML = playerStats.inventory.length > 0 
        ? playerStats.inventory.map(item => `<div>${item.name}</div>`).join('')
        : '<div>Empty</div>';
}

function addMessage(text, type = '') {
    const messages = document.getElementById('messages');
    const message = document.createElement('div');
    message.className = `message ${type}`;
    message.textContent = text;
    messages.appendChild(message);
    
    // Keep only last 5 messages
    while (messages.children.length > 5) {
        messages.removeChild(messages.firstChild);
    }
    
    // Auto scroll to bottom
    messages.scrollTop = messages.scrollHeight;
}

// Game flow
function startNewRun() {
    // Apply meta upgrades
    playerStats.maxHp = 100 + (metaUpgrades.upgrades.startingHealth * 20);
    playerStats.hp = playerStats.maxHp;
    playerStats.attack = 10 + (metaUpgrades.upgrades.startingAttack * 2);
    playerStats.defense = 5 + (metaUpgrades.upgrades.startingDefense * 2);
    playerStats.level = 1;
    playerStats.exp = 0;
    playerStats.expNeeded = 100;
    playerStats.gold = 0;
    playerStats.souls = 0;
    playerStats.inventory = [];
    
    gameState.dungeonFloor = 1;
    gameState.gameRunning = true;
    
    document.getElementById('mainMenu').style.display = 'none';
    document.getElementById('gameOver').style.display = 'none';
    
    generateDungeon();
    updateUI();
    addMessage('Welcome to the dungeon!', 'level');
    
    gameLoop();
}

function nextFloor() {
    gameState.dungeonFloor++;
    
    // Show upgrade choices
    showUpgradeChoices();
}

function showUpgradeChoices() {
    gameState.gameRunning = false;
    document.getElementById('levelComplete').style.display = 'block';
    
    const upgrades = [
        { name: 'Health Boost', effect: () => { playerStats.maxHp += 30; playerStats.hp += 30; } },
        { name: 'Attack Power', effect: () => { playerStats.attack += 5; } },
        { name: 'Defense Up', effect: () => { playerStats.defense += 3; } },
        { name: 'Full Heal', effect: () => { playerStats.hp = playerStats.maxHp; } }
    ];
    
    // Random 3 choices
    const choices = [];
    while (choices.length < 3) {
        const upgrade = upgrades[Math.floor(Math.random() * upgrades.length)];
        if (!choices.includes(upgrade)) {
            choices.push(upgrade);
        }
    }
    
    const choicesDiv = document.getElementById('upgradeChoices');
    choicesDiv.innerHTML = '';
    
    choices.forEach(upgrade => {
        const div = document.createElement('div');
        div.className = 'upgrade-option';
        div.textContent = upgrade.name;
        div.onclick = () => {
            upgrade.effect();
            document.getElementById('levelComplete').style.display = 'none';
            gameState.gameRunning = true;
            generateDungeon();
            updateUI();
            addMessage(`Floor ${gameState.dungeonFloor} - Enemies grow stronger!`, 'level');
        };
        choicesDiv.appendChild(div);
    });
}

function gameOver() {
    gameState.gameRunning = false;
    
    // Save souls to meta progression
    metaUpgrades.totalSouls += playerStats.souls;
    localStorage.setItem('totalSouls', metaUpgrades.totalSouls);
    
    document.getElementById('gameOver').style.display = 'block';
    document.getElementById('finalFloor').textContent = gameState.dungeonFloor;
    document.getElementById('soulsEarned').textContent = playerStats.souls;
}

function returnToMenu() {
    document.getElementById('gameOver').style.display = 'none';
    document.getElementById('mainMenu').style.display = 'block';
    updateMetaUpgradesUI();
}

function clearDungeon() {
    // Remove all objects from scene
    gameState.walls.forEach(wall => gameState.scene.remove(wall));
    gameState.enemies.forEach(enemy => gameState.scene.remove(enemy));
    gameState.items.forEach(item => gameState.scene.remove(item));
    
    gameState.walls = [];
    gameState.enemies = [];
    gameState.items = [];
    
    if (gameState.floor) {
        gameState.scene.remove(gameState.floor);
    }
    
    if (gameState.player) {
        gameState.scene.remove(gameState.player);
    }
    
    // Remove exit
    gameState.scene.traverse((child) => {
        if (child.userData && child.userData.type === 'exit') {
            gameState.scene.remove(child);
        }
    });
}

// Meta progression
function updateMetaUpgradesUI() {
    document.getElementById('totalSouls').textContent = metaUpgrades.totalSouls;
    
    const upgradesList = document.getElementById('upgradesList');
    upgradesList.innerHTML = '';
    
    const upgrades = [
        { 
            name: 'Starting Health', 
            cost: 50, 
            level: metaUpgrades.upgrades.startingHealth,
            key: 'startingHealth',
            description: '+20 starting HP per level'
        },
        { 
            name: 'Starting Attack', 
            cost: 40, 
            level: metaUpgrades.upgrades.startingAttack,
            key: 'startingAttack',
            description: '+2 starting attack per level'
        },
        { 
            name: 'Starting Defense', 
            cost: 40, 
            level: metaUpgrades.upgrades.startingDefense,
            key: 'startingDefense',
            description: '+2 starting defense per level'
        },
        { 
            name: 'Experience Bonus', 
            cost: 60, 
            level: metaUpgrades.upgrades.expBonus,
            key: 'expBonus',
            description: '+10% exp gain per level'
        },
        { 
            name: 'Gold Bonus', 
            cost: 30, 
            level: metaUpgrades.upgrades.goldBonus,
            key: 'goldBonus',
            description: '+20% gold gain per level'
        }
    ];
    
    upgrades.forEach(upgrade => {
        const div = document.createElement('div');
        div.className = 'upgrade-option';
        
        const cost = upgrade.cost * (upgrade.level + 1);
        const canAfford = metaUpgrades.totalSouls >= cost;
        
        div.innerHTML = `
            <strong>${upgrade.name}</strong> - Level ${upgrade.level}<br>
            <small>${upgrade.description}</small><br>
            Cost: ${cost} souls
        `;
        
        if (canAfford) {
            div.style.cursor = 'pointer';
            div.onclick = () => {
                metaUpgrades.totalSouls -= cost;
                metaUpgrades.upgrades[upgrade.key]++;
                localStorage.setItem('totalSouls', metaUpgrades.totalSouls);
                localStorage.setItem(upgrade.key, metaUpgrades.upgrades[upgrade.key]);
                updateMetaUpgradesUI();
            };
        } else {
            div.style.opacity = '0.5';
            div.style.cursor = 'not-allowed';
        }
        
        upgradesList.appendChild(div);
    });
}

// Main game loop
function gameLoop() {
    if (gameState.gameRunning) {
        requestAnimationFrame(gameLoop);
        
        updatePlayer();
        updateEnemies();
        animateItems();
        updateCamera();
        
        gameState.renderer.render(gameState.scene, gameState.camera);
    }
}

// Initialize game
window.onload = function() {
    initThreeJS();
    setupControls();
    updateMetaUpgradesUI();
    
    // Make functions globally accessible
    window.startNewRun = startNewRun;
    window.returnToMenu = returnToMenu;
};
