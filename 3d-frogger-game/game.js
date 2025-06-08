// 3D Frogger Game Implementation
// Complete game with Three.js 3D graphics

// Game state variables
let scene, camera, renderer, frog;
let gameState = 'start'; // 'start', 'playing', 'paused', 'gameOver'
let score = 0;
let lives = 3;
let level = 1;
let gameSpeed = 1;

// Game objects arrays
let cars = [];
let trucks = [];
let logs = [];
let lilyPads = [];
let powerUps = [];

// Input handling
let keys = {};
let lastMoveTime = 0;
const moveDelay = 200; // Milliseconds between moves

// Game dimensions
const GAME_WIDTH = 20;
const GAME_HEIGHT = 30;
const LANE_WIDTH = 2;

// Initialize the game
function init() {
    console.log('Initializing 3D Frogger...');

    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB); // Sky blue

    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 25, 15);
    camera.lookAt(0, 0, 0);

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    document.body.appendChild(renderer.domElement);

    // Add lighting
    setupLighting();

    // Create game world
    createGameWorld();

    // Create frog
    createFrog();

    // Setup event listeners
    setupEventListeners();

    // Start render loop
    animate();

    console.log('3D Frogger initialized successfully!');
}

function setupLighting() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    // Directional light (sun)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 20, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 50;
    directionalLight.shadow.camera.left = -25;
    directionalLight.shadow.camera.right = 25;
    directionalLight.shadow.camera.top = 25;
    directionalLight.shadow.camera.bottom = -25;
    scene.add(directionalLight);
}

function createGameWorld() {
    // Create ground plane
    const groundGeometry = new THREE.PlaneGeometry(GAME_WIDTH, GAME_HEIGHT);
    const groundMaterial = new THREE.MeshLambertMaterial({ color: 0x228B22 });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Create road sections
    createRoadSections();

    // Create river section
    createRiverSection();

    // Create safe zones
    createSafeZones();

    // Create lily pads (goal)
    createLilyPads();
}

function createRoadSections() {
    // Road lanes (y = -6 to -2)
    for (let y = -6; y <= -2; y += 2) {
        const roadGeometry = new THREE.PlaneGeometry(GAME_WIDTH, LANE_WIDTH);
        const roadMaterial = new THREE.MeshLambertMaterial({ color: 0x333333 });
        const road = new THREE.Mesh(roadGeometry, roadMaterial);
        road.rotation.x = -Math.PI / 2;
        road.position.set(0, 0.01, y);
        scene.add(road);

        // Add road markings
        const lineGeometry = new THREE.PlaneGeometry(GAME_WIDTH, 0.1);
        const lineMaterial = new THREE.MeshLambertMaterial({ color: 0xFFFFFF });
        const line = new THREE.Mesh(lineGeometry, lineMaterial);
        line.rotation.x = -Math.PI / 2;
        line.position.set(0, 0.02, y);
        scene.add(line);
    }
}

function createRiverSection() {
    // River (y = 2 to 6)
    const riverGeometry = new THREE.PlaneGeometry(GAME_WIDTH, 8);
    const riverMaterial = new THREE.MeshLambertMaterial({
        color: 0x0077BE,
        transparent: true,
        opacity: 0.8
    });
    const river = new THREE.Mesh(riverGeometry, riverMaterial);
    river.rotation.x = -Math.PI / 2;
    river.position.set(0, 0.01, 4);
    scene.add(river);
}

function createSafeZones() {
    // Starting zone (y = -8)
    const startGeometry = new THREE.PlaneGeometry(GAME_WIDTH, 2);
    const startMaterial = new THREE.MeshLambertMaterial({ color: 0x90EE90 });
    const startZone = new THREE.Mesh(startGeometry, startMaterial);
    startZone.rotation.x = -Math.PI / 2;
    startZone.position.set(0, 0.01, -8);
    scene.add(startZone);

    // Middle safe zone (y = 0)
    const middleGeometry = new THREE.PlaneGeometry(GAME_WIDTH, 2);
    const middleMaterial = new THREE.MeshLambertMaterial({ color: 0x90EE90 });
    const middleZone = new THREE.Mesh(middleGeometry, middleMaterial);
    middleZone.rotation.x = -Math.PI / 2;
    middleZone.position.set(0, 0.01, 0);
    scene.add(middleZone);
}

function createLilyPads() {
    lilyPads = [];
    const positions = [-6, -2, 2, 6];

    positions.forEach(x => {
        const padGeometry = new THREE.CylinderGeometry(0.8, 0.8, 0.1, 8);
        const padMaterial = new THREE.MeshLambertMaterial({ color: 0x228B22 });
        const pad = new THREE.Mesh(padGeometry, padMaterial);
        pad.position.set(x, 0.1, 8);
        pad.castShadow = true;
        scene.add(pad);
        lilyPads.push(pad);
    });
}

function createFrog() {
    // Frog body
    const bodyGeometry = new THREE.SphereGeometry(0.4, 8, 6);
    const bodyMaterial = new THREE.MeshLambertMaterial({ color: 0x228B22 });
    frog = new THREE.Mesh(bodyGeometry, bodyMaterial);
    frog.position.set(0, 0.4, -8);
    frog.castShadow = true;
    scene.add(frog);

    // Frog eyes
    const eyeGeometry = new THREE.SphereGeometry(0.1, 6, 4);
    const eyeMaterial = new THREE.MeshLambertMaterial({ color: 0x000000 });

    const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    leftEye.position.set(-0.2, 0.3, 0.3);
    frog.add(leftEye);

    const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    rightEye.position.set(0.2, 0.3, 0.3);
    frog.add(rightEye);
}

function createCar(lane, direction) {
    const carGeometry = new THREE.BoxGeometry(1.5, 0.6, 0.8);
    const carMaterial = new THREE.MeshLambertMaterial({
        color: Math.random() * 0xffffff
    });
    const car = new THREE.Mesh(carGeometry, carMaterial);

    const laneZ = -6 + (lane * 2);
    car.position.set(direction > 0 ? -12 : 12, 0.3, laneZ);
    car.castShadow = true;

    car.userData = {
        direction: direction,
        speed: (1 + Math.random() * 2) * gameSpeed,
        lane: lane
    };

    scene.add(car);
    cars.push(car);

    return car;
}

function createTruck(lane, direction) {
    const truckGeometry = new THREE.BoxGeometry(2.5, 0.8, 1.2);
    const truckMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
    const truck = new THREE.Mesh(truckGeometry, truckMaterial);

    const laneZ = -6 + (lane * 2);
    truck.position.set(direction > 0 ? -12 : 12, 0.4, laneZ);
    truck.castShadow = true;

    truck.userData = {
        direction: direction,
        speed: (0.5 + Math.random() * 1) * gameSpeed,
        lane: lane
    };

    scene.add(truck);
    trucks.push(truck);

    return truck;
}

function createLog(lane, direction) {
    const logGeometry = new THREE.CylinderGeometry(0.3, 0.3, 3, 8);
    const logMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
    const log = new THREE.Mesh(logGeometry, logMaterial);

    const laneZ = 2 + (lane * 2);
    log.position.set(direction > 0 ? -12 : 12, 0.2, laneZ);
    log.rotation.z = Math.PI / 2;
    log.castShadow = true;

    log.userData = {
        direction: direction,
        speed: (0.8 + Math.random() * 1.2) * gameSpeed,
        lane: lane,
        frogOnLog: false
    };

    scene.add(log);
    logs.push(log);

    return log;
}

function spawnVehicles() {
    // Spawn cars and trucks on road lanes
    if (Math.random() < 0.02 * gameSpeed) {
        const lane = Math.floor(Math.random() * 3);
        const direction = lane % 2 === 0 ? 1 : -1;

        if (Math.random() < 0.7) {
            createCar(lane, direction);
        } else {
            createTruck(lane, direction);
        }
    }

    // Spawn logs on river lanes
    if (Math.random() < 0.015 * gameSpeed) {
        const lane = Math.floor(Math.random() * 3);
        const direction = lane % 2 === 0 ? 1 : -1;
        createLog(lane, direction);
    }
}

function updateVehicles() {
    // Update cars
    cars.forEach((car, index) => {
        car.position.x += car.userData.direction * car.userData.speed * 0.1;

        if (Math.abs(car.position.x) > 15) {
            scene.remove(car);
            cars.splice(index, 1);
        }
    });

    // Update trucks
    trucks.forEach((truck, index) => {
        truck.position.x += truck.userData.direction * truck.userData.speed * 0.1;

        if (Math.abs(truck.position.x) > 15) {
            scene.remove(truck);
            trucks.splice(index, 1);
        }
    });

    // Update logs
    logs.forEach((log, index) => {
        log.position.x += log.userData.direction * log.userData.speed * 0.1;

        // Move frog with log if on it
        if (log.userData.frogOnLog) {
            frog.position.x += log.userData.direction * log.userData.speed * 0.1;
        }

        if (Math.abs(log.position.x) > 15) {
            if (log.userData.frogOnLog) {
                // Frog fell off the edge
                loseLife();
            }
            scene.remove(log);
            logs.splice(index, 1);
        }
    });
}

function checkCollisions() {
    const frogBox = new THREE.Box3().setFromObject(frog);

    // Check car collisions
    cars.forEach(car => {
        const carBox = new THREE.Box3().setFromObject(car);
        if (frogBox.intersectsBox(carBox)) {
            loseLife();
        }
    });

    // Check truck collisions
    trucks.forEach(truck => {
        const truckBox = new THREE.Box3().setFromObject(truck);
        if (frogBox.intersectsBox(truckBox)) {
            loseLife();
        }
    });

    // Check if frog is on a log in the river
    if (frog.position.z >= 2 && frog.position.z <= 6) {
        let onLog = false;

        logs.forEach(log => {
            const logBox = new THREE.Box3().setFromObject(log);
            if (frogBox.intersectsBox(logBox)) {
                onLog = true;
                log.userData.frogOnLog = true;
            } else {
                log.userData.frogOnLog = false;
            }
        });

        if (!onLog) {
            // Frog is in water without a log
            loseLife();
        }
    }

    // Check lily pad victory
    lilyPads.forEach(pad => {
        const padBox = new THREE.Box3().setFromObject(pad);
        if (frogBox.intersectsBox(padBox)) {
            levelComplete();
        }
    });
}

function setupEventListeners() {
    // Keyboard input
    document.addEventListener('keydown', (event) => {
        keys[event.code] = true;

        if (gameState === 'playing') {
            handleMovement(event.code);
        }

        if (event.code === 'Space') {
            event.preventDefault();
            if (gameState === 'playing') {
                pauseGame();
            } else if (gameState === 'paused') {
                resumeGame();
            }
        }
    });

    document.addEventListener('keyup', (event) => {
        keys[event.code] = false;
    });

    // Start button
    document.getElementById('startButton').addEventListener('click', startGame);

    // Window resize
    window.addEventListener('resize', onWindowResize);
}

function handleMovement(keyCode) {
    const currentTime = Date.now();
    if (currentTime - lastMoveTime < moveDelay) return;

    const moveDistance = 2;
    let moved = false;

    switch(keyCode) {
        case 'ArrowUp':
        case 'KeyW':
            if (frog.position.z < 8) {
                frog.position.z += moveDistance;
                moved = true;
            }
            break;
        case 'ArrowDown':
        case 'KeyS':
            if (frog.position.z > -8) {
                frog.position.z -= moveDistance;
                moved = true;
            }
            break;
        case 'ArrowLeft':
        case 'KeyA':
            if (frog.position.x > -8) {
                frog.position.x -= moveDistance;
                moved = true;
            }
            break;
        case 'ArrowRight':
        case 'KeyD':
            if (frog.position.x < 8) {
                frog.position.x += moveDistance;
                moved = true;
            }
            break;
    }

    if (moved) {
        lastMoveTime = currentTime;
        // Add points for forward movement
        if (keyCode === 'ArrowUp' || keyCode === 'KeyW') {
            score += 10;
            updateUI();
        }
    }
}

function startGame() {
    console.log('Starting game...');
    document.getElementById('startScreen').style.display = 'none';
    gameState = 'playing';
    resetGame();
}

function pauseGame() {
    gameState = 'paused';
    console.log('Game paused');
}

function resumeGame() {
    gameState = 'playing';
    console.log('Game resumed');
}

function loseLife() {
    if (gameState !== 'playing') return;

    lives--;
    updateUI();

    // Reset frog position
    frog.position.set(0, 0.4, -8);

    // Clear log associations
    logs.forEach(log => {
        log.userData.frogOnLog = false;
    });

    if (lives <= 0) {
        gameOver();
    } else {
        console.log(`Life lost! Lives remaining: ${lives}`);
    }
}

function levelComplete() {
    score += 1000 * level;
    level++;
    gameSpeed += 0.2;

    // Reset frog position
    frog.position.set(0, 0.4, -8);

    // Clear all vehicles and logs
    clearGameObjects();

    updateUI();

    // Show level complete notification
    showNotification(`Level ${level - 1} Complete!`);

    console.log(`Level ${level - 1} completed! Starting level ${level}`);
}

function gameOver() {
    gameState = 'gameOver';
    document.getElementById('finalScore').textContent = score;
    document.getElementById('gameOver').style.display = 'block';
    console.log(`Game Over! Final Score: ${score}`);
}

function restartGame() {
    document.getElementById('gameOver').style.display = 'none';
    document.getElementById('startScreen').style.display = 'flex';
    gameState = 'start';
}

function resetGame() {
    score = 0;
    lives = 3;
    level = 1;
    gameSpeed = 1;

    // Reset frog position
    frog.position.set(0, 0.4, -8);

    // Clear all game objects
    clearGameObjects();

    updateUI();
}

function clearGameObjects() {
    // Remove all cars
    cars.forEach(car => scene.remove(car));
    cars = [];

    // Remove all trucks
    trucks.forEach(truck => scene.remove(truck));
    trucks = [];

    // Remove all logs
    logs.forEach(log => scene.remove(log));
    logs = [];
}

function updateUI() {
    document.getElementById('score').textContent = `Score: ${score}`;
    document.getElementById('lives').textContent = `Lives: ${lives}`;
    document.getElementById('level').textContent = `Level: ${level}`;
}

function showNotification(text) {
    const notification = document.createElement('div');
    notification.className = 'powerup-notification';
    notification.textContent = text;
    document.body.appendChild(notification);

    setTimeout(() => {
        document.body.removeChild(notification);
    }, 2000);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);

    if (gameState === 'playing') {
        // Spawn new vehicles and logs
        spawnVehicles();

        // Update all moving objects
        updateVehicles();

        // Check for collisions
        checkCollisions();

        // Keep frog within bounds
        frog.position.x = Math.max(-8, Math.min(8, frog.position.x));
        frog.position.z = Math.max(-8, Math.min(8, frog.position.z));
    }

    // Render the scene
    renderer.render(scene, camera);
}

// Initialize the game when the page loads
window.addEventListener('load', () => {
    console.log('Page loaded, initializing 3D Frogger...');
    init();
});

// Global restart function for the restart button
window.restartGame = restartGame;