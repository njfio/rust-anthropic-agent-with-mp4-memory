import { CONFIG, COLORS } from '../config.js';
import { Utils } from '../utils.js';

export class DungeonSystem {
    constructor(events) {
        this.events = events;
        this.walls = [];
        this.floor = null;
        this.exit = null;
        this.dungeonGrid = null;
        this.rooms = [];
    }
    
    generateDungeon() {
        // Initialize grid
        this.dungeonGrid = Array(CONFIG.DUNGEON.SIZE).fill()
            .map(() => Array(CONFIG.DUNGEON.SIZE).fill(1));
        
        // Clear previous rooms
        this.rooms = [];
        
        // Generate rooms using BSP
        this.generateRooms(
            0, 0, 
            CONFIG.DUNGEON.SIZE - 1, 
            CONFIG.DUNGEON.SIZE - 1, 
            CONFIG.DUNGEON.BSP_DEPTH
        );
        
        // Connect rooms
        this.connectRooms();
        
        // Calculate room centers
        this.rooms.forEach(room => {
            room.center = {
                x: Math.floor((room.x1 + room.x2) / 2),
                z: Math.floor((room.y1 + room.y2) / 2)
            };
        });
        
        return {
            grid: this.dungeonGrid,
            rooms: this.rooms
        };
    }
    
    generateRooms(x1, y1, x2, y2, depth) {
        if (depth <= 0 || 
            (x2 - x1) < CONFIG.DUNGEON.MIN_ROOM_SIZE || 
            (y2 - y1) < CONFIG.DUNGEON.MIN_ROOM_SIZE) {
            // Create room
            const roomX1 = x1 + 1;
            const roomY1 = y1 + 1;
            const roomX2 = x2 - 1;
            const roomY2 = y2 - 1;
            
            this.rooms.push({ 
                x1: roomX1, 
                y1: roomY1, 
                x2: roomX2, 
                y2: roomY2 
            });
            
            // Carve out room
            for (let x = roomX1; x <= roomX2; x++) {
                for (let y = roomY1; y <= roomY2; y++) {
                    this.dungeonGrid[y][x] = 0;
                }
            }
            return;
        }
        
        // Split horizontally or vertically
        if (Math.random() > 0.5) {
            // Vertical split
            const splitX = x1 + Math.floor((x2 - x1) / 2);
            this.generateRooms(x1, y1, splitX, y2, depth - 1);
            this.generateRooms(splitX + 1, y1, x2, y2, depth - 1);
        } else {
            // Horizontal split
            const splitY = y1 + Math.floor((y2 - y1) / 2);
            this.generateRooms(x1, y1, x2, splitY, depth - 1);
            this.generateRooms(x1, splitY + 1, x2, y2, depth - 1);
        }
    }
    
    connectRooms() {
        for (let i = 0; i < this.rooms.length - 1; i++) {
            const room1 = this.rooms[i];
            const room2 = this.rooms[i + 1];
            
            const x1 = Math.floor((room1.x1 + room1.x2) / 2);
            const y1 = Math.floor((room1.y1 + room1.y2) / 2);
            const x2 = Math.floor((room2.x1 + room2.x2) / 2);
            const y2 = Math.floor((room2.y1 + room2.y2) / 2);
            
            // Create L-shaped corridor
            for (let x = Math.min(x1, x2); x <= Math.max(x1, x2); x++) {
                this.dungeonGrid[y1][x] = 0;
            }
            for (let y = Math.min(y1, y2); y <= Math.max(y1, y2); y++) {
                this.dungeonGrid[y][x2] = 0;
            }
        }
    }
    
    createDungeon3D(grid, scene) {
        // Create floor
        const floorGeometry = new THREE.PlaneGeometry(
            CONFIG.DUNGEON.SIZE * CONFIG.DUNGEON.CELL_SIZE,
            CONFIG.DUNGEON.SIZE * CONFIG.DUNGEON.CELL_SIZE
        );
        const floorMaterial = new THREE.MeshLambertMaterial({ 
            color: COLORS.FLOOR 
        });
        this.floor = new THREE.Mesh(floorGeometry, floorMaterial);
        this.floor.rotation.x = -Math.PI / 2;
        this.floor.receiveShadow = true;
        scene.add(this.floor);
        
        // Create walls
        const wallGeometry = new THREE.BoxGeometry(
            CONFIG.DUNGEON.CELL_SIZE, 
            CONFIG.DUNGEON.WALL_HEIGHT, 
            CONFIG.DUNGEON.CELL_SIZE
        );
        const wallMaterial = new THREE.MeshLambertMaterial({ 
            color: COLORS.WALL 
        });
        
        this.walls = [];
        
        for (let y = 0; y < CONFIG.DUNGEON.SIZE; y++) {
            for (let x = 0; x < CONFIG.DUNGEON.SIZE; x++) {
                if (grid[y][x] === 1) {
                    const wall = new THREE.Mesh(wallGeometry, wallMaterial);
                    wall.position.set(
                        x * CONFIG.DUNGEON.CELL_SIZE - 
                            (CONFIG.DUNGEON.SIZE * CONFIG.DUNGEON.CELL_SIZE) / 2 + 
                            CONFIG.DUNGEON.CELL_SIZE / 2,
                        CONFIG.DUNGEON.WALL_HEIGHT / 2,
                        y * CONFIG.DUNGEON.CELL_SIZE - 
                            (CONFIG.DUNGEON.SIZE * CONFIG.DUNGEON.CELL_SIZE) / 2 + 
                            CONFIG.DUNGEON.CELL_SIZE / 2
                    );
                    wall.castShadow = true;
                    wall.receiveShadow = true;
                    this.walls.push(wall);
                    scene.add(wall);
                }
            }
        }
        
        return { walls: this.walls, floor: this.floor };
    }
    
    createExit(x, z) {
        const exitGeometry = new THREE.CylinderGeometry(1.5, 1.5, 0.2, 8);
        const exitMaterial = new THREE.MeshLambertMaterial({ 
            color: COLORS.EXIT,
            emissive: COLORS.EXIT,
            emissiveIntensity: 0.5
        });
        this.exit = new THREE.Mesh(exitGeometry, exitMaterial);
        
        this.exit.position.set(x, 0.1, z);
        this.exit.userData = { type: 'exit' };
        
        // Add glow effect
        const glowGeometry = new THREE.RingGeometry(1.5, 2, 8);
        const glowMaterial = new THREE.MeshBasicMaterial({ 
            color: COLORS.EXIT,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        glow.rotation.x = -Math.PI / 2;
        glow.position.y = 0.2;
        this.exit.add(glow);
        
        // Animate in render loop
        const animate = () => {
            if (this.exit) {
                const time = Date.now() * 0.001;
                this.exit.rotation.y = time;
                if (glow.material) {
                    glow.material.opacity = 0.3 + Math.sin(time * 3) * 0.2;
                }
            }
        };
        
        // Store animation function
        this.exit.userData.animate = animate;
        
        return this.exit;
    }
    
    getWalls() {
        return this.walls;
    }
    
    getExit() {
        return this.exit;
    }
    
    clear() {
        // Clear walls
        this.walls.forEach(wall => {
            if (wall.parent) {
                wall.parent.remove(wall);
            }
            if (wall.geometry) wall.geometry.dispose();
            if (wall.material) wall.material.dispose();
        });
        this.walls = [];
        
        // Clear floor
        if (this.floor) {
            if (this.floor.parent) {
                this.floor.parent.remove(this.floor);
            }
            if (this.floor.geometry) this.floor.geometry.dispose();
            if (this.floor.material) this.floor.material.dispose();
            this.floor = null;
        }
        
        // Clear exit
        if (this.exit) {
            if (this.exit.parent) {
                this.exit.parent.remove(this.exit);
            }
            this.exit.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            this.exit = null;
        }
        
        // Clear data
        this.dungeonGrid = null;
        this.rooms = [];
    }
}