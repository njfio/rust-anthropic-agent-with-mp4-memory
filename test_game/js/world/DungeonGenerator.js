// Dungeon Generator
// Handles procedural dungeon generation using BSP

import { gameState } from '../core/GameState.js';

export class DungeonGenerator {
    constructor() {
        this.DUNGEON_SIZE = 20;
        this.CELL_SIZE = 4;
        this.WALL_HEIGHT = 5;
    }

    generateDungeon() {
        // Clear existing dungeon
        gameState.clearDungeon();
        
        // Create dungeon grid
        const dungeon = Array(this.DUNGEON_SIZE).fill().map(() => Array(this.DUNGEON_SIZE).fill(1));
        
        // Generate rooms and corridors using BSP
        const rooms = [];
        this.generateRooms(dungeon, rooms, 0, 0, this.DUNGEON_SIZE - 1, this.DUNGEON_SIZE - 1, 3);
        this.connectRooms(dungeon, rooms);
        
        // Create 3D representation
        this.createDungeon3D(dungeon);
        
        return rooms;
    }

    generateRooms(dungeon, rooms, x1, y1, x2, y2, depth) {
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
            this.generateRooms(dungeon, rooms, x1, y1, splitX, y2, depth - 1);
            this.generateRooms(dungeon, rooms, splitX + 1, y1, x2, y2, depth - 1);
        } else {
            // Horizontal split
            const splitY = y1 + Math.floor((y2 - y1) / 2);
            this.generateRooms(dungeon, rooms, x1, y1, x2, splitY, depth - 1);
            this.generateRooms(dungeon, rooms, x1, splitY + 1, x2, y2, depth - 1);
        }
    }

    connectRooms(dungeon, rooms) {
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

    createDungeon3D(dungeon) {
        // Floor
        const floorGeometry = new THREE.PlaneGeometry(
            this.DUNGEON_SIZE * this.CELL_SIZE,
            this.DUNGEON_SIZE * this.CELL_SIZE
        );
        const floorMaterial = new THREE.MeshLambertMaterial({ color: 0x444444 });
        gameState.floor = new THREE.Mesh(floorGeometry, floorMaterial);
        gameState.floor.rotation.x = -Math.PI / 2;
        gameState.floor.receiveShadow = true;
        gameState.scene.add(gameState.floor);
        
        // Walls
        const wallGeometry = new THREE.BoxGeometry(this.CELL_SIZE, this.WALL_HEIGHT, this.CELL_SIZE);
        const wallMaterial = new THREE.MeshLambertMaterial({ color: 0x666666 });
        
        for (let y = 0; y < this.DUNGEON_SIZE; y++) {
            for (let x = 0; x < this.DUNGEON_SIZE; x++) {
                if (dungeon[y][x] === 1) {
                    const wall = new THREE.Mesh(wallGeometry, wallMaterial);
                    wall.position.set(
                        x * this.CELL_SIZE - (this.DUNGEON_SIZE * this.CELL_SIZE) / 2 + this.CELL_SIZE / 2,
                        this.WALL_HEIGHT / 2,
                        y * this.CELL_SIZE - (this.DUNGEON_SIZE * this.CELL_SIZE) / 2 + this.CELL_SIZE / 2
                    );
                    wall.castShadow = true;
                    wall.receiveShadow = true;
                    gameState.walls.push(wall);
                    gameState.scene.add(wall);
                }
            }
        }
    }

    createExit(x, z) {
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

    getRoomCenter(room) {
        return {
            x: Math.floor((room.x1 + room.x2) / 2) * this.CELL_SIZE - (this.DUNGEON_SIZE * this.CELL_SIZE) / 2,
            z: Math.floor((room.y1 + room.y2) / 2) * this.CELL_SIZE - (this.DUNGEON_SIZE * this.CELL_SIZE) / 2
        };
    }

    getRandomPositionInRoom(room) {
        return {
            x: (room.x1 + Math.random() * (room.x2 - room.x1)) * this.CELL_SIZE - (this.DUNGEON_SIZE * this.CELL_SIZE) / 2,
            z: (room.y1 + Math.random() * (room.y2 - room.y1)) * this.CELL_SIZE - (this.DUNGEON_SIZE * this.CELL_SIZE) / 2
        };
    }
}

// Singleton instance
export const dungeonGenerator = new DungeonGenerator();