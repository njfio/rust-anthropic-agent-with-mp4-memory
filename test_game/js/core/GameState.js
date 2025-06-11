// Game State Management
// Centralized state management for the game

export class GameState {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.player = null;
        this.enemies = [];
        this.items = [];
        this.walls = [];
        this.floor = null;
        this.dungeonFloor = 1;
        this.gameRunning = false;
        
        this.controls = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            attack: false
        };
        
        this.mouseX = 0;
        this.mouseY = 0;
    }

    reset() {
        this.enemies = [];
        this.items = [];
        this.walls = [];
        this.floor = null;
        this.dungeonFloor = 1;
        this.gameRunning = false;
        
        this.controls = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            attack: false
        };
        
        this.mouseX = 0;
        this.mouseY = 0;
    }

    clearDungeon() {
        // Remove all objects from scene
        this.walls.forEach(wall => this.scene.remove(wall));
        this.enemies.forEach(enemy => this.scene.remove(enemy));
        this.items.forEach(item => this.scene.remove(item));
        
        this.walls = [];
        this.enemies = [];
        this.items = [];
        
        if (this.floor) {
            this.scene.remove(this.floor);
        }
        
        if (this.player) {
            this.scene.remove(this.player);
        }
        
        // Remove exit
        this.scene.traverse((child) => {
            if (child.userData && child.userData.type === 'exit') {
                this.scene.remove(child);
            }
        });
    }
}

// Singleton instance
export const gameState = new GameState();