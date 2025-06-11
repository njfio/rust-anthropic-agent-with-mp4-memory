/**
 * Input Manager for Shadowfall Depths
 * Handles keyboard and mouse input
 */

class InputManager {
    constructor() {
        this.keys = new Map();
        this.mousePos = { x: 0, y: 0 };
        this.mouseButtons = new Map();
        this.bindings = new Map();
    }
    
    init() {
        this.bindEvents();
        console.log('Input Manager initialized');
    }
    
    bindEvents() {
        // Keyboard events
        document.addEventListener('keydown', (event) => {
            this.keys.set(event.code, true);
            this.handleKeyDown(event);
        });
        
        document.addEventListener('keyup', (event) => {
            this.keys.set(event.code, false);
            this.handleKeyUp(event);
        });
        
        // Mouse events
        document.addEventListener('mousemove', (event) => {
            if (gameEngine.canvas) {
                this.mousePos = gameEngine.getCanvasMousePos(event);
            }
        });
        
        document.addEventListener('mousedown', (event) => {
            this.mouseButtons.set(event.button, true);
            this.handleMouseDown(event);
        });
        
        document.addEventListener('mouseup', (event) => {
            this.mouseButtons.set(event.button, false);
            this.handleMouseUp(event);
        });
        
        // Prevent context menu
        document.addEventListener('contextmenu', (event) => {
            event.preventDefault();
        });
    }
    
    handleKeyDown(event) {
        // Prevent default for game keys
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(event.code)) {
            event.preventDefault();
        }
        
        // Execute bindings
        const binding = this.bindings.get(event.code);
        if (binding && binding.onPress) {
            binding.onPress();
        }
    }
    
    handleKeyUp(event) {
        const binding = this.bindings.get(event.code);
        if (binding && binding.onRelease) {
            binding.onRelease();
        }
    }
    
    handleMouseDown(event) {
        if (event.button === 0) { // Left click
            this.handleAttack();
        }
    }
    
    handleMouseUp(event) {
        // Handle mouse up events
    }
    
    handleAttack() {
        if (window.player && window.player.alive && window.player.canAttack()) {
            // Find nearest enemy to attack
            const attackRange = 50;
            let nearestEnemy = null;
            let nearestDistance = attackRange;
            
            entities.forEach(entity => {
                if (entity.hasTag('enemy') && entity.alive) {
                    const distance = Utils.distance(
                        player.x + player.width/2, player.y + player.height/2,
                        entity.x + entity.width/2, entity.y + entity.height/2
                    );
                    
                    if (distance < nearestDistance) {
                        nearestEnemy = entity;
                        nearestDistance = distance;
                    }
                }
            });
            
            if (nearestEnemy) {
                player.attack(nearestEnemy);
                gameEngine.shakeCamera(5, 200);
            }
        }
    }
    
    bindPlayerControls() {
        // Movement keys
        this.bind('KeyW', {
            onPress: () => this.updatePlayerMovement(),
            onRelease: () => this.updatePlayerMovement()
        });
        
        this.bind('KeyS', {
            onPress: () => this.updatePlayerMovement(),
            onRelease: () => this.updatePlayerMovement()
        });
        
        this.bind('KeyA', {
            onPress: () => this.updatePlayerMovement(),
            onRelease: () => this.updatePlayerMovement()
        });
        
        this.bind('KeyD', {
            onPress: () => this.updatePlayerMovement(),
            onRelease: () => this.updatePlayerMovement()
        });
        
        // Arrow keys
        this.bind('ArrowUp', {
            onPress: () => this.updatePlayerMovement(),
            onRelease: () => this.updatePlayerMovement()
        });
        
        this.bind('ArrowDown', {
            onPress: () => this.updatePlayerMovement(),
            onRelease: () => this.updatePlayerMovement()
        });
        
        this.bind('ArrowLeft', {
            onPress: () => this.updatePlayerMovement(),
            onRelease: () => this.updatePlayerMovement()
        });
        
        this.bind('ArrowRight', {
            onPress: () => this.updatePlayerMovement(),
            onRelease: () => this.updatePlayerMovement()
        });
        
        // Action keys
        this.bind('Space', {
            onPress: () => this.handleAttack()
        });
        
        this.bind('KeyE', {
            onPress: () => this.handleInteract()
        });
    }
    
    updatePlayerMovement() {
        if (!window.player) return;
        
        let moveX = 0;
        let moveY = 0;
        
        // WASD movement
        if (this.isKeyPressed('KeyW') || this.isKeyPressed('ArrowUp')) {
            moveY -= 1;
        }
        if (this.isKeyPressed('KeyS') || this.isKeyPressed('ArrowDown')) {
            moveY += 1;
        }
        if (this.isKeyPressed('KeyA') || this.isKeyPressed('ArrowLeft')) {
            moveX -= 1;
        }
        if (this.isKeyPressed('KeyD') || this.isKeyPressed('ArrowRight')) {
            moveX += 1;
        }
        
        player.setMoveDirection(moveX, moveY);
    }
    
    handleInteract() {
        // Handle interaction with objects
        console.log('Interact');
    }
    
    bind(keyCode, actions) {
        this.bindings.set(keyCode, actions);
    }
    
    unbind(keyCode) {
        this.bindings.delete(keyCode);
    }
    
    isKeyPressed(keyCode) {
        return this.keys.get(keyCode) || false;
    }
    
    isMouseButtonPressed(button) {
        return this.mouseButtons.get(button) || false;
    }
    
    getMousePosition() {
        return { ...this.mousePos };
    }
    
    getWorldMousePosition() {
        if (gameEngine.canvas) {
            return gameEngine.screenToWorld(this.mousePos.x, this.mousePos.y);
        }
        return { x: 0, y: 0 };
    }
}

// Create global input manager
window.InputManager = new InputManager();

console.log('Input system loaded');