/**
 * Game Engine - Core game loop and state management
 * Professional game engine architecture for Shadowfall Depths
 */

class GameEngine {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.lastTime = 0;
        this.deltaTime = 0;
        this.fps = 60;
        this.targetFrameTime = 1000 / this.fps;
        
        this.isRunning = false;
        this.isPaused = false;
        this.gameState = 'menu'; // menu, playing, paused, gameOver, upgrades
        
        this.camera = {
            x: 0,
            y: 0,
            targetX: 0,
            targetY: 0,
            smoothing: 0.1,
            shake: { x: 0, y: 0, intensity: 0, duration: 0 }
        };
        
        this.viewport = {
            width: 800,
            height: 600,
            scale: 1
        };
        
        // Performance monitoring
        this.performance = {
            frameCount: 0,
            lastFpsUpdate: 0,
            currentFps: 0,
            updateTime: 0,
            renderTime: 0
        };
        
        // Event system
        this.eventListeners = new Map();
        
        this.init();
    }
    
    init() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas size
        this.canvas.width = this.viewport.width;
        this.canvas.height = this.viewport.height;
        
        // Disable image smoothing for pixel-perfect rendering
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.webkitImageSmoothingEnabled = false;
        this.ctx.mozImageSmoothingEnabled = false;
        this.ctx.msImageSmoothingEnabled = false;
        
        // Initialize systems
        this.initEventSystem();
        
        console.log('Game Engine initialized');
    }
    
    initEventSystem() {
        // Custom event system for game-wide communication
        this.on = (event, callback) => {
            if (!this.eventListeners.has(event)) {
                this.eventListeners.set(event, []);
            }
            this.eventListeners.get(event).push(callback);
        };
        
        this.off = (event, callback) => {
            if (this.eventListeners.has(event)) {
                const listeners = this.eventListeners.get(event);
                const index = listeners.indexOf(callback);
                if (index > -1) {
                    listeners.splice(index, 1);
                }
            }
        };
        
        this.emit = (event, data = null) => {
            if (this.eventListeners.has(event)) {
                this.eventListeners.get(event).forEach(callback => {
                    try {
                        callback(data);
                    } catch (error) {
                        console.error(`Error in event listener for ${event}:`, error);
                    }
                });
            }
        };
    }
    
    start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.lastTime = performance.now();
        this.gameLoop();
        
        console.log('Game engine started');
    }
    
    stop() {
        this.isRunning = false;
        console.log('Game engine stopped');
    }
    
    pause() {
        this.isPaused = true;
        this.emit('game:paused');
    }
    
    resume() {
        this.isPaused = false;
        this.emit('game:resumed');
    }
    
    gameLoop() {
        if (!this.isRunning) return;
        
        const currentTime = performance.now();
        this.deltaTime = currentTime - this.lastTime;
        this.lastTime = currentTime;
        
        // Cap delta time to prevent spiral of death
        this.deltaTime = Math.min(this.deltaTime, this.targetFrameTime * 3);
        
        // Update performance metrics
        this.updatePerformanceMetrics(currentTime);
        
        if (!this.isPaused) {
            // Update phase
            const updateStart = performance.now();
            this.update(this.deltaTime);
            this.performance.updateTime = performance.now() - updateStart;
        }
        
        // Render phase
        const renderStart = performance.now();
        this.render();
        this.performance.renderTime = performance.now() - renderStart;
        
        // Schedule next frame
        requestAnimationFrame(() => this.gameLoop());
    }
    
    update(deltaTime) {
        // Update camera
        this.updateCamera(deltaTime);
        
        // Emit update event for game systems
        this.emit('engine:update', deltaTime);
    }
    
    render() {
        // Clear canvas
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.viewport.width, this.viewport.height);
        
        // Save context state
        this.ctx.save();
        
        // Apply camera transform
        this.ctx.translate(
            -this.camera.x + this.camera.shake.x,
            -this.camera.y + this.camera.shake.y
        );
        
        // Emit render event for game systems
        this.emit('engine:render', this.ctx);
        
        // Restore context state
        this.ctx.restore();
        
        // Render UI elements that don't move with camera
        this.emit('engine:renderUI', this.ctx);
        
        // Render debug info if enabled
        if (window.DEBUG) {
            this.renderDebugInfo();
        }
    }
    
    updateCamera(deltaTime) {
        // Smooth camera movement
        this.camera.x = Utils.lerp(this.camera.x, this.camera.targetX, this.camera.smoothing);
        this.camera.y = Utils.lerp(this.camera.y, this.camera.targetY, this.camera.smoothing);
        
        // Update camera shake
        if (this.camera.shake.duration > 0) {
            this.camera.shake.duration -= deltaTime;
            const intensity = this.camera.shake.intensity * (this.camera.shake.duration / 1000);
            this.camera.shake.x = (Math.random() - 0.5) * intensity;
            this.camera.shake.y = (Math.random() - 0.5) * intensity;
        } else {
            this.camera.shake.x = 0;
            this.camera.shake.y = 0;
        }
    }
    
    updatePerformanceMetrics(currentTime) {
        this.performance.frameCount++;
        
        if (currentTime - this.performance.lastFpsUpdate >= 1000) {
            this.performance.currentFps = this.performance.frameCount;
            this.performance.frameCount = 0;
            this.performance.lastFpsUpdate = currentTime;
        }
    }
    
    renderDebugInfo() {
        this.ctx.save();
        this.ctx.fillStyle = '#00ff00';
        this.ctx.font = '12px monospace';
        
        const debugInfo = [
            `FPS: ${this.performance.currentFps}`,
            `Update: ${this.performance.updateTime.toFixed(2)}ms`,
            `Render: ${this.performance.renderTime.toFixed(2)}ms`,
            `State: ${this.gameState}`,
            `Camera: (${this.camera.x.toFixed(0)}, ${this.camera.y.toFixed(0)})`
        ];
        
        debugInfo.forEach((info, index) => {
            this.ctx.fillText(info, 10, 20 + index * 15);
        });
        
        this.ctx.restore();
    }
    
    // Camera control methods
    setCameraTarget(x, y) {
        this.camera.targetX = x - this.viewport.width / 2;
        this.camera.targetY = y - this.viewport.height / 2;
    }
    
    setCameraPosition(x, y) {
        this.camera.x = x - this.viewport.width / 2;
        this.camera.y = y - this.viewport.height / 2;
        this.camera.targetX = this.camera.x;
        this.camera.targetY = this.camera.y;
    }
    
    shakeCamera(intensity, duration) {
        this.camera.shake.intensity = intensity;
        this.camera.shake.duration = duration;
    }
    
    // Screen coordinate conversion
    screenToWorld(screenX, screenY) {
        return {
            x: screenX + this.camera.x,
            y: screenY + this.camera.y
        };
    }
    
    worldToScreen(worldX, worldY) {
        return {
            x: worldX - this.camera.x,
            y: worldY - this.camera.y
        };
    }
    
    // Viewport checks
    isInViewport(x, y, width = 0, height = 0) {
        const screenPos = this.worldToScreen(x, y);
        return screenPos.x + width >= 0 && 
               screenPos.x <= this.viewport.width &&
               screenPos.y + height >= 0 && 
               screenPos.y <= this.viewport.height;
    }
    
    // State management
    setState(newState) {
        const oldState = this.gameState;
        this.gameState = newState;
        this.emit('state:changed', { from: oldState, to: newState });
        console.log(`Game state changed: ${oldState} -> ${newState}`);
    }
    
    getState() {
        return this.gameState;
    }
    
    // Utility methods
    getCanvasMousePos(event) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
        };
    }
    
    getWorldMousePos(event) {
        const canvasPos = this.getCanvasMousePos(event);
        return this.screenToWorld(canvasPos.x, canvasPos.y);
    }
    
    // Resource loading utilities
    loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
        });
    }
    
    loadSound(src) {
        return new Promise((resolve, reject) => {
            const audio = new Audio();
            audio.oncanplaythrough = () => resolve(audio);
            audio.onerror = reject;
            audio.src = src;
        });
    }
    
    // Cleanup
    destroy() {
        this.stop();
        this.eventListeners.clear();
        console.log('Game engine destroyed');
    }
}

// Create global game engine instance
window.gameEngine = new GameEngine();