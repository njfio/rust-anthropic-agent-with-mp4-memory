// Input Manager
// Handles all input events and control state

import { gameState } from '../core/GameState.js';

export class InputManager {
    constructor() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.addEventListener('keydown', this.onKeyDown.bind(this));
        document.addEventListener('keyup', this.onKeyUp.bind(this));
        document.addEventListener('mousemove', this.onMouseMove.bind(this));
    }

    onKeyDown(e) {
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
                // Use item - handled by game controller
                this.onUseItem();
                break;
            case 'c':
                // Center camera - handled by game controller
                this.onCenterCamera();
                break;
        }
    }

    onKeyUp(e) {
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
    }

    onMouseMove(e) {
        // Reduce mouse sensitivity and add dampening
        const sensitivity = 0.5; // Reduced from 1.0
        gameState.mouseX = ((e.clientX / window.innerWidth) * 2 - 1) * sensitivity;
        gameState.mouseY = (-(e.clientY / window.innerHeight) * 2 + 1) * sensitivity;
    }

    onUseItem() {
        // Emit custom event for game controller to handle
        document.dispatchEvent(new CustomEvent('useItem'));
    }

    onCenterCamera() {
        // Emit custom event for game controller to handle
        document.dispatchEvent(new CustomEvent('centerCamera'));
    }
}

// Singleton instance
export const inputManager = new InputManager();