// 3D Roguelike Dungeon Crawler - Modular Version
// Main entry point using modular architecture

import { gameController } from './js/GameController.js';
import { uiManager } from './js/ui/UIManager.js';

// Initialize game when window loads
window.onload = function() {
    gameController.init();
    uiManager.updateMetaUpgradesUI();
    
    console.log('3D Roguelike Dungeon Crawler - Modular Version Loaded');
    console.log('Modules loaded: GameController, SceneManager, InputManager, UIManager');
    console.log('Use startNewRun() to begin a new game');
};