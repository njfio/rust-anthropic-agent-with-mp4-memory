// UI Manager
// Handles all UI updates and message display

import { gameState } from '../core/GameState.js';
import { playerStats, metaProgression } from '../core/PlayerStats.js';

export class UIManager {
    constructor() {
        this.messages = document.getElementById('messages');
    }

    updateUI() {
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

    addMessage(text, type = '') {
        const message = document.createElement('div');
        message.className = `message ${type}`;
        message.textContent = text;
        this.messages.appendChild(message);
        
        // Keep only last 5 messages
        while (this.messages.children.length > 5) {
            this.messages.removeChild(this.messages.firstChild);
        }
        
        // Auto scroll to bottom
        this.messages.scrollTop = this.messages.scrollHeight;
    }

    showUpgradeChoices(onUpgradeSelected) {
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
                onUpgradeSelected();
            };
            choicesDiv.appendChild(div);
        });
    }

    showGameOver() {
        gameState.gameRunning = false;
        document.getElementById('gameOver').style.display = 'block';
        document.getElementById('finalFloor').textContent = gameState.dungeonFloor;
        document.getElementById('soulsEarned').textContent = playerStats.souls;
    }

    showMainMenu() {
        document.getElementById('gameOver').style.display = 'none';
        document.getElementById('mainMenu').style.display = 'block';
        this.updateMetaUpgradesUI();
    }

    hideMainMenu() {
        document.getElementById('mainMenu').style.display = 'none';
        document.getElementById('gameOver').style.display = 'none';
    }

    updateMetaUpgradesUI() {
        document.getElementById('totalSouls').textContent = metaProgression.totalSouls;
        
        const upgradesList = document.getElementById('upgradesList');
        upgradesList.innerHTML = '';
        
        const upgrades = metaProgression.getUpgradeDefinitions();
        
        upgrades.forEach(upgrade => {
            const div = document.createElement('div');
            div.className = 'upgrade-option';
            
            const cost = upgrade.cost * (upgrade.level + 1);
            const canAfford = metaProgression.totalSouls >= cost;
            
            div.innerHTML = `
                <strong>${upgrade.name}</strong> - Level ${upgrade.level}<br>
                <small>${upgrade.description}</small><br>
                Cost: ${cost} souls
            `;
            
            if (canAfford) {
                div.style.cursor = 'pointer';
                div.onclick = () => {
                    if (metaProgression.purchaseUpgrade(upgrade.key, cost)) {
                        this.updateMetaUpgradesUI();
                    }
                };
            } else {
                div.style.opacity = '0.5';
                div.style.cursor = 'not-allowed';
            }
            
            upgradesList.appendChild(div);
        });
    }
}

// Singleton instance
export const uiManager = new UIManager();