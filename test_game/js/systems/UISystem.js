import { CONFIG } from '../config.js';
import { Utils } from '../utils.js';

export class UISystem {
    constructor(events) {
        this.events = events;
        this.messages = [];
    }
    
    init() {
        // Cache DOM elements
        this.elements = {
            // Stats
            playerLevel: Utils.getElementById('playerLevel'),
            dungeonFloor: Utils.getElementById('dungeonFloor'),
            playerHP: Utils.getElementById('playerHP'),
            playerMaxHP: Utils.getElementById('playerMaxHP'),
            playerExp: Utils.getElementById('playerExp'),
            expNeeded: Utils.getElementById('expNeeded'),
            playerAtk: Utils.getElementById('playerAtk'),
            playerDef: Utils.getElementById('playerDef'),
            playerGold: Utils.getElementById('playerGold'),
            healthBar: Utils.getElementById('healthBar'),
            expBar: Utils.getElementById('expBar'),
            
            // Inventory
            inventoryList: Utils.getElementById('inventoryList'),
            
            // Messages
            messages: Utils.getElementById('messages'),
            
            // Menus
            mainMenu: Utils.getElementById('mainMenu'),
            gameOver: Utils.getElementById('gameOver'),
            levelComplete: Utils.getElementById('levelComplete'),
            
            // Meta upgrades
            totalSouls: Utils.getElementById('totalSouls'),
            upgradesList: Utils.getElementById('upgradesList'),
            
            // Game over
            finalFloor: Utils.getElementById('finalFloor'),
            soulsEarned: Utils.getElementById('soulsEarned'),
            
            // Upgrade choices
            upgradeChoices: Utils.getElementById('upgradeChoices')
        };
    }
    
    updateStats(stats) {
        if (!stats) return;
        
        // Update text values
        Utils.updateElement('playerLevel', stats.level);
        Utils.updateElement('dungeonFloor', stats.floor || 1);
        Utils.updateElement('playerHP', Math.floor(stats.hp));
        Utils.updateElement('playerMaxHP', stats.maxHp);
        Utils.updateElement('playerExp', stats.exp);
        Utils.updateElement('expNeeded', stats.expNeeded);
        Utils.updateElement('playerAtk', stats.attack);
        Utils.updateElement('playerDef', stats.defense);
        Utils.updateElement('playerGold', stats.gold);
        
        // Update bars
        if (this.elements.healthBar) {
            const healthPercent = (stats.hp / stats.maxHp) * 100;
            this.elements.healthBar.style.width = healthPercent + '%';
        }
        
        if (this.elements.expBar) {
            const expPercent = (stats.exp / stats.expNeeded) * 100;
            this.elements.expBar.style.width = expPercent + '%';
        }
        
        // Update inventory
        this.updateInventory(stats.inventory);
    }
    
    updateInventory(inventory) {
        if (!this.elements.inventoryList) return;
        
        if (inventory && inventory.length > 0) {
            const html = inventory.map(item => 
                `<div class="inventory-item">${item.name}</div>`
            ).join('');
            Utils.setElementHTML('inventoryList', html);
        } else {
            Utils.setElementHTML('inventoryList', '<div>Empty</div>');
        }
    }
    
    addMessage(text, type = '') {
        const messagesElement = this.elements.messages;
        if (!messagesElement) return;
        
        // Create message element
        const message = document.createElement('div');
        message.className = `message ${type}`;
        message.textContent = text;
        messagesElement.appendChild(message);
        
        // Keep only last N messages
        while (messagesElement.children.length > CONFIG.UI.MESSAGE_LIMIT) {
            messagesElement.removeChild(messagesElement.firstChild);
        }
        
        // Auto scroll to bottom
        messagesElement.scrollTop = messagesElement.scrollHeight;
    }
    
    showMainMenu() {
        this.hideAllMenus();
        Utils.showElement('mainMenu');
    }
    
    showGameOver(floor, souls) {
        Utils.updateElement('finalFloor', floor);
        Utils.updateElement('soulsEarned', souls);
        Utils.showElement('gameOver');
    }
    
    showUpgradeChoices(choices, onSelect) {
        Utils.showElement('levelComplete');
        
        const choicesDiv = this.elements.upgradeChoices;
        if (!choicesDiv) return;
        
        choicesDiv.innerHTML = '';
        
        choices.forEach(upgrade => {
            const div = document.createElement('div');
            div.className = 'upgrade-option';
            div.textContent = upgrade.name;
            div.onclick = () => {
                onSelect(upgrade);
                Utils.hideElement('levelComplete');
            };
            choicesDiv.appendChild(div);
        });
    }
    
    updateMetaUpgrades(upgrades) {
        Utils.updateElement('totalSouls', upgrades.totalSouls);
        
        const upgradesList = this.elements.upgradesList;
        if (!upgradesList) return;
        
        upgradesList.innerHTML = '';
        
        const upgradeConfigs = [
            { 
                name: 'Starting Health', 
                key: 'startingHealth',
                ...CONFIG.META.UPGRADES.startingHealth
            },
            { 
                name: 'Starting Attack', 
                key: 'startingAttack',
                ...CONFIG.META.UPGRADES.startingAttack
            },
            { 
                name: 'Starting Defense', 
                key: 'startingDefense',
                ...CONFIG.META.UPGRADES.startingDefense
            },
            { 
                name: 'Experience Bonus', 
                key: 'expBonus',
                ...CONFIG.META.UPGRADES.expBonus
            },
            { 
                name: 'Gold Bonus', 
                key: 'goldBonus',
                ...CONFIG.META.UPGRADES.goldBonus
            }
        ];
        
        upgradeConfigs.forEach(upgrade => {
            const div = document.createElement('div');
            div.className = 'upgrade-option';
            
            const level = upgrades.upgrades[upgrade.key] || 0;
            const cost = upgrade.cost * (level + 1);
            const canAfford = upgrades.totalSouls >= cost;
            
            div.innerHTML = `
                <strong>${upgrade.name}</strong> - Level ${level}<br>
                <small>${upgrade.description}</small><br>
                Cost: ${cost} souls
            `;
            
            if (canAfford) {
                div.style.cursor = 'pointer';
                div.onclick = () => {
                    this.events.emit('meta:upgrade', { 
                        key: upgrade.key, 
                        cost: cost 
                    });
                };
            } else {
                div.style.opacity = '0.5';
                div.style.cursor = 'not-allowed';
            }
            
            upgradesList.appendChild(div);
        });
    }
    
    hideAllMenus() {
        Utils.hideElement('mainMenu');
        Utils.hideElement('gameOver');
        Utils.hideElement('levelComplete');
    }
    
    clearMessages() {
        if (this.elements.messages) {
            this.elements.messages.innerHTML = '';
        }
    }
}