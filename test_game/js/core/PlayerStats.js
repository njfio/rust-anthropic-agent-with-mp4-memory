// Player Statistics and Meta-Progression
// Handles player stats, leveling, and persistent upgrades

export class PlayerStats {
    constructor() {
        this.level = 1;
        this.exp = 0;
        this.expNeeded = 100;
        this.hp = 100;
        this.maxHp = 100;
        this.attack = 10;
        this.defense = 5;
        this.gold = 0;
        this.inventory = [];
        this.souls = 0;
    }

    reset() {
        this.level = 1;
        this.exp = 0;
        this.expNeeded = 100;
        this.hp = 100;
        this.maxHp = 100;
        this.attack = 10;
        this.defense = 5;
        this.gold = 0;
        this.inventory = [];
        this.souls = 0;
    }

    levelUp() {
        this.level++;
        this.exp -= this.expNeeded;
        this.expNeeded = Math.floor(this.expNeeded * 1.5);
        this.maxHp += 20;
        this.hp = this.maxHp;
        this.attack += 3;
        this.defense += 2;
        
        return {
            level: this.level,
            newMaxHp: this.maxHp,
            newAttack: this.attack,
            newDefense: this.defense
        };
    }

    gainExp(amount) {
        this.exp += amount;
        return this.exp >= this.expNeeded;
    }

    takeDamage(damage) {
        const actualDamage = Math.max(1, damage - this.defense);
        this.hp -= actualDamage;
        return {
            damage: actualDamage,
            isDead: this.hp <= 0
        };
    }

    heal(amount) {
        const healAmount = Math.min(amount, this.maxHp - this.hp);
        this.hp += healAmount;
        return healAmount;
    }

    addGold(amount) {
        this.gold += amount;
    }

    addSouls(amount) {
        this.souls += amount;
    }

    buffAttack(amount) {
        this.attack += amount;
    }

    buffDefense(amount) {
        this.defense += amount;
    }
}

export class MetaProgression {
    constructor() {
        this.totalSouls = parseInt(localStorage.getItem('totalSouls') || '0');
        this.upgrades = {
            startingHealth: parseInt(localStorage.getItem('startingHealth') || '0'),
            startingAttack: parseInt(localStorage.getItem('startingAttack') || '0'),
            startingDefense: parseInt(localStorage.getItem('startingDefense') || '0'),
            expBonus: parseInt(localStorage.getItem('expBonus') || '0'),
            goldBonus: parseInt(localStorage.getItem('goldBonus') || '0')
        };
    }

    addSouls(amount) {
        this.totalSouls += amount;
        this.save();
    }

    purchaseUpgrade(upgradeKey, cost) {
        if (this.totalSouls >= cost) {
            this.totalSouls -= cost;
            this.upgrades[upgradeKey]++;
            this.save();
            return true;
        }
        return false;
    }

    save() {
        localStorage.setItem('totalSouls', this.totalSouls.toString());
        Object.keys(this.upgrades).forEach(key => {
            localStorage.setItem(key, this.upgrades[key].toString());
        });
    }

    getUpgradeDefinitions() {
        return [
            { 
                name: 'Starting Health', 
                cost: 50, 
                level: this.upgrades.startingHealth,
                key: 'startingHealth',
                description: '+20 starting HP per level'
            },
            { 
                name: 'Starting Attack', 
                cost: 40, 
                level: this.upgrades.startingAttack,
                key: 'startingAttack',
                description: '+2 starting attack per level'
            },
            { 
                name: 'Starting Defense', 
                cost: 40, 
                level: this.upgrades.startingDefense,
                key: 'startingDefense',
                description: '+2 starting defense per level'
            },
            { 
                name: 'Experience Bonus', 
                cost: 60, 
                level: this.upgrades.expBonus,
                key: 'expBonus',
                description: '+10% exp gain per level'
            },
            { 
                name: 'Gold Bonus', 
                cost: 30, 
                level: this.upgrades.goldBonus,
                key: 'goldBonus',
                description: '+20% gold gain per level'
            }
        ];
    }
}

// Singleton instances
export const playerStats = new PlayerStats();
export const metaProgression = new MetaProgression();