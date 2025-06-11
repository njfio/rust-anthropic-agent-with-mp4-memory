// Game Configuration
export const CONFIG = {
    // Dungeon settings
    DUNGEON: {
        SIZE: 20,
        CELL_SIZE: 4,
        WALL_HEIGHT: 5,
        BSP_DEPTH: 3,
        MIN_ROOM_SIZE: 6
    },
    
    // Player settings
    PLAYER: {
        BASE_HP: 100,
        BASE_ATTACK: 10,
        BASE_DEFENSE: 5,
        BASE_SPEED: 0.1,
        ROTATION_SPEED: 0.05,
        ATTACK_RANGE: 3,
        ATTACK_ANGLE: Math.PI / 4,
        LEVEL_HP_BONUS: 20,
        LEVEL_ATTACK_BONUS: 3,
        LEVEL_DEFENSE_BONUS: 2
    },
    
    // Enemy settings
    ENEMY: {
        TYPES: {
            basic: { color: 0xff0000, hp: 30, attack: 5, exp: 20, speed: 0.02 },
            elite: { color: 0xff00ff, hp: 60, attack: 10, exp: 50, speed: 0.03 },
            boss: { color: 0xff8800, hp: 100, attack: 15, exp: 100, speed: 0.025 }
        },
        AGGRO_RANGE: 15,
        ATTACK_RANGE: 2,
        ATTACK_COOLDOWN: 1000,
        MAX_PER_ROOM: 3
    },
    
    // Item settings
    ITEM: {
        TYPES: {
            health: { color: 0x00ff00, effect: 'heal', value: 30 },
            attack: { color: 0xff0000, effect: 'buff_attack', value: 5 },
            defense: { color: 0x0000ff, effect: 'buff_defense', value: 3 },
            gold: { color: 0xffff00, effect: 'gold', value: 50 }
        },
        PICKUP_RANGE: 2,
        DROP_CHANCE: 0.3,
        MAX_PER_ROOM: 2
    },
    
    // Meta progression
    META: {
        UPGRADES: {
            startingHealth: { cost: 50, value: 20, description: '+20 starting HP per level' },
            startingAttack: { cost: 40, value: 2, description: '+2 starting attack per level' },
            startingDefense: { cost: 40, value: 2, description: '+2 starting defense per level' },
            expBonus: { cost: 60, value: 0.1, description: '+10% exp gain per level' },
            goldBonus: { cost: 30, value: 0.2, description: '+20% gold gain per level' }
        }
    },
    
    // UI settings
    UI: {
        MESSAGE_LIMIT: 5,
        MESSAGE_TYPES: {
            DAMAGE: 'damage',
            HEAL: 'heal',
            ITEM: 'item',
            LEVEL: 'level'
        }
    },
    
    // Rendering settings
    RENDER: {
        FOG_NEAR: 10,
        FOG_FAR: 50,
        CAMERA_FOV: 75,
        CAMERA_NEAR: 0.1,
        CAMERA_FAR: 1000,
        CAMERA_DISTANCE: 10,
        CAMERA_HEIGHT: 8,
        SHADOW_MAP_SIZE: 2048
    }
};

// Color palette
export const COLORS = {
    PLAYER: 0x00ff00,
    FLOOR: 0x444444,
    WALL: 0x666666,
    EXIT: 0x00ffff,
    SWORD: 0xcccccc,
    AMBIENT_LIGHT: 0x404040,
    DIRECTIONAL_LIGHT: 0xffffff
};

// Game states
export const GAME_STATES = {
    MENU: 'menu',
    PLAYING: 'playing',
    PAUSED: 'paused',
    GAME_OVER: 'game_over',
    LEVEL_COMPLETE: 'level_complete'
};

// Storage keys
export const STORAGE_KEYS = {
    TOTAL_SOULS: 'totalSouls',
    UPGRADES: {
        STARTING_HEALTH: 'startingHealth',
        STARTING_ATTACK: 'startingAttack',
        STARTING_DEFENSE: 'startingDefense',
        EXP_BONUS: 'expBonus',
        GOLD_BONUS: 'goldBonus'
    }
};