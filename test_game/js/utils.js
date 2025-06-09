// Utility functions
export const Utils = {
    // Math utilities
    clamp: (value, min, max) => {
        return Math.min(Math.max(value, min), max);
    },
    
    randomRange: (min, max) => {
        return Math.random() * (max - min) + min;
    },
    
    randomInt: (min, max) => {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    },
    
    randomChoice: (array) => {
        return array[Math.floor(Math.random() * array.length)];
    },
    
    distance3D: (pos1, pos2) => {
        return pos1.distanceTo(pos2);
    },
    
    // Storage utilities
    saveToStorage: (key, value) => {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.error('Failed to save to storage:', e);
            return false;
        }
    },
    
    loadFromStorage: (key, defaultValue = null) => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('Failed to load from storage:', e);
            return defaultValue;
        }
    },
    
    // Array utilities
    removeFromArray: (array, item) => {
        const index = array.indexOf(item);
        if (index > -1) {
            array.splice(index, 1);
        }
        return array;
    },
    
    // DOM utilities
    getElementById: (id) => {
        const element = document.getElementById(id);
        if (!element) {
            console.warn(`Element with id '${id}' not found`);
        }
        return element;
    },
    
    updateElement: (id, value) => {
        const element = Utils.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    },
    
    setElementHTML: (id, html) => {
        const element = Utils.getElementById(id);
        if (element) {
            element.innerHTML = html;
        }
    },
    
    showElement: (id) => {
        const element = Utils.getElementById(id);
        if (element) {
            element.style.display = 'block';
        }
    },
    
    hideElement: (id) => {
        const element = Utils.getElementById(id);
        if (element) {
            element.style.display = 'none';
        }
    },
    
    // Validation utilities
    isValidPosition: (position, bounds) => {
        return position.x >= bounds.minX && 
               position.x <= bounds.maxX && 
               position.z >= bounds.minZ && 
               position.z <= bounds.maxZ;
    },
    
    // Error handling
    tryExecute: (fn, errorMessage = 'An error occurred') => {
        try {
            return fn();
        } catch (error) {
            console.error(errorMessage, error);
            return null;
        }
    }
};

// Event system for decoupling
export class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }
    
    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }
    
    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(callback => {
                Utils.tryExecute(() => callback(data), `Error in event handler for ${event}`);
            });
        }
    }
    
    clear() {
        this.events = {};
    }
}

// Object pooling for performance
export class ObjectPool {
    constructor(createFn, resetFn, maxSize = 100) {
        this.createFn = createFn;
        this.resetFn = resetFn;
        this.maxSize = maxSize;
        this.pool = [];
        this.active = [];
    }
    
    acquire() {
        let obj;
        if (this.pool.length > 0) {
            obj = this.pool.pop();
        } else {
            obj = this.createFn();
        }
        this.active.push(obj);
        return obj;
    }
    
    release(obj) {
        const index = this.active.indexOf(obj);
        if (index !== -1) {
            this.active.splice(index, 1);
            if (this.pool.length < this.maxSize) {
                this.resetFn(obj);
                this.pool.push(obj);
            }
        }
    }
    
    releaseAll() {
        while (this.active.length > 0) {
            this.release(this.active[0]);
        }
    }
    
    clear() {
        this.pool = [];
        this.active = [];
    }
}