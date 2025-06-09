export class InputSystem {
    constructor(events) {
        this.events = events;
        this.keys = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            attack: false
        };
        this.mouse = {
            x: 0,
            y: 0
        };
    }
    
    init() {
        this.setupKeyboardControls();
        this.setupMouseControls();
    }
    
    setupKeyboardControls() {
        document.addEventListener('keydown', (e) => {
            switch(e.key.toLowerCase()) {
                case 'w':
                case 'arrowup':
                    this.keys.forward = true;
                    break;
                case 's':
                case 'arrowdown':
                    this.keys.backward = true;
                    break;
                case 'a':
                case 'arrowleft':
                    this.keys.left = true;
                    break;
                case 'd':
                case 'arrowright':
                    this.keys.right = true;
                    break;
                case ' ':
                    if (!this.keys.attack) {
                        this.keys.attack = true;
                        this.events.emit('input:attack');
                    }
                    e.preventDefault();
                    break;
                case 'e':
                    this.events.emit('input:use');
                    break;
            }
            
            this.emitMovement();
        });
        
        document.addEventListener('keyup', (e) => {
            switch(e.key.toLowerCase()) {
                case 'w':
                case 'arrowup':
                    this.keys.forward = false;
                    break;
                case 's':
                case 'arrowdown':
                    this.keys.backward = false;
                    break;
                case 'a':
                case 'arrowleft':
                    this.keys.left = false;
                    break;
                case 'd':
                case 'arrowright':
                    this.keys.right = false;
                    break;
                case ' ':
                    this.keys.attack = false;
                    break;
            }
            
            this.emitMovement();
        });
    }
    
    setupMouseControls() {
        document.addEventListener('mousemove', (e) => {
            this.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
            this.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
            
            this.emitMovement();
        });
        
        // Optional: mouse click for attack
        document.addEventListener('click', (e) => {
            // Only if clicking on game canvas
            if (e.target.id === 'gameCanvas') {
                this.events.emit('input:attack');
            }
        });
    }
    
    emitMovement() {
        this.events.emit('input:move', {
            forward: this.keys.forward,
            backward: this.keys.backward,
            left: this.keys.left,
            right: this.keys.right,
            mouseX: this.mouse.x,
            mouseY: this.mouse.y
        });
    }
    
    reset() {
        this.keys = {
            forward: false,
            backward: false,
            left: false,
            right: false,
            attack: false
        };
        this.mouse = {
            x: 0,
            y: 0
        };
    }
}