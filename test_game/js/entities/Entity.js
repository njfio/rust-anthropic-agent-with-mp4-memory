// Base Entity class
export class Entity {
    constructor(x, y, z) {
        this.position = new THREE.Vector3(x, y, z);
        this.mesh = null;
        this.velocity = new THREE.Vector3();
        this.health = 100;
        this.maxHealth = 100;
        this.alive = true;
    }
    
    update() {
        // Override in subclasses
    }
    
    takeDamage(amount) {
        this.health -= amount;
        if (this.health <= 0) {
            this.health = 0;
            this.alive = false;
        }
    }
    
    heal(amount) {
        this.health = Math.min(this.health + amount, this.maxHealth);
    }
    
    isDead() {
        return !this.alive;
    }
    
    destroy() {
        if (this.mesh) {
            if (this.mesh.parent) {
                this.mesh.parent.remove(this.mesh);
            }
            // Dispose of geometry and material
            if (this.mesh.geometry) {
                this.mesh.geometry.dispose();
            }
            if (this.mesh.material) {
                if (Array.isArray(this.mesh.material)) {
                    this.mesh.material.forEach(material => material.dispose());
                } else {
                    this.mesh.material.dispose();
                }
            }
        }
    }
    
    checkCollision(position, walls) {
        const radius = 0.5;
        
        for (const wall of walls) {
            const wallBox = new THREE.Box3().setFromObject(wall);
            const entitySphere = new THREE.Sphere(position, radius);
            
            if (wallBox.intersectsSphere(entitySphere)) {
                return true;
            }
        }
        return false;
    }
}