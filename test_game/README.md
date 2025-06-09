# 3D Roguelike Dungeon Crawler

A procedurally generated 3D roguelike dungeon crawler built with Three.js featuring meta-progression, combat, and exploration.

## Features

- **Procedural Dungeon Generation**: Each floor is uniquely generated using Binary Space Partitioning
- **3D Graphics**: Full 3D environment with dynamic lighting and shadows
- **Combat System**: Real-time combat with melee attacks and enemy AI
- **Meta-Progression**: Earn souls to purchase permanent upgrades between runs
- **Item System**: Collect health potions, attack/defense buffs, and gold
- **Level Progression**: Gain experience, level up, and grow stronger
- **Multiple Enemy Types**: Basic enemies, elite enemies with different stats
- **Floor Progression**: Complete floors to advance deeper into the dungeon

## How to Play

### Running the Game

#### Option 1: Direct File Access
1. Open `index.html` in a modern web browser (Chrome, Firefox, Safari recommended)
2. The game will load automatically

#### Option 2: Using Local Server (Recommended)
1. Run the Python server:
   ```bash
   python3 server.py
   ```
2. The game will automatically open in your browser at http://localhost:8000
3. Press Ctrl+C to stop the server when done

Note: Using the local server avoids any potential CORS or local file access issues.

### Controls

- **WASD/Arrow Keys**: Move your character
- **Mouse**: Look around / rotate character
- **Spacebar**: Attack enemies in front of you
- **E**: Use items from inventory (when implemented)

### Gameplay

1. **Start**: Click "Start New Run" from the main menu
2. **Explore**: Navigate through procedurally generated dungeons
3. **Combat**: Attack enemies to gain experience and loot
4. **Items**: Walk over items to pick them up automatically
5. **Level Up**: Gain experience to increase your stats
6. **Exit**: Find the glowing cyan portal to advance to the next floor
7. **Upgrades**: Choose an upgrade between floors
8. **Death**: When you die, spend souls on permanent upgrades

### Meta-Progression

Between runs, spend souls on permanent upgrades:
- **Starting Health**: +20 HP per level
- **Starting Attack**: +2 attack per level
- **Starting Defense**: +2 defense per level
- **Experience Bonus**: +10% exp gain per level
- **Gold Bonus**: +20% gold gain per level

### Tips

- Enemies get stronger on deeper floors
- Use the environment to your advantage - enemies can't walk through walls
- Save health potions for when you really need them
- Focus on meta-upgrades that match your playstyle
- Elite enemies (purple) give more experience but are tougher

## Technical Details

- Built with Three.js for 3D rendering
- Uses Binary Space Partitioning for dungeon generation
- Local storage for meta-progression persistence
- No external dependencies except Three.js (loaded from CDN)

## Browser Compatibility

Works best in modern browsers with WebGL support:
- Chrome 60+
- Firefox 60+
- Safari 12+
- Edge 79+

## Development

The game consists of:
- `index.html`: Main HTML structure and UI
- `game.js`: Complete game logic and Three.js implementation
- All game state is managed in JavaScript
- Meta-progression saved to localStorage

Enjoy your dungeon crawling adventure!