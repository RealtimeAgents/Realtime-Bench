# Gomoku (五子棋)

A polished, modern take on the classic **five-in-a-row** board game, built as a single
self-contained HTML file — no build step, no dependencies. Just open it in a browser and play.

![Gomoku](https://img.shields.io/badge/platform-web-blue) ![NoDeps](https://img.shields.io/badge/dependencies-0-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

### 🎮 Play Now

👉 **[https://mogoku.netlify.app/](https://mogoku.netlify.app/)**

No install required — open the link and start playing right away in any modern browser (desktop or mobile).

---

## ✨ Features

| | |
|---|---|
| 🎯 **Classic board** | 15×15 grid with star points (天元) and traditional styling |
| 🎮 **Two modes** | Player vs. Player (PVP) and Player vs. AI (PVE) |
| 🤖 **Adjustable AI** | Three difficulty levels using heuristic scoring + minimax |
| 🧠 **Smart opponent** | Offensive/defensive pattern evaluation with alpha-beta pruning on *Hard* |
| ⛔ **Renju rules** | Optional forbidden moves for Black (double-three, double-four, overline) |
| ↶ **Undo** | Take back moves (in PVE mode undoes both your move and the AI's reply) |
| 🔄 **Restart** | Reset the board at any time |
| 📱 **Responsive** | Mouse and touch input; adapts to any screen size |
| 🎨 **Modern visuals** | Wooden board with gradient stones, last-move marker, winning-line highlight |

---

## 🚀 Quick Start

This is a zero-setup project. Pick any method:

### Option 1 — Just open it
Double-click `index.html` to open it directly in your default browser.

### Option 2 — Local server (recommended)
For best behavior (especially on mobile), serve it over HTTP:

```bash
# Python 3
python -m http.server 8000
# then visit http://localhost:8000

# Node.js
npx serve .

# PHP
php -S localhost:8000
```

### Option 3 — Deploy
Drop `index.html` onto any static host: GitHub Pages, Netlify, Vercel, Cloudflare Pages, etc.

---

## 🎯 How to Play

1. **Black moves first**, then players alternate.
2. Click (or tap) on any intersection to place a stone.
3. The first player to line up **five stones** in a row — horizontally, vertically, or diagonally — wins.
4. The winning line is highlighted in red.

### Controls
- **Place stone** — click/tap an empty intersection
- **Undo** — take back the last move(s)
- **Restart** — clear the board and start over
- **Game Mode** — switch between 👥 PVP and 🤖 vs AI
- **AI Difficulty** — Easy / Normal / Hard (PVE only)
- **AI Plays** — choose whether the AI controls ⚫ Black or ⚪ White
- **Forbidden moves** — toggle Renju restrictions on Black

---

## 🤖 AI Details

The AI uses a combination of techniques, scaled by difficulty:

| Level | Strategy |
|-------|----------|
| **Easy** | Greedy heuristic scoring with randomized tie-breaking among top moves |
| **Normal** | 1-ply search combining offensive gain and defensive blocking; detects and blocks immediate threats (fours, open-fours) |
| **Hard** | Depth-2 **minimax with alpha-beta pruning** over the top-ranked candidate moves, plus immediate-win/threat detection |

Candidate moves are restricted to cells near existing stones (within radius 2), keeping
the search tractable. Stones are scored using pattern recognition across four directions
(open four, four, open three, split three, two, etc.).

---

## ⛔ Forbidden Moves (Renju)

When enabled, this rule set (traditionally applied to **Black only**, since Black has the
first-move advantage) prohibits:

- **Overline** — six or more in a row (a line of *exactly* five still wins)
- **Double-three (3-3)** — creating two open threes simultaneously
- **Double-four (4-4)** — creating two fours simultaneously

A move that forms exactly five always overrides the forbidden rule and wins immediately.
Violations are flagged in the status panel.

---

## 🛠 Tech Stack

- **HTML5 Canvas** for board rendering (with high-DPI support)
- **Vanilla JavaScript** (ES6+) for all game logic and AI — no frameworks
- **CSS3** with custom properties, gradients, and a responsive grid layout
- **Zero external dependencies** — everything lives in one ~30 KB file

---

## 📁 Project Structure

```
gomoku/
├── index.html      # The entire game (HTML + CSS + JS)
└── README.md       # This file
```

---

## 🧩 Customization

All tuning lives at the top of the `<script>` block and in the `:root` CSS variables:

- `SIZE` — change board dimensions (e.g. `19` for a Go-sized board)
- `--wood`, `--accent`, etc. — recolor the board and UI
- `PATTERNS` array — adjust AI pattern weights
- `aiDifficulty` search depth — extend for a stronger (slower) AI

---

## 📜 License

MIT © 2026. Free to use, modify, and distribute.

---

> *Built as a single-file showcase of classic game logic with a modern UI.*
