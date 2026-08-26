# ✏️ AI Air Pencil

> 🖐️ Draw in the air. 🎨 Create without touching the screen.

**AI Air Pencil** is a gesture-based digital drawing application that allows users to draw in the air using a webcam, computer vision, and real-time hand tracking.

Instead of using a traditional mouse, touchscreen, or drawing tablet, AI Air Pencil tracks the user's hand and index finger to transform natural hand movements into digital artwork.

---

## 🌟 Features

### 🖐️ Real-Time Hand Tracking
Uses **MediaPipe Hand Landmarker** to detect and track hand movements through a webcam in real time.

### ✏️ Air Drawing
Draw naturally by moving your index finger through the air.

### 🎨 Multiple Colors
Choose from multiple drawing colors, including:

- 🔴 Red
- 🟢 Green
- 🔵 Blue
- 🟡 Yellow
- ⚪ White
- 🟣 Purple
- ⚫ Black

### 📏 Dynamic Brush Size
Use a **pinch gesture** between your thumb and index finger to dynamically change the brush size.

### 🧹 Eraser
Use the erase gesture to remove parts of your drawing.

### ↶ Undo & ↷ Redo
Move backward and forward through your drawing history without losing previous states.

### 🗑️ Clear Canvas
Clear the current drawing with a single command.

### 💾 Save Drawings
Save your artwork directly into the project's `drawings` directory.

### 🖼️ Saved Drawings Gallery
Saved drawings automatically appear in the website's **Saved Drawings** section.

### 🗑️ Delete Saved Drawings
Remove previously saved drawings directly from the web interface.

### 🌐 Web Interface
A modern Flask-powered web application provides:

- 🏠 Landing page
- 🚀 Launch interface
- 🎥 Live webcam stream
- 🎨 Drawing controls
- 🖼️ Saved Drawings gallery
- ✨ Feature sections
- 📖 How It Works section
- 🎯 Creative showcase

---

# 🖐️ Gesture Controls

| Gesture / Key | Action |
|---|---|
| ☝️ Index finger | Draw |
| ✌️ Two fingers | Select color |
| 🤏 Pinch | Change brush size |
| 🖐️ Four fingers | Erase |
| 👍 Thumbs up | Save drawing |
| `Z` | Undo |
| `Y` | Redo |
| `C` | Clear canvas |
| `S` | Save drawing |
| `Q` | Quit |

---

# 🛠️ Technology Stack

AI Air Pencil is built using the following technologies:

| Technology | Purpose |
|---|---|
| 🐍 Python | Core application logic |
| 🌐 Flask | Web server and API |
| 👁️ OpenCV | Webcam processing and image manipulation |
| 🖐️ MediaPipe | Real-time hand tracking |
| 🔢 NumPy | Image and canvas operations |
| 🧠 TensorFlow Lite / MediaPipe Model | Hand landmark detection |
| 🌎 HTML5 | Web interface structure |
| 🎨 CSS3 | Website styling |
| ⚡ JavaScript | Web interactions and commands |

---

# 📁 Project Structure

```text
AI-Air-Pencil/
│
├── website/
│   ├── index.html
│   └── style.css
│
├── drawings/
│   └── Saved drawing files
│
├── app.py
├── app_backup.py
├── main.py
├── hand_landmarker.task
├── requirements.txt
├── README.md
└── .gitignore
