AI Air Pencil

AI Air Pencil is a gesture-based digital drawing application that lets
you draw in the air using a webcam, computer vision, and real-time hand
tracking.

Instead of using a physical mouse or touchscreen, the application
interprets hand gestures and turns your index-finger movement into
digital artwork.

✨ Features

🖐️ Real-time hand tracking

✏️ Air drawing using the index finger

🎨 Multiple drawing colors, including black

📏 Dynamic brush sizing using a pinch gesture

🧹 Eraser gesture

↶ Undo

↷ Redo

🗑️ Clear canvas

💾 Save drawings

🖼️ Saved Drawings section on the website

🗑️ Delete saved drawings

🌐 Flask-powered web interface

🎥 Live webcam feed

🖥️ Modern responsive website interface

🖐️ Gesture Controls

Gesture / Key          Action

1                      Draw
2                      Change color
Pinch                  Change brush size
4                      Erase
Thumb / Save gesture   Save drawing
Z                      Undo
Y                      Redo
C                      Clear
S                      Save
Q                      Quit

🛠️ Technology Stack

Python

Flask

OpenCV

MediaPipe

NumPy

HTML5

CSS3

JavaScript

TensorFlow Lite / MediaPipe hand-landmark model

📁 Project Structure

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

🚀 Run Locally

1. Clone the repository

git clone https://github.com/10AbdulRafay/AI-Air-Pencil.git
cd AI-Air-Pencil

2. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Start the application

python app.py

Then open:

http://127.0.0.1:5000

📷 Camera Access

AI Air Pencil requires access to a webcam for hand tracking.

When running the application locally, make sure:

A working webcam is connected.

Your browser has permission to use the camera where applicable.

Your hand is visible to the camera.

The application is running before launching the AI Air Pencil
interface.

💾 Saved Drawings

When a drawing is saved, the Flask application stores the drawing in the
project's drawings directory.

The website retrieves saved drawings through the application's drawings
API and displays them in the Saved Drawings section.

This means the website interface does not depend on manually placing
images into the HTML gallery.

For a multi-user production deployment, persistent storage and
user-specific storage should be added so each user's drawings remain
separate and survive server restarts or redeployments.

🧠 How It Works

The application follows this general flow:

Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Tracking
   ↓
Hand Landmarks
   ↓
Gesture Detection
   ↓
Drawing / Color / Eraser / Save Commands
   ↓
Digital Canvas
   ↓
Flask Web Interface

The browser communicates with the Python backend through Flask
endpoints. The backend provides the live camera stream, processes
commands, and exposes saved drawings to the website.

↶ Undo & Redo

AI Air Pencil maintains drawing history so users can move backward and
forward through drawing states.

Undo and redo restore the complete drawing state, allowing strokes to be
removed and restored without corrupting the drawing mask.

🎨 Drawing Experience

The application supports a range of drawing colors and uses a sharp
pencil-style stroke. Brush size can also be adjusted through the gesture
controls.

🌐 Web Application

The web interface contains:

Project landing page

AI Air Pencil launch button

Live application preview

Drawing controls

Saved Drawings section

Feature section

How It Works section

Creative gallery

Project information

⚠️ Deployment Note

The current project is designed and tested as a local Flask application
using a webcam and computer-vision processing.

Before deploying publicly, the application should be configured for a
production WSGI server and a hosting environment capable of supporting
the required Python computer-vision dependencies.

For a true multi-user deployment, additional work is recommended for:

User-specific saved drawings

Persistent storage

Secure file handling

Production server configuration

HTTPS

Browser camera permissions

Resource management for concurrent users

🔮 Future Improvements

Possible future improvements include:

User accounts

Cloud-based drawing storage

Per-user galleries

Download/share drawing buttons

More gesture controls

More brush styles

Drawing background options

Real-time collaborative drawing

Mobile/tablet support

Production cloud deployment

👨‍💻 Developer

Abdul Rafay
AI/ML Engineer

📜 License

This project does not currently specify an open-source license.
