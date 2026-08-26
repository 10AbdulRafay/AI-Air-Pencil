from flask import (
    Flask,
    send_from_directory,
    jsonify,
    Response,
    request
)

from pathlib import Path
from werkzeug.utils import secure_filename

import threading
import main


# ============================================================
# AI AIR PENCIL WEB SERVER
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WEBSITE_DIR = BASE_DIR / "website"

DRAWINGS_DIR = BASE_DIR / "drawings"

DRAWINGS_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# STATE
# ============================================================

camera_running = False

camera_lock = threading.Lock()


# ============================================================
# WEBSITE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        WEBSITE_DIR,
        "index.html"
    )


@app.route("/style.css")
def css():

    return send_from_directory(
        WEBSITE_DIR,
        "style.css"
    )


# ============================================================
# LAUNCH
# ============================================================

@app.route(
    "/launch",
    methods=["POST"]
)
def launch():

    global camera_running

    with camera_lock:

        camera_running = True

    return jsonify({
        "success": True,
        "message": "AI Air Pencil started successfully!"
    })


# ============================================================
# VIDEO STREAM
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        main.generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# WEB COMMANDS
# ============================================================

@app.route(
    "/command",
    methods=["POST"]
)
def command():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        command_name = data.get(
            "command",
            ""
        )

        if not command_name:

            return jsonify({
                "success": False,
                "message": "No command received."
            }), 400


        # ----------------------------------------------------
        # Allowed commands
        # ----------------------------------------------------

        allowed_commands = {
            "undo",
            "redo",
            "clear",
            "save",
            "quit"
        }


        # ----------------------------------------------------
        # Color command
        # ----------------------------------------------------

        if command_name.startswith(
            "color:"
        ):

            main.send_command(
                command_name
            )

            return jsonify({
                "success": True
            })


        # ----------------------------------------------------
        # Normal command
        # ----------------------------------------------------

        if command_name not in allowed_commands:

            return jsonify({
                "success": False,
                "message": "Invalid command."
            }), 400


        main.send_command(
            command_name
        )


        return jsonify({
            "success": True,
            "command": command_name
        })


    except Exception as error:

        print(
            "COMMAND ERROR:",
            error
        )

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# ============================================================
# SAVED DRAWINGS API
# ============================================================

@app.route("/api/drawings")
def get_drawings():

    try:

        drawings = []


        # ----------------------------------------------------
        # Find PNG drawings
        # ----------------------------------------------------

        files = sorted(
            DRAWINGS_DIR.glob("*.png"),
            key=lambda file: file.stat().st_mtime,
            reverse=True
        )


        for file in files:

            drawings.append({

                "filename": file.name,

                "url": (
                    "/drawings/"
                    + secure_filename(file.name)
                ),

                "created": file.stat().st_mtime

            })


        return jsonify({
            "success": True,
            "drawings": drawings
        })


    except Exception as error:

        print(
            "DRAWINGS ERROR:",
            error
        )

        return jsonify({
            "success": False,
            "drawings": [],
            "message": str(error)
        }), 500


# ============================================================
# SERVE SAVED DRAWINGS
# ============================================================

@app.route(
    "/drawings/<path:filename>"
)
def drawing_file(filename):

    safe_filename = secure_filename(
        filename
    )

    return send_from_directory(
        DRAWINGS_DIR,
        safe_filename
    )


# ============================================================
# DELETE DRAWING
# ============================================================

@app.route(
    "/api/drawings/<path:filename>",
    methods=["DELETE"]
)
def delete_drawing(filename):

    try:

        safe_filename = secure_filename(
            filename
        )

        file_path = (
            DRAWINGS_DIR /
            safe_filename
        )


        if not file_path.exists():

            return jsonify({
                "success": False,
                "message": "Drawing not found."
            }), 404


        file_path.unlink()


        return jsonify({
            "success": True,
            "message": "Drawing deleted."
        })


    except Exception as error:

        print(
            "DELETE ERROR:",
            error
        )

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("       AI AIR PENCIL WEB APP")
    print("======================================")
    print()

    print("Open this in your browser:")
    print("http://127.0.0.1:5000")

    print()
    print("Press CTRL+C to stop the server.")
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )