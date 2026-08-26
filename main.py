# ============================================================
# AI AIR PENCIL - WEB VERSION
# ============================================================
# This file is based on the original AI Air Pencil code.
# The original gesture, drawing, eraser, color, brush-size,
# save, undo/redo, clear, notification and UI rendering logic
# is preserved.
#
# The desktop cv2.imshow()/keyboard/mouse loop is replaced by
# a browser MJPEG stream and web commands.
# ============================================================

import queue
import threading


import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time
from pathlib import Path


# ============================================================
# AI AIR PENCIL
# Modern Gesture-Based Drawing Application
# ============================================================


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(PROJECT_DIR / "hand_landmarker.task")

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55
)


# ============================================================
# COLORS - BGR
# ============================================================

BG = (13, 17, 23)
PANEL = (22, 28, 36)
PANEL_LIGHT = (31, 39, 49)
PANEL_HOVER = (40, 50, 62)
BORDER = (48, 58, 70)

WHITE = (245, 247, 250)
MUTED = (145, 155, 168)

CYAN = (255, 210, 0)
GREEN = (90, 230, 120)
RED = (90, 90, 255)
ORANGE = (50, 170, 255)
PURPLE = (210, 100, 220)


# ============================================================
# COLOR PALETTE
# ============================================================

COLORS = [
    ("RED", (0, 0, 255)),
    ("GREEN", (0, 255, 0)),
    ("BLUE", (255, 0, 0)),
    ("YELLOW", (0, 255, 255)),
    ("WHITE", (255, 255, 255)),
    ("PURPLE", (255, 0, 180)),
    ("BLACK", (0, 0, 0)),
]


# ============================================================
# GLOBAL UI STATE
# ============================================================

mouse_x = -1
mouse_y = -1
mouse_clicked = False

hover_button = None


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global mouse_x
    global mouse_y
    global mouse_clicked

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True


# ============================================================
# FINGER DETECTION
# ============================================================

def fingers_up(hand):

    return [
        1 if hand[8].y < hand[6].y else 0,
        1 if hand[12].y < hand[10].y else 0,
        1 if hand[16].y < hand[14].y else 0,
        1 if hand[20].y < hand[18].y else 0
    ]


# ============================================================
# THUMBS UP
# ============================================================

def is_thumbs_up(hand):

    thumb_up = hand[4].y < hand[3].y

    index_down = hand[8].y > hand[6].y
    middle_down = hand[12].y > hand[10].y
    ring_down = hand[16].y > hand[14].y
    pinky_down = hand[20].y > hand[18].y

    return (
        thumb_up
        and index_down
        and middle_down
        and ring_down
        and pinky_down
    )


# ============================================================
# DISTANCE
# ============================================================

def distance(point1, point2, width, height):

    x1 = point1.x * width
    y1 = point1.y * height

    x2 = point2.x * width
    y2 = point2.y * height

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# ============================================================
# ROUNDED RECTANGLE
# ============================================================

def rounded_rectangle(
    image,
    top_left,
    bottom_right,
    color,
    radius=15
):

    x1, y1 = top_left
    x2, y2 = bottom_right

    radius = min(
        radius,
        abs(x2 - x1) // 2,
        abs(y2 - y1) // 2
    )

    cv2.rectangle(
        image,
        (x1 + radius, y1),
        (x2 - radius, y2),
        color,
        -1
    )

    cv2.rectangle(
        image,
        (x1, y1 + radius),
        (x2, y2 - radius),
        color,
        -1
    )

    cv2.circle(
        image,
        (x1 + radius, y1 + radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        image,
        (x2 - radius, y1 + radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        image,
        (x1 + radius, y2 - radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        image,
        (x2 - radius, y2 - radius),
        radius,
        color,
        -1
    )


# ============================================================
# TEXT
# ============================================================

def text(
    image,
    value,
    position,
    size=0.5,
    color=WHITE,
    thickness=1
):

    cv2.putText(
        image,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# SAVE DRAWING
# ============================================================

def save_drawing(
    canvas,
    drawings_folder,
    drawing_number
):

    filename = os.path.join(
        drawings_folder,
        f"drawing_{drawing_number:03d}.png"
    )

    success = cv2.imwrite(
        filename,
        canvas
    )

    return success, filename


# ============================================================
# HISTORY
# ============================================================

def push_undo(
    undo_stack,
    redo_stack,
    canvas,
    max_history
):

    undo_stack.append(
        canvas.copy()
    )

    if len(undo_stack) > max_history:

        undo_stack.pop(0)

    redo_stack.clear()


# ============================================================
# HIT TEST
# ============================================================

def inside_rect(
    x,
    y,
    x1,
    y1,
    x2,
    y2
):

    return (
        x >= x1
        and x <= x2
        and y >= y1
        and y <= y2
    )


# ============================================================
# DRAW TOP BAR
# ============================================================

def draw_top_bar(
    output,
    current_color,
    brush_size,
    gesture,
    colors
):

    height, width = output.shape[:2]

    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 88),
        BG,
        -1
    )

    output = cv2.addWeighted(
        overlay,
        0.94,
        output,
        0.06,
        0
    )

    # --------------------------------------------------------
    # Logo
    # --------------------------------------------------------

    cv2.circle(
        output,
        (30, 30),
        9,
        CYAN,
        -1
    )

    cv2.circle(
        output,
        (30, 30),
        4,
        WHITE,
        -1
    )

    text(
        output,
        "AIR PENCIL",
        (48, 32),
        0.68,
        WHITE,
        2
    )

    text(
        output,
        "AI GESTURE CANVAS",
        (49, 54),
        0.32,
        MUTED,
        1
    )

    # --------------------------------------------------------
    # Gesture status
    # --------------------------------------------------------

    gesture_names = {
        "DRAW": "DRAWING",
        "SELECT": "COLOR",
        "SIZE": "BRUSH SIZE",
        "ERASE": "ERASING",
        "SAVE": "SAVING",
        "IDLE": "READY",
        "UNKNOWN": "READY",
        "NO HAND": "NO HAND"
    }

    gesture_text = gesture_names.get(
        gesture,
        gesture
    )

    gesture_colors = {
        "DRAW": GREEN,
        "SELECT": CYAN,
        "SIZE": ORANGE,
        "ERASE": RED,
        "SAVE": GREEN,
        "NO HAND": MUTED
    }

    status_color = gesture_colors.get(
        gesture,
        MUTED
    )

    status_x = width // 2 - 85

    rounded_rectangle(
        output,
        (status_x, 17),
        (status_x + 170, 63),
        PANEL,
        12
    )

    cv2.circle(
        output,
        (status_x + 20, 40),
        5,
        status_color,
        -1
    )

    text(
        output,
        gesture_text,
        (status_x + 34, 45),
        0.42,
        WHITE,
        1
    )

    # --------------------------------------------------------
    # Brush
    # --------------------------------------------------------

    brush_x = width - 240

    rounded_rectangle(
        output,
        (brush_x, 13),
        (brush_x + 125, 68),
        PANEL,
        13
    )

    text(
        output,
        "BRUSH",
        (brush_x + 14, 34),
        0.30,
        MUTED,
        1
    )

    text(
        output,
        f"{brush_size}px",
        (brush_x + 14, 56),
        0.55,
        WHITE,
        2
    )

    preview_radius = max(
        3,
        min(18, brush_size // 2)
    )

    cv2.circle(
        output,
        (brush_x + 100, 42),
        preview_radius,
        current_color,
        -1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Live
    # --------------------------------------------------------

    live_x = width - 90

    cv2.circle(
        output,
        (live_x, 28),
        5,
        GREEN,
        -1
    )

    text(
        output,
        "LIVE",
        (live_x + 12, 33),
        0.38,
        WHITE,
        1
    )

    # --------------------------------------------------------
    # Color palette
    # FIXED: palette coordinates now match selection logic
    # --------------------------------------------------------

# --------------------------------------------------------
# LARGE COLOR PALETTE
# --------------------------------------------------------

    # --------------------------------------------------------
    # LARGE COLOR PALETTE
    # --------------------------------------------------------

    palette_y = 72
    palette_start_x = 230
    palette_spacing = 42
    palette_radius = 17

    for i, (_, color) in enumerate(colors):

        x = palette_start_x + i * palette_spacing

        # ----------------------------------------------------
        # Hover / selection area
        # ----------------------------------------------------

        if (
            abs(mouse_x - x) <= 24
            and abs(mouse_y - palette_y) <= 24
        ):

            cv2.circle(
                output,
                (x, palette_y),
                10,
                color,
                -1,
                cv2.LINE_AA
            )

            # Give dark colors a visible border
            if color == (0, 0, 0):
                cv2.circle(
                    output,
                    (x, palette_y),
                    11,
                    WHITE,
                    1,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # Outer dark ring
        # ----------------------------------------------------

        cv2.circle(
            output,
            (x, palette_y),
            palette_radius + 3,
            (10, 14, 20),
            -1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Color
        # ----------------------------------------------------

        cv2.circle(
            output,
            (x, palette_y),
            palette_radius,
            color,
            -1,
            cv2.LINE_AA
        )

        # Keep BLACK visible against the dark toolbar.
        if color == (0, 0, 0):
            cv2.circle(
                output,
                (x, palette_y),
                palette_radius,
                WHITE,
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Selected color
        # ----------------------------------------------------

        if color == current_color:

            cv2.circle(
                output,
                (x, palette_y),
                palette_radius + 5,
                WHITE,
                2,
                cv2.LINE_AA
            )
    return output

# ============================================================
# DRAW BOTTOM DOCK
# ============================================================

def draw_bottom_dock(
    output,
    undo_stack,
    redo_stack
):

    height, width = output.shape[:2]

    dock_width = min(
        850,
        width - 40
    )

    dock_height = 100

    x1 = (width - dock_width) // 2
    y1 = height - dock_height - 28

    x2 = x1 + dock_width
    y2 = y1 + dock_height

    # --------------------------------------------------------
    # Shadow
    # --------------------------------------------------------

    shadow = output.copy()

    rounded_rectangle(
        shadow,
        (x1 + 5, y1 + 7),
        (x2 + 5, y2 + 7),
        (0, 0, 0),
        20
    )

    output = cv2.addWeighted(
        shadow,
        0.35,
        output,
        0.65,
        0
    )

    # --------------------------------------------------------
    # Dock
    # --------------------------------------------------------

    rounded_rectangle(
        output,
        (x1, y1),
        (x2, y2),
        PANEL,
        20
    )

    # --------------------------------------------------------
    # Section title
    # --------------------------------------------------------

    text(
        output,
        "TOOLS",
        (x1 + 20, y1 + 25),
        0.30,
        MUTED,
        1
    )

    # --------------------------------------------------------
    # Buttons
    # --------------------------------------------------------

    buttons = [
        ("UNDO", "Z", len(undo_stack) > 0),
        ("REDO", "Y", len(redo_stack) > 0),
        ("CLEAR", "C", True),
        ("SAVE", "S", True),
        ("QUIT", "Q", True)
    ]

    start_x = x1 + 20

    for i, (
        name,
        shortcut,
        enabled
    ) in enumerate(buttons):

        bx = start_x + i * 105

        by = y1 + 38

        bx2 = bx + 88
        by2 = y1 + 79

        hovered = inside_rect(
            mouse_x,
            mouse_y,
            bx,
            by,
            bx2,
            by2
        )

        if not enabled:

            color = (24, 28, 34)

        elif hovered:

            color = PANEL_HOVER

        else:

            color = PANEL_LIGHT

        rounded_rectangle(
            output,
            (bx, by),
            (bx2, by2),
            color,
            10
        )

        # Border on hover
        if hovered and enabled:

            cv2.rectangle(
                output,
                (bx, by),
                (bx2, by2),
                BORDER,
                1,
                cv2.LINE_AA
            )

        label_color = (
            WHITE
            if enabled
            else MUTED
        )

        text(
            output,
            name,
            (bx + 10, y1 + 57),
            0.32,
            label_color,
            1
        )

        text(
            output,
            shortcut,
            (bx + 39, y1 + 72),
            0.27,
            MUTED,
            1
        )

    # --------------------------------------------------------
    # Gesture guide
    # --------------------------------------------------------

    guide_x = x1 + 560

    text(
        output,
        "GESTURES",
        (guide_x, y1 + 25),
        0.30,
        MUTED,
        1
    )

    text(
        output,
        "1 Draw   2 Color   Pinch Size",
        (guide_x, y1 + 49),
        0.30,
        WHITE,
        1
    )

    text(
        output,
        "4 Erase   Thumb Save",
        (guide_x, y1 + 70),
        0.30,
        WHITE,
        1
    )

    return output


# ============================================================
# DRAW SAVE NOTIFICATION
# ============================================================

def draw_notification(
    output,
    message,
    timer
):

    if timer <= 0:
        return output

    height, width = output.shape[:2]

    box_width = 220

    x1 = (width - box_width) // 2
    y1 = 105

    # Fade toward the end
    alpha = min(
        1.0,
        timer / 15.0
    )

    notification = output.copy()

    rounded_rectangle(
        notification,
        (x1, y1),
        (x1 + box_width, y1 + 48),
        (25, 62, 40),
        12
    )

    cv2.circle(
        notification,
        (x1 + 25, y1 + 24),
        7,
        GREEN,
        -1
    )

    text(
        notification,
        message,
        (x1 + 43, y1 + 30),
        0.42,
        WHITE,
        2
    )

    output = cv2.addWeighted(
        notification,
        alpha,
        output,
        1.0 - alpha,
        0
    )

    return output


# ============================================================
# DRAW HAND CURSOR
# ============================================================

def draw_cursor(
    frame,
    x,
    y,
    color,
    radius
):

    cv2.circle(
        frame,
        (x, y),
        radius,
        color,
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (x, y),
        3,
        color,
        -1,
        cv2.LINE_AA
    )


# ============================================================
# DRAW ERASER CURSOR
# ============================================================

def draw_eraser_cursor(
    frame,
    x,
    y,
    radius
):

    cv2.circle(
        frame,
        (x, y),
        radius,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (x, y),
        max(2, radius - 4),
        (60, 60, 60),
        1,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x - 7, y),
        (x + 7, y),
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x, y - 7),
        (x, y + 7),
        WHITE,
        1,
        cv2.LINE_AA
    )


# ============================================================
# PROJECT FOLDERS
# ============================================================

project_folder = os.path.dirname(
    os.path.abspath(__file__)
)

drawings_folder = os.path.join(
    project_folder,
    "drawings"
)

os.makedirs(
    drawings_folder,
    exist_ok=True
)


# ============================================================
# DRAWING NUMBER
# ============================================================

drawing_number = 1

while os.path.exists(
    os.path.join(
        drawings_folder,
        f"drawing_{drawing_number:03d}.png"
    )
):

    drawing_number += 1


# ============================================================
# HISTORY
# ============================================================

undo_stack = []
redo_stack = []

max_history = 20


# ============================================================


# ============================================================
# WEB COMMAND STATE
# ============================================================

command_queue = queue.Queue()
stop_event = threading.Event()


def send_command(command):
    command_queue.put(command)


def reset_web_state():
    stop_event.clear()

    while not command_queue.empty():
        try:
            command_queue.get_nowait()
        except queue.Empty:
            break


# ============================================================
# BROWSER FRAME GENERATOR
# ============================================================

def generate_frames():

    reset_web_state()
    global drawing_number

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

    
        # --------------------------------------------------------
        # Camera
        # --------------------------------------------------------
    
        camera = cv2.VideoCapture(0)
    
        if not camera.isOpened():
    
            print(
                "ERROR: Could not open webcam."
            )
    
            raise SystemExit
    
        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )
    
        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )
    
        # --------------------------------------------------------
        # Variables
        # --------------------------------------------------------
    
        canvas = None
        canvas_mask = None
    
        previous_x = None
        previous_y = None
    
        smooth_factor = 0.22
    
        current_color = WHITE
    
        brush_size = 8
    
        min_brush_size = 3
    
        max_brush_size = 50
    
        eraser_size = 40
    
        gesture = "NO HAND"
    
        save_message = ""
    
        save_message_timer = 0
    
        # --------------------------------------------------------
        # Gesture cooldowns
        # --------------------------------------------------------
    
        last_save_time = 0
        last_color_time = 0
    
        gesture_cooldown = 0.7
    
        # --------------------------------------------------------
        # Timestamp
        # --------------------------------------------------------
    
        last_timestamp = 0
    
        # --------------------------------------------------------
        # FPS
        # --------------------------------------------------------
    
        previous_time = time.perf_counter()
    
        fps = 0
    
        # --------------------------------------------------------

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "ERROR: Could not read webcam frame."
                )

                break

            # ----------------------------------------------------
            # Mirror camera
            # ----------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )

            height, width = frame.shape[:2]

            # ----------------------------------------------------
            # Canvas
            # ----------------------------------------------------

            if canvas is None:

                canvas = np.zeros_like(frame)

            if canvas_mask is None:

                canvas_mask = np.zeros(
                    frame.shape[:2],
                    dtype=np.uint8
                )
            # ----------------------------------------------------
            # Browser commands from the website
            # ----------------------------------------------------

            while not command_queue.empty():

                command = command_queue.get()

                if command == "undo":

                    if len(undo_stack) > 0:

                        redo_stack.append(
                            canvas.copy()
                        )

                        canvas = undo_stack.pop()

                        previous_x = None
                        previous_y = None

                        print("UNDO successful.")

                elif command == "redo":

                    if len(redo_stack) > 0:

                        undo_stack.append(
                            canvas.copy()
                        )

                        canvas = redo_stack.pop()

                        previous_x = None
                        previous_y = None

                        print("REDO successful.")

                elif command == "clear":

                    push_undo(
                        undo_stack,
                        redo_stack,
                        canvas,
                        max_history
                    )

                    canvas = np.zeros_like(frame)

                    previous_x = None
                    previous_y = None

                    save_message = "CANVAS CLEARED"
                    save_message_timer = 30

                    print("Canvas cleared.")

                elif command == "save":

                    success, filename = save_drawing(
                        canvas,
                        drawings_folder,
                        drawing_number
                    )

                    if success:

                        print(
                            f"Drawing saved: {filename}"
                        )

                        drawing_number += 1

                        save_message = "DRAWING SAVED"
                        save_message_timer = 45

                elif command.startswith("color:"):

                    try:

                        color_index = int(
                            command.split(":", 1)[1]
                        )

                        if 0 <= color_index < len(COLORS):

                            current_color = COLORS[
                                color_index
                            ][1]

                    except (ValueError, IndexError):

                        pass

                elif command == "quit":

                    stop_event.set()



            # ----------------------------------------------------
            # FPS
            # ----------------------------------------------------

            current_time = time.perf_counter()

            elapsed = (
                current_time -
                previous_time
            )

            if elapsed > 0:

                fps = int(
                    1 / elapsed
                )

            previous_time = current_time

            # ----------------------------------------------------
            # MediaPipe image
            # ----------------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # ----------------------------------------------------
            # Monotonic timestamp
            # ----------------------------------------------------

            timestamp = int(
                time.monotonic() * 1000
            )

            if timestamp <= last_timestamp:

                timestamp = (
                    last_timestamp +
                    1
                )

            last_timestamp = timestamp

            # ----------------------------------------------------
            # Detect hand
            # ----------------------------------------------------

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            gesture = "NO HAND"

            # ====================================================
            # HAND FOUND
            # ====================================================

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                # ------------------------------------------------
                # Finger states
                # ------------------------------------------------

                fingers = fingers_up(hand)

                index = fingers[0]
                middle = fingers[1]
                ring = fingers[2]
                pinky = fingers[3]

                # ------------------------------------------------
                # Thumbs up
                # ------------------------------------------------

                thumbs_up = is_thumbs_up(hand)

                # ------------------------------------------------
                # Fingertips
                # ------------------------------------------------

                thumb_tip = hand[4]
                index_tip = hand[8]

                # ------------------------------------------------
                # Pinch
                # ------------------------------------------------

                pinch_distance = distance(
                    thumb_tip,
                    index_tip,
                    width,
                    height
                )

                is_pinching = (
                    pinch_distance < 85
                )

                # ------------------------------------------------
                # Brush size
                # ------------------------------------------------

                if is_pinching:

                    brush_size = int(
                        np.interp(
                            pinch_distance,
                            [15, 85],
                            [
                                max_brush_size,
                                min_brush_size
                            ]
                        )
                    )

                    brush_size = max(
                        min_brush_size,
                        min(
                            max_brush_size,
                            brush_size
                        )
                    )

                # ------------------------------------------------
                # Gesture recognition
                # ------------------------------------------------

                if thumbs_up:

                    gesture = "SAVE"

                elif is_pinching:

                    gesture = "SIZE"

                elif (
                    index == 1
                    and middle == 0
                    and ring == 0
                    and pinky == 0
                ):

                    gesture = "DRAW"

                elif (
                    index == 1
                    and middle == 1
                    and ring == 0
                    and pinky == 0
                ):

                    gesture = "SELECT"

                elif (
                    index == 1
                    and middle == 1
                    and ring == 1
                    and pinky == 1
                ):

                    gesture = "ERASE"

                elif (
                    index == 0
                    and middle == 0
                    and ring == 0
                    and pinky == 0
                ):

                    gesture = "IDLE"

                else:

                    gesture = "UNKNOWN"

                # ------------------------------------------------
                # Index coordinates
                # ------------------------------------------------

                raw_x = int(
                    index_tip.x * width
                )

                raw_y = int(
                    index_tip.y * height
                )

                raw_x = max(
                    0,
                    min(
                        width - 1,
                        raw_x
                    )
                )

                raw_y = max(
                    0,
                    min(
                        height - 1,
                        raw_y
                    )
                )

                # ------------------------------------------------
                # Smoothing
                # ------------------------------------------------

                if previous_x is not None:

                    x = int(
                        previous_x +
                        (
                            raw_x -
                            previous_x
                        ) *
                        smooth_factor
                    )

                    y = int(
                        previous_y +
                        (
                            raw_y -
                            previous_y
                        ) *
                        smooth_factor
                    )

                else:

                    x = raw_x
                    y = raw_y

                # =================================================
                # SAVE
                # =================================================

                if gesture == "SAVE":

                    previous_x = None
                    previous_y = None

                    now = time.time()

                    if (
                        now -
                        last_save_time
                        > gesture_cooldown
                    ):

                        success, filename = save_drawing(
                            canvas,
                            drawings_folder,
                            drawing_number
                        )

                        if success:

                            print(
                                f"Drawing saved: {filename}"
                            )

                            drawing_number += 1

                            save_message = (
                                "DRAWING SAVED"
                            )

                            save_message_timer = 45

                            last_save_time = now

                # =================================================
                # SIZE
                # =================================================

                elif gesture == "SIZE":

                    previous_x = None
                    previous_y = None

                    draw_cursor(
                        frame,
                        x,
                        y,
                        current_color,
                        max(
                            6,
                            brush_size // 2
                        )
                    )

                # =================================================
                # COLOR SELECT
                # =================================================

                elif gesture == "SELECT":

                    previous_x = None
                    previous_y = None

                    now = time.time()

                    # ------------------------------------------------
                    # Correct palette coordinates
                    # ------------------------------------------------

                    palette_start_x = 230
                    palette_spacing = 42

                    if raw_y < 95:

                        color_index = round(
                            (
                                raw_x -
                                palette_start_x
                            ) /
                            palette_spacing
                        )

                        if (
                            0 <= color_index
                            < len(COLORS)
                        ):

                            color_x = (
                                palette_start_x +
                                color_index *
                                palette_spacing
                            )

                            # Only select when actually near color
                            if (
                                abs(raw_x - color_x)
                                <= 24
                            ):

                                if (
                                    now -
                                    last_color_time
                                    > 0.4
                                ):

                                    current_color = (
                                        COLORS[
                                            color_index
                                        ][1]
                                    )

                                    last_color_time = now

                # =================================================
                # DRAW
                # =================================================

                elif gesture == "DRAW":

                    # ------------------------------------------------
                    # Protect top and bottom UI
                    # ------------------------------------------------

                    bottom_ui_start = (
                        height -
                        155
                    )

                    if (
                        y > 90
                        and y < bottom_ui_start
                    ):

                        # New stroke
                        if (
                            previous_x is None
                            or previous_y is None
                        ):

                            push_undo(
                                undo_stack,
                                redo_stack,
                                canvas,
                                max_history
                            )

                        # Continue stroke
                        if (
                            previous_x is not None
                            and previous_y is not None
                        ):

# ------------------------------------------------
# Draw smooth pencil stroke
# ------------------------------------------------

                            if (
                                previous_x is not None
                                and previous_y is not None
                            ):

                                cv2.line(
                                    canvas,
                                    (
                                        previous_x,
                                        previous_y
                                    ),
                                    (
                                        x,
                                        y
                                    ),
                                    current_color,
                                    brush_size,
                                    cv2.LINE_AA
                                )

                                cv2.line(
                                    canvas_mask,
                                    (
                                        previous_x,
                                        previous_y
                                    ),
                                    (
                                        x,
                                        y
                                    ),
                                    255,
                                    brush_size,
                                    cv2.LINE_AA
                                )


                            # ------------------------------------------------
                            # Sharper pencil tip
                            # ------------------------------------------------

                            tip_radius = max(
                                1,
                                int(brush_size * 0.20)
                            )

                            cv2.circle(
                                canvas,
                                (x, y),
                                tip_radius,
                                current_color,
                                -1,
                                cv2.LINE_AA
                            )

                        previous_x = x
                        previous_y = y

                    else:

                        previous_x = None
                        previous_y = None

                        cv2.circle(
                            canvas_mask,
                            (x, y),
                            tip_radius,
                            255,
                            -1,
                            cv2.LINE_AA
                        )

                # =================================================
                # ERASE
                # =================================================

                elif gesture == "ERASE":

                    # Save state at beginning of erase
                    if (
                        previous_x is None
                        and previous_y is None
                    ):

                        push_undo(
                            undo_stack,
                            redo_stack,
                            canvas,
                            max_history
                        )

                    # Protect top/bottom UI
                    if (
                        y > 90
                        and y < height - 155
                    ):

                        cv2.circle(
                            canvas,
                            (x, y),
                            eraser_size,
                            (0, 0, 0),
                            -1,
                            cv2.LINE_AA
                        )
                        cv2.circle(
                            canvas_mask,
                            (x, y),
                            eraser_size,
                            0,
                            -1,
                            cv2.LINE_AA
                        )

                        draw_eraser_cursor(
                            frame,
                            x,
                            y,
                            eraser_size
                        )

                        previous_x = x
                        previous_y = y

                    else:

                        previous_x = None
                        previous_y = None

                # =================================================
                # IDLE / UNKNOWN
                # =================================================

                else:

                    previous_x = None
                    previous_y = None

                # =================================================
                # IMPORTANT LANDMARKS
                # =================================================

                important_points = [
                    4,
                    8,
                    12,
                    16,
                    20
                ]

                for point_index in important_points:

                    landmark = hand[
                        point_index
                    ]

                    lx = int(
                        landmark.x *
                        width
                    )

                    ly = int(
                        landmark.y *
                        height
                    )

                    cv2.circle(
                        frame,
                        (lx, ly),
                        4,
                        current_color,
                        -1,
                        cv2.LINE_AA
                    )

            else:

                previous_x = None
                previous_y = None

            # ====================================================
            # COMBINE CAMERA + CANVAS
            # ====================================================

            output = frame.copy()

            if canvas_mask is not None:

                mask = canvas_mask > 0

                output[mask] = canvas[mask]

            # ====================================================
            # TOP UI
            # ====================================================

            output = draw_top_bar(
                output,
                current_color,
                brush_size,
                gesture,
                COLORS
            )

            # ====================================================
            # BOTTOM UI
            # ====================================================
            # The website renders the bottom tool dock itself.
            # Do not draw another copy into the MJPEG stream.

            # ====================================================
            # SAVE NOTIFICATION
            # ====================================================

            if save_message_timer > 0:

                output = draw_notification(
                    output,
                    save_message,
                    save_message_timer
                )

                save_message_timer -= 1

            # ====================================================
            # FPS
            # ====================================================

            text(
                output,
                f"{fps} FPS",
                (20, height - 12),
                0.32,
                MUTED,
                1
            )

            # ====================================================

            # ====================================================
            # STREAM FRAME TO BROWSER
            # ====================================================

            success, encoded = cv2.imencode(
                ".jpg",
                output,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    85
                ]
            )

            if success:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded.tobytes()
                    + b"\r\n"
                )

            if stop_event.is_set():

                break

        camera.release()


# ============================================================
# DIRECT TEST
# ============================================================
# The Flask app imports this module and calls generate_frames().
# Running this file directly starts the generator for testing.
# It will not open an OpenCV window; the browser is the UI.
# ============================================================

if __name__ == "__main__":

    print("AI Air Pencil web engine loaded.")
    print("Start the website with: python app.py")