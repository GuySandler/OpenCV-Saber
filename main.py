import cv2
import mediapipe as mp
import numpy as np
import random
import time
from flask import Flask, Response, render_template_string, redirect, url_for

app = Flask(__name__)

# Global variables for game state
cap = None
hands = None
mp_hands = None
mp_draw = None
spawn_interval, gravity = 1.5, 6  # Default to Medium difficulty
blocks = []
hits = 0
misses = 0
last_spawn_time = 0
start_time = 0
game_duration = 60
game_over_flag = False


def initialize_game():
    global cap, mp_hands, hands, mp_draw
    global blocks, hits, misses, last_spawn_time, start_time, game_over_flag

    # spawn_interval and gravity are global, set by default or set_difficulty route.
    # This function will use those current values.

    # Initialize camera only if not already done or if it was closed
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

    # Initialize mediapipe only once
    if mp_hands is None:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_draw = mp.solutions.drawing_utils

    # Reset game-specific state
    blocks = []
    hits = 0
    misses = 0
    last_spawn_time = time.time()
    start_time = time.time()
    game_over_flag = False
    print(f"Game initialized/reset. Difficulty: spawn_interval={spawn_interval}, gravity={gravity}.")


def generate_frames():
    global blocks, hits, misses, last_spawn_time, start_time, game_duration, game_over_flag

    if cap is None or not cap.isOpened():
        print("Webcam not initialized or not open.")
        # Try to initialize again if not already
        try:
            initialize_game()
        except IOError as e:
            print(f"Error initializing game: {e}")
            # Yield a black frame with an error message
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "Error: Cannot open webcam", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            return  # Stop generation if webcam fails

    # Ensure game variables are reset if starting a new stream after game over
    if game_over_flag:
        initialize_game()  # Re-initialize for a new game session on new stream connection

    while True:
        if game_over_flag:  # If game ended, stop sending frames for this stream
            print("Game over. Stopping frame generation for this stream.")
            # Optionally, yield a "Game Over" frame
            img = np.zeros((480, 640, 3), dtype=np.uint8)  # Assuming 640x480, adjust if needed
            final_score_text = f"Game Over! Score: {hits}/{hits + misses}"
            cv2.putText(img, final_score_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            break

        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            # If frames stop, could indicate end of stream or camera issue
            # For simplicity, we break; a more robust app might try to re-init camera
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_time = time.time()
        elapsed_time = int(current_time - start_time)

        if elapsed_time >= game_duration:
            game_over_flag = True  # Set game over flag
            # Don't break here, let it yield one last "Game Over" frame if desired, or handle above

        if not game_over_flag and current_time - last_spawn_time > spawn_interval:
            blocks.append(
                [random.randint(100, frame.shape[1] - 100), 0, random.choice(["Left", "Right"]), gravity])
            last_spawn_time = current_time

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_label = handedness.classification[0].label
                hand_pos = hand_landmarks.landmark[8]  # Finger tip
                hand_x, hand_y = int(hand_pos.x * frame.shape[1]), int(hand_pos.y * frame.shape[0])

                cv2.putText(frame, hand_label, (hand_x, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)

                blocks_to_remove = []
                for i, block in enumerate(blocks):
                    block_x, block_y, block_hand, speed = block
                    if abs(hand_x - block_x) < 40 and abs(hand_y - block_y) < 40:
                        if not game_over_flag:  # Only score if game is not over
                            if hand_label == block_hand:
                                hits += 1
                            else:
                                misses += 1
                        blocks_to_remove.append(block)

                for block in blocks_to_remove:
                    if block in blocks:  # Check if block still exists
                        blocks.remove(block)

        # Move and draw blocks
        for block in blocks[:]:
            if not game_over_flag:
                block[1] += block[3]  # Move down only if game is not over

            color = (0, 0, 255) if block[2] == "Right" else (255, 0, 0)  # Blue for Left, Red for Right
            cv2.rectangle(frame, (block[0] - 30, block[1] - 30), (block[0] + 30, block[1] + 30), color, -1)
            cv2.putText(frame, block[2], (block[0] - 20, block[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)

            if block[1] > frame.shape[0]:  # Block missed
                if not game_over_flag:  # Only count miss if game is not over
                    misses += 1
                blocks.remove(block)

        score_ratio = hits / max(hits + misses, 1)
        cv2.putText(frame, f"Score: {hits}/{hits + misses} ({score_ratio:.2%})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2, cv2.LINE_AA)

        time_remaining = game_duration - elapsed_time
        if game_over_flag:
            time_remaining = 0  # Show 0 when game is over
            cv2.putText(frame, "GAME OVER", (frame.shape[1] // 2 - 100, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 0, 255), 3)

        cv2.putText(frame, f"Time: {max(0, time_remaining)}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                    cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    # HTML to display the video stream and difficulty selector
    difficulty_name = 'Custom'
    if spawn_interval == 2.0 and gravity == 4:
        difficulty_name = 'Easy'
    elif spawn_interval == 1.5 and gravity == 6:
        difficulty_name = 'Medium'
    elif spawn_interval == 1.0 and gravity == 8:
        difficulty_name = 'Hard'
    elif spawn_interval == 0.7 and gravity == 10:
        difficulty_name = 'Extreme'

    return render_template_string("""
        <html>
            <head>
                <title>OpenCV Saber Game</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; color: #333; text-align: center; }
                    h1 { color: #333; }
                    h2 { color: #555; margin-top: 30px; }
                    .container { max-width: 800px; margin: auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 8px; }
                    .section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #fff; }
                    .section a { 
                        text-decoration: none; padding: 10px 18px; margin: 5px; background-color: #007bff; color: white; border-radius: 4px; display: inline-block;
                        transition: background-color 0.3s ease;
                    }
                    .section a:hover { background-color: #0056b3; }
                    img { display: block; margin: 20px auto; border: 2px solid #ccc; border-radius: 4px; }
                    p { line-height: 1.6; }
                    .status-display p { font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>OpenCV Saber Game</h1>

                    <div class="section difficulty-selector">
                        <h2>Select Difficulty:</h2>
                        <a href="{{ url_for('set_difficulty', difficulty_level='easy') }}">Easy</a>
                        <a href="{{ url_for('set_difficulty', difficulty_level='medium') }}">Medium</a>
                        <a href="{{ url_for('set_difficulty', difficulty_level='hard') }}">Hard</a>
                        <a href="{{ url_for('set_difficulty', difficulty_level='extreme') }}">Extreme</a>
                    </div>

                    <div class="section status-display">
                        <p>Current Difficulty: {{ current_difficulty_name }}</p>
                        <p>(Spawn Interval: {{ current_spawn_interval }}, Gravity: {{ current_gravity }})</p>
                    </div>

                    <img src="{{ url_for('video_feed') }}" width="640" height="480">

                    <div class="section game-controls">
                        <p><a href="{{ url_for('start_game_route') }}">Start/Restart Game</a></p>
                    </div>

                    <p>Point your hands at the blocks matching the hand label (Left/Right).</p>
                </div>
            </body>
        </html>
    """, current_spawn_interval=spawn_interval, current_gravity=gravity, current_difficulty_name=difficulty_name)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/set_difficulty/<difficulty_level>')
def set_difficulty(difficulty_level):
    global spawn_interval, gravity, game_over_flag
    if difficulty_level == 'easy':
        spawn_interval, gravity = 2.0, 4
    elif difficulty_level == 'medium':
        spawn_interval, gravity = 1.5, 6
    elif difficulty_level == 'hard':
        spawn_interval, gravity = 1.0, 8
    elif difficulty_level == 'extreme':
        spawn_interval, gravity = 0.7, 10
    else:  # Default to medium if invalid
        spawn_interval, gravity = 1.5, 6

    print(f"Difficulty set to {difficulty_level}: spawn_interval={spawn_interval}, gravity={gravity}")

    # Optionally, force current game to end so next start uses new difficulty
    game_over_flag = True

    return redirect(url_for('index'))


@app.route('/start_game')
def start_game_route():
    global game_over_flag
    print("Attempting to start/restart game...")

    try:
        initialize_game()
    except IOError as e:
        return f"Failed to initialize camera: {e}", 500

    print(f"Game started/restarted with difficulty: spawn_interval={spawn_interval}, gravity={gravity}")

    return redirect(url_for('index'))


def run_app():
    try:
        initialize_game()
    except IOError as e:
        print(f"CRITICAL: Could not initialize webcam on startup: {e}")
        print("The application will run, but video feed may not work until camera is available.")

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)


if __name__ == "__main__":
    run_app()
