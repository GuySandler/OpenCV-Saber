import cv2
import mediapipe as mp
import numpy as np
import random
import time


def select_difficulty():
    print("Select Difficulty:")
    print("1. Easy (Slow blocks, longer spawn time)")
    print("2. Medium (Normal speed and spawn rate)")
    print("3. Hard (Fast blocks, quick spawn rate)")
    print("4. Impossible (Super fast blocks, Very quick spawn rate)")
    choice = input("Enter 1, 2, 3, or 4 (ignore other text): ")

    if choice == "1":
        return 2.0, 3  # Spawn interval, gravity
    elif choice == "2":
        return 1.5, 6
    elif choice == "3":
        return 1.0, 8
    elif choice == "4":
        return 0.5, 15
    else:
        print("Invalid choice, defaulting to Medium.")
        return 1.8, 6


def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    spawn_interval, gravity = select_difficulty()
    blocks = []
    hits = 0
    misses = 0
    last_spawn_time = time.time()
    start_time = time.time()
    game_duration = 60

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally for a mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_time = time.time()
        elapsed_time = int(current_time - start_time)

        if elapsed_time >= game_duration:
            break

        if current_time - last_spawn_time > spawn_interval:
            blocks.append(
                [random.randint(100, 500), 0, random.choice(["Left", "Right"]), gravity])  # X, Y, hand target, speed
            last_spawn_time = current_time

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_label = handedness.classification[0].label  # "Left" or "Right"
                hand_pos = hand_landmarks.landmark[8]
                hand_x, hand_y = int(hand_pos.x * frame.shape[1]), int(hand_pos.y * frame.shape[0])

                cv2.putText(frame, hand_label, (hand_x, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)

                blocks_to_remove = []
                for block in blocks:
                    block_x, block_y, block_hand, speed = block
                    if abs(hand_x - block_x) < 40 and abs(hand_y - block_y) < 40:
                        if hand_label == block_hand:
                            hits += 1
                        else:
                            misses += 1  # Penalize for wrong hand hit
                        blocks_to_remove.append(block)

                # Remove the blocks that were hit
                for block in blocks_to_remove:
                    blocks.remove(block)

        # Move and draw blocks
        for block in blocks[:]:
            block[1] += block[3]  # Move down
            color = (0, 0, 255) if block[2] == "Right" else (255, 0, 0)
            cv2.rectangle(frame, (block[0] - 30, block[1] - 30), (block[0] + 30, block[1] + 30), color, -1)
            cv2.putText(frame, block[2], (block[0] - 20, block[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),2)

            # Remove missed blocks
            if block[1] > frame.shape[0]:
                blocks.remove(block)
                misses += 1

        # score
        score_ratio = hits / max(hits + misses, 1)
        cv2.putText(frame, f"Score: {hits}/{hits + misses} ({score_ratio:.2%})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 2, cv2.LINE_AA)

        # timer
        cv2.putText(frame, f"Time: {game_duration - elapsed_time}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Beat Saber Hand Game", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Final Score: {hits}/{hits + misses} ({score_ratio:.2%})")


if __name__ == "__main__":
    main()
