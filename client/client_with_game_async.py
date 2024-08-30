import threading
import asyncio
import aiohttp
from PIL import Image
import io
import cv2
import mediapipe as mp
import pygame
import sys
import time
import os

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 900, 900
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe!")

# Load images
BOARD = pygame.image.load("assets/Board.png")
X_IMG = pygame.image.load("assets/X.png")
O_IMG = pygame.image.load("assets/O.png")
cursor_size = 200  # Adjust the cursor size
cursor_surface = pygame.Surface((cursor_size, cursor_size), pygame.SRCALPHA)
cursor_surface.fill((0, 0, 0, 128))

BG_COLOR = (214, 201, 227)

# Initialize the game state
board = [["", "", ""], ["", "", ""], ["", "", ""]]
graphical_board = [
    [[None, None], [None, None], [None, None]],
    [[None, None], [None, None], [None, None]],
    [[None, None], [None, None], [None, None]],
]

to_move = "X"
cursor_pos = [0, 0]  # Cursor starts at the top-left corner of the grid
game_finished = False

# Global variable to hold the current frame and detected hand region
current_frame = None
hand_detected = False  # Indicates if a hand is detected

# Lock to ensure thread safety
frame_lock = threading.Lock()

# Directory to save the images
saved_images_dir = "sent_images"
os.makedirs(saved_images_dir, exist_ok=True)


def render_board():
    SCREEN.fill(BG_COLOR)
    SCREEN.blit(BOARD, (64, 64))

    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                graphical_board[i][j][0] = X_IMG
                graphical_board[i][j][1] = X_IMG.get_rect(
                    center=(j * 300 + 150, i * 300 + 150)
                )
            elif board[i][j] == "O":
                graphical_board[i][j][0] = O_IMG
                graphical_board[i][j][1] = O_IMG.get_rect(
                    center=(j * 300 + 150, i * 300 + 150)
                )

    for i in range(3):
        for j in range(3):
            if graphical_board[i][j][0] is not None:
                SCREEN.blit(graphical_board[i][j][0], graphical_board[i][j][1])

    # Draw cursor
    if not game_finished:  # Only draw the cursor if the game is not finished
        i, j = cursor_pos
        cursor_x = j * 300 + 150 - cursor_size // 2  # Center cursor horizontally
        cursor_y = i * 300 + 150 - cursor_size // 2  # Center cursor vertically
        SCREEN.blit(cursor_surface, (cursor_x + 64, cursor_y + 64))

    pygame.display.update()


def add_XO():
    global to_move
    i, j = cursor_pos
    if board[i][j] == "":  # Only place if the spot is empty
        board[i][j] = to_move
        if to_move == "O":
            to_move = "X"
        else:
            to_move = "O"

    render_board()


def check_win():
    global game_finished
    winner = None
    for row in range(0, 3):
        if (board[row][0] == board[row][1] == board[row][2]) and board[row][0] != "":
            winner = board[row][0]
            for i in range(0, 3):
                graphical_board[row][i][0] = pygame.image.load(
                    f"assets/Winning {winner}.png"
                )
                SCREEN.blit(graphical_board[row][i][0], graphical_board[row][i][1])
            pygame.display.update()
            game_finished = True
            return winner

    for col in range(0, 3):
        if (board[0][col] == board[1][col] == board[2][col]) and board[0][col] != "":
            winner = board[0][col]
            for i in range(0, 3):
                graphical_board[i][col][0] = pygame.image.load(
                    f"assets/Winning {winner}.png"
                )
                SCREEN.blit(graphical_board[i][col][0], graphical_board[i][col][1])
            pygame.display.update()
            game_finished = True
            return winner

    if (board[0][0] == board[1][1] == board[2][2]) and board[0][0] != "":
        winner = board[0][0]
        graphical_board[0][0][0] = pygame.image.load(f"assets/Winning {winner}.png")
        SCREEN.blit(graphical_board[0][0][0], graphical_board[0][0][1])
        graphical_board[1][1][0] = pygame.image.load(f"assets/Winning {winner}.png")
        SCREEN.blit(graphical_board[1][1][0], graphical_board[1][1][1])
        graphical_board[2][2][0] = pygame.image.load(f"assets/Winning {winner}.png")
        SCREEN.blit(graphical_board[2][2][0], graphical_board[2][2][1])
        pygame.display.update()
        game_finished = True
        return winner

    if (board[0][2] == board[1][1] == board[2][0]) and board[0][2] != "":
        winner = board[0][2]
        graphical_board[0][2][0] = pygame.image.load(f"assets/Winning {winner}.png")
        SCREEN.blit(graphical_board[0][2][0], graphical_board[0][2][1])
        graphical_board[1][1][0] = pygame.image.load(f"assets/Winning {winner}.png")
        SCREEN.blit(graphical_board[1][1][0], graphical_board[1][1][1])
        graphical_board[2][0][0] = pygame.image.load(f"assets/Winning {winner}.png")
        SCREEN.blit(graphical_board[2][0][0], graphical_board[2][0][1])
        pygame.display.update()
        game_finished = True
        return winner

    if winner is None:
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == "":
                    return None
        game_finished = True  # Game ends in a draw
        return "DRAW"

    return winner


def move_cursor(gesture):
    if not game_finished:  # Only move cursor if the game is not finished
        if gesture == "up":
            cursor_pos[0] = max(0, cursor_pos[0] - 1)
        elif gesture == "down":
            cursor_pos[0] = min(2, cursor_pos[0] + 1)
        elif gesture == "left":
            cursor_pos[1] = max(0, cursor_pos[1] - 1)
        elif gesture == "right":
            cursor_pos[1] = min(2, cursor_pos[1] + 1)
        elif gesture == "select":
            add_XO()
            check_win()


def send_image_to_server(img_bytes):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_send_image_to_server(img_bytes))
    loop.close()


async def _send_image_to_server(img_bytes):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://15.222.254.223:80/predict", data={"image": img_bytes}
        ) as response:
            if response.status == 200:
                json_response = await response.json()
                gesture = json_response["gesture"]
                print(f"Gesture received: {gesture}")
                move_cursor(gesture)
            else:
                print(f"Server returned status code {response.status}")


def capture_and_show_frame():
    global current_frame, hand_detected
    while True:
        ret, frame = cap.read()
        if ret:
            # Capture and process the frame for hand detection
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            hand_detected = False  # Reset hand_detected flag for each frame

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                    x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                    y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                    y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                    h, w, _ = frame.shape
                    x_min = int(x_min * w) - 30
                    x_max = int(x_max * w) + 30
                    y_min = int(y_min * h) - 30
                    y_max = int(y_max * h) + 30

                    x_min = max(0, x_min)
                    x_max = min(w, x_max)
                    y_min = max(0, y_min)
                    y_max = min(h, y_max)

                    # Draw the rectangle on the frame
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    with frame_lock:
                        hand_detected = (
                            True  # Set hand_detected to True when a hand is detected
                        )

            # Show the frame
            with frame_lock:
                current_frame = frame.copy() if hand_detected else None
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def start_frame_thread():
    frame_thread = threading.Thread(target=capture_and_show_frame)
    frame_thread.daemon = (
        True  # This ensures the thread will close when the main program exits
    )
    frame_thread.start()


def poll_for_frame():
    global game_finished
    frame_counter = 0
    while not game_finished:
        with frame_lock:
            if (
                current_frame is not None
            ):  # Only process gestures if the game is not finished
                # Capture the hand region from the current frame
                img_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_min = min(
                            [landmark.x for landmark in hand_landmarks.landmark]
                        )
                        x_max = max(
                            [landmark.x for landmark in hand_landmarks.landmark]
                        )
                        y_min = min(
                            [landmark.y for landmark in hand_landmarks.landmark]
                        )
                        y_max = max(
                            [landmark.y for landmark in hand_landmarks.landmark]
                        )

                        h, w, _ = current_frame.shape
                        x_min = int(x_min * w)
                        x_max = int(x_max * w)
                        y_min = int(y_min * h)
                        y_max = int(y_max * h)

                        x_min = max(0, x_min - 15)
                        x_max = min(w, x_max + 15)
                        y_min = max(0, y_min - 10)
                        y_max = min(h, y_max + 10)

                        # Crop the hand region, excluding the green box
                        hand_img = current_frame[y_min:y_max, x_min:x_max]

                        # Convert the hand image back to RGB for saving and sending
                        pil_img = Image.fromarray(hand_img)
                        pil_img = pil_img.resize(
                            (600, 600), Image.LANCZOS
                        )  # Resize to 600x600 before sending
                        img_bytes = io.BytesIO()
                        pil_img.save(img_bytes, format="JPEG", quality=100)
                        img_bytes = img_bytes.getvalue()

                        # Save the hand region locally
                        save_path = f"hand_{frame_counter}.jpg"
                        pil_img.save(save_path)
                        frame_counter += 1

                        # Send the hand image to the server
                        send_image_to_server(img_bytes)

        time.sleep(2)  # Poll every 2 seconds


def start_polling_thread():
    polling_thread = threading.Thread(target=poll_for_frame)
    polling_thread.daemon = True  # Ensure the thread closes when the main program exits
    polling_thread.start()


def main_loop():
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Render the game board and cursor
        render_board()

        time.sleep(0.2)  # Slight delay to reduce CPU usage


def run_game():
    start_frame_thread()
    start_polling_thread()
    main_loop()


if __name__ == "__main__":
    run_game()
    cap.release()
    cv2.destroyAllWindows()
