import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Function to collect and preprocess images
def collect_and_preprocess_images(gesture, num_images, output_folder):
    cap = cv2.VideoCapture(0)  # Initialize webcam
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output directory if it doesn't exist
    
    count = 0
    while count < num_images:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            print("Failed to capture image")
            break
        
        # Convert the frame to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)  # Detect hands
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])
                
                h, w, _ = frame.shape
                x_min = int(x_min * w) - 15 # Expand the box by 20 pixels on each side
                x_max = int(x_max * w) + 15
                y_min = int(y_min * h) - 15
                y_max = int(y_max * h) + 15
                # Ensure coordinates are within image bounds
                x_min = max(0, x_min)
                x_max = min(w, x_max)
                y_min = max(0, y_min)
                y_max = min(h, y_max)
                
                # Crop the hand region
                hand_img = frame[y_min:y_max, x_min:x_max]

                if hand_img.size == 0:
                    print("Empty hand image, skipping...")
                    continue
                
                # Resize and save the image
                hand_img = cv2.resize(hand_img, (600, 600))
                img_path = os.path.join(output_folder, f"{gesture}_{count}.jpg")
                cv2.imwrite(img_path, hand_img)
                print(f"Saved image: {img_path}")
                count += 1

                # Draw a rectangle around the hand region
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display the gesture name on the frame
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the count of captured images
        cv2.putText(frame, f'Count: {count}/{num_images}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Frame', frame)  # Display the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Collect images for each gesture
gestures = ['up', 'down', 'left', 'right', 'select']
num_images_per_gesture = 2500 # Adjust as needed

for gesture in gestures:
    collect_and_preprocess_images(gesture, num_images_per_gesture, f'dataset/{gesture}')


