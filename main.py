import cv2
import mediapipe as mp

# Initializing the Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initializing the video source (0 for the built-in camera)
cap = cv2.VideoCapture(0)

while True:
    # Reading the image from the camera
    ret, frame = cap.read()

    # Converting the image to BGR format (Mediapipe expects BGR format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting hands using Mediapipe
    results = hands.process(rgb_frame)

    # If hands are found, display the detected points
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Coordinates of the center of the hand
            cx, cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]), \
                     int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

            # Determining the positions of the index finger and thumb tips
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Checking hand openness: if the fingers are far apart, the hand is open
            distance = abs(index_finger.x - thumb.x) + abs(index_finger.y - thumb.y)
            hand_open = distance > 0.07  # Threshold value can be adjusted as needed

            # Displaying the center point
            cv2.putText(frame, f'Hand Center: ({cx}, {cy})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            for landmark in hand_landmarks.landmark:
                cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Displaying the state
            if hand_open:
                cv2.putText(frame, 'Open', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Closed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Displaying the image
    cv2.imshow('Hand Detection', frame)

    # Exiting the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing resources and closing the window
cap.release()
cv2.destroyAllWindows()
