#!/usr/bin/python3

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#declare global variables for tracking finger
index_finger_pos_x = 0.0
index_finger_pos_y = 0.0

# Webcamera no 0 is used to capture the frames
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get image height and image width
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        index_finger_pos_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
        index_finger_pos_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Print debug position of index finger
    print('position of index finger:' + str(index_finger_pos_x) + ',' + str(index_finger_pos_y))

    # Draw a diagonal green line with thickness of 9 px
    image = cv2.line(image, (int(image_width / 2), 0), (int(image_width / 2), image_height), (0,255,0), 2)

    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image,1)    

    # compare if index finger x position on the left or right side of the screen
    if index_finger_pos_x <= image_width / 2:
      image = cv2.putText(image, 'Right', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2, cv2.LINE_AA)
    if index_finger_pos_x > image_width / 2:
      image = cv2.putText(image, 'Left', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detect Index Finger Position', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()