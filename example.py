import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from categories import categories

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = tf.keras.models.load_model('saved_model/my_model')
print('Press q to quit.')


def get_coord(lm):
    hand_lm = lm[0].landmark
    data_arr = []

    # PROCESS DATA (Calculate relative pos from lm 0)
    for i, lm in enumerate(hand_lm):
        if i == 0:
            refx, refy, refz = lm.x, lm.y, lm.z

        else:
            x = lm.x - refx
            y = lm.y - refy
            z = lm.z - refz
            data_arr.append([x, y, z])

    return data_arr


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:

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
        hand_lm = results

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks
            lm_arr = get_coord(hand_lm)
            lm_arr = np.array([lm_arr, ])

            pred_vals = model.predict(lm_arr)
            pred = np.argmax(pred_vals)
            conf = np.max(pred_vals)

            pred_char = categories[pred]
            if conf > 0.8:
                cv2.putText(image, f"{pred_char}|{conf}", (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
