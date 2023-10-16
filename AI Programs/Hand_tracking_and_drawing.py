import cv2
import mediapipe as mp
import pyautogui
import math
from google.protobuf.json_format import MessageToDict

class Hand:
    # Total number of hands
    hands = 0

    def __init__(self, hand_lms, handedness):
        self.hand_lms = hand_lms
        self.landmarks = hand_lms.landmark
        self.handedness = handedness
        
        # Increment total number of hands
        Hand.hands += 1

    # Retrieve landmark location based on index
    def landmark(self, index):
        return self.landmarks[index]
    
    def scaled_2d_landmark(self, index, w, h):
        l = self.landmark(index)

        return {
            'x': l.x * w,
            'y': l.y * h
        }

    # Get a finger node based on finger index (0 - 4) and finger node index from tip to stem (0 - 3)
    def get_finger_node(self, finger, f_index):
        if finger == 0:
            return self.landmark(4 - f_index)
        elif finger == 1:
            return self.landmark(8 - f_index)
        elif finger == 2:
            return self.landmark(12 - f_index)
        elif finger == 3:
            return self.landmark(16 - f_index)
        elif finger == 4:
            return self.landmark(20 - f_index)

    # Determines if two designated fingers are approximately touching based on 2D distance
    def fingertips_touching(self, f1, f2, threshold = 0.04):
       tip1 = self.get_finger_node(f1, 0)
       tip2 = self.get_finger_node(f2, 0)

       return math.dist([tip1.x, tip1.y], [tip2.x, tip2.y]) < threshold

# Capture video stream
cap = cv2.VideoCapture(0)

# Mediapipe hand detection setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
Hands = mp_hands.Hands()

# Device display size
display_size = pyautogui.size()

def run():
    drawings = []
    drawing = []
    prevDraw = False
    draw = False

    while(True):
        # Array of hands seen in the image (max is 2)
        hands = {
            'Left': False,
            'Right': False
        }

        # Read the image from the video stream
        success, image = cap.read()
        # Convert to RGB
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageRGB.flags.writeable = False
        # Search for hands
        results = Hands.process(imageRGB)
        # Size of the image
        img_size = image.shape

        # If results are found...
        if results.multi_hand_landmarks:
            # Loop through and add a new hand to the array for each instance
            for index in range(len(results.multi_hand_landmarks)):
                hand_lms = results.multi_hand_landmarks[index]
                handedness = MessageToDict(results.multi_handedness[index])['classification'][0]['label']
                hands[handedness] = Hand(hand_lms, handedness)
            
            l_hand = hands['Left']
            r_hand = hands['Right']

            if l_hand and r_hand:
                pass
            
            if l_hand:
                # Thumb tip landmark
                n4 = l_hand.landmark(4)
                # Index fingertip landmark
                n8 = l_hand.landmark(8)

                # Compute image coordinates
                s8 = l_hand.scaled_2d_landmark(8, img_size[1], img_size[0])
                s4 = l_hand.scaled_2d_landmark(4, img_size[1], img_size[0])
                d8 = l_hand.scaled_2d_landmark(8, display_size[0], display_size[1])

                mp_drawing.draw_landmarks(image, l_hand.hand_lms, mp_hands.HAND_CONNECTIONS)

                if(l_hand.fingertips_touching(0, 1)):
                    draw = True
                else:
                    draw = False
                
                if(draw):
                    drawing.append([int(s8['x']), int(s8['y'])])

                if(draw != prevDraw):
                    if(not draw):
                        drawings.append(drawing)
                        drawing = []
                    
                    prevDraw = draw
                
            if r_hand:
                mp_drawing.draw_landmarks(image, r_hand.hand_lms, mp_hands.HAND_CONNECTIONS)

        for d in drawings:
            for index in range(len(d)):
                if index == 0: continue

                point1 = d[index - 1]
                point2 = d[index]

                cv2.line(image, point1, point2, (0, 0, 180), 10)
            
        for index in range(len(drawing)):
            if index == 0: continue

            point1 = drawing[index - 1]
            point2 = drawing[index]

            cv2.line(image, point1, point2, (0, 0, 255), 10)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("image", image)
  
run()

# Release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()