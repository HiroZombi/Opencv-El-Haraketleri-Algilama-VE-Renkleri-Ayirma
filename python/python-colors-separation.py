import cv2
import mediapipe
import pyttsx3
import numpy as np

camera = cv2.VideoCapture(0)

mpHands = mediapipe.solutions.hands

hands = mpHands.Hands()

mpDraw = mediapipe.solutions.drawing_utils

pozitif_ifade = False

negatif_ifade = False

ilk_kosul_calisti = False

engine = pyttsx3.init()

while True:

    sucess, frame = camera.read()
    frame = cv2.flip(frame, 1)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hlms = hands.process(imgRGB)

    height, width, channel = frame.shape

    # HSV Hue(TON) ,Saturation(DOYGUNLUK) ,Value (DEĞER)
    # RGB DEĞERLERİ , ([kırmızı, yeşil, mavi])
    #renklerimiz
    lower_red = np.array([161, 155, 84])
    upper_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    lower_green = np.array([45, 100, 50])
    upper_green = np.array([75, 255, 255])
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Combine komutu ile 3 rengimizi bir ekrana taşıyoruz
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
    combined = cv2.bitwise_and(frame, frame, mask=combined_mask)


    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks:

            for fingerNum, landmark in enumerate(handlandmarks.landmark):
                positionX, positionY = int(landmark.x * width), int(landmark.y * height)

                cv2.putText(frame, str(fingerNum), (positionX, positionY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) # parmaklarımıda ki eklemlerin sayısal değerini kameraya yazdırmak için

                #if fingerNum == 4:
                    #cv2.circle(frame, (positionX, positionY), 30, (255,255,255), cv2.FILLED)  #=> 4 numaralı baş parmağımızı temsil eden kod satırı

                #if fingerNum == 2:
                    #print(positionY) #=> 2 numaralı konumun y ekseni üzerinde ki konumunun sayısal değerini görmemiz için

                if fingerNum > 4 and landmark.y < handlandmarks.landmark[2].y:
                    break

                if fingerNum == 20 and landmark.y > handlandmarks.landmark[2].y and not ilk_kosul_calisti:
                    print("bi laykını aldım!")
                    pozitif_ifade = True
                    ilk_kosul_calisti = True

                    if pozitif_ifade:
                        engine.say("Bravo!")
                        engine.runAndWait()
                        break

            mpDraw.draw_landmarks(frame, handlandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Camera", frame)

    if pozitif_ifade:
        #cv2.imshow("Red Mask (Siyah Beyaz)", red_mask)
        cv2.imshow("Red Mask (Renkli)", red)

        #cv2.imshow("Green Mask (Siyah Beyaz)", green_mask)
        cv2.imshow("Green Mask (Renkli)", green)

        #cv2.imshow("Blue Mask (Siyah Beyaz)", blue_mask)
        cv2.imshow("Blue Mask (Renkli)", blue)

        cv2.imshow("Combined Mask (Siyah Beyaz)", combined_mask)
        cv2.imshow("Combined Mask (Renkli)", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
