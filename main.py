from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0)  
cap.set(3, 1280)
cap.set(4, 720)


model = YOLO("playingCards.pt")
classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']

def calculate_blackjack_value(hand):

    value = 0
    aces = 0

    for card in hand:
        rank = card[:-1]  
        if rank in ['K', 'Q', 'J']:
            value += 10
        elif rank == 'A':
            aces += 1
            value += 11  
        else:
            value += int(rank)


    while value > 21 and aces:
        value -= 10
        aces -= 1

    return value

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    hand = []


    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))


            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            if conf > 0.5:
                hand.append(classNames[cls])


    hand = list(set(hand))

    if hand:  # Process only if cards are detected
        value = calculate_blackjack_value(hand)
        if value == 21:
            result = "BLACKJACK!!!"
        elif value > 21:
            result = "Bust!"
        else:
            result = f"Value: {value}"

        print(f"Hand: {hand}, {result}")
        cvzone.putTextRect(img, result, (300, 75), scale=3, thickness=5, colorR=(0, 255, 0) if value == 21 else (255, 0, 0))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
