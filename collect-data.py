import cv2
import numpy as np
import os
import string
import config

# Create the directory structure
if not os.path.exists(config.DATA_DIR):
    os.makedirs(config.DATA_DIR)
if not os.path.exists(config.TRAIN_DIR):
    os.makedirs(config.TRAIN_DIR)
if not os.path.exists(config.TEST_DIR):
    os.makedirs(config.TEST_DIR)

# Create directories for 0-9 and A-Z
labels = list(string.ascii_uppercase) + [str(i) for i in range(10)]
for label in labels:
    train_path = os.path.join(config.TRAIN_DIR, label)
    test_path = os.path.join(config.TEST_DIR, label)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

# Train or test 
mode = 'train'
directory = config.TRAIN_DIR if mode == 'train' else config.TEST_DIR
minValue = config.MIN_VALUE

cap = cv2.VideoCapture(0)
interrupt = -1  

while True:
    _, frame = cap.read()
    if frame is None:
        break
        
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {}
    for label in labels:
        count[label] = len(os.listdir(os.path.join(directory, label)))
    
    # Display counts on frame
    y_pos = 70
    x_pos = 10
    for i, label in enumerate(labels):
        # Simple layout logic
        if y_pos > 400:
            y_pos = 70
            x_pos += 150
        
        cv2.putText(frame, f"{label} : {count[label]}", (x_pos, y_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        y_pos += 10
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    
    # Drawing the ROI
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    cv2.imshow("Frame", frame)
    
    # Image processing for preview
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    test_image = cv2.resize(test_image, (config.ROI_SIZE, config.ROI_SIZE))
    cv2.imshow("test", test_image)
        
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    
    # Check for key presses to save images
    # 0-9
    for i in range(10):
        if interrupt & 0xFF == ord(str(i)):
            save_path = os.path.join(directory, str(i), f"{count[str(i)]}.jpg")
            cv2.imwrite(save_path, roi)
            
    # A-Z
    for char in string.ascii_lowercase:
        if interrupt & 0xFF == ord(char):
            label = char.upper()
            save_path = os.path.join(directory, label, f"{count[label]}.jpg")
            cv2.imwrite(save_path, roi)

cap.release()
cv2.destroyAllWindows()
