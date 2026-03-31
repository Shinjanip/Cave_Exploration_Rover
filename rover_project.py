import cv2
from ultralytics import YOLO

# 1. Load your trained brain! 
# (Make sure best.pt is in the same folder as this script)
model = YOLO('best.pt')

# 2. Turn on the laptop webcam (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

print("Starting Rover Vision System... Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Ask the model to look at the frame and find cracks
    # conf=0.5 means it will only show boxes if it's 50%+ sure it's a crack
    results = model(frame, conf=0.5, verbose=False)

    # 4. Draw the boxes on the frame
    annotated_frame = results[0].plot()

    # 5. Add your Rover Safety Logic!
    # If the model found ANY cracks (boxes), trigger the Danger alert
    if len(results[0].boxes) > 0:
        cv2.putText(annotated_frame, "DANGER: STRUCTURAL CRACK DETECTED!", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(annotated_frame, "STATUS: SAFE - NO CRACKS", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6. Show the live feed on your screen
    cv2.imshow('Rover Autonomous Vision', annotated_frame)

    # Press 'q' to stop the rover
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up when done
cap.release()
cv2.destroyAllWindows()