import cv2
from tracker import *

if __name__== "__main__":
    # Create tracker object
    tracker = EuclideanDistanceTracker()

    capture = cv2.VideoCapture("highway.mp4")

    # Object detection
    object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 40)
    # object_detector = cv2.createBackgroundSubtractorKNN()

    while True:
        ret, frame = capture.read()
        height, width, _ = frame.shape

        # Extract region of interest
        roi = frame[340:720, 500:800]

        # 1. Object Detection
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(contour)
            if area > 100:
                # cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)

                detections.append([x, y, w, h])

        # 2. Object Tracking
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("roi", roi)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(30)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
