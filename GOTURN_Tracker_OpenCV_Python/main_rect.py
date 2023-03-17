import cv2
import sys

if __name__ == "__main__":

    # OpenCV tracker types
    TRACKER_BOOSTING = "BOOSTING"
    TRACKER_MIL = "MIL"
    TRACKER_KCF = "KCF"
    TRACKER_TLD = "TLD"
    TRACKER_MEDIANFLOW = "MEDIANFLOW"
    TRACKER_GOTURN = "GOTURN"
    TRACKER_MOSSE = "MOSSE"
    TRACKER_CSRT = "CSRT"

    tracker_type = TRACKER_GOTURN

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == TRACKER_BOOSTING:
            tracker = cv2.TrackerBoosting_create()

        if tracker_type == TRACKER_MIL:
            tracker = cv2.TrackerMIL_create()

        if tracker_type == TRACKER_KCF:
            tracker = cv2.TrackerKCF_create()

        if tracker_type == TRACKER_TLD:
            tracker = cv2.TrackerTLD_create()

        if tracker_type == TRACKER_MEDIANFLOW:
            tracker = cv2.TrackerMedianFlow_create()

        if tracker_type == TRACKER_GOTURN:
            tracker = cv2.TrackerGOTURN_create()

        if tracker_type == TRACKER_MOSSE:
            tracker = cv2.TrackerMOSSE_create()

        if tracker_type == TRACKER_CSRT:
            tracker = cv2.TrackerCSRT_create()

    # Open video file
    capture = cv2.VideoCapture("../Test_Video_Files/highway.mp4")

    if not capture.isOpened():
        print("Cannot open video file")
        sys.exit()

    ok, frame = capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # Define an initial bounding box
    bounding_box = (10, 10, 100, 100)

    # Select a bounding box
    bounding_box = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bounding_box)

    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bounding_box = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break