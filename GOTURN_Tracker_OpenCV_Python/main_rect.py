import cv2
import sys
import time

def compare_opencv_version(major, minor, subminor):
    (current_major, current_minor, current_subminor) = cv2.__version__.split(".")

    current_major = int(current_major)
    current_minor = int(current_minor)
    current_subminor = int(current_subminor)

    if major >= 0:
        if major > current_major:
            return -1
        elif major < current_major:
            return 1

    if minor >= 0:
        if minor > current_minor:
            return -1
        elif minor < current_minor:
            return 1

    if subminor >= 0:
        if subminor > current_subminor:
            return -1
        elif subminor < current_subminor:
            return 1

    return 0

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

    # tracker_type = TRACKER_GOTURN
    tracker_type = TRACKER_KCF

    # if int(minor_ver) < 3:
    if compare_opencv_version(-1, 3, -1) < 0:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == TRACKER_BOOSTING:
            if compare_opencv_version(4, 5, 1) < 0:
                tracker = cv2.TrackerBoosting_create()
            else:
                tracker = cv2.legacy.upgradeTrackingAPI(cv2.legacy.TrackerBoosting_create())

        if tracker_type == TRACKER_MIL:
            tracker = cv2.TrackerMIL_create()

        if tracker_type == TRACKER_KCF:
            tracker = cv2.TrackerKCF_create()

        if tracker_type == TRACKER_TLD:
            if compare_opencv_version(4, 5, 1) < 0:
                tracker = cv2.TrackerTLD_create()
            else:
                tracker = cv2.legacy.upgradeTrackingAPI(cv2.legacy.TrackerTLD_create())

        if tracker_type == TRACKER_MEDIANFLOW:
            if compare_opencv_version(4, 5, 1) < 0:
                tracker = cv2.TrackerMedianFlow_create()
            else:
                tracker = cv2.legacy.upgradeTrackingAPI(cv2.legacy.TrackerMedianFlow_create())

        if tracker_type == TRACKER_GOTURN:
            tracker = cv2.TrackerGOTURN_create()

        if tracker_type == TRACKER_MOSSE:
            if compare_opencv_version(4, 5, 1) < 0:
                tracker = cv2.TrackerMOSSE_create()
            else:
                tracker = cv2.legacy.upgradeTrackingAPI(cv2.legacy.TrackerMOSSE_create())

        if tracker_type == TRACKER_CSRT:
            tracker = cv2.TrackerCSRT_create()

    # Open video file
    capture = cv2.VideoCapture("../Test_Video_Files/cars.mp4")

    if not capture.isOpened():
        print("Cannot open video file")
        sys.exit()

    ok, frame = capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    # Initialize calculating FPS
    start = time.time_ns()
    frame_count = 0
    fps = -1

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

        # Increase frame count
        frame_count += 1

        # Update tracker
        ok, bounding_box = tracker.update(frame)

        # Calculate Frames per second (FPS)
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bounding_box[0]), int(bounding_box[1]))
            p2 = (int(bounding_box[0] + bounding_box[2]), int(bounding_box[1] + bounding_box[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break