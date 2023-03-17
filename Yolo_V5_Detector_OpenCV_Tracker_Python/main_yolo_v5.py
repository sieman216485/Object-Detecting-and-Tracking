import cv2
import time
import sys
import numpy as np

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

COLORS = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

def build_model(is_cuda):
    net = cv2.dnn.readNet("yolov5s.onnx")

    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB = True, crop = False)
    net.setInput(blob)
    output = net.forward()
    # output = net.forward(net.getUnconnectedOutLayersNames())

    return output

def wrap_detection(image, detections):
    class_ids = []
    confidences = []
    boxes = []

    rows = detections.shape[0]

    image_width, image_height, _ = image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = detections[r]
        confidence = row[4]

        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            _, _, _, max_index = cv2.minMaxLoc(classes_scores)
            class_id = max_index[1]

            if (classes_scores[class_id] >= SCORE_THRESHOLD):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()

                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, SCORE_THRESHOLD)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

# Main function
if __name__== "__main__":
    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

    net = build_model(is_cuda)

    # Read class list
    class_list = []
    with open("classes.txt", "r") as classes_file:
        class_list = [class_name.strip() for class_name in classes_file.readlines()]

    capture = cv2.VideoCapture("../Test_Video_Files/road.mp4")

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

    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        if frame is None:
            break

        image = format_yolov5(frame)
        detections = detect(image, net)

        class_ids, confidences, boxes = wrap_detection(image, detections[0])

        frame_count += 1

        for (class_id, confidence, box) in zip(class_ids, confidences, boxes):
            color = COLORS[int(class_id) % len(COLORS)]
            label = "%s (%d%%)" % (class_list[class_id], int(confidence * 100))

            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(30)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
