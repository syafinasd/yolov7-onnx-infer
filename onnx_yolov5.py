import numpy as np
import cv2

# Step 1 - Load the model
net = cv2.dnn.readNet('best_5N.onnx')

# Helper function to format the image
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

# Load class names
class_list = []
with open("data.txt", "r") as f:
    class_list = [cname.strip() for cname in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Step 2 - Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    input_image = format_yolov5(frame)  # Making the image square
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    # Step 3 - Unwrap the predictions to get the object detections
    class_ids = []
    confidences = []
    boxes = []

    output_data = predictions[0]
    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    for i in range(len(result_class_ids)):
        box = result_boxes[i]
        class_id = result_class_ids[i]
        confidence = result_confidences[i]
        label = f"{class_list[class_id]}: {confidence:.2f}"
        cv2.rectangle(frame, box, (0, 255, 255), 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the output
    cv2.imshow("YOLOv5 ONNX STREAM", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
