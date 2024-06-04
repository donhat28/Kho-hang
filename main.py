from inference import get_model
import supervision as sv
import cv2
import numpy as np
from detector import ColorDetector

model = get_model(model_id="khohangv2/1")
color_detector = ColorDetector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error")
    exit()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Error")
        break

    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)

    confidences = detections.confidence
    masks = detections.mask

    for i in range(len(confidences)):
        mask = masks[i]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                masked_image = cv2.bitwise_and(frame, frame, mask=mask.astype(np.uint8))
                masked_hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

                detect_color = color_detector.detect_color_in_mask(masked_hsv)

                if detect_color == 'orange':
                    print(1)
                elif detect_color == 'yellow':
                    print(2)
                elif detect_color == 'green':
                    print(3)
                else:
                    print(0)
            else:
                print(0)
        else:
            print(0)

    bbox_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_img = bbox_annotator.annotate(scene=frame, detections=detections)
    annotated_label = label_annotator.annotate(scene=annotated_img, detections=detections)

    cv2.imshow("Camera", annotated_img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()