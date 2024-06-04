from inference import get_model
import supervision as sv
import cv2
import numpy as np

def define_color_ranges():
    orange_lower = np.array([10, 100, 20])
    orange_upper = np.array([25, 255, 255])

    yellow_lower = np.array([30, 100, 20])
    yellow_upper = np.array([60, 255, 255])

    green_lower = np.array([90, 100, 20])
    green_upper = np.array([120, 255, 255])

    return {
        'orange': (orange_lower, orange_upper),
        'yellow': (yellow_lower, yellow_upper),
        'green': (green_lower, green_upper)
    }

def detect_color_in_mask(masked_hsv, color_ranges):
    for color, (lower, upper) in color_ranges.items():
        color_mask = cv2.inRange(masked_hsv, lower, upper)

        if np.any(color_mask > 0):
            return color
    return None

img_file = "test_image_2.jpg"
image = cv2.imread(img_file)

model = get_model(model_id="khohangv2/1")

results = model.infer(image)[0]

detections = sv.Detections.from_inference(results)

# print(detections)

# xyxy = detections.xyxy
class_names = detections.data['class_name']
confidences = detections.confidence
masks = detections.mask

labels = []

for i in range(len(confidences)):
    # x1, y1, x2, y2 =xyxy[i]
    # x = x1
    # y = y1
    # width = x2 - x1
    # height = y2 - y1
    # label = f"{class_names[i]} {confidences[i]:.2f} (x: {x}, y: {y}, width: {width}, height: {height})"
    label = f"{class_names[i]} {confidences[i]:.2f}"
    labels.append(label)

    mask = masks[i]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
            masked_hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

            color_ranges = define_color_ranges()
            detect_color = detect_color_in_mask(masked_hsv, color_ranges)

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
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

annotated_img = mask_annotator.annotate(scene=image, detections=detections)
annotated_label = label_annotator.annotate(scene=annotated_img, detections=detections, labels=labels)

# print(labels)
sv.plot_image(annotated_img)

