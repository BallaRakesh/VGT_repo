import cv2
import json
import numpy as np

def draw_bboxes(image_path, image_id, annotations, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Iterate over annotations
    for ann in annotations:
        if ann["image_id"] == image_id:
            x, y, w, h = map(int, ann["bbox"])
            score = ann["score"]
            
            # Draw bounding box
            color = (0, 255, 0)  # Green color
            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(image, (x, y), (x + h, y + w), color, 2)
            
            # Put confidence score
            cv2.putText(image, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save and display the output image
    cv2.imwrite(output_path, image)
    print(f"Image saved at {output_path}")

res_json = '/home/ng6309/datascience/rakesh/VGT/outputs/sample_vgt_data/model_output/inference/coco_instances_results.json'
with open(res_json, 'r') as r_j:
    annotations = json.load(r_j)
print(annotations[0])

image_path = "/home/ng6309/datascience/rakesh/VGT/outputs/sample_vgt_data/images/Bill_Of_Landing_5_page_2.png"

draw_bboxes(image_path, 0, annotations, "output.jpg")




exit('DONEEEEEEEEE')









import cv2

# Load the image
image_path = "/home/ng6309/datascience/rakesh/VGT/outputs/sample_vgt_data/images/Bill_Of_Landing_5_page_2.png"
image = cv2.imread(image_path)



def denormalize(points: list, width: int, height: int) -> list:
    val = 1000  # Scaling factor used in normalization

    # Extract values (assuming [x1, y1, w, h])
    x1, y1, w, h = points

    # Reverse the normalization
    x1 = int((x1 / val) * width)
    y1 = int((y1 / val) * height)
    w = int((w / val) * width)
    h = int((h / val) * height)

    # Convert to [x_min, y_min, x_max, y_max]
    x_max = x1 + w
    y_max = y1 + h

    return [x1, y1, x_max, y_max]

# Check if image is loaded correctly
if image is None:
    raise FileNotFoundError(f"Error: Could not load image at {image_path}")

# Get image dimensions
height, width, _ = image.shape


# Image dimensions

denormalized_bbox = [
                627.9992400000001,
                662.0004094999999,
                144.00059199999998,
                39.999737
            ]

# Get original pixel coordinates
# denormalized_bbox = denormalize(normalized_bbox, width, height)
# print(denormalized_bbox)

# x2, y2 = x1 + h, y1 + w
# cv2.rectangle(
#     image, 
#     (int(denormalized_bbox[0]), int(denormalized_bbox[1])), 
#     (int(denormalized_bbox[0]) + int(denormalized_bbox[3]), int(denormalized_bbox[1]) + int(denormalized_bbox[2])), 
#     (0, 255, 0), 
#     2
# )

# cv2.rectangle(
#     image, 
#     (int(denormalized_bbox[0]), int(denormalized_bbox[1])), 
#     (int(denormalized_bbox[2]), int(denormalized_bbox[3])), 
#     (0, 255, 0), 
#     2
# )

# x2, y2 = x1 + w, y1 + h
cv2.rectangle(
    image, 
    (int(denormalized_bbox[0]), int(denormalized_bbox[1])), 
    (int(denormalized_bbox[0]) + int(denormalized_bbox[2]), int(denormalized_bbox[1]) + int(denormalized_bbox[3])), 
    (0, 255, 0), 
    2
)

# cv2.rectangle(image, (denormalized_bbox[0], denormalized_bbox[1]), (denormalized_bbox[2], denormalized_bbox[3]), (0, 255, 0), 2)
cv2.imwrite("output.jpg", image)

exit('OK')
# Normalized bounding box coordinates (x_min, y_min, x_max, y_max)

# # If the bbox values are in the range [0, 1], multiply with width & height
# x_min = int(normalized_bbox[0] * width if normalized_bbox[0] < 1 else normalized_bbox[0])
# y_min = int(normalized_bbox[1] * height if normalized_bbox[1] < 1 else normalized_bbox[1])
# x_max = int(normalized_bbox[2] * width if normalized_bbox[2] < 1 else normalized_bbox[2])
# y_max = int(normalized_bbox[3] * height if normalized_bbox[3] < 1 else normalized_bbox[3])
# print((x_min, y_min), (x_max, y_max))
# Draw the bounding box (BGR color: Green, Thickness: 2)
cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Optionally, save the image
cv2.imwrite("output.jpg", image)
