import cv2

# Given image path
image_path = "/home/ng6309/datascience/rakesh/VGT/outputs/outputs/downloads/extracted/samples/small_dataset/val/images/ec2330f403b735c37ebeb6997bc351e05539bacdbb91e302cf77a57a5b563410.png"

# Load the image
image = cv2.imread(image_path)

# List of bounding boxes (Rectangles)
rectangles = [
    (73, 39, 82, 75), (422, 59, 538, 68), (110, 70, 115, 105), (224, 102, 355, 110),
    (99, 114, 167, 122), (99, 126, 167, 134), (83, 142, 85, 202), (99, 184, 204, 192),
    (99, 196, 146, 204), (161, 196, 307, 204), (99, 208, 188, 216), (224, 208, 397, 216),
    (99, 220, 261, 228), (99, 232, 225, 240), (78, 244, 83, 252), (78, 265, 214, 273),
    (78, 277, 141, 285), (99, 289, 130, 297), (224, 289, 402, 297), (92, 303, 146, 532),
    (224, 301, 397, 309), (78, 337, 83, 345), (268, 413, 276, 432), (224, 440, 397, 448),
    (77, 458, 81, 550), (224, 527, 232, 573), (78, 591, 177, 599), (80, 601, 85, 678),
    (224, 625, 355, 633), (99, 637, 167, 645), (99, 649, 167, 657), (78, 682, 366, 690),
    (78, 694, 156, 702), (99, 706, 282, 714), (99, 718, 146, 726), (161, 718, 307, 726),
    (99, 730, 188, 738), (203, 730, 370, 738), (99, 742, 183, 750), (78, 754, 83, 762),
    (290, 779, 305, 788)
]

# Draw rectangles on the image
for (left, top, right, bottom) in rectangles:
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)  # Green color with thickness 2

# Save or show the image
output_path = "output_image.jpg"
cv2.imwrite(output_path, image)
