import argparse

import cv2

from ditod import add_vit_config

import torch

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from ditod.VGTTrainer import DefaultPredictor
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import cv2
import torch

import cv2
import torch

import cv2
import torch
import os
import numpy as np
from detectron2.structures import Instances



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






def draw_detections(image_path, output_folder, predictions, classes_):
    """
    Draws bounding boxes and class labels on an image based on Detectron2 predictions.

    Parameters:
    - image_path (str): Path to the input image.
    - output_folder (str): Directory where the output image will be saved.
    - predictions (Instances): Detectron2 Instances object containing 'pred_boxes', 'scores', and 'pred_classes'.
    """
    # Load the image
    print(image_path)
    # print('???????????????????')
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    # Ensure output folder exists
    # os.makedirs(output_folder, exist_ok=True)

    # Extract bounding boxes, classes, and scores
    if predictions.has("pred_boxes"):
        boxes = predictions.pred_boxes.tensor.cpu().numpy()  # Convert to NumPy array
    else:
        boxes = np.array([])

    if predictions.has("scores"):
        scores = predictions.scores.cpu().numpy()
    else:
        scores = np.array([])

    if predictions.has("pred_classes"):
        classes = predictions.pred_classes.cpu().numpy()
    else:
        classes = np.array([])

    # Define colors for bounding boxes
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red (Can be extended)

    # Draw bounding boxes and labels
    for i, box in enumerate(boxes):
        # if scores[i] > 0.4:
        x1, y1, x2, y2 = map(int, box)  # Convert width-height format to x2, y2
        # x1, y1, w, h = map(int, box)  # Convert width-height format to x2, y2
        # denormalized_bbox = denormalize([x1, y1, x2, y2], height, width)
        # x1, y1, x2, y2 = denormalized_bbox[0], denormalized_bbox[1], denormalized_bbox[2], denormalized_bbox[3]
        # x2, y2 = x1 + w, y1 + h
        color = colors[i % len(colors)]  # Cycle through colors
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"{classes_[classes[i]]}: {scores[i]:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the output image
    # output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_folder, image)
    print(f"Saved annotated image: {output_folder}")



model_info = {
    'D4LA' : {
        'config_file': '/home/ng6309/datascience/rakesh/VGT/object_detection/Configs/cascade/D4LA_VGT_cascade_PTM.yaml',
        'model_path' : '/home/ng6309/datascience/rakesh/VGT/finetuned_models/D4LA_VGT_model.pth',
        'thing_classes': ["DocTitle","ParaTitle","ParaText","ListText","RegionTitle", "Date", "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText", "RegionKV", "Regionlist", "Abstract", "Author", "TableName", "Table", "Figure", "FigureName", "Equation", "Reference", "Footnote", "PageHeader", "PageFooter", "Number", "Catalog", "PageNumber"]
    },
    'publaynet' : {
        'config_file': '/home/ng6309/datascience/rakesh/VGT/object_detection/Configs/cascade/publaynet_VGT_cascade_PTM.yaml',
        'model_path' : '/home/ng6309/datascience/rakesh/VGT/finetuned_models/publaynet_VGT_model.pth',
        'thing_classes': ["text","title","list","table","figure"]
    },
    'doclaynet' : {
        'config_file': '/home/ng6309/datascience/rakesh/VGT/object_detection/Configs/cascade/doclaynet_VGT_cascade_PTM.yaml',
        'model_path' : '/home/ng6309/datascience/rakesh/VGT/finetuned_models/doclaynet_VGT_model.pth',
        'thing_classes': ["Caption","Footnote","Formula","List-item","Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]
    },
    'docbank' : {
        'config_file': '/home/ng6309/datascience/rakesh/VGT/object_detection/Configs/cascade/docbank_VGT_cascade_PTM.yaml',
        'model_path' : '/home/ng6309/datascience/rakesh/VGT/finetuned_models/docbank_VGT_model.pth',
        'thing_classes': ["abstract","author","caption","date","equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"]
    }
}

def main(image_path, model_name, grid_folder, output_folder):
    parser = argparse.ArgumentParser(description="Detectron2 inference script")
    image_name_with_extension = os.path.basename(image_path)
    image_name_only = os.path.splitext(image_name_with_extension)[0]
    
    grid_path = os.path.join(grid_folder, image_name_only) + ".pdf.pkl"
    
    output_file_name = os.path.join(output_folder, image_name_with_extension)
    
    config_file = model_info.get(model_name).get('config_file')
    model_path = model_info.get(model_name).get('model_path')
    thing_classes = model_info.get(model_name).get('thing_classes')
    
    '''parser.add_argument(
        "--image_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--grid_root",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--image_name",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_root",
        help="Name of the output visualization file.",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="Path to input image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )'''
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    
    '''if args.dataset in ('D4LA', 'doclaynet', 'publaynet', 'docbank'):
        image_path = args.image_root + args.image_name + ".png"
    else:
        image_path = args.image_root + args.image_name + ".jpg"
    
    if args.dataset == 'publaynet':
        grid_path = args.grid_root + args.image_name + ".pdf.pkl"
    elif args.dataset == 'docbank':
        # grid_path = args.grid_root + args.image_name + ".pkl"
        grid_path = args.grid_root + args.image_name + ".pdf.pkl"
    elif args.dataset == 'D4LA':
        # grid_path = args.grid_root + args.image_name + ".pkl"
        grid_path = args.grid_root + args.image_name + ".pdf.pkl"
    elif args.dataset == 'doclaynet':
        grid_path = args.grid_root + args.image_name + ".pdf.pkl"
        
    output_file_name = args.output_root + args.image_name + ".jpg"'''
    
    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(config_file)
    
    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)
    
    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = model_path
    # Step 4: define model
    predictor = DefaultPredictor(cfg)
    print(image_path)
    # Step 5: run inference
    img = cv2.imread(image_path)
    
    '''md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if args.dataset == 'publaynet':
        md.set(thing_classes=["text","title","list","table","figure"])
    elif args.dataset == 'docbank':
        md.set(thing_classes=["abstract","author","caption","date","equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"])
    elif args.dataset == 'D4LA':
        md.set(thing_classes=["DocTitle","ParaTitle","ParaText","ListText","RegionTitle", "Date", "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText", "RegionKV", "Regionlist", "Abstract", "Author", "TableName", "Table", "Figure", "FigureName", "Equation", "Reference", "Footnote", "PageHeader", "PageFooter", "Number", "Catalog", "PageNumber"])
    elif args.dataset == 'doclaynet':
        md.set(thing_classes=["Caption","Footnote","Formula","List-item","Page-footer", "Page-header", "Picture", "Section-header", "Table", "Text", "Title"])

    # Set the mapping of class labels
    thing_classes = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", 
                    "Picture", "Section-header", "Table", "Text", "Title"]
    
    thing_classes = ["text","title","list","table","figure"]
    thing_classes = ["abstract","author","caption","date","equation", "figure", "footer", "list", "paragraph", "reference", "section", "table", "title"]
    thing_classes = ["DocTitle","ParaTitle","ParaText","ListText","RegionTitle", "Date", "LetterHead", "LetterDear", "LetterSign", "Question", "OtherText", "RegionKV", "Regionlist", "Abstract", "Author", "TableName", "Table", "Figure", "FigureName", "Equation", "Reference", "Footnote", "PageHeader", "PageFooter", "Number", "Catalog", "PageNumber"]
    '''
    
    output = predictor(img, grid_path)["instances"]
    print(output.scores.cpu().numpy())
    draw_detections(image_path, output_file_name, output, thing_classes)

    # # import ipdb;ipdb.set_trace()
    # v = Visualizer(img[:, :, ::-1],
    #                 md,
    #                 scale=1.0,
    #                 instance_mode=ColorMode.SEGMENTATION)
    # result = v.draw_instance_predictions(output.to("cpu"))
    # result_image = result.get_image()[:, :, ::-1]

    # # step 6: save
    # cv2.imwrite(output_file_name, result_image)
import os
if __name__ == '__main__':
    output_folder = '/home/ng6309/datascience/rakesh/VGT/outputs/itf_doc_testing/output'
    grid_folder = '/home/ng6309/datascience/rakesh/VGT/outputs/itf_doc_testing/sample_grid_files'
    images_folder = '/home/ng6309/datascience/rakesh/VGT/outputs/itf_doc_testing/dit_test'
    model_name = 'docbank' #'publaynet' #'D4LA' #'doclaynet'         #D4LA publaynet docbank
    
    os.makedirs(output_folder, exist_ok=True)
    for doc_type in os.listdir(images_folder):
        images_root = os.path.join(images_folder, doc_type)
        doc_output_fol = os.path.join(output_folder, doc_type)
        os.makedirs(doc_output_fol, exist_ok=True)
        for img_path in os.listdir(images_root):
            main(os.path.join(images_root, img_path), model_name, grid_folder, doc_output_fol)

