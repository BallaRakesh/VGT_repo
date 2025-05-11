import json
import os
import random
from pathlib import Path
import shutil

categories1 = {
        "Caption": 0,
        "Footnote": 1,
        "Formula": 2,
        "List-item": 3,
        "Page-footer": 4,
        "Page-header": 5,
        "Picture": 6,
        "Section-header": 7,
        "Table": 8,
        "Text": 9,
        "Title": 10
    }

categories2 = {
    "transaction_date": 0,
    "bill_of_lading_number": 1,
    "bill_of_lading_issue_date": 2,
    "shipper_name": 3,
    "shipper_address": 4,
    "consignee_name": 5,
    "consignee_address": 6,
    "notify_party_name": 7,
    "notify_party_address": 8,
    "carrier_name": 9,
    "carrier_country": 10,
    "agent_name": 11,
    "agent_country": 12,
    "pre_carriage_by": 13,
    "vessel_name": 14,
    "voyage_number": 15,
    "container_number": 16,
    "place_of_receipt": 17,
    "port_of_loading": 18,
    "port_of_discharge": 19,
    "final_destination": 20,
    "shipped_onboard_date": 21,
    "bol_original_number": 22,
    "bol_original_or_copy": 23,
    "freight_collect_at": 24,
    "freight_collect_or_prepaid": 25,
    "place_of_issue": 26,
    "signed_by_carrier": 27,
    "signed_By_agent": 28,
    "lc_ref_number": 29,
    "lc_date": 30,
    "signature": 31,
    "stamp": 32,
    "net_weight": 33,
    "gross_weight": 34,
    "page_no": 35,
    "country_of_origin": 36,
    "goods_description": 37,
    "goods_quantity": 38,
    "dimension": 39,
    "goods_marks_and_nos": 40,
    "mode_of_transport": 41,
    "measurement": 42,
    "master_name": 43,
    "signed_by_master": 44,
    "total_original_no": 45,
    "stamped_vessel_or_cargo_name": 46,
    "charter": 47,
    "stamped_port_of_loading": 48,
    "stamped_onboard_date": 49,
    "carriage_condition": 50,
    "stamped_shipped_onboard_date": 51,
    "signed_by_master": 52,
    "shipment_date": 53
}


categories1_list =[
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title"
    ]

categories2_list = [
    "transaction_date",
    "bill_of_lading_number",
    "bill_of_lading_issue_date",
    "shipper_name",
    "shipper_address",
    "consignee_name",
    "consignee_address",
    "notify_party_name",
    "notify_party_address",
    "carrier_name",
    "carrier_country",
    "agent_name",
    "agent_country",
    "pre_carriage_by",
    "vessel_name",
    "voyage_number",
    "container_number",
    "place_of_receipt",
    "port_of_loading",
    "port_of_discharge",
    "final_destination",
    "shipped_onboard_date",
    "bol_original_number",
    "bol_original_or_copy",
    "freight_collect_at",
    "freight_collect_or_prepaid",
    "place_of_issue",
    "signed_by_carrier",
    "signed_By_agent",
    "lc_ref_number",
    "lc_date",
    "signature",
    "stamp",
    "net_weight",
    "gross_weight",
    "page_no",
    "country_of_origin",
    "goods_description",
    "goods_quantity",
    "dimension",
    "goods_marks_and_nos",
    "mode_of_transport",
    "measurement",
    "master_name",
    "signed_by_master",
    "total_original_no",
    "stamped_vessel_or_cargo_name",
    "charter",
    "stamped_port_of_loading",
    "stamped_onboard_date",
    "carriage_condition",
    "stamped_shipped_onboard_date",
    "signed_by_master",
    "shipment_date"
]


def create_coco_annotation(annotation_data, annotation_id):
    """Convert single annotation to COCO format"""
    boxes = []
    for form in annotation_data["form"]:
        # box = form["box"]
        box = form["box_line"]
        category = form["category"]
        text = form["text"]
        
        # Convert box coordinates [x, y, width, height]
        boxes.append({
            "id": annotation_id,  # Use the running counter for unique IDs
            "image_id": annotation_data["metadata"]["page_hash"],
            # "image_id": os.path.splitext(os.path.basename(annotation_data["metadata"]["original_filename"]))[0], ##### change1
            "category_id": get_category_id(category),
            "bbox": box,
            "area": box[2] * box[3],  # width * height
            # "segmentation": [[
            #     box[0], 
            #     box[1],  # top-left
            #     box[0] + box[2], 
            #     box[1],  # top-right
            #     box[0] + box[2], 
            #     box[1] + box[3],  # bottom-right
            #     box[0], 
            #     box[1] + box[3]  # bottom-left
            # ]],
            "segmentation": [],
            "iscrowd": 0,
            "precedence": 0  # Default value, adjust if needed
        })
        annotation_id += 1  # Increment the counter
    return boxes, annotation_id

def get_category_id(category_name):
    """Map category names to IDs (0-based indexing)"""

    return categories1.get(category_name, -1)

def create_coco_image_info(annotation_data, image_filename):
    """Create image info in COCO format"""
    metadata = annotation_data["metadata"]
    return {
        "id": metadata["page_hash"],
        # "id": os.path.splitext(os.path.basename(metadata["original_filename"]))[0], ##change2
        "file_name": image_filename,
        # "width": metadata["coco_width"],
        "width": metadata["original_width"],
        # "height": metadata["coco_height"],
        "height": metadata["original_height"],
        "doc_category": metadata["doc_category"],
        "collection": metadata["collection"],
        "doc_name": metadata["original_filename"],
        "page_no": metadata["page_no"]
    }

def generate_coco_json(annotations_dir, images_dir, output_dir, train_split=0.8):
    """Generate COCO format JSON files for training and testing"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize COCO format dictionaries with 0-based indexing
    categories = [
        {"id": i, "name": name} for i, name in enumerate(categories1_list)
    ]
    
    all_annotations = []
    all_images = []
    annotation_id = 0  # Start annotation IDs from 0
    
    # Process all annotation files
    for ann_file in Path(annotations_dir).glob("*.json"):
    # for ann_file in Path(annotations_dir).glob("*.txt"): #change 3
        # file_name = ann_file.stem.removesuffix("_labels")  #change 4
        file_name = ann_file.stem
        print(ann_file)
        print(file_name)
        image_file = Path(images_dir) / f"{file_name}.png"
        
        if not image_file.exists():
            print(f"Warning: Missing image for annotation {ann_file}")
            continue
            
        with open(ann_file) as f:
            ann_data = json.load(f)
        print(image_file.name)
        # Create image info
        image_info = create_coco_image_info(ann_data, image_file.name)
        print(image_info)
        all_images.append(image_info)
        print(ann_data["metadata"]["page_hash"])
        # Create annotations with running counter for unique IDs
        boxes, annotation_id = create_coco_annotation(ann_data, annotation_id)
        all_annotations.extend(boxes)
    
    # Randomly split into train and test
    total_images = len(all_images)
    train_size = int(total_images * train_split)
    
    # Create index mapping to keep annotations with their images
    image_id_to_idx = {img["id"]: i for i, img in enumerate(all_images)}
    
    # Randomly shuffle images
    random.seed(42)  # for reproducibility
    image_indices = list(range(total_images))
    random.shuffle(image_indices)
    
    train_indices = set(image_indices[:train_size])
    test_indices = set(image_indices[train_size:])
    
    # Split images and annotations
    train_images = [img for i, img in enumerate(all_images) if i in train_indices]
    test_images = [img for i, img in enumerate(all_images) if i in test_indices]
    
    train_annotations = [ann for ann in all_annotations 
                        if image_id_to_idx[ann["image_id"]] in train_indices]
    test_annotations = [ann for ann in all_annotations 
                       if image_id_to_idx[ann["image_id"]] in test_indices]
    
    # Create COCO format dictionaries
    train_coco = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }
    
    test_coco = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": categories
    }
    
    # Save JSON files
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_coco, f, indent=2)
        
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_coco, f, indent=2)
    
    print(f"Generated train.json with {len(train_images)} images and {len(train_annotations)} annotations")
    print(f"Generated test.json with {len(test_images)} images and {len(test_annotations)} annotations")

if __name__ == "__main__":
    
    # code validation is done , need to change the path and get the train.json and test.json and retrain it on doclaynet!!
    # Example usage
    annotations_dir = "/home/ng6309/datascience/rakesh/VGT/outputs/outputs/downloads/extracted/samples/small_dataset/test/annotations"
    images_dir = "/home/ng6309/datascience/rakesh/VGT/outputs/outputs/downloads/extracted/samples/small_dataset/val/images"
    output_dir = "/home/ng6309/datascience/rakesh/VGT/outputs/outputs"
    
    generate_coco_json(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        output_dir=output_dir,
        train_split= 1 # 80% train, 20% test split
    )