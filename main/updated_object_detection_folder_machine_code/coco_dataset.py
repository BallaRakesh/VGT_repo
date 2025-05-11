from pathlib import Path
import json
from globox import AnnotationSet, Annotation, BoundingBox
from typing import List, Dict
import glob

def parse_custom_annotation(json_path: Path, image_path: Path) -> Annotation:
    """Parse a single custom JSON annotation file."""
    
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get image dimensions from metadata
    image_width = data['metadata']['original_width']
    image_height = data['metadata']['original_height']
    
    # Create an Annotation object with the image ID and size
    image_id = image_path.name  # Use image filename as ID
    annotation = Annotation(
        image_id=image_id,
        image_size=(image_width, image_height)
    )
    
    # Add each form element as a bounding box
    for form_element in data['form']:
        # Extract bounding box coordinates
        x, y, w, h = form_element['box']
        
        # Create a BoundingBox object
        # Using the 'category' field as the label
        box = BoundingBox.create(
            label=form_element['category'],
            coords=(x, y, w, h),
            box_format=BoxFormat.LTWH  # Left, Top, Width, Height format
        )
        
        # Add the box to the annotation
        annotation.add(box)
    
    return annotation

def convert_to_coco(images_dir: str, annotations_dir: str, output_file: str):
    """Convert custom JSON annotations to COCO format."""
    
    # Convert string paths to Path objects
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    
    # Get all image files
    image_files = list(images_dir.glob('*.png'))
    
    # Create annotations for each image
    annotations = []
    for image_file in image_files:
        # Find corresponding JSON file
        json_file = annotations_dir / f"{image_file.stem}.json"
        
        if json_file.exists():
            try:
                # Parse the annotation
                annotation = parse_custom_annotation(json_file, image_file)
                annotations.append(annotation)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    # Create AnnotationSet
    annotation_set = AnnotationSet(annotations)
    
    # Save as COCO format
    # Using auto_ids=True to automatically generate category and image IDs
    annotation_set.save_coco(output_file, auto_ids=True)
    
    print(f"Successfully converted {len(annotations)} annotations to COCO format")
    print(f"Output saved to: {output_file}")
    
    # Show statistics about the dataset
    annotation_set.show_stats()

if __name__ == "__main__":
    # Example usage
    convert_to_coco(
        images_dir="/home/ng6309/datascience/rakesh/VGT/outputs/outputs/downloads/extracted/samples/small_dataset/val/images",
        annotations_dir="/home/ng6309/datascience/rakesh/VGT/outputs/outputs/downloads/extracted/samples/small_dataset/val/annotations",
        output_file="output.json"
    )