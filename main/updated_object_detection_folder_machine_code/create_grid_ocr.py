import os
import pickle
import numpy as np
import pdfplumber
import argparse
from transformers import AutoTokenizer


def return_word_grid(pdf_path):
    """Return the word information from a PDF file using pdfplumber.

    Parameters
    ----------
    pdf_path : str
        The path to the input PDF file.

    Returns
    -------
    List
        Returns a list of shape (num_pages, num_words, 8)
    """
    pdf = pdfplumber.open(pdf_path)

    word_data = list()
    for page in pdf.pages:
        # extracts words and their bounding boxes
        word_data.append(page.extract_words())

    return word_data


def tokenize(tokenizer, text_body):
    """Tokenize the input text using the provided tokenizer.

    Parameters
    ----------
    tokenizer : HuggingFace Tokenizer
        The tokenizer to be used.
    text_body : List
        List of text to be tokenized.

    Returns
    -------
    np.array
        Return the tokenized input_ids.
    """
    # tokenize entire list of words
    tokenized_inputs = tokenizer.batch_encode_plus(
        text_body,
        return_token_type_ids=False,
        return_attention_mask=False,
        add_special_tokens=False
    )
    return tokenized_inputs["input_ids"]


def readjust_bbox_coords(bounding_box, tokens):
    """Readjust the bounding box coordinates based on the tokenized input.

    Parameters
    ----------
    bounding_box : List
        List of bounding box coordinates in the format (x, y, width, height).
    tokens : List
        List of input_ids from the tokenizer.

    Returns
    -------
    List
        Returns a list of the adjusted bounding box coordinates.
    """
    adjusted_boxes = []
    for box, _id in zip(bounding_box, tokens):
        if len(_id) > 1:
            # Adjust the width and x-coordinate for each part
            new_width = box[2] / len(_id)
            for i in range(len(_id)):
                adjusted_boxes.append(
                    (box[0] + i * new_width, box[1], new_width, box[3])
                )
        else:
            adjusted_boxes.append((box[0], box[1], box[2], box[3]))
    return adjusted_boxes


def create_grid_dict(tokenizer, page_data):
    """Create a dictionary with the tokenized input,
    bounding box coordinates, and text.

    Parameters
    ----------
    tokenizer : HuggingFace Tokenizer
        The tokenizer to be used.
    page_data : List
        List of word information from pdfplumber.

    Returns
    -------
    Dict
        Returns a dictionary with the tokenized input,
        bounding box coordinates, and text.
    """
    grid = {
        "input_ids": [],
        "bbox_subword_list": [],
        "texts": [],
        "bbox_texts_list": []
    }
    #{"word": "Consignor", "left": 115, "top": 135, "width": 121, "height": 24, "x1": 115, "y1": 135, "x2": 236, "y2": 159}
    for ele in page_data:
        grid["texts"].append(ele["word"])

        # since expected bbox format is (x,y,width,height)
        grid["bbox_texts_list"].append(
            (ele["x1"],
             ele["top"],
             ele["x2"]-ele["x1"],
             ele["y2"]-ele["top"]))

    input_ids = tokenize(tokenizer, grid["texts"])

    # flatten the input_ids
    grid["input_ids"] = np.concatenate(input_ids)

    grid["bbox_subword_list"] = np.array(
        readjust_bbox_coords(
            grid["bbox_texts_list"],
            input_ids
            )
        )

    grid["bbox_texts_list"] = np.array(grid["bbox_texts_list"])

    return grid


def save_pkl_file(grid, output_dir, output_file, model="doclaynet"):
    """Save the grid dictionary as a pickle file.

    Parameters
    ----------
    grid : Dict
        The grid dictionary to be saved.
    output_dir : str
        The path to the output folder.
    output_file : str
        The name of the output file.
    model : str, optional
        Model that will be used by VGT further
        , by default "doclaynet"

    Returns
    -------
    None
    """
    if model == "doclaynet" or model == "publaynet":
        extension = "pdf.pkl"
    else:
        extension = "pkl"

    pkl_save_location = os.path.join(
        output_dir,
        f'{output_file}.{extension}')

    with open(pkl_save_location, 'wb') as handle:
        pickle.dump(grid, handle)


def select_tokenizer(tokenizer):
    """Select the tokenizer to be used.

    Parameters
    ----------
    tokenizer : str
        The name of the tokenizer to be used.

    Returns
    -------
    tokenizer: HuggingFace Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer

import os
import pickle
import glob
import json
import ast

def save_pkl_file_(grid, output_dir, output_file, model="doclaynet"):
    """Save the grid dictionary as a pickle file."""
    extension = "pdf.pkl" if model in ["doclaynet", "publaynet"] else "pkl"
    pkl_save_location = os.path.join(output_dir, f'{output_file}.{extension}')
    
    with open(pkl_save_location, 'wb') as handle:
        pickle.dump(grid, handle)

def process_pdfs(root_ocr_folder, output_dir, tokenizer_model="google-bert/bert-base-uncased", model="doclaynet"):
    """Iterate over PDFs in the root folder and process them."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tokenizer = select_tokenizer(tokenizer_model)
    # pdf_files = glob.glob(os.path.join(root_pdf_folder, "*.pdf"))
    page = 0
    for ocr_file in os.listdir(root_ocr_folder):
        print(ocr_file)
        with open(os.path.join(root_ocr_folder, ocr_file), 'r', encoding='utf-8') as f:
            # wc_content = json.load(f).get('word_coordinates', {})
            wc_content = f.read()
            wc_content = ast.literal_eval(wc_content).get('word_coordinates', {})
            # print(wc_content)
        f.close()
        if len(wc_content):
            file_name = os.path.splitext(os.path.basename(ocr_file))[0]  # Extract file name without extension
            file_name = file_name.removesuffix("_text")
            # word_grid = return_word_grid(pdf_path)
            grid = create_grid_dict(tokenizer, wc_content)
            # save_pkl_file_(grid, output_dir, f"{file_name}_page_{page}", model)
            save_pkl_file_(grid, output_dir, f"{file_name}", model)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--pdf",
    #                     required=True,
    #                     help="Path to the PDF file")
    # parser.add_argument("--output",
    #                     required=False,
    #                     default="grid",
    #                     help="Path to the output folder")
    # parser.add_argument("--tokenizer",
    #                     required=False,
    #                     default="google-bert/bert-base-uncased",
    #                     help="Tokenizer to be used")
    # parser.add_argument("--model",
    #                     required=False,
    #                     default="doclaynet",
    #                     help="VGT fine-tuned model to use")

    # args = parser.parse_args()

    # # Create the output folder if it doesn't exist
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)

    # word_grid = return_word_grid(args.pdf)
    # tokenizer = select_tokenizer(args.tokenizer)

    # for page in range(len(word_grid)):

    #     grid = create_grid_dict(tokenizer, word_grid[page])
    #     # save_pkl_file(grid, args.output, f"page_{page}", page, args.model)
    #     save_pkl_file(grid, args.output, f"page_{page}", args.model)

    # Example usage
    root_ocr_folder = "/home/ng6309/datascience/rakesh/VGT/outputs/sample_vgt_data/OCR"
    output_dir = "/home/ng6309/datascience/rakesh/VGT/outputs/sample_vgt_data/sample_grid_files"
    process_pdfs(root_ocr_folder, output_dir)
