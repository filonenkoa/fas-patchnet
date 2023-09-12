"""
Prepare a dataset to be used in the training/validation process:
1. Crop the faces from full images if needed.
2. Generate .csv file with file paths and classes for the dataset.
"""
import argparse
from pathlib import Path
import sys
from typing import List, Tuple
import cv2
import csv
from loguru import logger
from tqdm.auto import tqdm

sys.path.append(Path(__file__).resolve().parent.as_posix())
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())
from preprocessor import PreprocessorBoxExpansion
from utils.misc import get_all_file_paths
from containers import Rect2i


def read_agruments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dataset preparation')
    parser.add_argument('--input_folder', type=Path, required=True,
                        help='Path to the folder with input data')
    parser.add_argument('--output_folder', type=Path, required=False,
                        help='Path to the folder with output data (markup + images)')
    parser.add_argument('--output_file', type=str, default="markup.csv",
                        help='Name of the output markup file')
    parser.add_argument('--spoofing_names', nargs='+', default=['spoofing', 'spoof', 'print', 'replay'],
                        help='Names of subdirectories contents of which should have a "spoofing" class')
    parser.add_argument('--crop', action='store_true',
                        help='Should crop?')
    parser.add_argument('--markup_format', type=str, default="insightface",
                        help='The format of faces markup')
    args = parser.parse_args()
    return args


def decode_markup_insightface(markup_text: list) -> Rect2i:
    markup_text[0] = markup_text[0].replace("\n", "")
    markup_text[1] = markup_text[1].replace("\n", "")
    left, top = int(markup_text[0].split(" ")[0]), int(markup_text[0].split(" ")[1])
    right, bottom = int(markup_text[1].split(" ")[0]), int(markup_text[1].split(" ")[1])
    
    bbox = Rect2i(left, top, right - left, bottom - top)
    return bbox


def read_markup(image_path: Path, markup_format: str) -> Rect2i:
    markup = None
    if markup_format == "insightface":
        markup_file = image_path.with_name(f"{image_path.stem}.txt")
        with open(markup_file, "r") as f:
            markup_text = f.readlines()
            markup = decode_markup_insightface(markup_text)
    else:
        raise NotImplementedError
    return markup


if __name__ == "__main__":
    args = read_agruments()
    preprocessor = None
    if args.crop:
        preprocessor = PreprocessorBoxExpansion()
    if args.output_folder is None:
        args.output_folder = args.input_folder
    else:
        args.output_folder.mkdir(parents=True, exist_ok=True)
    markup_path = args.output_folder / args.output_file
    
    image_files = get_all_file_paths(args.input_folder)
    logger.info(f"Found {len(image_files)} image files")
    classes = []
    markup: List[Tuple[str, int]] = []
    for image_path in tqdm(image_files):
        residual_image_path = Path( image_path.as_posix().replace(args.input_folder.as_posix() + "/", "") )
        class_id = 1 # not spoofing
        for spoof_folder in args.spoofing_names:
            if spoof_folder in residual_image_path.as_posix():
                class_id = 0 # spoofing
        if args.crop and preprocessor is not None:
            image = cv2.imread(image_path.as_posix())
            bounding_box = read_markup(image_path, args.markup_format)
            image = preprocessor(image, bounding_box)
            residual_image_path = residual_image_path.with_suffix(".jpg")
            output_image_path = Path(args.output_folder, residual_image_path)
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_image_path.as_posix(), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        markup.append( (residual_image_path.as_posix(), class_id) )
    with open(markup_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(markup)