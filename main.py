#!/usr/bin/env python
"""
WildVision Wildlife Classifier - Main Script
Handles both image collection and model training with smart skipping.
"""

# Standard Imports
import os
import argparse
from datetime import datetime

# Local Imports
from src.data_loader import download_species_images, prepare_data
from src.train import train_and_evaluate
from src.utils import create_directory_structure
from config import TARGET_SPECIES, MODEL_CONFIG


# NEW: Command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='WildVision Training Pipeline')
    parser.add_argument('--redownload', action='store_true',
                        help='Force re-download of all images')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip image download phase entirely')
    parser.add_argument('--max-images', type=int, default=200,
                        help='Max images per species (default: 200)')
    return parser.parse_args()


def main():
    args = parse_args()
    create_directory_structure()

    # NEW: Smart Download Logic
    download_flag = True
    metadata_file = "data/species_metadata.csv"

    if args.skip_download:
        print("[Status] Skipping download as requested")
        download_flag = False
    elif os.path.exists(metadata_file) and not args.redownload:
        print("[Status] Using existing images (add --redownload to refresh)")
        download_flag = False

    if download_flag:
        print(f"[{datetime.now()}] Downloading images...")
        download_species_images(
            TARGET_SPECIES,
            max_images=args.max_images
        )

    # Proceed to training
    print(f"[{datetime.now()}] Starting model training...")
    train_and_evaluate()


if __name__ == "__main__":
    main()