import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import tqdm
import json
# !{sys.executable} -m pip install opencv-python matplotlib
# !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image
import random
import pickle
from segmentAnythingUtils import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"
random.seed(41)

max_dim = 2000
min_dim = 600

sam_checkpoint = "/trunk/shared/cuneiform/CuneiformDating/image_classification/segmentation/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=5,
    pred_iou_thresh=0.94,
    stability_score_thresh=0.90,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=10000,  # Requires open-cv to run post-processing
)

# ids_path = "/trunk/shared/cuneiform/CuneiformDating/image_classification/segmentation/code/temp_results/run_segmentation_again.json"
ids_path = "/trunk/shared/cuneiform/full_data/all_ids.json'

with open(ids_path, 'r') as f:
    all_ids = json.load(f)

print("Total images to segment:", len(all_ids))

image_anno = json.load(open("/trunk2/datasets/cuneiform/image_anno.json", 'r'))


for pid in tqdm.tqdm(all_ids):
    try:

        image_path = "/trunk/shared/cuneiform/full_data/images/"+ "P"+ str(pid).zfill(6)+".jpg"
        masks_filepath = "/trunk/shared/cuneiform/full_data/segmented_mask_info_compressed/P" + str(pid).zfill(6) +".pkl"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width = image.shape[1]
        height = image.shape[0]

        if not ("RGB" in image_anno[pid].keys() and image_anno[pid]["RGB"]): #if non RGB image, no segmentation
            cv2.imwrite("/trunk/shared/cuneiform/full_data/segmented_images/"+image_name, image)
            continue
            
#         #if low resolution, use original image
#         if height<=min_dim and width<=min_dim:
#             cv2.imwrite("/trunk/shared/cuneiform/full_data/segmented_images/"+image_name, image)
#             continue

        #if very high resolution image, then resize and save the resized image
        if height>max_dim or width>max_dim:
            image = resizeImage(image, max_dim)
            cv2.imwrite("/trunk/shared/cuneiform/full_data/images/"+ "P"+ str(pid).zfill(6)+".jpg", image)

        #adjust contrast for better segmentation
        enhanced_image = image.copy()
        black_mask = image < 30
        enhanced_image[black_mask] = 0

        masks = mask_generator.generate(enhanced_image)
        masks = sorted(masks, key = lambda d: d['area'], reverse = True)
        topFive = masks[:5]

        with open(masks_filepath, 'wb') as f:
            pickle.dump(topFive,f)

        cutout = getFrontCutout(topFive, image)
        cutout = cv2.cvtColor(cutout, cv2. COLOR_BGR2RGB)
        cv2.imwrite("/trunk/shared/cuneiform/full_data/segmented_images/"+ "P"+ str(pid).zfill(6)+".jpg", cutout)

    except:
        print(pid)
        

