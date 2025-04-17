from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import roi_align
from torchvision.io import read_image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import torch.nn as nn
import os
from PIL import Image
from tqdm import tqdm
# Only run the wget below once to get utils.py
# os.system("wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py")
import utils

class bounding_boxes:

    def __init__(self):
        # self.dataset = foz.load_zoo_dataset("coco-2017", split="validation", max_samples=1)
        self.IMAGE_DIR = './data/images/train2017'
        self.OUTPUT_DIR = './data/features/train2017'
        self.TOP_K = 36

        self.model = self.get_model_instance_segmentation()
        # Check device availability
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print("You are using device: %s" % self.device)
        self.model.to(self.device)

    def get_model_instance_segmentation(self):
        # faster r-cnn citation: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        # load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT")

        return model

    def get_transform(self, train=False):
        transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
        ])

        return transform

    def visualize_output(self):

        # output citation: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        eval_transform = self.get_transform(train=False)

        # Loop through images
        for i, image_file in enumerate(tqdm(os.listdir(self.IMAGE_DIR))):
            # Setting a low number of images for testing
            if i >= 5:
                break
            image_path = os.path.join(self.IMAGE_DIR, image_file)
            image = Image.open(image_path).convert("RGB")
            image_tensor = eval_transform(image).to(self.device)
            images = [image_tensor]

            # get first image in dataset citation: https://docs.voxel51.com/api/fiftyone.core.dataset.html?highlight=first#fiftyone.core.dataset.Dataset.first
            # image = read_image(self.dataset.first().filepath)

            self.model.eval()
            with torch.no_grad():
                # convert RGBA -> RGB and move to device
                image_tensor = image_tensor[:3, ...].to(self.device)
                predictions = self.model([image_tensor, ])
                pred = predictions[0]

                # Boolean tensor mask citation: https://www.tutorialspoint.com/index-based-operation-in-pytorch
                start_filter = pred["scores"] > 0.2

                # Output the top 36 results
                pred["boxes"] = pred["boxes"][start_filter][:self.TOP_K]

            image = (255.0 * (image_tensor - image_tensor.min()) /
                    (image_tensor.max() - image_tensor.min())).to(torch.uint8)
            pred_boxes = pred["boxes"].long()

            output_image = draw_bounding_boxes(image, pred_boxes, colors="red")


            # Extract features
            outputs = self.model(images)
            boxes = outputs[0]['boxes'][:self.TOP_K]

            all_feature_maps = self.model.backbone(image_tensor.unsqueeze(0))
            first_feature_map = all_feature_maps['0']

            region_feats = roi_align(
                first_feature_map,
                [boxes],
                output_size=(7, 7),
                spatial_scale=1/32,
                sampling_ratio=2
            )

            region_feats = region_feats.view(region_feats.size(0), -1)

            save_path = os.path.join(self.OUTPUT_DIR, image_file.replace('.jpg', '.pt'))
            torch.save(region_feats.cpu(), save_path)

        return output_image

    def display_output(self, output_image):
        plt.figure(figsize=(12, 12))
        plt.imshow(output_image.permute(1, 2, 0))


if __name__ == "__main__":
    bb = bounding_boxes()
    output = bb.visualize_output()
    bb.display_output(output)
