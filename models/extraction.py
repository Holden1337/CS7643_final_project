import os
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import roi_align
from PIL import Image
from tqdm import tqdm

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
# https://pytorch.org/vision/main/generated/torchvision.ops.roi_align.html


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
IMAGE_DIR = './data/images/train2017'
OUTPUT_DIR = './data/features/train2017'
TOP_K = 36

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

resnet = torchvision.models.resnet50(weights='IMAGENET1K_V2')
backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
backbone.out_channels = 2048

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

model = FasterRCNN(
    backbone=backbone,
    num_classes=91,  # COCO default, doesn't matter if you're just extracting
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
).to(DEVICE).eval()

def extract_features():
    for image_file in tqdm(os.listdir(IMAGE_DIR)):
        image_path = os.path.join(IMAGE_DIR, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).to(DEVICE)
        images = [image_tensor]

        with torch.no_grad():
            outputs = model(images)
            boxes = outputs[0]['boxes'][:TOP_K]

            feature_maps = model.backbone(image_tensor.unsqueeze(0))

            region_feats = roi_align(
                feature_maps,
                [boxes],
                output_size=(7, 7),
                spatial_scale=1/32,
                sampling_ratio=2
            )

            region_feats = region_feats.view(region_feats.size(0), -1)

            save_path = os.path.join(OUTPUT_DIR, image_file.replace('.jpg', '.pt'))
            torch.save(region_feats.cpu(), save_path)

if __name__ == "__main__":
    extract_features()
