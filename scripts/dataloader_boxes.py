from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms

# maybe class name is too long
class CocoBoundingBoxDataset(CocoDetection):
    def __init__(self, img_folder, annotations_file, transform=None):
        super().__init__(img_folder, annotations_file)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        if self.transform:
            img = self.transform(img)

        return img, target  


if __name__=='__main__':
    
    # test to make sure it works
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # replace with where you downloaded the files
    dataset = CocoBoundingBoxDataset(img_folder='/home/holden/github/CS7643_final_project/scripts/val2017',
                                   annotations_file='/home/holden/github/CS7643_final_project/scripts/annotations_trainval2017/annotations/instances_val2017.json',
                                   transform=transform)
    
    # need collate_fn since bounding boxes can be variable length
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

    for images, targets in dataloader:
        print("Batch of images shape:", [img.shape for img in images])
        print("Target sample:", targets[0])
        break
