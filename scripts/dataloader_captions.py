import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CocoCaptionDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        # need ToTensor transform to get png images suitable for pytorch
        self.transform = transform

        # load captions
        with open(captions_file, 'r') as f:
            data = json.load(f)

        # image_id to filename dict
        self.id_to_filename = {img['id']: img['file_name'] for img in data['images']}

        self.captions = data['annotations']

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption_data = self.captions[idx]
        image_id = caption_data['image_id']
        caption = caption_data['caption']
        image_filename = self.id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, image_filename)

        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, caption
    

if __name__=='__main__':

    # test to see if it works
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # replace with whereever you decided to download the files
    val_dataset = CocoCaptionDataset(
        image_dir='/home/holden/github/CS7643_final_project/scripts/val2017',
        captions_file='/home/holden/github/CS7643_final_project/scripts/annotations_trainval2017/annotations/captions_val2017.json',
        transform=transform
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    for images, captions in val_loader:
        print(images.shape)  
        print(captions[0])   
        break
