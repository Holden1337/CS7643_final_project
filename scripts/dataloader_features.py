import os
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.build_vocab import Vocabulary

class COCODatasetWithFeatures(Dataset):
    def __init__(self, captions_file, features_dir, vocab, transform=None):
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)

        self.features_dir = features_dir
        self.captions_file = captions_file

        
        self.image_id_to_filename = {
            img['id']: img['file_name'] for img in captions_data['images']
        }

        self.annotations = [
            annotation for annotation in captions_data['annotations']
            if self.is_valid_sample(annotation)
        ]

        self.features_dir = features_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def is_valid_sample(self, annotation):
        """
        only pull captions for images that exist in the temp. We shouldn't need this once we have all the 
        .pt files but for now it's necessary.
        """
        image_id = annotation['image_id']
        image_filename = self.image_id_to_filename[image_id]
        # made temp folder with images that also have .pt files available
        image_path = os.path.join("../models/data/images/train2017_temp/", image_filename)

        return os.path.exists(image_path) 

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        caption = annotation['caption']
        image_id = annotation['image_id']
        image_filename = self.image_id_to_filename[image_id]
        feature_path = os.path.join(self.features_dir, image_filename.replace('.jpg', '.pt'))

        # RCNN features / bounding boxes
        features = torch.load(feature_path)

        tokens = self.vocab.numericalize(caption)
        caption_tensor = torch.tensor(tokens)

        #features.requires_grad_(False)
        #caption_tensor.requires_grad_(False)

        return features, caption_tensor
    

if __name__ == "__main__":

    # test out dataloader

    coco_json = '../models/data/annotations/captions_train2017.json'

    vocab = Vocabulary(min_freq=5)
    captions = vocab.load_coco_captions(coco_json)
    vocab.build_vocab(captions)
    captions_file = "../models/data/annotations/captions_train2017.json"
    features_dir = "../models/data/image_features/train2017/"


    def collate_fn(batch):
        """
        need this to deal with images that have fewer than 36 features.
        """
        features, captions = zip(*batch)

        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True)

        return list(features), captions  # return features as a list to handle variable shapes

    test = COCODatasetWithFeatures(captions_file=captions_file, features_dir=features_dir, vocab=vocab)

    val_loader = DataLoader(test, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    for features, captions in val_loader:
        print(f"features[1].shape: {features[1].shape}")
        print(captions[1])
        break
