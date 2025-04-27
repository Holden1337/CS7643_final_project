import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import torchvision
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
# do this so we can import the dataloader code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.dataloader_features import COCODatasetWithFeatures
from scripts.build_vocab import Vocabulary
from LSTM import UpDownCaptionerText
import time
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)


coco_json = '../models/data/annotations/captions_train2017.json'

vocab = Vocabulary(min_freq=5)
captions = vocab.load_coco_captions(coco_json)
vocab.build_vocab(captions)
captions_file = "../models/data/annotations/captions_train2017.json"
features_dir = "../models/data/image_features/train2017/train2017"
images_dir = "../models/data/images/train2017/"


model = UpDownCaptionerText(vocab_size=len(vocab), feature_dim=256, attention_dim=1024)
model.load_state_dict(torch.load('model_weights.pth'))



with open(captions_file, 'r') as f:
    captions_data = json.load(f)

image_id_to_filename = {img['id']: img['file_name'] for img in captions_data['images']}

softmax = nn.Softmax(dim=1)

def is_valid_sample(annotation):
    """
    only pull captions for images that exist in the temp. We shouldn't need this once we have all the 
    .pt files but for now it's necessary.
    """
    image_id = annotation['image_id']
    image_filename = image_id_to_filename[image_id]
    feature_filename = image_filename.replace(".jpg", ".pt")
    # made temp folder with images that also have .pt files available
    image_path = os.path.join("../models/data/images/train2017/", image_filename)
    feature_path = os.path.join("../models/data/image_features/train2017/train2017/" + feature_filename)
    return os.path.exists(image_path) and os.path.exists(feature_path)



annotations = [annotation for annotation in captions_data['annotations'] if is_valid_sample(annotation)]

#print(annotations[0])

n = random.randint(0, len(annotations))

def grab_image_caption_features(n, model, vocab, device='cuda'):
    annotation = annotations[n]
    print(annotation)
    image_id = annotation['image_id']
    image_filename = image_id_to_filename[image_id]
    feature_filename = image_filename.replace(".jpg", ".pt")
    image_path = os.path.join("../models/data/images/train2017/", image_filename)
    feature_path = os.path.join("../models/data/image_features/train2017/train2017/" + feature_filename)

    # Load the image
    image_tensor = torchvision.io.read_image(image_path)

    padded_features = []
    feature_masks = []

    f = torch.load(feature_path)

    max_len = 36
    pad_len = max_len - f.shape[0]
    padded = torch.cat([f, torch.zeros(pad_len, f.shape[1])], dim=0)
    padded_features.append(padded)
    mask = torch.cat([torch.ones(f.shape[0]), torch.zeros(pad_len)])
    feature_masks.append(mask)

    features_tensor = torch.stack(padded_features)
    batch_size, num_boxes, feat_dim = features_tensor.shape  
    features_tensor = features_tensor.view(batch_size, num_boxes, 256, 7, 7)  
    features_tensor = features_tensor.mean(dim=[3, 4])  
    feature_masks_tensor = torch.stack(feature_masks)
    feature_masks_tensor = feature_masks_tensor.to(torch.bool)

    feature_masks_tensor = feature_masks_tensor.to(device)
    features_tensor = features_tensor.to(device)
    model.to(device)


    start_idx = vocab.word2idx['<start>']
    pad_idx = vocab.word2idx['<pad>']
    #print(f"vocab.idx2word[1]: {vocab.idx2word[1]}")
    #print(f"pad_idx: {pad_idx}")
    #print(f"start_idx: {start_idx}")
    #print(f"padded_features.shape: {features_tensor.shape}")
    #print(f"mask.shape: {feature_masks_tensor.shape}")

    #print(len(vocab))
    #print(f"features_tensor.shape: {features_tensor.shape}")
    #print(f"features_mask_tensor: {feature_masks_tensor.shape}")
    #features_tensor = torch.zeros_like(features_tensor)
    #feature_masks_tensor = torch.zeros_like(feature_masks_tensor)
    features_tensor = features_tensor.to(device)
    print(f"torch.sum(features_tensor): {torch.sum(features_tensor)}")
    feature_masks_tensor = feature_masks_tensor.to(device)
    outputs = model.predict_caption(features_tensor, feature_masks_tensor, 1, max_len=20)
    #softmax_output = softmax(outputs)
    #idxs = softmax_output.argmax(dim=1, keepdim=True)
    print(outputs[0])
    idxs = [int(x) for x in outputs[0]]
    guess = [vocab.idx2word[idx] for idx in idxs]
    print(guess)
    #idxs_real = [int(x) for x in captions]
    #real = [vocab.idx2word[idx] for idx in idxs_real]
    #print(guess)
    #print(f"test[0][0]: {test[0][0]}")
    #print(f"word = {vocab.idx2word[int(test[0][0])]}")
    #l = test[0].tolist()
    #for i in l:
#        print(vocab.idx2word[i])
    # Display the image using Matplotlib
    #plt.imshow(image_tensor.permute(1, 2, 0))
    #plt.axis('off') # Turn off axis labels
    #plt.show()


grab_image_caption_features(n, model=model, vocab=vocab)











