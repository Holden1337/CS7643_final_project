import os
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import matplotlib.pyplot as plt
# do this so we can import the dataloader code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.dataloader_features import COCODatasetWithFeatures
from scripts.build_vocab import Vocabulary
from LSTM import UpDownCaptionerText
import time
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("You are using device: %s" % device)


coco_json = '../models/data/annotations/captions_train2017.json'

vocab = Vocabulary(min_freq=5)
captions = vocab.load_coco_captions(coco_json)
vocab.build_vocab(captions)
captions_file = "../models/data/annotations/captions_train2017.json"
features_dir = "../models/data/image_features/train2017/train2017"


def collate_fn_with_padding(batch):
    """
    Need to do this since not all images have 36 features. Just 
    pad some features with zero tensors
    probably need to incorporate this into LSTM.py somehow
    """
    features, captions = zip(*batch)

    max_len = max(f.shape[0] for f in features)
    padded_features = []
    feature_masks = []

    for f in features:
        # Need to detach before stacking since pytorch is weird about that
        # that and create the padding tensors
        f = f.detach() if f.requires_grad else f
        pad_len = max_len - f.shape[0]
        padded = torch.cat([f, torch.zeros(pad_len, f.shape[1])], dim=0)
        padded_features.append(padded)
        mask = torch.cat([torch.ones(f.shape[0]), torch.zeros(pad_len)])
        feature_masks.append(mask)

    features_tensor = torch.stack(padded_features)
    feature_masks_tensor = torch.stack(feature_masks)

    feature_masks_tensor = feature_masks_tensor.to(torch.bool)

    batch_size, num_boxes, feat_dim = features_tensor.shape  
    features_tensor = features_tensor.view(batch_size, num_boxes, 256, 7, 7)  
    features_tensor = features_tensor.mean(dim=[3, 4])  

    captions_tensor = pad_sequence(captions, batch_first=True, padding_value=0)

    return features_tensor, feature_masks_tensor, captions_tensor


train_set = COCODatasetWithFeatures(captions_file=captions_file, features_dir=features_dir, vocab=vocab)
train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn_with_padding)


captions_file_val = "../models/data/annotations/captions_val2017.json"
features_dir_val = "../models/data/image_features/val2017/val2017"

# don't have .pt files for val set
val_set = COCODatasetWithFeatures(captions_file=captions_file_val, features_dir=features_dir_val, vocab=vocab, train=False)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn_with_padding)

print(f"val set len: {len(val_loader.dataset)}")

for features, features_masks, captions in train_loader:
    # use this to check that the data loader went ok
    #print(f"features shape: {features.shape}")
    #print(captions[0])
    break

model = UpDownCaptionerText(vocab_size=len(vocab), feature_dim=256)

# using Adam to start, might try other stuff later.
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
PAD_IDX = 0
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

NUM_EPOCHS = 10
train_loss_arr = []
val_loss_arr = []

def train_model(model, train_loader, val_loader, vocab_size, criterion, optimizer, num_epochs, device='cuda'):
    model.to(device)
    print("BEGINNING MODEL TRAINING...")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for features, features_mask_tensor, captions in train_loader:
            
            
            features, features_mask_tensor, captions = features.to(device),features_mask_tensor.to(device), captions.to(device)

            optimizer.zero_grad()
            outputs = model(features, captions, feature_mask=features_mask_tensor)

            # print(f"outputs.shape before reshape: {outputs.shape}")
            # print(f"captions.shape before reshape: {captions.shape}")
            # print(f"captions[0]: {captions[0]}")
            outputs = outputs.view(-1, vocab_size)
            captions = captions.view(-1)

            # print(f"outputs.shape: {outputs.shape}")
            # print(f"captions.shape: {captions.shape}")
            # print(captions)
            # print(f"outputs[0]: {outputs[0]}")
            # time.sleep(50)


            loss = criterion(outputs, captions)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * features.size(0)  # sum over batch
            #print(running_train_loss)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_arr.append(epoch_train_loss)

        # waiting until we get .pt files to uncomment this
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features_val, features_mask_tensor_val, captions_val in val_loader:
                features_val, features_mask_tensor_val, captions_val = features_val.to(device), features_mask_tensor_val.to(device), captions_val.to(device)
                outputs_val = model.predict_caption(features_val, captions_val, feature_mask=features_mask_tensor_val, start_idx=1, max_len=20)
                outputs_val = outputs_val.view(-1, vocab_size)
                captions_val = captions_val.view(-1)
                loss = criterion(outputs_val, captions_val)
                running_val_loss += loss.item() * features.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_arr.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} "
              f"Val Loss: {0}")
    loss_df = pd.DataFrame()
    loss_df['train_loss'] = train_loss_arr
    loss_df['val_loss'] = val_loss_arr
    loss_df.to_csv("loss_df.csv", index=False)
    torch.save(model.state_dict(), 'model_weights.pth')
        
#val_loader = None
train_model(model, train_loader, val_loader, len(vocab), criterion, optimizer, num_epochs=NUM_EPOCHS, device='cuda')
