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

softmax = nn.Softmax(dim=1)

def collate_fn_with_padding(batch):
    """
    Pads feature tensors to the same number of boxes (spatial features).
    Also computes valid_box_counts to indicate how many boxes each image has.
    """
    features, captions = zip(*batch)

    max_len = max(f.shape[0] for f in features)
    padded_features = []
    valid_box_counts = []

    for f in features:
        f = f.detach() if f.requires_grad else f
        f = f.view(f.shape[0], 256, 7, 7)
        f = f.mean(dim=[2, 3])
        pad_len = max_len - f.shape[0]
        valid_box_counts.append(f.shape[0])
        padded = torch.cat([f, torch.zeros(pad_len, f.shape[1])], dim=0)
        padded_features.append(padded)

    features_tensor = torch.stack(padded_features)
    valid_box_counts_tensor = torch.tensor(valid_box_counts, dtype=torch.long, device=features_tensor.device)

    # Reshape and mean-pool
    batch_size, num_boxes, feat_dim = features_tensor.shape
    #features_tensor = features_tensor.view(batch_size, num_boxes, 256, 7, 7)
    #features_tensor = features_tensor.mean(dim=[3, 4])  # Resulting shape: [batch_size, num_boxes, 256]

    captions_tensor = pad_sequence(captions, batch_first=True, padding_value=0)

    return features_tensor, valid_box_counts_tensor, captions_tensor


train_set = COCODatasetWithFeatures(captions_file=captions_file, features_dir=features_dir, vocab=vocab)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn_with_padding)


captions_file_val = "../models/data/annotations/captions_val2017.json"
features_dir_val = "../models/data/image_features/val2017/val2017"

# don't have .pt files for val set
val_set = COCODatasetWithFeatures(captions_file=captions_file_val, features_dir=features_dir_val, vocab=vocab, train=False)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn_with_padding)

#print(f"val set len: {len(val_loader.dataset)}")

for features, features_masks, captions in train_loader:
    # use this to check that the data loader went ok
    #print(f"features shape: {features.shape}")
    #print(captions[0])
    break

model = UpDownCaptionerText(vocab_size=len(vocab), feature_dim=256, attention_dim=1024, hidden_dim=1024, embed_dim=1024)

# using Adam to start, might try other stuff later.
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
PAD_IDX = 0
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


NUM_EPOCHS = 3
train_loss_arr = []
val_loss_arr = []

def train_model(model, train_loader, val_loader, vocab_size, criterion, optimizer, num_epochs, device='cuda'):
    model.to(device)
    print("Beginning model training. Surely it'll work this time...")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for features, valid_box_counts, captions in train_loader:
            features, valid_box_counts, captions = features.to(device), valid_box_counts.to(device), captions.to(device)
            batch_size, num_boxes, _ = features.shape

            feature_mask_tensor = torch.arange(num_boxes, device=features.device).unsqueeze(0) < valid_box_counts.unsqueeze(1)
            feature_mask_tensor = feature_mask_tensor.bool()

            inputs = captions[:, :-1]  # Remove last token
            targets = captions[:, 1:]  # Remove start token

            optimizer.zero_grad()
            outputs = model(features, inputs, feature_mask=feature_mask_tensor)
    
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            running_train_loss += loss.item() * features.size(0)  # sum over batch

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_arr.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features_val, valid_box_counts_val, captions_val in val_loader:
                features_val, valid_box_counts_val, captions_val = features_val.to(device), valid_box_counts_val.to(device), captions_val.to(device)

                batch_size, num_boxes_val, _ = features_val.shape

                feature_mask_tensor_val = torch.arange(num_boxes_val, device=features_val.device).unsqueeze(0) < valid_box_counts_val.unsqueeze(1)
                feature_mask_tensor_val = feature_mask_tensor_val.bool()


                # again, this might be causing the issue
                inputs_val = captions_val[:, :-1]  # Remove last token
                targets_val = captions_val[:, 1:]  # Remove start token

                outputs_val = model.forward(features_val, inputs_val, feature_mask=feature_mask_tensor_val)

                val_loss = criterion(outputs_val.reshape(-1, vocab_size), targets_val.reshape(-1))
                running_val_loss += val_loss.item() * features_val.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_loss_arr.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} "
              f"Val Loss: {epoch_val_loss:.4f}")
    loss_df = pd.DataFrame()
    loss_df['train_loss'] = train_loss_arr
    loss_df['val_loss'] = val_loss_arr
    loss_df.to_csv("loss_df_2.csv", index=False)
    torch.save(model.state_dict(), 'model_weights_2.pth')
        
train_model(model, train_loader, val_loader, len(vocab), criterion, optimizer, num_epochs=NUM_EPOCHS, device='cuda')
