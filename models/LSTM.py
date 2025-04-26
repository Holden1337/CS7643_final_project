import torch
import torch.nn as nn
import torch.nn.functional as F
import time

softmax = nn.Softmax(dim=1)
class UpDownCaptionerAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super(UpDownCaptionerAttention, self).__init__()
        self.feature_att = nn.Linear(feature_dim, attention_dim)
        self.hidden_att = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state, feature_mask=None):
        # print(f"hidden_state.shape: {hidden_state.shape}")
        # print(f"features.shape: {features.shape}")
        att1 = self.feature_att(features)
        att2 = self.hidden_att(hidden_state).unsqueeze(1)
        att = F.tanh(att1 + att2)
        e = self.full_att(att).squeeze(2)

        # print(f"e.shape: {e.shape}")
        # print(f"feature_mask.shape: {feature_mask.shape}")
 
        if feature_mask is not None:
            e = e.masked_fill(~feature_mask, -1e9)
        alpha = F.softmax(e, dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class UpDownCaptionerText(nn.Module):
    def __init__(self, vocab_size, feature_dim=256, embed_dim=512, hidden_dim=512, attention_dim=512, dropout=0.5):
        super(UpDownCaptionerText, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.feature_proj = nn.Linear(feature_dim, 2048)
        self.attention = UpDownCaptionerAttention(2048, hidden_dim, attention_dim)
        self.att_lstm = nn.LSTMCell(embed_dim + 2048 + hidden_dim, hidden_dim)
        self.lang_lstm = nn.LSTMCell(2048 + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, features, captions, feature_mask=None):
        batch_size = features.size(0)
        device = features.device

        embeddings = self.embedding(captions)
        h_att = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_att = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_lang = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_lang = torch.zeros(batch_size, self.hidden_dim).to(device)

        features = self.feature_proj(features)
        #print(f"FEATURES SHAPE: {features.shape}")
        #features = features.mean(dim=1)
        outputs = []
        for t in range(captions.size(1)):
            word_embed = embeddings[:, t, :]
            # print(f"features.shape: {features.shape}")
            # print(f"features_mask: {feature_mask}")
            #time.sleep(40)

            # print(f"h_lang.shape: {h_lang.shape}")
            # print(f"features.shape: {features.shape}")
            # print(f"word_embed.shape: {word_embed.shape}")
            # time.sleep(50)
            #features = features.mean(dim=1)
            mean_features = features.mean(dim=1)

            att_lstm_input = torch.cat([h_lang, mean_features, word_embed], dim=1)
            h_att, c_att = self.att_lstm(att_lstm_input, (h_att, c_att))
            h_att_dropout = self.dropout(h_att)

            # Now attend using the current attention LSTM hidden state
            context, _ = self.attention(features, h_att, feature_mask)

            lang_lstm_input = torch.cat([context, h_att_dropout], dim=1)
            h_lang, c_lang = self.lang_lstm(lang_lstm_input, (h_lang, c_lang))
            output = self.fc(h_lang)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        #print(f"outputs.shape: {outputs.shape}")
        return outputs
    
    def predict_caption(self, features, feature_mask, pad_idx, max_len=20):
        batch_size = features.size(0)
        device = features.device

       
        #features = self.feature_proj(features)
        outputs = torch.zeros((batch_size, max_len, self.vocab_size), device=device)

        captions = torch.full((batch_size, max_len), pad_idx, device=device)
        captions[:, 0:2] = 1 # make captions start with start token
        

        for t in range(max_len):
            preds = self.forward(features, captions, feature_mask)
            outputs[:, t, :] = preds[:, t, :]

            if t < max_len - 1:
                indices = softmax(preds[:, t, :]).argmax(dim=1, keepdim=True)
                captions[:, t+1] = indices
        print(f"captions: {captions}")

        print(f"outputs.shape: {outputs.shape}")

        return outputs
        


            

