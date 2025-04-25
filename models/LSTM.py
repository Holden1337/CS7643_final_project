import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class UpDownCaptionerAttention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super(UpDownCaptionerAttention, self).__init__()
        self.feature_att = nn.Linear(feature_dim, attention_dim)
        self.hidden_att = nn.Linear(hidden_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state, feature_mask=None):
        att1 = self.feature_att(features)
        att2 = self.hidden_att(hidden_state).unsqueeze(1)
        att = F.tanh(att1 + att2)
        e = self.full_att(att).squeeze(2)
 
        if feature_mask is not None:
            e = e.masked_fill(~feature_mask, -1e9)
        alpha = F.softmax(e, dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class UpDownCaptionerText(nn.Module):
    def __init__(self, vocab_size, feature_dim=256, embed_dim=512, hidden_dim=512, attention_dim=512, dropout=0.5):
        super(UpDownCaptionerText, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.feature_proj = nn.Linear(feature_dim, 2048)
        self.attention = UpDownCaptionerAttention(2048, hidden_dim, attention_dim)
        self.att_lstm = nn.LSTMCell(embed_dim + 2048 + hidden_dim, hidden_dim)
        self.lang_lstm = nn.LSTMCell(2048 + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions, feature_mask=None):
        batch_size = features.size(0)
        device = features.device

        embeddings = self.embedding(captions)
        h_att = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_att = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_lang = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_lang = torch.zeros(batch_size, self.hidden_dim).to(device)

        features = self.feature_proj(features)
        outputs = []
        for t in range(captions.size(1)):
            word_embed = embeddings[:, t, :]
            # print(f"features.shape: {features.shape}")
            # print(f"features_mask: {feature_mask}")
            #time.sleep(40)
            context, _ = self.attention(features, h_att, feature_mask)
            att_lstm_input = torch.cat([h_lang, context, word_embed], dim=1)
            h_att, c_att = self.att_lstm(att_lstm_input, (h_att, c_att))
            h_att_dropout = self.dropout(h_att)
            lang_lstm_input = torch.cat([context, h_att_dropout], dim=1)
            h_lang, c_lang = self.lang_lstm(lang_lstm_input, (h_lang, c_lang))
            output = self.fc(h_lang)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def predict_caption(self, features, feature_mask, start_idx, max_len=20):
        batch_size = features.size(0)
        device = features.device

        h_att = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_att = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_lang = torch.zeros(batch_size, self.hidden_dim).to(device)
        c_lang = torch.zeros(batch_size, self.hidden_dim).to(device)
        features = self.feature_proj(features)
        

        inputs = torch.tensor([[start_idx]] * batch_size, device='cuda')
        pred_caption = [] 

        for _ in range(max_len):
            word_embed = self.embedding(inputs).squeeze(1)
            context, _ = self.attention(features, h_att, feature_mask)

            att_lstm_input = torch.cat([h_lang, context, word_embed], dim=1)
            h_att, c_att = self.att_lstm(att_lstm_input, (h_att, c_att))

            lang_lstm_input = torch.cat([context, h_att], dim=1)
            h_lang, c_lang = self.lang_lstm(lang_lstm_input, (h_lang, c_lang))

            output = self.fc(h_lang)
            predicted_word = output.argmax(dim=1, keepdim=True)
            #print(predicted_word)
            pred_caption.append(predicted_word)

            inputs = predicted_word

        pred_caption = torch.cat(pred_caption, dim=1)
        return pred_caption

