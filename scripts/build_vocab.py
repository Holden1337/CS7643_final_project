import json
from collections import Counter

class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.pad_token = '<pad>'
        self.start_token = '<start>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'
        self.add_special_tokens()

    def add_special_tokens(self):
        special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        for token in special_tokens:
            self.add_word(token)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def tokenize(self, text):
        return text.lower().strip().split()

    def build_vocab(self, captions):
        counter = Counter()
        for caption in captions:
            tokens = self.tokenize(caption)
            counter.update(tokens)

        for word, count in counter.items():
            if count >= self.min_freq:
                self.add_word(word)

    def numericalize(self, caption):
        tokens = [self.start_token] + self.tokenize(caption) + [self.end_token]
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.idx2word.get(idx, self.unk_token) for idx in token_ids]
        return ' '.join(tokens)

    def __len__(self):
        return len(self.word2idx)

    def load_coco_captions(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Only need the captions for the vocab
        captions = [ann['caption'] for ann in data['annotations']]
        return captions

if __name__=='__main__':
    vocab = Vocabulary(min_freq=5)

    coco_json = '../models/data/annotations/captions_train2017.json'
    captions = vocab.load_coco_captions(coco_json)

    vocab.build_vocab(captions)

    print(vocab)
