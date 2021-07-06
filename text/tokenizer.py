import torch


class Tokenizer:
    def __init__(self, dictionary_path='./filelists/word_index.txt', state_size=3):
        self.a1_coef = 15
        self.state_size = state_size
        self.dictionary = self.load_dictionary(dictionary_path)
        self.accent_dict = self.build_num_dict(start=0, end=16)
        self.distance_dict = self.build_num_dict(start=-15, end=9)

    def __call__(self, phonemes, a1s, f2s, split='_'):
        phonemes = [self.dictionary[s] for s in phonemes.split(split)]
        a1s = [self.distance_dict[a1] for a1 in a1s.split(split)]
        f2s = [self.accent_dict[f2] for f2 in f2s.split(split)]

        phonemes = torch.LongTensor(phonemes)
        a1s = torch.LongTensor(a1s)
        f2s = torch.LongTensor(f2s)
        return phonemes, a1s, f2s

    @staticmethod
    def load_dictionary(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        dictionary = dict()
        for i, w in enumerate([w.strip() for w in lines]):
            dictionary[w] = i
        return dictionary

    @staticmethod
    def build_num_dict(start, end):
        d = {str(k): i for i, k in enumerate(range(start, end+1), start=1)}
        d['xx'] = len(d)+1
        return d

    def __len__(self):
        return len(self.dictionary)