import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from text import Tokenizer
from transform import spectrogram_torch
from .utils import load_data, load_wav


class TextAudioDataset(Dataset):

    def __init__(self, data_file_path, hparams):
        self.audiopaths_and_text = load_data(data_file_path)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.tokenizer = Tokenizer()
        _, sr = load_wav(self.audiopaths_and_text[0][0])
        if sr != self.sampling_rate:
            raise ValueError(f"{sr} {self.sampling_rate} SR doesn't match target SR")

        self.spec_dir = Path(self.audiopaths_and_text[0][0]).parent / 'mel'
        self.spec_dir.mkdir(exist_ok=True)

        self.spec_dict = dict()

    def get_audio_text_pair(self, audiopath_and_text):
        file_path, *text = audiopath_and_text
        text, a1, f2 = self.get_text(text)
        spec, wav = self.get_audio(file_path)
        return text, a1, f2, spec, wav

    def get_audio(self, filename):
        audio, sampling_rate = load_wav(filename)
        spec_filename = os.path.basename(filename)
        if spec_filename in self.spec_dict:
            spec = self.spec_dict[spec_filename]
        else:
            spec = spectrogram_torch(
                audio, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False
             )
            spec = torch.squeeze(spec, 0).transpose(0, 1)
            self.spec_dict[spec_filename] = spec
        audio = audio.squeeze()
        return spec, audio

    def get_text(self, text):
        phoneme, a1, f2 = self.tokenizer(*text)
        return phoneme, a1, f2

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)
