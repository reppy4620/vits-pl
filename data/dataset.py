import os

import torch
from torch.utils.data import Dataset

from text import Tokenizer
from transform import mel_spectrogram_torch
from .utils import load_data, load_wav


class TextAudioDataset(Dataset):

    def __init__(self, data_file_path, hparams):
        self.audiopaths_and_text = load_data(data_file_path)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.mel_size = hparams.n_mel_channels
        self.f_min = hparams.mel_fmin
        self.f_max = hparams.mel_fmax
        self.tokenizer = Tokenizer()
        _, sr = load_wav(self.audiopaths_and_text[0][0])
        if sr != self.sampling_rate:
            raise ValueError(f"{sr} {self.sampling_rate} SR doesn't match target SR")

        self.mel_dict = dict()

    def get_audio_text_pair(self, audiopath_and_text):
        file_path, *text = audiopath_and_text
        text, a1, f2 = self.get_text(text)
        mel, wav = self.get_audio(file_path)
        return text, a1, f2, mel, wav

    def get_audio(self, filename):
        audio, sampling_rate = load_wav(filename)
        spec_filename = os.path.basename(filename)
        if spec_filename in self.mel_dict:
            mel = self.mel_dict[spec_filename]
        else:
            mel = mel_spectrogram_torch(
                audio,
                self.filter_length,
                self.mel_size,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                self.f_min,
                self.f_max
            )
            mel = torch.squeeze(mel, 0).transpose(0, 1)
            self.mel_dict[spec_filename] = mel
        audio = audio.squeeze()
        return mel, audio

    def get_text(self, text):
        phoneme, a1, f2 = self.tokenizer(*text)
        return phoneme, a1, f2

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)
