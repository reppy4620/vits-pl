import torch
from torch.nn.utils.rnn import pad_sequence


class TextAudioCollate:

    def __call__(self, batch):
        phoneme, a1, f2, mel, wav = tuple(zip(*batch))

        phoneme_padded = pad_sequence(phoneme, batch_first=True)
        a1_padded = pad_sequence(a1, batch_first=True)
        f2_padded = pad_sequence(f2, batch_first=True)
        text_lengths = torch.LongTensor([len(x) for x in phoneme])

        mel_padded = pad_sequence(mel, batch_first=True).transpose(-1, -2)
        mel_lengths = torch.LongTensor([x.size(0) for x in mel])

        wav_padded = pad_sequence(wav, batch_first=True)
        wav_padded = wav_padded.unsqueeze(1)
        wav_lengths = torch.LongTensor([x.size(0) for x in wav])

        return (
            phoneme_padded, a1_padded, f2_padded, text_lengths,
            mel_padded, mel_lengths,
            wav_padded, wav_lengths
        )
