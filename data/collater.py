import torch
from torch.nn.utils.rnn import pad_sequence


class TextAudioCollate:

    def __call__(self, batch):
        phoneme, a1, f2, spec, wav = tuple(zip(*batch))

        phoneme_padded = pad_sequence(phoneme, batch_first=True)
        a1_padded = pad_sequence(a1, batch_first=True)
        f2_padded = pad_sequence(f2, batch_first=True)
        text_lengths = torch.LongTensor([len(x) for x in phoneme])

        spec_padded = pad_sequence(spec, batch_first=True).transpose(-1, -2)
        spec_lengths = torch.LongTensor([x.size(0) for x in spec])

        wav_padded = pad_sequence(wav, batch_first=True)
        wav_padded = wav_padded.unsqueeze(1)
        wav_lengths = torch.LongTensor([x.size(0) for x in wav])

        return (
            phoneme_padded, a1_padded, f2_padded, text_lengths,
            spec_padded, spec_lengths,
            wav_padded, wav_lengths
        )
