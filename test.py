import os.path
import warnings
warnings.filterwarnings('ignore')

import pyaudio
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

from models import load_vits
from text import Tokenizer

noise_scale = .667
noise_scale_w = 0.8
length_scale = 1.0
MAX_WAV_VALUE = 32768.0
sr = 24000


if __name__ == '__main__':
    file_path = 'filelists/test.txt'

    parser = ArgumentParser()
    parser.add_argument('--vits', required=True, type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--play_audio', action='store_true')
    parser.add_argument('--output_dir', default='./outputs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vits = load_vits(args.vits)
    vits = vits.eval().to(device)

    tokenizer = Tokenizer()

    if args.play_audio:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            frames_per_buffer=1024,
            output=True
        )

    def infer(inputs):
        phoneme, a1, f2 = tokenizer(*inputs)
        length = torch.tensor([phoneme.size(-1)], dtype=torch.long)
        phoneme, a1, f2 = phoneme.unsqueeze(0), a1.unsqueeze(0), f2.unsqueeze(0)
        phoneme, a1, f2, length = phoneme.to(device), a1.to(device), f2.to(device), length.to(device)
        with torch.no_grad():
            wav, *_ = vits.infer(
                phoneme, a1, f2, length,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=1
            )
        wav = wav.cpu().squeeze(0)
        return wav

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, total=len(lines)):
        gt_path, *inputs = line.strip().split('|')
        fn = gt_path.split('/')[-1]
        if hasattr(args, 'gt_root'):
            gt_path = f'{gt_path}/{fn}'
        wav_gen = infer(inputs)
        d = output_dir / os.path.splitext(fn)[0]
        d.mkdir(exist_ok=True, parents=True)

        wav_gt, _ = torchaudio.load(gt_path)
        torchaudio.save(
            str(d / 'gt.wav'),
            wav_gt,
            sr,
            encoding='PCM_S',
            bits_per_sample=16
        )
        torchaudio.save(
            str(d / 'gen.wav'),
            wav_gen,
            sr,
            encoding='PCM_S',
            bits_per_sample=16
        )

        if args.play_audio:
            stream.write((wav_gt.squeeze().numpy() * MAX_WAV_VALUE).astype(np.int16).tobytes())
            stream.write((wav_gen.squeeze().numpy() * MAX_WAV_VALUE).astype(np.int16).tobytes())
