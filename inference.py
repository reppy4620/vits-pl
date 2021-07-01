import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
from pathlib import Path

import pyaudio
import torch
import torchaudio

from models import VITSModule
from text import TokenizerForInfer

noise_scale = .667
noise_scale_w = 0.8
length_scale = 1.0
MAX_WAV_VALUE = 32768.0
sr = 24000


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vits_ckpt_path', required=True, type=str)
    parser.add_argument('--play_audio', action='store_true')
    parser.add_argument('--save_wav', action='store_true')
    parser.add_argument('--output_dir', default='./outputs')
    args = parser.parse_args()

    if args.save_wav:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vits = VITSModule.load_from_checkpoint(args.vits_ckpt_path).to(device).eval().freeze()

    tokenizer = TokenizerForInfer()

    if args.play_audio:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            frames_per_buffer=1024,
            output=True
        )

    def infer(s):
        phoneme, a1, f2, length = tokenizer(s)
        phoneme, a1, f2, length = phoneme.to(device), a1.to(device), f2.to(device), length.to(device)
        with torch.no_grad():
            wav, *_ = vits(phoneme, a1, f2, length)
        wav = wav.squeeze(0)
        return wav

    try:
        while True:
            s = input('文章を入力してください >> ')
            try:
                wav = infer(s)
            except:
                wav = infer('有効な文章を入力してください．')
            if args.save_wav:
                torchaudio.save(f'{str(output_dir)}/{s}.wav', wav, sr)
            if args.play_audio:
                stream.write(wav.tobytes())
    except KeyboardInterrupt:
        stream.close()
