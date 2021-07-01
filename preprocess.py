import argparse
import warnings
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


class PreProcessor:
    def preprocess(self):
        pass


class AudioProcessor(PreProcessor):
    def __init__(self, dataset_path, output_path):
        self.fns = list(Path(dataset_path).glob('*.wav'))
        self.output_dir = Path(output_path) / 'wav'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resample = torchaudio.transforms.Resample(48000, 24000)

    def process_file(self, fn):
        wav, sr = torchaudio.load(fn)
        wav = self.resample(wav)
        torchaudio.save(str(self.output_dir / fn.name), wav, self.resample.new_freq, encoding='PCM_S', bits_per_sample=16)

    def preprocess(self):
        Parallel(n_jobs=-1)(
            delayed(self.process_file)(fn) for fn in tqdm(self.fns, total=len(self.fns))
        )


if __name__ == '__main__':
    try:
        import torchaudio
        torchaudio.set_audio_backend('sox_io')
    except RuntimeError:
        torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
        torchaudio.set_audio_backend('soundfile')
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--wav_output_dir', type=str, required=True)
    args = parser.parse_args()

    ap = AudioProcessor(args.wav_dir, args.wav_output_dir)

    print('Start audio preprocessing')
    ap.preprocess()
