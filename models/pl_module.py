import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from .commons import slice_segments
from .models import SynthesizerTrn, MultiPeriodDiscriminator
from .losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss
)
from transform import spec_to_mel_torch, mel_spectrogram_torch
from text import Tokenizer


class VITSModule(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.net_g = SynthesizerTrn(
            len(Tokenizer()),
            spec_channels=params.data.filter_length // 2 + 1,
            segment_size=params.train.segment_size // params.data.hop_length,
            **params.model)
        self.net_d = MultiPeriodDiscriminator(params.model.use_spectral_norm)

    def forward(self, phoneme, a1, f2, length, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0):
        """
        :return: wav: torch.FloatTensor => [1, T]
        """
        wav, *_ = self.net_g.infer(
            phoneme, a1, f2, length,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale
        )
        wav = wav.squeeze(0)
        return wav

    def training_step(self, batch, batch_idx):
        phoneme, a1, f2, x_lengths, spec, spec_lengths, y, y_lengths = batch

        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()

        y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
        (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(phoneme, a1, f2, x_lengths, spec, spec_lengths)

        mel = spec_to_mel_torch(
            spec,
            self.params.data.filter_length,
            self.params.data.n_mel_channels,
            self.params.data.sampling_rate,
            self.params.data.mel_fmin,
            self.params.data.mel_fmax
        )
        y_mel = slice_segments(mel, ids_slice, self.params.train.segment_size // self.params.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.params.data.filter_length,
            self.params.data.n_mel_channels,
            self.params.data.sampling_rate,
            self.params.data.hop_length,
            self.params.data.win_length,
            self.params.data.mel_fmin,
            self.params.data.mel_fmax
        )
        y = slice_segments(y, ids_slice * self.params.data.hop_length, self.params.train.segment_size)

        # for Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
        loss_d, *_ = discriminator_loss(y_d_hat_r, y_d_hat_g)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        # for Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.params.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.params.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % n == 0:
            sch_g.step()
            sch_d.step()

        self.log_dict({
            'train/d': loss_d,
            'train/g': loss_g,
            'train/gen': loss_gen,
            'train/fm': loss_fm,
            'train/mel': loss_mel,
            'train/dur': loss_dur,
            'train/kl': loss_kl
        })

    def validation_step(self, batch, batch_idx):
        phoneme, a1, f2, x_lengths, spec, spec_lengths, y, y_lengths = batch

        y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
        (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(phoneme, a1, f2, x_lengths, spec, spec_lengths)

        mel = spec_to_mel_torch(
            spec,
            self.params.data.filter_length,
            self.params.data.n_mel_channels,
            self.params.data.sampling_rate,
            self.params.data.mel_fmin,
            self.params.data.mel_fmax
        )
        y_mel = slice_segments(mel, ids_slice, self.params.train.segment_size // self.params.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            self.params.data.filter_length,
            self.params.data.n_mel_channels,
            self.params.data.sampling_rate,
            self.params.data.hop_length,
            self.params.data.win_length,
            self.params.data.mel_fmin,
            self.params.data.mel_fmax
        )
        y = slice_segments(y, ids_slice * self.params.data.hop_length, self.params.train.segment_size)

        # for Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
        loss_d, *_ = discriminator_loss(y_d_hat_r, y_d_hat_g)

        # for Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.params.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.params.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

        self.log_dict({
            'valid/d': loss_d,
            'valid/g': loss_g,
            'valid/gen': loss_gen,
            'valid/fm': loss_fm,
            'valid/mel': loss_mel,
            'valid/dur': loss_dur,
            'valid/kl': loss_kl
        })

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.net_g.parameters(),
            self.params.optimizer.lr,
            betas=self.params.optimizer.betas,
            eps=1e-9)
        opt_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.params.optimizer.lr,
            betas=self.params.optimizer.betas,
            eps=1e-9)
        scheduler_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.params.optimizer.lr_decay)
        scheduler_d = optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.params.optimizer.lr_decay)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]
