import logging

import torch
from omegaconf import OmegaConf

from minimaxspeech.modules.vq_vae.dvae import DiscreteVAE, TorchMelSpectrogram
from minimaxspeech.modules.gpt.gpt import GPT
from minimaxspeech.modules.flow_vae.flow_vae import DAC_FLOW_VAE
from minimaxspeech.modules.flow_matching.flow_matching import MaskedDiffWithXvec


class MiniMaxSpeechModel:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.load_checkpoint()

    def load_checkpoint(self):
        self.vq_vae.load_state_dict(torch.load(self.config.model.vq_vae.checkpoint, map_location="cpu"))
        logging.info(f"VQVAE model loaded")

        checkpoint = torch.load(self.config.model.gpt.checkpoint, map_location="cpu")
        if 'gpt' in checkpoint: # self-host checkpoint
            self.gpt.load_state_dict(checkpoint['gpt'])
        else: # from xtts2 checkpoint
            new_dict = {k[4:]: v for k, v in checkpoint['model'].items() if k.startswith('gpt')}
            self.gpt.load_state_dict(new_dict)
        self.gpt.eval()
        logging.info(f"GPT model loaded")

        self.flow_matching.load_state_dict(
            torch.load(self.config.model.flow_matching.checkpoint, map_location="cpu")['flow_matching']
        )
        logging.info(f"FlowMatching model loaded")

        self.flow_vae.load_state_dict(torch.load(self.config.model.flow_vae.checkpoint, map_location="cpu")['flow_vae'])
        logging.info(f"FlowVAE model loaded")
        logging.info(f"Model loaded successfully")

    def setup_model(self):
        self.torch_mel_spectrogram_vq_vae = TorchMelSpectrogram(
            mel_norm_file=self.config.model.vq_vae.mel_norm_file,
            sampling_rate=self.config.model.vq_vae.sample_rate
        ).to(self.device)

        vq_vae_config = OmegaConf.load(self.config.model.vq_vae.config)
        self.vq_vae = DiscreteVAE(**vq_vae_config.model.vq_vae).to(self.device)
        self.vq_vae.eval()

        gpt_config = OmegaConf.load(self.config.model.gpt.config)
        self.gpt = GPT(**gpt_config.model.gpt).to(self.device)
        self.gpt.eval()
        
        flow_matching_config = OmegaConf.load(self.config.model.flow_matching.config)
        self.flow_matching = MaskedDiffWithXvec(**flow_matching_config.model.flow_matching).to(self.device)
        self.flow_matching.eval()

        flow_vae_config = OmegaConf.load(self.config.model.flow_vae.config)
        self.flow_vae = DAC_FLOW_VAE(**flow_vae_config.model.flow_vae).to(self.device)
        self.flow_vae.eval()

    def generate(self, prompt_audio, text_tokens):
        """
        Args:
            prompt_audio: (b, 1, t)
            text: (b, t)

        Returns:
            audio: (b, 1, t)
        """
        with torch.no_grad():
            feat = self.torch_mel_spectrogram_vq_vae(prompt_audio)
            
            z = self.flow_vae.encode(prompt_audio).transpose(1, 2)
            z_len = prompt_audio.shape[2] // self.flow_vae.downsample_rate
            z_len = torch.tensor([z_len]).to(self.device)

            prompt_token = self.vq_vae.get_codebook_indices(feat)
            prompt_token_len = torch.ceil(torch.tensor([(prompt_audio.shape[2] + 1) / 1024])).long()
            prompt_token_len = prompt_token_len.to(self.device)

            speaker_embedding = self.gpt.get_style_emb(feat).transpose(1, 2)
            self.gpt.init_gpt_for_inference()
            audio_tokens = self.gpt.generate(cond_latents=speaker_embedding, text_inputs=text_tokens)

            flow_cache = torch.zeros(1, self.flow_matching.out_channels, 0, 2)

            latent, flow_cache = self.flow_matching.inference(
                token=audio_tokens,
                token_len=audio_tokens.shape[1],
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=z,
                prompt_feat_len=z_len,
                embedding=speaker_embedding.mean(dim=1),
                flow_cache=flow_cache
            )
            audio = self.flow_vae.decode(latent)
            return audio