import logging

import matplotlib.pylab as plt
import torch
import torchaudio
import torch.nn.functional as F
from omegaconf import OmegaConf

from minimaxspeech.modules.common.utils import wav_to_mel_cloning
from minimaxspeech.modules.flow_matching.flow_matching import MaskedDiffWithXvec
from minimaxspeech.modules.flow_vae.flow_vae import DAC_FLOW_VAE
from minimaxspeech.modules.gpt.gpt import GPT
from minimaxspeech.modules.hifigan.generator import HiFTGenerator
from minimaxspeech.modules.vq_vae.dvae import DiscreteVAE, TorchMelSpectrogram
from minimaxspeech.utils.audio_utils.matcha_mel import mel_spectrogram

class MiniMaxSpeechModel:
    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config.model.sample_rate
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

        ckpt_path = self.config.model.hift.checkpoint
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(ckpt_path, map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        
    def setup_model(self):
        self.torch_mel_spectrogram_vq_vae = TorchMelSpectrogram(
            mel_norm_file=self.config.model.vq_vae.mel_norm_file,
            sampling_rate=self.config.model.vq_vae.sample_rate
        ).to(self.device)

        self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
            filter_length=2048,
            hop_length=256,
            win_length=1024,
            normalize=False,
            sampling_rate=self.config.model.vq_vae.sample_rate,
            mel_fmin=0,
            mel_fmax=8000,
            n_mel_channels=80,
            mel_norm_file=self.config.model.vq_vae.mel_norm_file,
        )

        vq_vae_config = OmegaConf.load(self.config.model.vq_vae.config)
        self.vq_vae = DiscreteVAE(**vq_vae_config.model.vq_vae).to(self.device)
        self.mel_stats = torch.load(self.config.model.vq_vae.mel_norm_file)
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

        hift_config = OmegaConf.load(self.config.model.hift.config)
        self.hift = HiFTGenerator(**hift_config.model.hift).to(self.device)

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 4):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio: (b, 1, t)
            sr (int): Sample rate of the audio.
            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.
            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio
                is being used without chunking. It must be < `length`. Defaults to 6.
        """
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        if self.gpt.use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]

                # if the chunk is too short ignore it 
                if audio_chunk.size(-1) < 22050 * 0.33:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.gpt.get_style_emb(mel_chunk.to(audio_chunk.device), None) # (b, 1024, 32)
                style_embs.append(style_emb)

            # mean style embedding
            cond_latent = torch.stack(style_embs).mean(dim=0) # (1, 1024, 32)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    def generate(self, prompt_audio, text_tokens):
        """
        Args:
            prompt_audio: (1, t)
            text: (t_text)

        Returns:
            audio: (b, 1, t)
        """
        with torch.no_grad():
            """ GPT inference """
            gpt_cond_latent = self.get_gpt_cond_latents(prompt_audio, self.sample_rate)
            
            self.gpt.init_gpt_for_inference()
            hf_generate_kwargs = {
                "input_tokens": None,
                "do_sample": False,
                "top_p": 0.85,
                "top_k": 50,
                "temperature": 0.75,
                "num_return_sequences": 1,
                "num_beams": 1,
                "length_penalty": 1.0,
                "repetition_penalty": 5.0,
                "output_attentions": False,
                "voice_dirs": None,
                "d_vector": None
            }
            audio_tokens = self.gpt.generate(
                cond_latents=gpt_cond_latent,
                text_inputs=text_tokens.unsqueeze(0),
                **hf_generate_kwargs
            )[:,:-1] # remove the last token
            logging.info(f"audio_tokens.shape: {audio_tokens.shape}, audio_tokens: {audio_tokens}")

            """ Flow matching inference """
            prompt_audio = prompt_audio.unsqueeze(0)
            prompt_feat = self.torch_mel_spectrogram_vq_vae(prompt_audio[:, :, :int(22050 * 1 // 1024 * 1024 - 1)])

            # tmp_feat = self.torch_mel_spectrogram_vq_vae(prompt_audio)
            # tmp_token = self.vq_vae.get_codebook_indices(tmp_feat)

            prompt_token = self.vq_vae.get_codebook_indices(prompt_feat)
            prompt_token_len = torch.tensor([prompt_token.shape[1]]).to(self.device)

            # prompt_z = self.flow_vae.encode(prompt_audio[:, :, :int(22050 * 1 // 1024 * 1024)]).transpose(1, 2) # (1, t, d)
            prompt_z = mel_spectrogram(prompt_audio.squeeze(0)[:, :int(22050 * 1 // 1024 * 1024)]).transpose(1, 2) # (1, t, d)
            prompt_z_len = torch.tensor([prompt_z.shape[1]]).to(self.device)

            cond_mels = self.torch_mel_spectrogram_style_encoder(prompt_audio[:, :, :22050*6])
            embedding = self.gpt.get_style_emb(cond_mels)
            embedding = torch.mean(embedding, dim=2) # (b, d, 32) -> (b, d)

            flow_cache = torch.zeros(1, self.flow_matching.output_size, 0, 2)

            latent, flow_cache = self.flow_matching.inference(
                token=audio_tokens,
                token_len=audio_tokens.shape[1],
                prompt_token=prompt_token,
                prompt_token_len=prompt_token_len,
                prompt_feat=prompt_z, # (1, t', d)
                prompt_feat_len=prompt_z_len,
                embedding=embedding,
                flow_cache=flow_cache
            )

            # latent, flow_cache = self.flow_matching.inference_without_prompt(
            #     token=audio_tokens,
            #     token_len=torch.tensor([audio_tokens.shape[1]]).to(self.device),
            #     embedding=embedding,
            #     flow_cache=flow_cache
            # )

            fig, ax = plt.subplots(figsize=(10, 2))
            im = ax.imshow(latent[0].cpu(), aspect="auto", origin="lower", interpolation="none")
            plt.colorbar(im, ax=ax)
            plt.savefig('data/LJ001-0001-infer-matcha.png')

            pred_gt, _ = self.hift.inference(latent)

            # """ FlowVAE inference """
            # audio = self.flow_vae.decode(latent)
            return pred_gt