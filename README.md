# Open-Minimax-Speech(WIP)
An Unofficial Implementation of [MiniMax-Speech](https://arxiv.org/abs/2505.07916)
![An overview of the architecture of MiniMax-Speech](./assets/fig/minimax-speech.png)
|SubModule|Params|Details|
|---------|---------|---------|
|VQ-VAE|51M (Encoder: 24M)|DVAE from [Tortoise](https://github.com/neonbjb/DL-Art-School/blob/master/codes/models/audio/tts/lucidrains_dvae.py)|
|LLM|441M|GPT-2 from [XTTS](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/xtts/gpt.py)|
|Flow Matching|112M|Conditional Flow Matching from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py)|
|Flow-VAE|61M (Encoder: 18M, Decoder: 41M)|Encoder(Conv), Decoder(Conv, MRF) and Discriminator(MPD, MSD, MRD) are from [DAC](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py) <br> Flow(RealNVP) from [VITS](https://github.com/heatz123/naturalspeech/blob/main/models/models.py#L577)|

## Installation

```bash

docker run --gpus all --net=host --shm-size=64g -it \
    --name minimaxspeech_tmp \
    -v /path/to/code:/path/to/code \
    pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime \
    bash

conda create -n minimaxspeech python=3.10
conda activate minimaxspeech
pip install -r requirements.txt
```

## Usage
### Data Preparation
For LibriTTS dataset preparation, please refer to [scripts/libritts_data_prepare.sh](scripts/libritts_data_prepare.sh).
This script will download the dataset and prepare the metadata files.

```bash
bash scripts/libritts_data_prepare.sh
```

### Training
#### Step1. VQ-VAE
To train from scratch on LibriTTS dataset, you can run the following command:

```bash
python minimaxspeech/trainers/vq_vae_trainer.py \
    --config configs/vq_vae_config_libritts.yaml

# Multi-GPU with DDP
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun --nproc_per_node=8  minimaxspeech/trainers/vq_vae_trainer.py \
      --config configs/vq_vae_config_libritts.yaml
```

You can also finetune a pretrained model by runing:

```bash
# Download Pretrained Models
wget -O ./checkpoints/xtts2/dvae.pth https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth
wget -O ./checkpoints/xtts2/mel_stats.pth https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth

python minimaxspeech/trainers/vq_vae_trainer.py \
    --config configs/vq_vae_config_libritts_ft.yaml
```
#### Step2. GPT2
To train the GPT model, you can run the following command:

```bash
python minimaxspeech/trainers/gpt_trainer.py \
    --config configs/gpt_config_libritts.yaml

# Multi-GPU with DDP
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun --nproc_per_node=8 minimaxspeech/trainers/gpt_trainer.py \
    --config configs/gpt_config_libritts.yaml
```

To finetune the GPT model, you can run the following command:

```bash
# Download Pretrained Models
wget -O ./checkpoints/xtts2/gpt.pth https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth
wget -O ./checkpoints/xtts2/vocab.json https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json

python minimaxspeech/trainers/gpt_trainer.py \
    --config configs/gpt_config_libritts_ft.yaml
```

#### Step3. Flow-VAE

##### Train

```bash
# Single GPU
python minimaxspeech/trainers/flow_vae_trainer.py \
    --config configs/flow_vae_config_libritts.yaml

# Multi-GPU with DDP
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun --nproc_per_node=8 minimaxspeech/trainers/flow_vae_trainer.py \
    --config configs/flow_vae_config_libritts.yaml
```

##### Finetune
If you want to finetune the flow-vae by utilizing the official encoder and decoder weights of DAC, please download the [official weights](https://github.com/descriptinc/audiotools?tab=readme-ov-file) and move them to the `checkpoints/dac` from the `~/.cache/descript/dac/`. Then you can run the following command:

```bash
python minimaxspeech/trainers/flow_vae_trainer.py \
    --config configs/flow_vae_config_ft.yaml
```

##### Evaluation
The following command can be run to evaluate the reconstruction quality of Flow-VAE and obtain reconstructed audio:

```bash
python minimaxspeech/utils/audio_utils/flow_vae_audio_reconstruct.py \
    --config configs/flow_vae_config.yaml \
    --ckpt_file output/flow_vae/checkpoint_090000.pth \
    --input data/dac/evaluate/valid \
    --output data/dac/evaluate/valid_recon_flow_vae
```

Then compute the following audio quality metrics: PESQ, STOI, ViSQOL and Mel distance.

```bash
python minimaxspeech/utils/audio_utils/reconstruction_evaluate.py \
    --input data/dac/evaluate/valid \
    --output data/dac/evaluate/valid_recon_flow_vae
```

#### Step4. Flow Matching
To train the Flow Matching model (requires trained VQ-VAE, GPT, and Flow-VAE), you can run the following command:

```bash
python minimaxspeech/trainers/flow_matching_trainer.py \
    --config configs/flow_matching_config_libritts.yaml

# Multi-GPU with DDP
export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
torchrun --nproc_per_node=8 minimaxspeech/trainers/flow_matching_trainer.py \
    --config configs/flow_matching_config_libritts.yaml
```

### Inference

```bash
python minimaxspeech/cli/inference.py \
    --config configs/minimaxspeech_config.yaml \
    --prompt_audio data/LJ001-0001.wav \
    --text "Hello, how are you?" \
    --lang en \
    --output_file output/audio.wav
```

## Acknowledgements
- [XTTS](https://github.com/coqui-ai/TTS)
- [DAC](https://github.com/descriptinc/descript-audio-codec)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [Tortoise](https://github.com/neonbjb/tortoise-tts/tree/main/tortoise)

## Citation
If you use this work, please cite the original Minimax-Speech paper:

```
@article{zhang2025minimaxspeech,
      title={MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder}, 
      author={Bowen Zhang and Congchao Guo and Geng Yang and Hang Yu and Haozhe Zhang and Heidi Lei and Jialong Mai and Junjie Yan and Kaiyue Yang and Mingqi Yang and Peikai Huang and Ruiyang Jin and Sitan Jiang and Weihua Cheng and Yawei Li and Yichen Xiao and Yiying Zhou and Yongmao Zhang and Yuan Lu and Yucen He},
      year={2025},
      eprint={2505.07916},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.07916}, 
}
```