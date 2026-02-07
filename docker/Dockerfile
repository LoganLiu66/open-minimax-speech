FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN apt update && apt install git vim -y
RUN bash minimaxspeech/utils/audio_utils/visqol_install.sh
RUN pip install -r requirements.txt