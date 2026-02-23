import argparse
import glob
import json
import os
import random
from collections import defaultdict
from tqdm import tqdm


def main():
    wavs = list(glob.glob('{}/*/*/*wav'.format(args.src_dir)))

    f_w = open(args.output_file, 'w', encoding='utf-8')

    spk2audio = defaultdict(list)
    audios = []
    for wav in tqdm(wavs):
        txt = wav.replace('.wav', '.normalized.txt')
        if not os.path.exists(txt):
            print('{} do not exsist'.format(txt))
            continue
        with open(txt) as f:
            content = ''.join(l.replace('\n', '') for l in f.readline())

        audio_id = os.path.basename(wav).replace('.wav', '')
        spk = audio_id.split('_')[0]
        audio = {
            'audio_id': audio_id,
            'audio_file': wav,
            'spk': spk,
            'lang': 'en',
            'text': content.strip()
        }
        spk2audio[spk].append(wav)
        audios.append(audio)

    for audio in audios:
        if len(spk2audio[audio['spk']]) > 1:
            audio['ref_audio_file'] = random.choice(spk2audio[audio['spk']])
            f_w.write(json.dumps(audio) + '\n')
    
    f_w.close()
    print(f'Saved {len(audios)} audio files to {args.output_file}')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    main()