stage=1
stop_stage=3

data_url=www.openslr.org/resources/60
data_dir=./data/tts/openslr/libritts

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Data Download"
  for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
    scripts/download_and_untar.sh ${data_dir} ${data_url} ${part}
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Data preparation, prepare audio files and metadata"
  for x in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    output_file=$data_dir/$x.jsonl
    python ./scripts/prepare_data.py --src_dir $data_dir/LibriTTS/$x --output_file $output_file
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Data preparation, prepare train/valid/test metadata"
  cat $data_dir/{train-clean-100,train-clean-360,train-other-500}.jsonl > $data_dir/libritts_train.jsonl
  cat $data_dir/{dev-clean,dev-other}.jsonl > $data_dir/libritts_valid.jsonl
  cat $data_dir/{test-clean,test-other}.jsonl > $data_dir/libritts_test.jsonl
fi