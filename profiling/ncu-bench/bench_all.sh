bsizes=(1 2 4 8 16)

for bsize in "${bsizes[@]}"; do
  echo "Running bsize=$bsize"
  sudo env "PATH=$PATH" "VIRTUAL_ENV=$VIRTUAL_ENV" ncu --profile-from-start off --set full --launch-count 500 -o whisper_b$bsize python3 bench.py whisper $bsize
done

for bsize in "${bsizes[@]}"; do
  echo "Running bsize=$bsize"
  sudo env "PATH=$PATH" "VIRTUAL_ENV=$VIRTUAL_ENV" ncu --profile-from-start off --set full --launch-count 1200 -o qwen_b$bsize python3 bench.py qwen $bsize
done
