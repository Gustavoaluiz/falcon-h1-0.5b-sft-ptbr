accelerate launch \
    --config_file /home/jovyan/falcon-h1-0.5b-sft-ptbr/configs/zero3.yaml \
    sft.py \
    --config /home/jovyan/falcon-h1-0.5b-sft-ptbr/configs/sft_full.yaml \
    --model_name_or_path tiiuae/Falcon-H1-0.5B-Base \
    --run_name falcon-h1-0.5b-base-ptbr-sft \
    --attn_implementation="flash_attention_2" \
    --dataset_sample_size 11000 \
    --dataset_eval_fraction 0.05