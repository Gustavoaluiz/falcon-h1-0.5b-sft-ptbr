accelerate launch \
    --config_file configs/zero3.yaml \
    sft.py \
    --config configs/sft_full.yaml \
    --model_name_or_path tiiuae/Falcon-H1-0.5B-Base \
    --run_name falcon-h1-0.5b-base-ptbr-sft \
    --attn_implementation="kernels-community/flash-attn" \
    --dataset_sample_size 200000 \
    --dataset_eval_fraction 0.05