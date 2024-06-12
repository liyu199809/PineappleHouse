if [ ! -d "/mnt/bn/liyullm2/kdd_output" ]; then
    mkdir -p /mnt/bn/liyullm2/kdd_output
    echo "mkdir output"
fi

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo "mkdir logs"
fi


accelerate launch --main_process_port=25641 --config_file ../../config/default_config.yaml train_embedding.py  \
--model_name_or_path "/mnt/bn/liyullm2/gte-large-en-v1.5" \
--dataset "../../example_data/pretrain_data.json" \
--output_dir "/mnt/bn/liyullm2/kdd_output/pretrain" \
--batch_size 24 \
--lr 1e-5 \
--epochs 5 \
--save_on_epoch_end 1 \
--gradient_accumulation_steps 24  \
--log_with 'wandb' \
--warmup_proportion 0.1 \
--neg_nums 5 \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 512 \