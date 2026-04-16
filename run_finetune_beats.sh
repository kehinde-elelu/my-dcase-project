#!/bin/bash
# ============================================================
# Fine-tune BEATs with contrastive learning for DCASE 2026 Task 2
# ============================================================
# Usage: CUDA_VISIBLE_DEVICES=4 bash run_finetune_beats.sh -d
# ============================================================

TZ=JST-9 date
echo "$0 $*"

dev_eval=$1
echo -e "\tdev_eval = '$dev_eval'"
echo

if [ "${dev_eval}" != "-d" ] && [ "${dev_eval}" != "-e" ] \
    && [ "${dev_eval}" != "--dev" ] && [ "${dev_eval}" != "--eval" ]; then
    echo "$0: argument error"
    echo "usage: $0 ['-d' | '--dev' | '-e' | '--eval']"
    exit 1
fi

if [ "${dev_eval}" = "-d" ] || [ "${dev_eval}" = "--dev" ]; then
    DEV_FLAG="-d"
else
    DEV_FLAG="-e"
fi

BEATS_CKPT="./BEATs/BEATs_iter3_plus_AS2M.pt"

EPOCHS=50
BATCH_SIZE=16
LR=1e-4
UNFREEZE_LAYERS=4
SEGMENT_SEC=4.0

dataset_list="DCASE2026T2ToyCar"

for dataset in $dataset_list; do
    echo ""
    echo "========================================"
    echo "Fine-tuning BEATs on: ${dataset}"
    echo "========================================"
    python3 finetune_beats.py \
        --dataset=${dataset} \
        ${DEV_FLAG} \
        --beats_ckpt=${BEATS_CKPT} \
        --epochs=${EPOCHS} \
        --batch_size=${BATCH_SIZE} \
        --lr=${LR} \
        --unfreeze_layers=${UNFREEZE_LAYERS} \
        --segment_sec=${SEGMENT_SEC}

    # Evaluate with fine-tuned model
    FINETUNED_CKPT="./models/saved_model/beats_finetuned/${dataset}/beats_finetuned_best.pt"
    if [ -f "${FINETUNED_CKPT}" ]; then
        echo ""
        echo "----------------------------------------"
        echo "Evaluating fine-tuned model: ${dataset}"
        echo "----------------------------------------"
        python3 beats_knn.py \
            --dataset=${dataset} \
            ${DEV_FLAG} \
            --beats_ckpt=${FINETUNED_CKPT} \
            --k=1 \
            --scoring=ensemble \
            --decision_threshold=0.5
    fi
done

echo ""
echo "========================================"
echo "All done!"
TZ=JST-9 date