#!/bin/bash
# ============================================================
# BEATs + kNN Anomaly Detection for DCASE 2026 Task 2
# ============================================================
# Usage: CUDA_VISIBLE_DEVICES=4 bash run_beats_knn.sh -d
# ============================================================

TZ=JST-9 date
echo "$0 $*"

dev_eval=$1
echo -e "\tdev_eval = '$dev_eval'"
echo

# Check args
if [ "${dev_eval}" != "-d" ] && [ "${dev_eval}" != "-e" ] \
    && [ "${dev_eval}" != "--dev" ] && [ "${dev_eval}" != "--eval" ]; then
    echo "$0: argument error"
    echo "usage: $0 ['-d' | '--dev' | '-e' | '--eval']"
    exit 1
fi

# Convert to flag
if [ "${dev_eval}" = "-d" ] || [ "${dev_eval}" = "--dev" ]; then
    DEV_FLAG="-d"
else
    DEV_FLAG="-e"
fi

# BEATs checkpoint path
BEATS_CKPT="./BEATs/BEATs_iter3_plus_AS2M.pt"

# Check if BEATs checkpoint exists
if [ ! -f "${BEATS_CKPT}" ]; then
    echo "BEATs checkpoint not found. Downloading..."
    mkdir -p BEATs
    cd BEATs
    # Download BEATs model code
    if [ ! -f "BEATs.py" ]; then
        wget -q https://raw.githubusercontent.com/microsoft/unilm/master/beats/BEATs.py
        echo "Downloaded BEATs.py"
    fi
    # Download pre-trained checkpoint (HuggingFace mirror)
    wget -q "https://huggingface.co/lpepino/beats_ckpts/resolve/main/BEATs_iter3_plus_AS2M.pt"
    echo "Downloaded BEATs_iter3_plus_AS2M.pt"
    cd ..
fi

# Dataset list
# dataset_list="DCASE2026T2ToyCar"
dataset_list+=" DCASE2026T2Generator"

# kNN parameters (k=1 cosine gives best balanced source/target AUC)
K=5

for dataset in $dataset_list; do
    echo ""
    echo "========================================"
    echo "Processing: ${dataset}"
    echo "========================================"
    python3 beats_knn.py \
        --dataset=${dataset} \
        ${DEV_FLAG} \
        --beats_ckpt=${BEATS_CKPT} \
        --k=${K} \
        --scoring=mahalanobis \
        --decision_threshold=0.9
done

# mahalanobis
# ensemble
# knn


echo ""
echo "========================================"
echo "All done!"
TZ=JST-9 date
