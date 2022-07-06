#!/bin/bash
set -eu

# jsquad needs a preprocessing:
# ```
# python convert_dataset.py jsquad \
#   -i /pathto/JGLUE/datasets/jsquad-v1.0 \
#   -o /pathto/JGLUE/datasets/jsquad-v1.0-preprocessed
# ```

# set your own dir
SCRIPT_DIR="./scripts"
MODEL_ROOT="./bert"
JGLUE_DATA_DIR="./datasets/JGLUE/datasets"
JGLUE_VERSION="v1.0"
OUTPUT_ROOT="./out"
LOG_DIR="./logs"


# model to search
MODEL_NAMES=("chitra_v1.0")
DATASETS=("marc-ja" "jsts" "jnli" "jsquad" "jcommonsenseqa")

# Hyperparameters from Appendix A.3, Devlin et al., 2019
BATCHES=(32)
LRS=(5e-5 3e-5 2e-5)
EPOCHS=(3 4)

# set path to the model files
declare -A MODEL_DIRS=(
  ["chitra_v1.0"]="${MODEL_ROOT}/chitra-v1.0/"
)

declare -A DATASET_DIRS=(
  ["marc-ja"]="${JGLUE_DATA_DIR}/marc_ja-${JGLUE_VERSION}"
  ["jsts"]="${JGLUE_DATA_DIR}/jsts-${JGLUE_VERSION}"
  ["jnli"]="${JGLUE_DATA_DIR}/jnli-${JGLUE_VERSION}"
  ["jsquad"]="${JGLUE_DATA_DIR}/jsquad-${JGLUE_VERSION}-preprocessed"
  ["jcommonsenseqa"]="${JGLUE_DATA_DIR}/jcommonsenseqa-${JGLUE_VERSION}"
)

declare -A MAX_SEQ_LEN_MAP=(
  ["marc-ja"]=512
  ["jsts"]=128
  ["jnli"]=128
  ["jsquad"]=384
  ["jcommonsenseqa"]=64
)

function set_model_args() {
  MODEL=$1
  DATASET=$2
  MODEL_DIR="${MODEL_DIRS[$1]}"
  DATASET_DIR="${DATASET_DIRS[$2]}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}_${DATASET}/${LR}_${BATCH}_${EPOCH}/"
  export MODEL DATASET MODEL_DIR DATASET_DIR OUTPUT_DIR

  MAX_SEQ_LEN=${MAX_SEQ_LEN_MAP[$2]}
  export MAX_SEQ_LEN

  # pretokenizer
  PRETOKENIZER="identity"
  if [ ${MODEL} = "kyoto" ] ; then
    PRETOKENIZER="juman"
  elif [ ${MODEL} = "nict" ] ; then
    PRETOKENIZER="mecab-juman"
  fi
  export PRETOKENIZER

  # tokenizer (sudachi)
  TOKENIZER=${MODEL_DIR}
  if [ ${MODEL:0:6} = "chitra" ] ; then
    TOKENIZER="sudachi"
  fi
  export TOKENIZER
}

export SCRIPT_PATH="${SCRIPT_DIR}/run_evaluation.py"

function run_script() {
  python ${SCRIPT_PATH} \
    --model_name_or_path          ${MODEL_DIR} \
    --pretokenizer_name           ${PRETOKENIZER} \
    --tokenizer_name              ${TOKENIZER} \
    --dataset_name                ${DATASET} \
    --do_train --do_eval --do_predict \
    --train_file                  "${DATASET_DIR}/train-${JGLUE_VERSION}.json" \
    --validation_file             "${DATASET_DIR}/valid-${JGLUE_VERSION}.json" \
    --test_file                   "${DATASET_DIR}/valid-${JGLUE_VERSION}.json" \
    --output_dir                  ${OUTPUT_DIR} \
    --gradient_accumulation_steps $((BATCH / 8)) \
    --per_device_eval_batch_size  64 \
    --per_device_train_batch_size 8 \
    --learning_rate               ${LR} \
    --num_train_epochs            ${EPOCH} \
    --warmup_ratio                0.1 \
    --evaluation_strategy         epoch \
    --overwrite_cache             \
    --save_total_limit            1 \
    --max_seq_length              ${MAX_SEQ_LEN} \
    # --max_train_samples           100 \
    # --max_val_samples             100 \
    # --max_test_samples            100 \
}

# mkdir for log
mkdir -p ${LOG_DIR}

for DATASET in ${DATASETS[@]}; do
  for MODEL in ${MODEL_NAMES[@]}; do
    for BATCH in ${BATCHES[@]}; do
      for LR in ${LRS[@]}; do
        for EPOCH in ${EPOCHS[@]}; do
          export BATCH LR EPOCH
          set_model_args ${MODEL} ${DATASET}

          run_script 2> ${LOG_DIR}/${MODEL}_${DATASET}_batch${BATCH}_lr${LR}_epochs${EPOCH}.log
        done
      done
    done
  done
done
