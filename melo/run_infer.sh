#!/bin/bash

OUTPUT_DIR="data/infer"
LANGUAGE="EN"

# Frases según tipo
declare -A TEXTS_BY_TYPE
TEXTS_BY_TYPE["VAL"]="If Pinocchio cried much longer, the little woman thought he would melt away"
TEXTS_BY_TYPE["TEST"]="the chief of staff helped me out, and at last the baron directed him to telephone the commandant"
TEXTS_BY_TYPE["BALANCED"]="The box was thrown beside the parked truck."

# Modelos y sus checkpoints
MODELS=(
  # "logs/config_male_losscomb2/G_0.pth"
  # "logs/config_male_losscomb2/G_1000.pth"
  # "logs/config_male_losscomb2/G_2000.pth"
  # "logs/config_male_losscomb2/G_4000.pth"
  # "logs/config_male_losscomb2/G_6000.pth"
  # "logs/config_male_losscomb2/G_7000.pth"
  # "logs/config_male_losscomb2/G_8000.pth"
  # "logs/config_male_losscomb2/G_10000.pth"
  "logs/config_male_losscomb2/G_12000.pth"
  # "logs/config_male_losscomb2/G_14000.pth"
)

# Inferencia por tipo
for TYPE in VAL TEST BALANCED; do
  TEXT="${TEXTS_BY_TYPE[$TYPE]}"
  echo "[INFO] Frase para $TYPE: $TEXT"

  for CKPT_PATH in "${MODELS[@]}"; do
    CKPT_FILE=$(basename "$CKPT_PATH")           # Ej: G_1000.pth
    CKPT_NAME="${CKPT_FILE%.*}"                  # Ej: G_1000
    TYPE_LOWER=$(echo "$TYPE" | tr '[:upper:]' '[:lower:]')

    # Carpeta de destino
    DEST_DIR="$OUTPUT_DIR/$TYPE_LOWER"
    mkdir -p "$DEST_DIR"

    echo "[INFO] Ejecutando: tipo=$TYPE, modelo=$CKPT_NAME"

    python3 infer.py \
      --ckpt_path "$CKPT_PATH" \
      --text "$TEXT" \
      --language "$LANGUAGE" \
      --output_dir "$DEST_DIR" \
      --type "$TYPE" \
      --output_name "${TYPE_LOWER}_${CKPT_NAME}.wav"
  done
done

echo "All the inferences completed."

