#!/bin/bash

OUTPUT_DIR="data/infer"
LANGUAGE="EN"

# Frases según tipo
declare -A TEXTS_BY_TYPE
# MALE
# TEXTS_BY_TYPE["VAL"]="If Pinocchio cried much longer, the little woman thought he would melt away"
# TEXTS_BY_TYPE["TEST"]="the chief of staff helped me out, and at last the baron directed him to telephone the commandant"
# TEXTS_BY_TYPE["BALANCED"]="The box was thrown beside the parked truck."

# FEMALE
TEXTS_BY_TYPE["VAL"]="Fanny is the only one who is judged rightly throughout, who has been consistent."
TEXTS_BY_TYPE["TEST"]="Crawford has too much sense to stay here if he found himself in any danger from maria."
TEXTS_BY_TYPE["BALANCED"]="The source of the huge river is the clear spring."
TEXTS_BY_TYPE["BALANCED"]="Glue the sheet to the dark blue background."
# BALANCED: varias frases
# TEXTS_BY_TYPE["BALANCED"]="The birch canoe slid on the smooth planks.|Glue the sheet to the dark blue background.|It's easy to tell the depth of a well.|These days a chicken leg is a rare dish.|Rice is often served in round bowls.|The juice of lemons makes fine punch.|The box was thrown beside the parked truck.|The hogs were fed chopped corn and garbage.|Four hours of steady work faced us.|A large size in stockings is hard to sell."



# Modelos y sus checkpoints
MODELS=(
  "logs/config_female_newfunc_large_v2/G_1000.pth"
  "logs/config_female_newfunc_large_v2/G_2000.pth"
  "logs/config_female_newfunc_large_v2/G_3000.pth"
  "logs/config_female_newfunc_large_v2/G_4000.pth"
  "logs/config_female_newfunc_large_v2/G_5000.pth"
  "logs/config_female_newfunc_large_v2/G_6000.pth"
  "logs/config_female_newfunc_large_v2/G_7000.pth"
  "logs/config_female_newfunc_large_v2/G_8000.pth"
  "logs/config_female_newfunc_large_v2/G_9000.pth"
  "logs/config_female_newfunc_large_v2/G_10000.pth"
  "logs/config_female_newfunc_large_v2/G_12000.pth"
  "logs/config_female_newfunc_large_v2/G_13000.pth"
  "logs/config_female_newfunc_large_v2/G_15000.pth"
  "logs/config_female_newfunc_large_v2/G_20000.pth"
  "logs/config_female_newfunc_large_v2/G_25000.pth"
)

# Inferencia por tipo
# for TYPE in VAL TEST BALANCED; do
#   TEXT="${TEXTS_BY_TYPE[$TYPE]}"
#   echo "[INFO] Frase para $TYPE: $TEXT"

#   for CKPT_PATH in "${MODELS[@]}"; do
#     CKPT_FILE=$(basename "$CKPT_PATH")           # Ej: G_1000.pth
#     CKPT_NAME="${CKPT_FILE%.*}"                  # Ej: G_1000
#     TYPE_LOWER=$(echo "$TYPE" | tr '[:upper:]' '[:lower:]')

#     # Carpeta de destino
#     DEST_DIR="$OUTPUT_DIR/$TYPE_LOWER"
#     mkdir -p "$DEST_DIR"

#     echo "[INFO] Ejecutando: tipo=$TYPE, modelo=$CKPT_NAME"

#     python3 infer.py \
#       --ckpt_path "$CKPT_PATH" \
#       --text "$TEXT" \
#       --language "$LANGUAGE" \
#       --output_dir "$DEST_DIR" \
#       --type "$TYPE" \
#       --output_name "${TYPE_LOWER}_${CKPT_NAME}.wav"
#   done
# done

for TYPE in VAL TEST BALANCED; do
  # Convertir cadena a array usando '|'
  IFS='|' read -r -a TEXTS <<< "${TEXTS_BY_TYPE[$TYPE]}"

  for i in "${!TEXTS[@]}"; do
    TEXT="${TEXTS[$i]}"
    echo "[INFO] Frase $i para $TYPE: $TEXT"

    for CKPT_PATH in "${MODELS[@]}"; do
      CKPT_FILE=$(basename "$CKPT_PATH")
      CKPT_NAME="${CKPT_FILE%.*}"
      TYPE_LOWER=$(echo "$TYPE" | tr '[:upper:]' '[:lower:]')
      DEST_DIR="$OUTPUT_DIR/$TYPE_LOWER"
      mkdir -p "$DEST_DIR"

      python3 infer.py \
        --ckpt_path "$CKPT_PATH" \
        --text "$TEXT" \
        --language "$LANGUAGE" \
        --output_dir "$DEST_DIR" \
        --type "$TYPE" \
        --output_name "${TYPE_LOWER}_${i}_${CKPT_NAME}.wav"
    done
  done
done


echo "All the inferences completed."

