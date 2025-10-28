#!/bin/bash
#
# Script: copiar.sh
# Uso: ./copiar.sh -n NUM_LINEAS -s LOCUTOR_ID
#
# Ejemplo:
#   ./copiar.sh -n 100 -s 11
#

# ==== CONFIGURACIÓN GENERAL ====
BASE_ORIGEN="data/dataset_10"      # Carpeta raíz de todos los locutores
DESTINO="data/example"              # Carpeta de destino donde montar el paquete
TRAIN_LIST="train.list"
VAL_LIST="val.list"
METADATA="metadata.list"
METADATA_CLEAN="metadata.list.cleaned"
CONFIG="config.json"

# ==== PARÁMETROS ====
NUM_LINEAS=0
LOCUTOR_ID=1

while getopts "n:s:" opt; do
  case $opt in
    n) NUM_LINEAS=$OPTARG ;;
    s) LOCUTOR_ID=$OPTARG ;;
    *) echo "Uso: $0 -n NUM_LINEAS -s LOCUTOR_ID"
       exit 1 ;;
  esac
done

# ==== DICCIONARIO DE LOCUTORES ====
declare -A LOCUTORES
LOCUTORES[11]="K_Savage"
LOCUTORES[12]="L_Caldwell"
LOCUTORES[13]="LA_Walden"
LOCUTORES[14]="S_Denton"
LOCUTORES[15]="JM_Smallheer"
LOCUTORES[21]="D_Clarke"
LOCUTORES[22]="J_Greenman"
LOCUTORES[23]="M_Geeson"
LOCUTORES[24]="MF_Smith"
LOCUTORES[25]="P_Chenevert"

LOCUTOR_NAME=${LOCUTORES[$LOCUTOR_ID]}
if [[ -z "$LOCUTOR_NAME" ]]; then
    echo "Locutor ID '$LOCUTOR_ID' no está definido en el diccionario"
    exit 1
fi

# Carpeta origen de este locutor
ORIGEN="$BASE_ORIGEN/$LOCUTOR_NAME"

echo "Preparando listas/config"
echo "Locutor: $LOCUTOR_ID → $LOCUTOR_NAME"
echo "Origen:  $ORIGEN"
echo "Destino: $DESTINO"

mkdir -p "$DESTINO"

# ==== LANZAR MONITOREO EN SEGUNDO PLANO ====
if [[ -x ./monitor.sh ]]; then
    echo "Lanzando monitor de recursos..."
    ./monitor.sh 30 &
    MONITOR_PID=$!
    echo "Monitor corriendo en segundo plano con PID $MONITOR_PID"
else
    echo "No se encontró monitor.sh"
fi

# ==== CREAR NUEVOS LISTADOS ====
# 1. Seleccionar primeras N líneas de train.list
if [[ $NUM_LINEAS -gt 0 ]]; then
    head -n "$NUM_LINEAS" "$ORIGEN/$TRAIN_LIST" > "$DESTINO/$TRAIN_LIST"
else
    cp "$ORIGEN/$TRAIN_LIST" "$DESTINO/$TRAIN_LIST"
fi

# 2. Copiar val.list completo
cp "$ORIGEN/$VAL_LIST" "$DESTINO/$VAL_LIST"

# 3. metadata.list reducido
if [[ $NUM_LINEAS -gt 0 ]]; then
    head -n "$NUM_LINEAS" "$ORIGEN/$METADATA" > "$DESTINO/$METADATA"
else
    cp "$ORIGEN/$METADATA" "$DESTINO/$METADATA"
fi

# 4. metadata_cleaned.list reducido
if [[ $NUM_LINEAS -gt 0 ]]; then
    head -n "$NUM_LINEAS" "$ORIGEN/$METADATA_CLEAN" > "$DESTINO/$METADATA_CLEAN"
else
    cp "$ORIGEN/$METADATA_CLEAN" "$DESTINO/$METADATA_CLEAN"
fi

# 5. Copiar config.json tal cual
cp "$ORIGEN/$CONFIG" "$DESTINO/$CONFIG"

echo "Paquete generado en $DESTINO"

# ==== LLAMAR A OTRO SCRIPT ====
if [[ -x ./train.sh ]]; then
    echo "Executing train.sh"
    ./train.sh
else
    echo "No se encontró procesar_copiados.sh o no tiene permisos de ejecución"
fi