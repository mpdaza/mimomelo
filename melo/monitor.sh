#!/bin/bash
#
# Monitor completo CPU + GPU con archivo incremental
#

# Carpeta de salida
OUTDIR="logs/config/monitoring"
mkdir -p "$OUTDIR"

# Buscar siguiente número disponible
INDEX=0
while [[ -f "$OUTDIR/monitor${INDEX}.csv" ]]; do
    INDEX=$((INDEX + 1))
done

OUTFILE="$OUTDIR/monitor${INDEX}.csv"
INTERVAL=${1:-5}   # intervalo en segundos (default: 5s)

echo "Guardando monitoreo en: $OUTFILE (cada $INTERVAL s)"

# Cabecera CSV
echo "timestamp,cpu_percent,cpu_temp_C,mem_total_MB,mem_used_MB,mem_free_MB,gpu_percent,gpu_temp_C,gpu_mem_used_MB,gpu_mem_total_MB,gpu_mem_percent" > "$OUTFILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # ==== CPU ====
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}')
    
    if command -v sensors >/dev/null 2>&1; then
        CPU_TEMP=$(sensors | grep -m1 "Package id 0" | awk '{print $4}' | tr -d '+°C')
        [ -z "$CPU_TEMP" ] && CPU_TEMP=NA
    else
        CPU_TEMP=NA
    fi

    MEM_TOTAL=$(free -m | awk '/Mem:/ {print $2}')
    MEM_USED=$(free -m | awk '/Mem:/ {print $3}')
    MEM_FREE=$(free -m | awk '/Mem:/ {print $4}')

    # ==== GPU ====
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_QUERY=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.total,memory.used \
                    --format=csv,noheader,nounits | head -n1)
        GPU_UTIL=$(echo $GPU_QUERY | awk -F',' '{print $1}' | xargs)
        GPU_TEMP=$(echo $GPU_QUERY | awk -F',' '{print $2}' | xargs)
        GPU_MEM_TOTAL=$(echo $GPU_QUERY | awk -F',' '{print $3}' | xargs)
        GPU_MEM_USED=$(echo $GPU_QUERY | awk -F',' '{print $4}' | xargs)

        # Validar que sean números antes de calcular
        if [[ "$GPU_MEM_USED" =~ ^[0-9]+$ ]] && [[ "$GPU_MEM_TOTAL" =~ ^[0-9]+$ ]]; then
            GPU_MEM_PERCENT=$((100 * GPU_MEM_USED / GPU_MEM_TOTAL))
        else
            GPU_UTIL=NA
            GPU_TEMP=NA
            GPU_MEM_TOTAL=NA
            GPU_MEM_USED=NA
            GPU_MEM_PERCENT=NA
        fi
    else
        GPU_UTIL=NA
        GPU_TEMP=NA
        GPU_MEM_TOTAL=NA
        GPU_MEM_USED=NA
        GPU_MEM_PERCENT=NA
    fi

    # ==== Guardar en CSV ====
    echo "$TIMESTAMP,$CPU,$CPU_TEMP,$MEM_TOTAL,$MEM_USED,$MEM_FREE,$GPU_UTIL,$GPU_TEMP,$GPU_MEM_USED,$GPU_MEM_TOTAL,$GPU_MEM_PERCENT" >> "$OUTFILE"

    sleep "$INTERVAL"
done
