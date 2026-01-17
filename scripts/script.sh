GNU nano 7.2                                                                ML.sh
RAW_DIR="data/raw"
PROCESSED_DIR="data/processed"
RAW_FILE="$RAW_DIR/phishing_dataset.csv"
PROCESSED_FILE="$PROCESSED_DIR/clean_dataset.csv"

echo "=============================================="
echo " 🚀 Iniciando preparación de datos para MLPhish"
echo "=============================================="

if [ ! -f "$RAW_FILE" ]; then
    echo "❌ Error: No se encontró el archivo $RAW_FILE"
    echo "Por favor, coloca tu dataset CSV en la carpeta data/raw/"
    exit 1
fi

mkdir -p "$PROCESSED_DIR"

echo "🧹 Limpiando dataset..."
awk 'NF' "$RAW_FILE" > "$PROCESSED_FILE"

LINES=$(wc -l < "$PROCESSED_FILE")
echo "✅ Dataset procesado correctamente."
echo "📦 Guardado en: $PROCESSED_FILE"
echo "📊 Total de filas: $LINES"

echo "=============================================="
echo " ✅ Preparación de datos completada"
echo " Ahora puedes ejecutar: python src/train.py"
echo "=============================================="
