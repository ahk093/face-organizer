#!/bin/bash
# Kullanım: ./export_person.sh Kisi_001 HedefKlasorAdi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Kullanım: ./export_person.sh <kisi_klasoru> <hedef_klasor_adi>"
    echo "Örnek: ./export_person.sh Kisi_001 Ahmet"
    exit 1
fi

PERSON_DIR="/Volumes/BACKUP/backup/_Kisiler/$1"
OUTPUT_DIR="/Volumes/BACKUP/backup/_Export/$2"

if [ ! -d "$PERSON_DIR" ]; then
    echo "Hata: $PERSON_DIR bulunamadı"
    exit 1
fi

# Hedef klasörü oluştur
mkdir -p "$OUTPUT_DIR"

echo "Kopyalanıyor: $1 -> $2"

count=0
for link in "$PERSON_DIR"/*; do
    if [ -L "$link" ] || [ -f "$link" ]; then
        filename=$(basename "$link")
        # Gizli dosyaları ve metadata'yı atla
        if [[ "$filename" == "_YUZ.jpg" ]] || [[ "$filename" == .* ]] || [[ "$filename" == ._* ]]; then
            continue
        fi
        cp -L "$link" "$OUTPUT_DIR/" 2>/dev/null && ((count++))
    fi
done

# macOS metadata dosyalarını temizle
find "$OUTPUT_DIR" -name '._*' -delete 2>/dev/null
find "$OUTPUT_DIR" -name '.DS_Store' -delete 2>/dev/null

echo "✅ $count fotoğraf kopyalandı"
echo "📁 Klasör: $OUTPUT_DIR"
open "$OUTPUT_DIR"
