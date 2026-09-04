#!/usr/bin/env bash
set -e

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 file1.webm [file2.webm ...]"
    exit 1
fi

FPS=15
WIDTH=640

for infile in "$@"; do
    if [ ! -f "$infile" ]; then
        echo "skipping $infile (not found)"
        continue
    fi
    base="${infile%.*}"
    palette="/tmp/$(basename "$base")_palette.png"
    outfile="${base}.gif"

    echo "converting $infile -> $outfile"
    ffmpeg -y -i "$infile" -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos,palettegen" "$palette" -loglevel error
    ffmpeg -y -i "$infile" -i "$palette" -filter_complex "fps=$FPS,scale=$WIDTH:-1:flags=lanczos[x];[x][1:v]paletteuse" "$outfile" -loglevel error
    rm -f "$palette"
    echo "  done: $outfile ($(du -h "$outfile" | cut -f1))"
done
