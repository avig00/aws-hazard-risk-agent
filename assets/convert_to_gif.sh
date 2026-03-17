#!/usr/bin/env bash
# Usage: ./assets/convert_to_gif.sh assets/demo.mov
# Output: assets/demo.gif
#
# Requires ffmpeg: brew install ffmpeg

INPUT="${1:-assets/demo.mov}"
OUTPUT="${INPUT%.mov}.gif"

if [ ! -f "$INPUT" ]; then
  echo "Error: file not found: $INPUT"
  exit 1
fi

echo "Converting $INPUT → $OUTPUT ..."

ffmpeg -i "$INPUT" \
  -vf "fps=12,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer" \
  -loop 0 \
  "$OUTPUT"

echo "Done: $OUTPUT"
