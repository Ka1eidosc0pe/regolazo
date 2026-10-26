#!/usr/bin/env bash
# Verifies ffmpeg can encode H.264 video and mux audio (needed for demo video assembly).
set -euo pipefail
out="$(mktemp -d)"
trap 'rm -r "$out"' EXIT
ffmpeg -hide_banner -loglevel error -f lavfi -i testsrc2=size=1920x1080:rate=60 -f lavfi -i sine=frequency=440 \
  -t 2 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$out/check.mp4"
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height -of compact "$out/check.mp4"
echo "ffmpeg OK"
