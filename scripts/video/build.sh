#!/usr/bin/env bash
# Rebuilds data/video/regolazo_rough_cut.mp4 from the captured assets in data/video/raw/:
#   live.webm (+json)        live broadcast with overlays, narration caption and the Spotter panel  (capture.mjs)
#   rewind.webm              Rewind & Rewrite take from web/e2e/demo.spec.ts (copy of data/demo/regolazo_demo_mo_08.webm)
#   split_left/right.webm    Analyst EN and Casual ES captures of the same moment                    (capture.mjs)
#   annotated.mp4            Spotter's annotated clip (copied from data/matches/candidate-2004/spotter/)
# Cards (title, architecture, close, split-screen furniture) are rendered with Playwright from cards.html.
# Usage: scripts/video/build.sh        (needs ffmpeg, node, python3 and web/node_modules/playwright)
set -euo pipefail
cd "$(dirname "$0")/../.."
V=data/video; R=$V/raw; C=$V/cards; S=$V/seg; AUD=data/matches/candidate-2004/narratives/audio
mkdir -p "$R" "$C" "$S"
FFLOG="-hide_banner -loglevel error -y"
ENC="-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30"

# --- inputs ---
if [ -f data/matches/candidate-2004/spotter/annotated.mp4 ]; then cp data/matches/candidate-2004/spotter/annotated.mp4 "$R/annotated.mp4"; fi
[ -f data/demo/regolazo_demo_mo_08.webm ] && [ ! -f "$R/rewind.webm" ] && cp data/demo/regolazo_demo_mo_08.webm "$R/rewind.webm"
node scripts/video/render_cards.mjs >/dev/null
node scripts/video/render_split.mjs >/dev/null
jget() { python3 -c "import json,sys; d=json.load(open('$1')); print(eval(sys.argv[1]))" "$2"; }

# still PNG -> silent clip with 0.3 s fades.  still NAME PNG SECONDS
still() { ffmpeg $FFLOG -loop 1 -framerate 30 -t "$3" -i "$2" -vf "fade=in:0:9,fade=out:st=$(python3 -c "print($3-0.3)"):d=0.3,format=yuv420p" $ENC "$S/$1.v.mp4"; }
# add audio.  audio NAME [MP3 DELAY_S]: silent track when no mp3; otherwise normalised to -18 LUFS, delayed, faded out at the end
audio() {
  local n=$1 dur; dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$S/$n.v.mp4")
  if [ $# -ge 3 ]; then
    local d; d=$(python3 -c "print(int(max(0,$3)*1000))")
    ffmpeg $FFLOG -i "$S/$n.v.mp4" -i "$2" -filter_complex "[1:a]loudnorm=I=-18:TP=-2:LRA=7,aformat=sample_rates=48000:channel_layouts=stereo,adelay=$d|$d,apad,atrim=duration=$dur,afade=t=out:st=$(python3 -c "print($dur-1)"):d=1[a]" \
      -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 160k -shortest "$S/$n.mp4"
  else
    ffmpeg $FFLOG -i "$S/$n.v.mp4" -f lavfi -i anullsrc=r=48000:cl=stereo -c:v copy -c:a aac -b:a 160k -shortest "$S/$n.mp4"
  fi
}

# --- beat 1: title (8.0 s) ---
still b1a $C/hook1.png 3.2; still b1b $C/hook2.png 2.4; still b1c $C/wordmark.png 2.4
for n in b1a b1b b1c; do audio $n; done

# --- beat 2: live broadcast 17.5 s + Spotter annotated clip 5 s, EN casual narration under it ---
LS=$(jget $R/live.json "d['playStart']+0.3"); LAT=$(jget $R/live.json "d['captions'][0]['at']-0.3")
# annotated clip: t=1230 s of the match (the 21' chip) = offset 50 s into the 1180-1300 window; 960x540 -> 1080p
if [ -f "$R/annotated.mp4" ]; then
  ffmpeg $FFLOG -ss "$LS" -t 17.5 -i $R/live.webm -ss 50 -t 5 -i $R/annotated.mp4 -filter_complex \
    "[0:v]fps=30,scale=1920:1080,setpts=PTS-STARTPTS[a];[1:v]fps=30,scale=1920:1080:flags=lanczos,setpts=PTS-STARTPTS,fade=in:0:6[b];[a][b]concat=n=2:v=1:a=0[v]" -map "[v]" $ENC $S/b2.v.mp4
else
  # placeholder slot for the Spotter clip
  ffmpeg $FFLOG -ss "$LS" -t 17.5 -i $R/live.webm -f lavfi -t 5 -i "color=c=0x05070d:s=1920x1080:r=30" -filter_complex "[0:v]fps=30[a];[a][1:v]concat=n=2:v=1:a=0[v]" -map "[v]" $ENC $S/b2.v.mp4
fi
audio b2 "$AUD/mom_ev_005291_en_casual.mp3" "$LAT"

# --- beat 3: Rewind & Rewrite (cut from the 69 s demo take to 39.5 s) ---
CUTS="9,12 12,22 24,31 34,38 44,50 59.4,68.9"
F=""; N=0; CC=""
for c in $CUTS; do s=${c%,*}; e=${c#*,}; F+="[0:v]trim=$s:$e,setpts=PTS-STARTPTS,fps=30[v$N];"; CC+="[v$N]"; N=$((N+1)); done
ffmpeg $FFLOG -i $R/rewind.webm -filter_complex "${F}${CC}concat=n=$N:v=1:a=0[v]" -map "[v]" $ENC $S/b3.v.mp4
audio b3

# --- beat 4: split screen, Analyst EN (left, no audio) vs Casual ES (right, with TTS) -- 20 s ---
LP=$(jget $R/split_left.json "d['playStart']+0.3"); RP=$(jget $R/split_right.json "d['playStart']+0.3")
LA=$(jget $C/split_timing.json "d['left']['at']-0.3"); LE=$(jget $C/split_timing.json "d['left']['at']-0.3+d['left']['hold']")
RA=$(jget $C/split_timing.json "d['right']['at']-0.3"); RE=$(jget $C/split_timing.json "d['right']['at']-0.3+d['right']['hold']")
ffmpeg $FFLOG -loop 1 -framerate 30 -t 20 -i $C/split_base.png \
  -ss "$LP" -t 20 -i $R/split_left.webm -ss "$RP" -t 20 -i $R/split_right.webm \
  -loop 1 -framerate 30 -t 20 -i $C/split_cap_left.png -loop 1 -framerate 30 -t 20 -i $C/split_cap_right.png -filter_complex \
  "[1:v]fps=30,crop=1440:1080:0:0,scale=960:720,setpts=PTS-STARTPTS[l];[2:v]fps=30,crop=1440:1080:0:0,scale=960:720,setpts=PTS-STARTPTS[r];\
   [0:v][l]overlay=0:70:shortest=1[x];[x][r]overlay=960:70:shortest=1[y];\
   [y][3:v]overlay=0:0:enable='between(t,$LA,$LE)'[z];[z][4:v]overlay=0:0:enable='between(t,$RA,$RE)',format=yuv420p,fade=in:0:6[v]" \
  -map "[v]" -t 20 $ENC $S/b4.v.mp4
audio b4 "$AUD/mom_ev_022296_es_casual.mp3" "$RA"

# --- beat 5: architecture card (18 s), beat 6: close (8 s) ---
still b5 $C/arch.png 18; audio b5
still b6 $C/close.png 8; audio b6

# --- assemble ---
: > $S/list.txt
for n in b1a b1b b1c b2 b3 b4 b5 b6; do echo "file '$PWD/$S/$n.mp4'" >> $S/list.txt; done
ffmpeg $FFLOG -f concat -safe 0 -i $S/list.txt -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30 -c:a aac -b:a 160k -ar 48000 -movflags +faststart $V/regolazo_rough_cut.mp4
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate -of default=nw=1 $V/regolazo_rough_cut.mp4
for n in b1a b1b b1c b2 b3 b4 b5 b6; do echo "$n $(ffprobe -v error -show_entries format=duration -of csv=p=0 $S/$n.mp4)"; done
