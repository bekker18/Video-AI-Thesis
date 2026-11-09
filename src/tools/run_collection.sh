#!/usr/bin/env bash
# The pipeline on every video of a folder, each output moved into a collection; finished videos are skipped.
# A count takes a fixed random sample, and a larger count adds to it.
: "${2:?usage: run_collection.sh <folder of .mp4 files> <collection under data/processed> [how many]}"
folder=$1 collection=$2 count=${3:-1000000}

mkdir -p "data/processed/$collection" "data/logs/$collection"
LC_ALL=C ls "$folder"/*.mp4 | awk 'BEGIN { srand(0) } { print rand(), $0 }' | LC_ALL=C sort -n |
  cut -d' ' -f2- | head -n "$count" | while read -r video; do
  id=$(basename "$video" .mp4)
  [ -f "data/processed/$collection/$id/identikit.json" ] && continue
  if python3 main.py "$video" > "data/logs/$collection/$id.log" 2>&1 < /dev/null; then
    mv "data/processed/$id" "data/processed/$collection/"
  else
    echo "FAILED $id"
  fi
done
