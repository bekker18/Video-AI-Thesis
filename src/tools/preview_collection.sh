#!/usr/bin/env bash
# Draws previews for every video of a collection; identikit pages unless views are named.
: "${1:?usage: preview_collection.sh <collection under data/processed> [view ...]}"
collection=$1
shift
views=${*:-identikit}

for folder in data/processed/"$collection"/*/; do
  python3 -m src.tools.preview "$collection/$(basename "$folder")" --views $views
done
