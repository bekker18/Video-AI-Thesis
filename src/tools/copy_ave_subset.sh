#!/usr/bin/env bash
# Copies the first 50 processed AVE scenes by name into data/processed/ave: those no threshold was fitted on.
mkdir -p data/processed/ave
LC_ALL=C ls data/processed/ave_processed | head -n 50 | while read -r id; do
  cp -r "data/processed/ave_processed/$id" data/processed/ave/
done
