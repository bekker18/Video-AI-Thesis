#!/usr/bin/env bash
# Downloads UCO-LAEO (4.1 GB) and unpacks it into data/raw/ucolaeo/.
set -e
archive=data/raw/ucolaeo/ucolaeodb_v1.1.tgz
mkdir -p data/raw/ucolaeo
python3 -c "import urllib.request as u; u.urlretrieve('http://rabinf24.uco.es/ucolaeo/ucolaeodb_v1.1.tgz', '$archive')"
tar -xzf "$archive" -C data/raw/ucolaeo
rm "$archive"
