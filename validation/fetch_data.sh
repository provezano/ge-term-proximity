#!/usr/bin/env bash
# Download the text collections used for validation (ARFF bag-of-words files from the
# ICMC-USP LABIC repository, Rossi, Marcacini and Rezende 2013). The original site is
# offline, so the files come from the Internet Archive snapshot of 2020-06-23.
set -euo pipefail
dest="${1:-data}"
mkdir -p "$dest"
base="http://web.archive.org/web/20200623195015id_/http://sites.labic.icmc.usp.br/text_collections"
while read -r name sha; do
  curl -sSL -o "$dest/$name.arff.zip" "$base/$name.arff.zip"
  got=$(shasum -a 256 "$dest/$name.arff.zip" | cut -d' ' -f1)
  [ "$got" = "$sha" ] || { echo "checksum mismatch for $name" >&2; exit 1; }
  unzip -o -q "$dest/$name.arff.zip" -d "$dest"
done <<'LIST'
CSTR 0a53ece82320dbc51ce25901c68504f542d4a5870dbd5b4c3e8a31ca574c2eb6
tr23 7ebf7dc535401d5d4e231732b6b792a2faf71bfe53dc146b69940a82162e8c76
tr12 327d1755972decb9170683078a78c2aceb641b882c684bce106ce193752d1208
tr11 a7e41542ec4801ce5313955b0300502d934930a5a74c6308a85aa47f29a09b02
oh15 d45295cf516c767c0ac7b037b8a98b9e58f6381ee2372d50295f98f6550b1797
LIST
