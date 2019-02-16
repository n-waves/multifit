#!/usr/bin/env bash
set -x
ZIPNAME="preprocessed_wiki_8langs.zip"
OUTDIR="data/wiki"
wget -nc 'https://www.dropbox.com/sh/srfwvur6orq0cre/AAAQc36bcD17C1KM1mneXN7fa/data/wiki?dl=1' -O "${ZIPNAME}"
mkdir -p "${OUTDIR}"
unzip "${ZIPNAME}" -d "${OUTDIR}"

for archive in "${OUTDIR}"/??-100.tar.gz; do tar xvf "${archive}" -C "${OUTDIR}" && rm "${archive}"; done

#optionally download models
MODELS="pretrained_lm_models.zip"
read -r -p "Download pretrained lm models? [y/N] " response
if [[ "$response" =~ ^[yY]$ ]]
then
    wget -nc 'https://www.dropbox.com/sh/srfwvur6orq0cre/AAABRFdrCNHmbpf4nNcMiJwJa/models/data/wiki?dl=1' -O "${MODELS}"
    unzip "${MODELS}" -d "${OUTDIR}"
fi
