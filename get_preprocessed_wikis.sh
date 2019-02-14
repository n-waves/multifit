#!/usr/bin/env bash
set -x
ZIPNAME="preprocessed_wiki_8langs.zip"
OUTDIR="data/wiki"
wget -nc 'https://www.dropbox.com/sh/srfwvur6orq0cre/AAAQc36bcD17C1KM1mneXN7fa/data/wiki?dl=1' -O "${ZIPNAME}"
mkdir -p "${OUTDIR}"
unzip "${ZIPNAME}" -d "${OUTDIR}"

for archive in "${OUTDIR}"/??-100.tar.gz; do tar xvf "${archive}" -C "${OUTDIR}" && rm "${archive}"; done
