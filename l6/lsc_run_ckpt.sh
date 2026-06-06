#!/bin/bash
# Zglasza 3 NIEZALEZNE joby (po jednym na aktywnosc) pipeline'u checkpoint -> DT.
# Kazdy job: record (1800 ep) -> train DT (40k) -> eval DT vs checkpoint -> plot.
#
# Checkpointy ~3/4 drogi do modelu koncowego (dostosuj STEPS jesli chcesz inny):
#   Slide        final ~1.1M  -> 800000
#   Push         final ~425k  -> 300000   (UWAGA: Push zbiega wczesnie, 300k bywa ~97% -> rozwaz 100000)
#   PickAndPlace final 600k   -> 400000
#
# Uruchom z katalogu l6:  bash lsc_run_ckpt.sh

rm -f GRID_log*

set -e

sbatch --parsable --job-name=DT_slide \
  --export=ALL,CKPT_ENV=FetchSlide-v4,CKPT_STEPS=800000 \
  lsc_job_ckpt | tee latest_job_slide

sbatch --parsable --job-name=DT_push \
  --export=ALL,CKPT_ENV=FetchPush-v4,CKPT_STEPS=300000 \
  lsc_job_ckpt | tee latest_job_push

sbatch --parsable --job-name=DT_pnp \
  --export=ALL,CKPT_ENV=FetchPickAndPlace-v4,CKPT_STEPS=400000 \
  lsc_job_ckpt | tee latest_job_pnp

echo "Zgloszono 3 joby. Podglad: squeue --me   |   logi: GRID_log_out_<jobid>.txt"
