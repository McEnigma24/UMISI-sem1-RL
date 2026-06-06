#!/bin/bash
# Zglasza joby pipeline'u checkpoint -> DT (record -> train DT -> eval -> plot)
# dla PODANYCH krokow checkpointu, osobno dla kazdej z 3 aktywnosci.
# Wszystkie joby leca ROWNOLEGLE (osobny job na (env, steps)).
#
# NIE nadpisuje wczesniejszych wynikow: katalogi i dataset-y sa kluczowane krokami
# (np. dt_weights_ckpt/FetchSlide-v4-ckpt200000, dataset l6/fetchslide-v4/ckpt200000-sac-v0),
# a pipeline pomija etapy, ktorych wyniki juz istnieja (resumowalny).
#
# Uzycie (z katalogu l6):
#   bash lsc_run_ckpt.sh 200000 400000 600000   # pelny sweep slabszych checkpointow
#   bash lsc_run_ckpt.sh 200000                 # tylko jeden zestaw krokow
#
# UWAGA: Push ma checkpointy tylko do 400k -> wyzsze kroki dla Push sa POMIJANE
# (z ostrzezeniem), zeby nie zglaszac joba skazanego na blad braku pliku.

set -u

STEPS_LIST=("$@")
if [ ${#STEPS_LIST[@]} -eq 0 ]; then
  echo "Uzycie: bash lsc_run_ckpt.sh STEPS [STEPS2 ...]   (np. 200000 400000 600000)"
  exit 1
fi

ENVS=(FetchSlide-v4 FetchPush-v4 FetchPickAndPlace-v4)

submit() {
  local env="$1" steps="$2"
  local ckpt="weights/besties/${env}/checkpoints/sac_her_${steps}_steps.zip"
  if [ ! -f "$ckpt" ]; then
    echo "POMIJAM ${env} @ ${steps}: brak pliku ${ckpt}"
    return 0
  fi
  local jid
  jid=$(sbatch --parsable --job-name="DT_${env}_${steps}" \
        --export=ALL,CKPT_ENV="${env}",CKPT_STEPS="${steps}" \
        lsc_job_ckpt)
  if [ -n "${jid}" ]; then
    echo "${jid}  <-  ${env} @ ${steps}"
    echo "${jid}" > "latest_job_${env}_${steps}"
  else
    echo "BLAD sbatch dla ${env} @ ${steps}"
  fi
}

echo "=== Zgłaszam joby dla krokow: ${STEPS_LIST[*]} ==="
for steps in "${STEPS_LIST[@]}"; do
  for env in "${ENVS[@]}"; do
    submit "${env}" "${steps}"
  done
done

echo "Gotowe. Podglad: squeue --me   |   logi: GRID_log_out_<jobid>.txt"
