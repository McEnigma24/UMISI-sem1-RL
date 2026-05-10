#!/bin/bash

rm -f GRID_log*

# sbatch lsc_job
sbatch --parsable lsc_job | tee latest_job_started
