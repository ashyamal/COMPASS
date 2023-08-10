#! /bin/bash

#sbatch --mem 100G -c 8 -t 5-12:00 -p priority  ./run.sh

Rscript tcga_download_process.R 
