#!/bin/bash

cd /mnt/beegfs/home/jseepaul2023/testing_zone/

echo "Hi everyone!"


echo "This is a test script to run the testing job on the cluster. I hope it works!!"

#SBATCH --job-name=testing_job
#SBATCH --output=testing_job_output.txt
#SBATCH --error=testing_job_error.txt
#SBATCH --cpus-per-task=1 --nodes=1
#SBATCH --mem=1G


python3 test_job.py
