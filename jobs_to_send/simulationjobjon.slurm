#!/bin/bash


cd 

# do i include the thingy about removing previous script?

echo "Hi everyone!!"


echo "This is a test script to run the simulation job on the cluster. I hope it works!!"



#SBATCH --job-name=fluxonium_simulation
#SBATCH --output=fluxonium_simulation_output.txt
#SBATCH --error=fl