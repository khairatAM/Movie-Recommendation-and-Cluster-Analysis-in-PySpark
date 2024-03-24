#!/bin/bash
#$ -l h_rt=6:00:00  #time needed
#$ -pe smp 15 #number of cores
#$ -l rmem=20G #number of memery
#$ -P rse-com6012 # require a com6012-reserved node
#$ -q rse-com6012.q # specify com6012 queue
#$ -o Output/output.txt  #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M kamonguno1@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 20G --executor-memory 20G --conf spark.driver.maxResultSize=4G Code/code.py

