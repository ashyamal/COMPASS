#! /bin/bash


## fastp installation
# wget http://opengene.org/fastp/fastp
# chmod a+x ./fastp
# export PATH=$PATH:/home/was966/Software
# module load fastqc
# module load multiqc


cd /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GideBulkmRNA/data/rawdata

#fastqc
nohup fastqc -t 10 -o . /ERR*.fastq.gz > fastqc.log &

#multiqc
multiqc ./*.zip



#>>>fastp.sh>>>
rawdata_dir=/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GideBulkmRNA/data/rawdata
cleandata_dir=/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GideBulkmRNA/data/cleandata/fastp
n_jobs=10
#cut -f 4 ${rawdata_dir}/filereport_read_run_PRJNA229998_tsv.txt | sed '1d' > ${rawdata_dir}/run_accession.txt

cat ${rawdata_dir}/run_accession.txt | while read i
do
fastp \
--in1 ${rawdata}/${i}_1.fastq.gz \
--in2 ${rawdata}/${i}_2.fastq.gz  \
--out1 ${cleandata}/${i}_1.fastp.fq.gz \
--out2 ${cleandata}/${i}_2.fastp.fq.gz \
--json ${cleandata}/${i}.fastp.json \
--html ${cleandata}/${i}.fastp.html \
--report_title ${cleandata}/${i} \
--thread ${n_jobs}
done
#<<<fastp.sh<<<

cd /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GideBulkmRNA/data/cleandata/fastp
# sbatch --mem 10G -c 10 -t 5-12:00 -p priority ./fastp.sh
