#! /bin/bash

work_dir=/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/fastaq/Gide
index_dir=/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/star_index/gdc_index/star-2.7.5c_GRCh38.d1.vd1_gencode.v36

rawdata_dir=${work_dir}/data/rawdata
cleandata_dir=${work_dir}/data/cleandata/fastp
out_dir=${work_dir}/STAR

n_jobs=40

# fastqc
fastqc -t ${n_jobs} -o ${work_dir}/data/rawdata/ ${work_dir}/data/rawdata/ERR*.fastq.gz

# multiqc
multiqc -o ${work_dir}/data/rawdata/ ${work_dir}/data/rawdata/*.zip

# fastp
cd ${rawdata_dir}
ls *.fastq.gz | awk -F'[_.]' '{print $1}' | sort -u > run_accession.txt
cat ${rawdata_dir}/run_accession.txt | while read i
do
fastp \
--in1 ${rawdata_dir}/${i}_1.fastq.gz \
--in2 ${rawdata_dir}/${i}_2.fastq.gz  \
--out1 ${cleandata_dir}/${i}_1.fastp.fq.gz \
--out2 ${cleandata_dir}/${i}_2.fastp.fq.gz \
--json ${cleandata_dir}/${i}.fastp.json \
--html ${cleandata_dir}/${i}.fastp.html \
--report_title ${cleandata_dir}/${i} \
--thread ${n_jobs}
done

# star
input_dir=${work_dir}/data/cleandata/fastp

cat ${rawdata_dir}/run_accession.txt | while read i
do
STAR \
--readFilesIn ${input_dir}/${i}_1.fastp.fq.gz ${input_dir}/${i}_2.fastp.fq.gz \
--outSAMattrRGline ID:sample SM:sample PL:ILLUMINA \
--genomeDir ${index_dir} \
--readFilesCommand zcat \
--runThreadN ${n_jobs} \
--twopassMode Basic \
--outFileNamePrefix ${out_dir}/${i} \
--quantMode GeneCounts \
--outSAMtype BAM Unsorted \
--outSAMunmapped None \

done

# sbatch --mem 24G -c 40 -t 5-12:00 -p priority ./pipeline.sh



