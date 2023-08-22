#!/bin/sh

#>>>star.sh>>>
rawdata_dir=/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GideBulkmRNA/data/rawdata
index_dir=/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/star_index/human
input_dir=/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GideBulkmRNA/data/cleandata/fastp
out_dir=/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GideBulkmRNA/STAR
n_jobs=10


cat ${rawdata_dir}/run_accession.txt | while read i
do
STAR \
--readFilesIn ${input_dir}/${i}_1.fastp.fq.gz ${input_dir}/${i}_2.fastp.fq.gz \
--outSAMattrRGline ID:sample SM:sample PL:ILLUMINA \
--genomeDir ${index_dir} \
--readFilesCommand zcat \
--runThreadN ${n_jobs} \
--twopassMode Basic \
--outFilterMultimapNmax 20 \
--alignSJoverhangMin 8 \
--alignSJDBoverhangMin 1 \
--outFilterMismatchNmax 999 \
--outFilterMismatchNoverLmax 0.1 \
--alignIntronMin 20 \
--alignIntronMax 1000000 \
--alignMatesGapMax 1000000 \
--outFilterType BySJout \
--outFilterScoreMinOverLread 0.33 \
--outFilterMatchNminOverLread 0.33 \
--limitSjdbInsertNsj 1200000 \
--outFileNamePrefix ${out_dir}/${i} \
--outSAMstrandField intronMotif \
--outFilterIntronMotifs None \
--alignSoftClipAtReferenceEnds Yes \
--quantMode TranscriptomeSAM GeneCounts \
--outSAMtype BAM SortedByCoordinate \
--outSAMunmapped Within \
--genomeLoad NoSharedMemory \
--chimSegmentMin 15 \
--chimJunctionOverhangMin 15 \
--chimOutType Junctions SeparateSAMold WithinBAM SoftClip \
--chimOutJunctionFormat 1 \
--chimMainSegmentMultNmax 1 \
--outSAMattributes NH HI AS nM NM ch
done
#<<<star.sh<<<

#sbatch --mem 64G -c 10 -t 5-12:00 -p priority ./03.star_align_counts.sh
