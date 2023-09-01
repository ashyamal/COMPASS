#! /bin/bash

<< COMMENTS
"""
Created on Sat Aug 19 13:45:37 2023

@author: wanxiang_shen@hms.harvard.edu


This step is unnecessary, your can download the index file and GTF file from GDC:
<可以不用执行这一步，直接从GDC下载：>

https://gdc.cancer.gov/about-data/gdc-data-processing/gdc-reference-files

GDC.h38.d1.vd1 STAR2 Index Files (v36)
•	star-2.7.5c_GRCh38.d1.vd1_gencode.v36.tgz
o	md5: acafb76bba5e3e80eb028dc05f002ffc
o	file size: 25 GB
after that unzip:
tar -xvzf star-2.7.5c_GRCh38.d1.vd1_gencode.v36.tgz

GDC.h38 GENCODE v36 GTF
•	gencode.v36.annotation.gtf.gz
o	md5: c03931958d4572148650d62eb6dec41a
o	file size: 44.5 MB

COMMENTS



################################################################################
# 01.准备和下载人类reference基因组及注释文件
mkdir -p /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_36
cd /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_36

#>>>downGENCODE.sh
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_36/gencode.v36.annotation.gtf.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_36/GRCh38.p13.genome.fa.gz

mv gencode.v36.annotation.gtf.gz HS.gencode.v36.annotation.gtf.gz
mv GRCh38.p13.genome.fa.gz HS.GRCh38.p13.genome.fa.gz

gzip -d HS.gencode.v36.annotation.gtf.gz
gzip -d HS.GRCh38.p13.genome.fa.gz
#<<<downGENCODE.sh

################################################################################

# 02. 构建star的索引
mkdir -p /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/star_index/human
cd /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/star_index/human

#>>>star_index.sh>>>
n_jobs=32
STAR \
--runMode genomeGenerate \
--genomeDir /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/star_index/human \
--genomeFastaFiles /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_36/HS.GRCh38.p13.genome.fa \
--sjdbOverhang 100 \
--sjdbGTFfile /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_36/HS.gencode.v36.annotation.gtf \
--runThreadN ${n_jobs} 
#<<<star_index.sh<<<

#sbatch --mem 8G -c 32 -t 5-12:00 -p priority ./star_index.sh

################################################################################

