#! /bin/bash

<< COMMENTS
"""
Created on Sat Aug 19 13:45:37 2023

@author: wanxiang_shen@hms.harvard.edu
COMMENTS

################################################################################

# 01. 环境准备 load tools
module load gcc/6.2.0
module load star/2.7.9a
module load fastqc 
## 安装下载利器aspera， 安装后 ascp -i /home/was966/anaconda3/envs/aggmap/etc/asperaweb_id_dsa.openssh
#conda install -c hcc aspera-cli



# 02.准备和下载人类reference基因组及注释文件
mkdir -p /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_44
cd /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_44

#>>>downGENCODE.sh
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/GRCh38.p14.genome.fa.gz

mv gencode.v44.annotation.gtf.gz HS.gencode.v44.annotation.gtf.gz
mv GRCh38.p14.genome.fa.gz HS.GRCh38.p14.genome.fa.gz

gzip -d HS.gencode.v44.annotation.gtf.gz
gzip -d HS.GRCh38.p14.genome.fa.gz
#<<<downGENCODE.sh

################################################################################

# 03. 构建star的索引
mkdir -p /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/star_index/human
cd /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/star_index/human

#>>>star_index.sh>>>
n_jobs=32
STAR \
--runMode genomeGenerate \
--genomeDir /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/star_index/human \
--genomeFastaFiles /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_44/HS.GRCh38.p14.genome.fa \
--sjdbOverhang 100 \
--sjdbGTFfile /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/GENCODE/release_44/HS.gencode.v44.annotation.gtf \
--runThreadN ${n_jobs} 
#<<<star_index.sh<<<

#sbatch --mem 8G -c 32 -t 5-12:00 -p priority ./star_index.sh

################################################################################

