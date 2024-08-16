set.seed(123)
# author: Monica Arniella
# step 1: table cleanup
library(magrittr)
library(data.table)
library(tidyverse)
library(readxl)

# Load gene expression and gene annotation tables
# Set this to your source_data directory
source_data_path <- ".../Source Data"

load(paste0(source_data_path, "/Limma_Fgsea/gene_names.rds"))
master_annotations_df <- read_excel(paste0(source_data_path, "/Clinical/Table_S1_Clinical_Annotations.xlsx"), sheet= "Table_S1_Clinical_Annotations", skip = 2)
su2c_rna_counts_mat <-  read_delim(paste0(source_data_path, "/Limma_Fgsea/SU2C-MARK_Supplementary_Table_12_RNA_Counts_v4.gct"), skip=2)
row.names(su2c_rna_counts_mat) <- su2c_rna_counts_mat$Name
su2c_rna_log2tpm_mat <- read_delim(paste0(source_data_path, "/Limma_Fgsea/SU2C-MARK_Supplementary_Table_14_RNA_TPM.gct"), skip=2)
row.names(su2c_rna_log2tpm_mat) <- su2c_rna_log2tpm_mat$Name

# filter by QC
rna_master_annotations_df <- master_annotations_df[master_annotations_df$`Pre-treatment_RNA_Sample_QC` == "Keep" | master_annotations_df$`Pre-treatment_RNA_Sample_QC` == "Flag",]
all_rna_samples <- as.character(na.omit(rna_master_annotations_df$Harmonized_SU2C_RNA_Tumor_Sample_ID_v2) )

rna_counts_mat <- su2c_rna_counts_mat[,all_rna_samples]
row.names(rna_counts_mat)<- row.names(su2c_rna_counts_mat)

#rna_tpm_mat <- su2c_rna_tpm_mat[,all_rna_samples_fc]
#colnames(rna_tpm_mat) <- row.names(su2c_rna_counts_mat)

rna_log2tpm_mat <- su2c_rna_log2tpm_mat[,all_rna_samples]
row.names(rna_log2tpm_mat)<- row.names(su2c_rna_log2tpm_mat)
rna_log2tpm_mat <- log2(rna_log2tpm_mat+1)

# filter out the samples that have mixed small-cell histology
rna_training_samples <- as.character(na.omit(rna_master_annotations_df$Harmonized_SU2C_RNA_Tumor_Sample_ID_v2[rna_master_annotations_df$RNA_Cohort_1 == 1]))
rna_validation_samples <- as.character(na.omit(rna_master_annotations_df$Harmonized_SU2C_RNA_Tumor_Sample_ID_v2[rna_master_annotations_df$RNA_Cohort_2 == 1]))
rna_training_samples <- setdiff(rna_training_samples, c("SU2CLC-MGH-1151-T1"))
training_rna_log2tpm_mat <- rna_log2tpm_mat[, rna_training_samples]
row.names(training_rna_log2tpm_mat) <- row.names(rna_log2tpm_mat)

# set aside validation + flagged input for BNMF
rna_validation_flag_samples <- union(
  rna_validation_samples,
  na.omit(rna_master_annotations_df$Harmonized_SU2C_RNA_Tumor_Sample_ID_v2[rna_master_annotations_df$`Pre-treatment_RNA_Sample_QC` == "Flag"])
)
validation_rna_log2tpm_mat <- rna_log2tpm_mat[, as.character(rna_validation_flag_samples)]
validation_rna_log2tpm_df <- as.data.frame(validation_rna_log2tpm_mat) %>%
  mutate(ensembl_gene_id_version = rownames(validation_rna_log2tpm_mat))
validation_rna_log2tpm_df <- validation_rna_log2tpm_df[,c("ensembl_gene_id_version", rna_validation_flag_samples)]
write.csv(validation_rna_log2tpm_df, row.names = FALSE, paste0(source_data_path, "/Limma_Fgsea/SU2C_validation+flag.classifier.input.csv"))


