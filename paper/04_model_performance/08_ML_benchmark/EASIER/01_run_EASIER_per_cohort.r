library(easier)



############################# 1 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Kim_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Kim_counts.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "STAD",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "STAD"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "STAD",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Kim_easier.csv", row.names = TRUE)



############################# 2 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Gide_pipe_tpm_short.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Gide_pipe_counts_short.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "SKCM",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "SKCM"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "SKCM",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Gide_easier.csv", row.names = TRUE)




############################# 3 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Imvigor210_rpkg_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Imvigor210_rpkg_counts.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "BLCA",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "BLCA"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "BLCA",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/IMvigor210_easier.csv", row.names = TRUE)


############################# 4 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Liu_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Liu_counts.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "SKCM",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "SKCM"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "SKCM",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Liu_easier.csv", row.names = TRUE)



############################# 5 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Allen_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Allen_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "SKCM",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "SKCM"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "SKCM",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Allen_easier.csv", row.names = TRUE)

############################# 6 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Hugo_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Hugo_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "SKCM",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "SKCM"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "SKCM",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Hugo_easier.csv", row.names = TRUE)


############################# 7 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/MGH_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/MGH_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "SKCM",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "SKCM"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "SKCM",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/MGH_easier.csv", row.names = TRUE)


############################# 8 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Riaz_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Riaz_counts.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "SKCM",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "SKCM"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "SKCM",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Riaz_easier.csv", row.names = TRUE)


############################# 9 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Choueiri_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Choueiri_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "KIRC",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "KIRC"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "KIRC",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Choueiri_easier.csv", row.names = TRUE)

############################# 10 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/IMmotion150_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/IMmotion150_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "KIRC",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "KIRC"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "KIRC",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/IMmotion150_easier.csv", row.names = TRUE)

############################# 11 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Miao_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Miao_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "KIRC",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "KIRC"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "KIRC",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Miao_easier.csv", row.names = TRUE)

############################# 12 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Rose_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Rose_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "BLCA",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "BLCA"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "BLCA",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Rose_easier.csv", row.names = TRUE)

############################# 13 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Snyder_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Snyder_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "BLCA",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "BLCA"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "BLCA",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Snyder_easier.csv", row.names = TRUE)

############################# 14 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/Zhao_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/Zhao_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "GBM",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "GBM"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "GBM",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/Zhao_easier.csv", row.names = TRUE)

############################# 15 #####################################
# 1. Load TPM and count data
tpm_raw <- read.csv("./data/SU2CLC1_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/SU2CLC1_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "LUAD",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "LUAD"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "LUAD",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/SU2CLC1_easier.csv", row.names = TRUE)

############################# 16 #####################################
tpm_raw <- read.csv("./data/SU2CLC2_tpm.csv", row.names = 1, check.names = FALSE)
tpm <- t(tpm_raw)  # 'easier' expects samples as rows and genes as columns

counts_raw <- read.csv("./data/SU2CLC2_counts_impu.csv", row.names = 1, check.names = FALSE)
RNA_counts <- t(counts_raw)  # 'easier' expects samples as rows and genes as columns

# 2. Compute cell fractions (requires TPM)
cell_fractions <- compute_cell_fractions(RNA_tpm = tpm, verbose = TRUE)

# 3. Compute pathway activity (requires counts)
pathway_activity <- compute_pathway_activity(
  RNA_counts = RNA_counts,
  remove_sig_genes_immune_response = TRUE
)

# 4. Compute transcription factor activity (requires TPM)
tf_activity <- compute_TF_activity(
  RNA_tpm = tpm,
  verbose = TRUE
)

# 5. Compute ligand–receptor pair scores (requires TPM)
lrpairs_weights <- compute_LR_pairs(
  RNA_tpm = tpm,
  cancer_type = "LUSC",
  verbose = TRUE
)

# 6. Compute cell–cell communication pair scores
ccpair_scores <- compute_CC_pairs(
  lrpairs = lrpairs_weights,
  cancer_type = "LUSC"
)

# 7. Predict immune response
predictions_immune_response <- predict_immune_response(
  pathways = pathway_activity,
  immunecells = cell_fractions,
  tfs = tf_activity,
  lrpairs = lrpairs_weights,
  ccpairs = ccpair_scores,
  cancer_type = "LUSC",
  verbose = TRUE
)

# Save results
write.csv(predictions_immune_response, file = "./results/SU2CLC2_easier.csv", row.names = TRUE)
