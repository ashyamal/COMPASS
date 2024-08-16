set.seed(123)

# author: Monica Arniella
# step 2: differential expression analysis of just the training samples
library(fgsea)
library(biomaRt)
library(edgeR)
library("DESeq2")
library("parallel")

hallmark_pathways <- gmtPathways(paste0(source_data_path,"/Limma_Fgsea/h.all.v2022.1.Hs.entrez.gmt"))
names(hallmark_pathways) <- gsub("HALLMARK_", "", names(hallmark_pathways))

ensembl = useEnsembl(biomart="ensembl", dataset="hsapiens_gene_ensembl")

# what is the distribution of gene median TPMs in high-quality samples?
gene_median_log2tpm <- apply(training_rna_log2tpm_mat, 1, median)
gene_median_log2tpm_df <- data.frame(
  median_log2tpm = gene_median_log2tpm, 
  ensembl_gene_id_version = rownames(training_rna_log2tpm_mat)
)

# in what percent of samples are genes expressed above certain threshold?
log2tpm_threshold <- 0.5
percent_above_tpm <- function(gene_vector){ return((sum(gene_vector > log2tpm_threshold))/length(gene_vector)) }
get_percent_cat <- function(percent){
  if(percent < 0.15){
    return("less than 15%")
  } else if(percent < 0.3){
    return("15-30%")
  } else if(percent < 0.5){
    return("30-50%")
  } else if(percent < 0.8){
    return("50-80%")
  } else {
    return("80-100%")
  }
}
gene_expression_percentiles <- apply(training_rna_log2tpm_mat, 1, percent_above_tpm)
gene_expression_percentiles_df <- data.frame(
  ensembl_gene_id_version = rownames(training_rna_log2tpm_mat),
  gexp_gt1_percent = gene_expression_percentiles
)
gene_expression_percentiles_df$gexp_gt1_cat <- as.character(unlist(
  lapply(gene_expression_percentiles_df$gexp_gt1_percent, get_percent_cat)))

# compile info in one df
gene_median_log2tpm_df <- left_join(gene_median_log2tpm_df, gene_expression_percentiles_df)
gene_median_log2tpm_df$gexp_gt1_cat <- factor(gene_median_log2tpm_df$gexp_gt1_cat, 
                                              levels = c("less than 15%","15-30%","30-50%","50-80%","80-100%"))

# filter out genes that are all 0's or have 0 variance, or are expressed above 0.5 in less than 30% of samples
zero_sum_filter <- apply(rna_counts_mat[,colnames(training_rna_log2tpm_mat)], 1, sum) != 0
zero_var_filter <- apply(rna_counts_mat[,colnames(training_rna_log2tpm_mat)], 1, var) != 0
pc_filter <- rownames(rna_counts_mat) %in% gene_names$ensembl_gene_id_version[gene_names$gene_biotype == "protein_coding"]
gexp_filter <- rownames(rna_counts_mat) %in% 
  gene_median_log2tpm_df$ensembl_gene_id_version[
    gene_median_log2tpm_df$gexp_gt1_cat %in% c("30-50%","50-80%","80-100%")]

su2c_all_cts_filtered <- rna_counts_mat[zero_sum_filter & zero_var_filter & gexp_filter & pc_filter, ] # all samples that pass filters

# exclude samples with missing best overall response (BOR) information 
get_BOR_cat <- function(BOR){
  if(is.na(BOR)){
    return(NA)
  } else if(BOR=="PR" | BOR=="CR"){
    return("response")
  } else if(BOR=="PD" | BOR=="SD"){
    return("resistance")
  }
  return(NA)
}



# load lists of genes that might be of interest, to highlight in volcano plot
# * Lung epithelial subtypes from [Laughney et al. 2020](https://pubmed.ncbi.nlm.nih.gov/32042191/)
# * Developmental lineage from [Tata et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29587142/)
# * CoCA subtype markers from [Chen et al. 2017](https://pubmed.ncbi.nlm.nih.gov/27775076/)

lung_epithelial_subtypes <- data.frame(
  hgnc_symbol = c("SOX17", "HHEX", "KRT5", "VIM", "SNAI2", 
                  "BMP4", "DKK1", "SOX2", "FGFR2", "FOXA2", "NKX2-1",
                  "WNT7B", "SOX9", "TM4SF1"),
  subtype = c(rep("type I-P", 3), rep("type I-Q", 2), 
              rep("type II", 6), rep("type III", 3))) %>% 
  left_join(gene_names[,c("hgnc_symbol", "ensembl_gene_id_version")])

lung_lineage_markers <- data.frame(
  hgnc_symbol  = c("TP63", "KRT5", "KRT1", "KRT14", "DSG3", "KRT6C", "KRT6A", "KRT6B", "PAX9", "SOX2",
                   "NKX2-1", "SFTA3", "LMO3", "NAPSA", "SFTPC", "FOXA2", "HNF1B", 
                   "KRT7", "FGA", "PRSS1", "CDX2", "MUC2", "PDX1", "TFF2", "MUC6", "REG4", "TFF1", "HNF4A", "F2", "CPS1"),
  lineage = c(rep("Esophagus",10), rep("Lung",7), rep("Midgut/Hindgut",13)),
  histology = c(rep("Squamous",10), rep("Adeno",7), rep("Mucinous Adeno",13)))

read.only.cols <- function(file_path, skipped_lines, cols_to_select){
  col_classes <- read.csv(
    file_path, header = FALSE, nrows = 1, skip = skipped_lines,
    colClasses = "character", check.names = FALSE, stringsAsFactors = FALSE)[1,] %in% cols_to_select 
  col_classes[col_classes == TRUE] <- NA
  col_classes[col_classes == FALSE] <- "NULL"
  
  file_df <- read.csv(
    file_path, header = TRUE, skip = skipped_lines,
    colClasses = col_classes, check.names = FALSE, stringsAsFactors = FALSE)
  return(file_df)
}

coca_subtypes_markers <- read.only.cols(paste0(source_data_path,"/Limma_Fgsea/Chen_et_al_2017_histology700markers.csv"),
                                        1, c("Gene", "Entrez ID")) %>% plyr::rename(c("Entrez ID" = "entrezgene"))

get.split.first.element <- function(vec, spl){
  return(strsplit(vec,spl)[[1]][1])
}

coca_subtypes_markers$hgnc_symbol <- as.character(unlist(mcmapply(get.split.first.element, coca_subtypes_markers$Gene, "\\|")))
coca_subtypes_markers <- dplyr::select(coca_subtypes_markers, -Gene) 


BOR_samples_metadata <- rna_master_annotations_df[
  ,c("Pre-treatment RNA sample Harmonized","Harmonized_Confirmed_BOR","Histology_Harmonized", "Agent_PD1_Category",
     "Patient_Age_at_Diagnosis", "Patient_Sex", "PDL1_TPS", "Patient_Smoking_Pack_Years_Harmonized")] %>%
  mutate(BOR_cat = as.character(unlist(lapply(Harmonized_Confirmed_BOR, get_BOR_cat)))) %>%
  subset(`Pre-treatment RNA sample Harmonized` %in% colnames(rna_log2tpm_mat))
BOR_samples_metadata.origALL <- BOR_samples_metadata
BOR_samples_metadata <- BOR_samples_metadata[!is.na(BOR_samples_metadata$BOR_cat),]
BOR_samples <- as.character(BOR_samples_metadata$`Pre-treatment RNA sample Harmonized`[!is.na(BOR_samples_metadata$BOR_cat)])
BOR_samples_cts <- su2c_all_cts_filtered[,BOR_samples]

# Design matrix for differential expression using BOR 
BOR_samples_metadata$BOR_cat <- factor(BOR_samples_metadata$BOR_cat, levels=c("resistance","response"))
BOR_samples_metadata <- BOR_samples_metadata[!is.na(BOR_samples_metadata$BOR_cat),]
bor_design <- model.matrix(~BOR_cat, data = BOR_samples_metadata)

# DGE + TMM normalization 
dge_bor <- DGEList(counts=BOR_samples_cts)
dge_bor <- calcNormFactors(dge_bor, method="TMM")

#####
# voom with sample quality weights
#####
vwts <- voomWithQualityWeights(dge_bor, design=bor_design, normalize.method ="none", plot=TRUE)
bor_fit2 <- lmFit(vwts)
bor_fit2 <- eBayes(bor_fit2)
limma_voom_deg_all_keep_samples_bor <- topTable(bor_fit2, coef="BOR_catresponse", sort="none",n=Inf)
limma_voom_deg_all_keep_samples_bor <- mutate(limma_voom_deg_all_keep_samples_bor, ensembl_gene_id_version = rownames(limma_voom_deg_all_keep_samples_bor)) %>%
  left_join(., name_desc, by = c('ensembl_gene_id_version' = 'Name')) %>%
  left_join(., gene_median_log2tpm_df) %>%
  plyr::rename(c("Description" = "hgnc_symbol"))

print("Write limma results")
write.csv(limma_voom_deg_all_keep_samples_bor, paste0(source_data_path,"/Limma_Fgsea/pre_SU2C_limma_out.csv"))

genes_to_highlight_all <- as.character(limma_voom_deg_all_keep_samples_bor$hgnc_symbol[ 
  limma_voom_deg_all_keep_samples_bor$hgnc_symbol %in% lung_epithelial_subtypes$hgnc_symbol |
    limma_voom_deg_all_keep_samples_bor$hgnc_symbol %in% coca_subtypes_markers$hgnc_symbol |
    limma_voom_deg_all_keep_samples_bor$hgnc_symbol %in% lung_lineage_markers$hgnc_symbol])







#####
# fast gene set enrichment 
#####
print("run fgsea")
deg_bor_by_clust <- limma_voom_deg_all_keep_samples_bor
deg_bor_by_clust$ensembl_gene_id <- gsub("\\.[0-9]+$", "", deg_bor_by_clust$ensembl_gene_id_version)

deg_bor_by_clust_genes <- unique(deg_bor_by_clust$ensembl_gene_id)
deg_bor_by_clust_entrez_genes <- getBM(attributes=c('ensembl_gene_id', 'entrezgene_id', 'gene_biotype'),
                                       filters = 'ensembl_gene_id',
                                       values = deg_bor_by_clust_genes,
                                       mart = ensembl)

# Order by p-value and logFC
deg_bor_by_clust <- left_join(deg_bor_by_clust, deg_bor_by_clust_entrez_genes) %>%
  subset(gene_biotype == "protein_coding")
deg_bor_by_clust$enrichment_score <- (2*as.numeric(deg_bor_by_clust$logFC > 0) - 1) * -log10(deg_bor_by_clust$P.Value) # signed p-value

# positive scores enriched in upregulated genes in the cluster
deg_bor_by_clust <- deg_bor_by_clust[order(deg_bor_by_clust$enrichment_score, decreasing = TRUE),]
deg_bor_by_clust_pval_ranks <- deg_bor_by_clust[!is.na(deg_bor_by_clust$entrezgene_id),]$enrichment_score
names(deg_bor_by_clust_pval_ranks) <- deg_bor_by_clust[!is.na(deg_bor_by_clust$entrezgene_id),]$entrezgene_id

# Get pathways ###############
# using number of permutations 
deg_bor_by_clust_pathways <- hallmark_pathways 
deg_bor_by_clust_gene_ranks_reactome <- fgsea(pathways = deg_bor_by_clust_pathways, stats = deg_bor_by_clust_pval_ranks,
                                              minSize=15, maxSize=500, nperm=100000) 

topPathwaysUp <- deg_bor_by_clust_gene_ranks_reactome[ES > 0][head(order(pval), n=15), pathway]
topPathwaysDown <- deg_bor_by_clust_gene_ranks_reactome[ES < 0][head(order(pval), n=15), pathway]
topPathways <- c(topPathwaysUp, rev(topPathwaysDown))

deg_bor_by_clust_gene_ranks_reactome.copy <-deg_bor_by_clust_gene_ranks_reactome
deg_bor_by_clust_gene_ranks_reactome.copy$leadingEdge <- lapply(deg_bor_by_clust_gene_ranks_reactome.copy$leadingEdge, paste, collapse=",")
deg_bor_by_clust_gene_ranks_reactome.copy$leadingEdge <- unlist(deg_bor_by_clust_gene_ranks_reactome.copy$leadingEdge)
write_delim(deg_bor_by_clust_gene_ranks_reactome.copy, delim = "\t",  paste0(source_data_path,"/Limma_Fgsea/pre_fgsea_out.tsv"))




