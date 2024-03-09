## data description

 * 00_clinical_label_orignal.csv: the orignal TCGA clinical labels
 * 00_clinical_label.csv: the preprocessed labels, binary labels
 * 01_readouts_geneset.csv: the conceptor's readouts (scores) for 140 genesets  <for survival analysising>
 * 02_readouts_celltype.csv: the conceptor's readouts (scores) for 45 celltypes, the first column is cancer type, the last column is reference <for survival analysising>
 * 03_features_geneset.csv: the conceptor's features for 140 genesets, for each patient, there are 140 genesets, for each geneset, there are 32-dim features. <for UMAP, TSNE 2D visulization>
 * 04_features_celltype.csv: the conceptor's features for 45 celltypes, for each patient, there are 45 celltypes, for each cell type, there are 32-dim features. <for UMAP, TSNE 2D visulization>