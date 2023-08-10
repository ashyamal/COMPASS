# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("BioinformaticsFMRP/TCGAbiolinksGUI.data") ## adapte to new version GDC
# BiocManager::install("BioinformaticsFMRP/TCGAbiolinks")
# BiocManager::install("SummarizedExperiment")

library(TCGAbiolinks)
library(SummarizedExperiment)


setwd('/n/data1/hms/dbmi/zitnik/lab/users/was966/TCGA')
directory = '/n/data1/hms/dbmi/zitnik/lab/users/was966/TCGA/GDC_v37'

#melanoma：'TCGA-SKCM'

#version_https://docs.gdc.cancer.gov/Data/Release_Notes/Data_Release_Notes/#data-release-370
tcga_list = c('TCGA-BRCA','TCGA-GBM','TCGA-OV','TCGA-LUAD','TCGA-UCEC','TCGA-KIRC',
              'TCGA-HNSC','TCGA-LGG','TCGA-THCA','TCGA-LUSC','TCGA-PRAD','TCGA-SKCM',
              'TCGA-COAD','TCGA-STAD','TCGA-BLCA','TCGA-LIHC','TCGA-CESC','TCGA-KIRP',
              'TCGA-SARC','TCGA-LAML','TCGA-ESCA','TCGA-PAAD','TCGA-PCPG','TCGA-READ',
              'TCGA-TGCT','TCGA-THYM','TCGA-KICH','TCGA-ACC','TCGA-MESO','TCGA-UVM',
              'TCGA-DLBC','TCGA-UCS','TCGA-CHOL')


for(i in tcga_list){
    
    ## expression ###
    query <- GDCquery(project = i,
                    data.category = 'Transcriptome Profiling',
                    data.type = 'Gene Expression Quantification',
                    workflow.type = 'STAR - Counts',) #, "Primary Tumor" #sample.type = c("Metastatic")
    
    if(is.null(query) != TRUE){
        GDCdownload(query,directory = directory)
        data <- GDCprepare(query, directory = directory)
        #use assayNames(data) to select the tpm_unstrand column
        data_counts <- assay(data,i = 1)#
        data_tpm <- assay(data,i = 4)#
        
        rowdata <- rowData(data) ## gene table
        coldata <- colData(data) ## patient table
    
        #treatments <- coldata$treatments
        #primary_site <- coldata$primary_site
        #primary_site_df = as.data.frame(do.call(rbind, primary_site))
        #treatments_df = as.data.frame(do.call(rbind, treatments))
        
        # remove \n in treatments col of coldata
        coldata$treatments <- gsub("\\n","",coldata$treatments )
        coldata$primary_site <- gsub("\\n","",coldata$primary_site )
        
        write.csv(data_counts, file = paste0(directory, '/', i,'/rna_seq_counts_matrix.csv'))
        write.csv(data_tpm, file = paste0(directory, '/', i, '/rna_seq_tpm_matrix.csv'))
        write.csv(rowdata, file = paste0(directory, '/', i, '/gene_table.csv'))
        write.csv(coldata, file = paste0(directory, '/', i, '/sample_table.csv'))
    }
}





# ### simple nucleotide variation ###
# for(j in c('muse','mutect2','somaticsniper','varscan2')){
#   maf <- GDCquery_Maf(strsplit(i, '-')[[1]][2], pipelines = j)
#   write.csv(maf, file = paste0('/n/data1/hms/dbmi/zitnik/lab/users/was966/TCGAbiolinks/GDCdata/',i,'/SNV_',j,'.csv'))
# }

### Copy Number Variation ###
# query <- GDCquery(project = i,
#                   data.category = 'Copy Number Variation',
#                   data.type = 'Copy Number Segment',
#                   sample.type = 'Primary Tumor')
# if(is.null(query) != TRUE){
#   GDCdownload(query)
#   data <- GDCprepare(query)
#   eset <- assay(data)
#   write.csv(eset, file = paste0('/n/data1/hms/dbmi/zitnik/lab/users/was966/TCGAbiolinks/GDCdata/',i,'/CNV_table.csv'))
# }

### clinical ###
## drug response ##
# query <- GDCquery(project = i,
#                 data.category = 'Clinical',
#                 data.type = 'Clinical Supplement',
#                 data.format = 'BCR Biotab')
# if(is.null(query) != TRUE){
# GDCdownload(query)
# data <- GDCprepare(query)
# eset <- data[[paste0('clinical_drug_',tolower(strsplit(i,'-')[[1]][2]))]]
# write.csv(eset, file = paste0('/n/data1/hms/dbmi/zitnik/lab/users/was966/TCGAbiolinks/GDCdata/',i,'/clinical_drug_table.csv'))
# }

# ## patient information ##
# query <- GDCquery(project = i,
#                 data.category = 'Clinical',
#                 data.type = 'Clinical Supplement',
#                 data.format = 'BCR Biotab')
# if(is.null(query) != TRUE){
# GDCdownload(query)
# data <- GDCprepare(query)
# eset <- data[[paste0('clinical_patient_',tolower(strsplit(i,'-')[[1]][2]))]]
# write.csv(eset, file = paste0('/n/data1/hms/dbmi/zitnik/lab/users/was966/TCGAbiolinks/GDCdata/',i,'/clinical_patient_table.csv'))
# }


