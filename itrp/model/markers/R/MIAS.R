

##### ssGSEA using GSVA
MIAS.Score.GSVA<-function (DataM_EX,Signatures_M) 
{
	##
	library(GSVA)
	Score.pos<- gsva(DataM_EX, Signatures_M, verbose=FALSE, parallel.sz=1)[1,]
	MIAS_Score=Score.pos
	return (MIAS_Score)
}


##### ssGSEA using GSVA
GEP.Score.GSVA<-function (DataM_EX) 
{
	##
	library(GSVA)
	GEP.genes<-NULL
	GEP.genes$GEP<-c("CCL5","CD27","CD274","CD276","CD8A","CMKLR1","CXCL9","CXCR6","HLA-DQA1"
		,"HLA-DRB1","HLA-E","IDO1","LAG3","NKG7","PDCD1LG2","PDL2","PSMB10","STAT1","TIGTT")
	Score<- gsva(DataM_EX, GEP.genes, verbose=FALSE, parallel.sz=1)[1,]
	return (Score)
}


#######  Calculate IMPRES.Score Scores
IMPRES.Score<-function (DataM_EX) {
	## !!! DataM_EX is the gene expression matrix (genes vs samples)
	## IMPRES genes based on Auslander et al., Nat Med. 2018
	IMPRES.Gene1<-c("PDCD1","CD27","CTLA4","CD40","CD86","CD28","CD80","CD274","CD86","CD40"
		,"CD86","CD40","CD28","CD40","TNFRSF14")
	IMPRES.Gene2<-c("TNFSF4","PDCD1","TNFSF4","CD28","TNFSF4","CD86","TNFSF9","VSIR","HAVCR2"
		,"PDCD1","CD200","CD80","CD276","CD274","CD86")
	
	## match the IMPRES genes
	SYMBOL<-rownames(DataM_EX)
	loc1<-match(IMPRES.Gene1,SYMBOL)
	loc2<-match(IMPRES.Gene2,SYMBOL)
	pos1<-which(loc1>0)
	pos2<-which(loc2>0)
	pos<-intersect(pos1,pos2)
	loc1<-loc1[pos]
	loc2<-loc2[pos]
	EX1<-DataM_EX[loc1,]
	EX2<-DataM_EX[loc2,]
	
	##
	DIM<-dim(DataM_EX)	
	IMPRES_Score<-rep(0,DIM[2])
	for (kk in 1:DIM[2]) {
		EX1b<-EX1[,kk]
		EX2b<-EX2[,kk]
		IMPRES_Score[kk]<-length(which(EX1b>EX2b))
	}

	##
	names(IMPRES_Score)<-colnames(DataM_EX)
	return (IMPRES_Score)
}



####### Calculate AUC
ROCF<-function(prediction,score)
{
	##
	N<-length(score)
	N_pos<-sum(prediction)
	N_neg<-N-N_pos
	df<-data.frame(pred=prediction,score=score)
	df<-df[order(-df$score),]
	df$above=(1:N)-cumsum(df$pred)
	AUC<-1-sum(df$above*df$pred/(N_pos*(N-N_pos)))
	ranking<-seq(N,1,by=-1)
	TP<-cumsum(df$pred)
	FP =df$above
	threshold_indx = which(diff(ranking)!=0); 
	TPR = TP[threshold_indx]/N_pos;
	FPR = FP[threshold_indx]/N_neg;
	ROC=list(TPR=TPR,FPR=FPR,AUC=AUC)
	
	##
	return(ROC)
}






























	