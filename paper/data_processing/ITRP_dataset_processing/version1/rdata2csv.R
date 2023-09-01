
#Download data of[Kong et al.Nat. Comm. 2022](https://www.nature.com/articles/s41591-018-0136-1)  
#```python
#import gdown
#gdown.download(id='1hMK9d4icJHeQHu2BvbWl0umtwJihK2Fh', output = '/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/RData/data.zip')
#```
#unzip data.zip into /n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/RData/ folder



################# preprocess-tcga #########################
load('/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/RData/prob.TCGA.ICI.RData')
save_dir = '/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/RData2csv/TCGA'
if (!dir.exists(save_dir)){dir.create(save_dir)}

dat_items <- names(prob)
for(j in dat_items){
    mydat = prob[[j]]
    write.table(mydat, file = paste0(save_dir, '/', j, '.txt'), sep = "\t", quote = F)
}


################# preprocess-task data #########################
directory = '/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/RData/immuno/'
save_dir = '/n/data1/hms/dbmi/zitnik/lab/users/was966/ITRP/RData2csv/immuno'
if (!dir.exists(save_dir)){dir.create(save_dir)}

file_list <- list.files(directory)

for(i in file_list){
    save_path = paste0(save_dir, '/', i)
    if (!dir.exists(save_path)){dir.create(save_path)}
    rdata = paste0(directory, '/', i)
    load(rdata)
    dat_items <- names(dat)
    for(j in dat_items){
        mydat = dat[[j]]
        write.table(mydat, file = paste0(save_path, '/', j, '.txt'), sep = "\t", quote = F)
    }


}


