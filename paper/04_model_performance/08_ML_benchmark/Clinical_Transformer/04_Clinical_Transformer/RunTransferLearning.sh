#usr/bin/bash
#nohup bash RunTransferLearning.sh > RunTransferLearning.log 2>&1 &
export LD_LIBRARY_PATH=/home/shenwanxiang/anaconda3/envs/IRnet_env/lib:$LD_LIBRARY_PATH
/home/shenwanxiang/anaconda3/envs/IRnet_env/bin/python 05_RunTransferLearning.py
