#usr/bin/bash
#nohup bash RunBaseline.sh > RunBaseline.log 2>&1 &
export LD_LIBRARY_PATH=/home/shenwanxiang/anaconda3/envs/IRnet_env/lib:$LD_LIBRARY_PATH
/home/shenwanxiang/anaconda3/envs/IRnet_env/bin/python 06_RunBaseline.py
# /home/shenwanxiang/anaconda3/envs/IRnet_env/bin/python 06_RunBaseline_2.py
# /home/shenwanxiang/anaconda3/envs/IRnet_env/bin/python 06_RunBaseline_3.py