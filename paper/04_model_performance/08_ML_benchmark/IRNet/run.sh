## install IRnet via: https://github.com/yuexujiang/IRnet
conda activate IRnet_env


python /home/shenwanxiang/Research/IRnet/predict.py -input /home/shenwanxiang/Research/IRnet/example_expression.txt -output ./prediction_results/ -treatment anti-PD1


python /home/shenwanxiang/Research/IRnet/predict.py -input ./data/Gide_pipe_counts.txt -output ./results/Gide_prediction_results/ -treatment anti-PD1

python /home/shenwanxiang/Research/IRnet/predict.py -input ./data/Kim_counts_irnet.txt -output ./results/Kim_prediction_results/ -treatment anti-PD1


python /home/shenwanxiang/Research/IRnet/predict.py -input ./data/Imvigor210_counts_irnet.txt -output ./results/Imvigor210_prediction_results/ -treatment anti-PDL1