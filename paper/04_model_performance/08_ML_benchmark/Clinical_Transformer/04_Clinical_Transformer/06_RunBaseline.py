# export LD_LIBRARY_PATH=/home/shenwanxiang/anaconda3/envs/IRnet_env/lib:$LD_LIBRARY_PATH
# nohup /home/shenwanxiang/anaconda3/envs/IRnet_env/bin/python 06_RunBaseline.py > RunBaseline.log 2>&1 &

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/shenwanxiang/anaconda3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
tf.test.is_gpu_available()
tf.config.run_functions_eagerly(True)
tf.__version__
import gc

import sys
sys.path.append('../codeocean/environment/clinical_transformer/')
from xai.models import Trainer
from xai.models import SurvivalTransformer
from xai.models import OptimizedSurvivalDataGenerator as SurvivalDataGenerator
from xai.losses.survival import cIndex_SigmoidApprox as cindex_loss
from xai.metrics.survival import sigmoid_concordance as cindex_metric
from xai.models.explainer import TransformerSurvivalEvaluator
from xai.models.explainer import survival_attention_scores
from xai.models import load_transformer

import pandas as pd
from samecode.random import set_seed

import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt
from samecode.survival.plot import KMPlot
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index as lfcindex
import seaborn as sns
from shap.plots import colors
from glob import glob
import os
colors = [colors.blue_rgb, colors.red_rgb]

## PARAMETERS
max_features_percentile=100
test_size=0.1 # fraction of samples used for validation
repetitions=10  # number replicates (training / validation) random splits to evaluate variability.
mode='survival'
learning_rate=0.0001
epochs=300
verbose=1

embedding_size = 128
num_heads = 2
num_layers = 8


SEED = [24, 42, 64]


def cleanup_model_checkpoints(outdir):
    for i in range(repetitions):
        cmd = f"rm -r {outdir}/fold-{i}_id-{i}/model.*.h5"
        os.system(cmd)

        
def get_best_epoch(train_data, features, pretrained_path, 
                   work_dir = './results/model/cohort'):
    
    outdir = work_dir
    os.system('rm -r %s' % outdir)

    set_seed(0)

    trainer = Trainer(
        #from_pretrained=pretrained_path,
        out_dir = outdir,
        max_features_percentile=max_features_percentile,
        test_size=test_size,
        mode=mode,
        model=SurvivalTransformer, 
        dataloader=SurvivalDataGenerator,
        loss=cindex_loss,
        metrics=[cindex_metric]
    )

    trainer.setup_data(
        train_data, 
        discrete_features = [],
        continuous_features = features,
        target=['time', 'event']
    )

    trainer.setup_model(
        learning_rate=learning_rate,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_layers=num_layers,
        batch_size_max=True,
        save_best_only=False
    )

    trainer.fit(repetitions=repetitions, 
                epochs=epochs, 
                verbose=verbose, 
                seed=0)
    
    del trainer; tf.keras.backend.clear_session(); gc.collect()
    best_es = []
    for i in range(repetitions):
        dfh = pd.read_csv(f"{outdir}/fold-{i}_id-{i}/history.csv", index_col=0)
        best_es.append(dfh.val_sigmoid_concordance)
    best_epoch = pd.concat(best_es, axis=1).mean(axis=1).idxmax() + 1

    return best_epoch


def train(train_data, features, pretrained_path, best_epoch, 
                  work_dir = './results/TransferLearningSurvival/'
                  ):

    all_dirs = []
    for seed in SEED:
        set_seed(seed)
        
        outdir = f'{work_dir}/full_{seed}'
        os.system('rm -r %s' % outdir)
        
        trainer = Trainer(
            #from_pretrained=pretrained_path,
            out_dir = outdir,
            max_features_percentile=max_features_percentile,
            test_size=0.,
            mode=mode,
            model=SurvivalTransformer, 
            dataloader=SurvivalDataGenerator,
            loss=cindex_loss,
            metrics=[cindex_metric]
        )

        trainer.setup_data(
            train_data, 
            discrete_features = [],
            continuous_features = features,
            target=['time', 'event']
        )

        trainer.setup_model(
            learning_rate=learning_rate,
            embedding_size=embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
            batch_size_max=True,
            save_best_only=False
        )

        trainer.fit(repetitions=1, 
                    epochs=best_epoch, 
                    verbose=verbose, 
                    seed=seed)
        
        del trainer; tf.keras.backend.clear_session(); gc.collect()
        
        all_dirs.append(outdir)
        
    return all_dirs


def predict(train_data, test_data, features, best_epoch, model_dirs):

    predicted_paths = []
    for path_dir in model_dirs:
        sample_id = 'Index'
        run = 'fold-0_id-0'
        path = f'{path_dir}'
        
        repeat = path_dir.split('/')[-1]
        cohort = path_dir.split('/')[-2]  
        model = path_dir.split('/')[-3]        
        
        trainer = load_transformer(path, run, epoch=best_epoch)
        transformed_data = trainer.data_converter.transform(train_data).reset_index(drop=True)
        set_seed(0)
        evaluator = TransformerSurvivalEvaluator(model=trainer)
        train_data, outputs, features, patient_ids, iters, attention_scores = survival_attention_scores(
            transformed_data, evaluator, iterations=1, sample_id=sample_id
        )
        train_cindex = lfcindex(train_data['time'], train_data['β'], train_data['event'])
        transformed_test_data = trainer.data_converter.transform(test_data).reset_index(drop=True)
        evaluator = TransformerSurvivalEvaluator(model=trainer)

        test_data, outputs, features, patient_ids, iters, attention_scores = survival_attention_scores(
            transformed_test_data, evaluator, iterations=10, sample_id=sample_id
        )
        test_cindex = lfcindex(test_data['time'], test_data['β'], test_data['event'])

        del trainer; tf.keras.backend.clear_session(); gc.collect()
        median_cutoff = np.quantile(train_data.β, 0.75)


        test_data['population'] = (test_data.β >= median_cutoff).replace([False, True], ['Low Score', 'High Score'])
        train_data['population'] = (train_data.β >= median_cutoff).replace([False, True], ['Low Score', 'High Score'])

        fig, axs = plt.subplots(ncols=2, nrows=1, figsize = (12,5))
        KMPlot(train_data, time='time', event='event', label=['population']).plot(ax=axs[0], 
                                                                              colors = colors, 
                                                                              title='Training set',
                                                                              ci_show=True,ci_alpha=0.10,

                                                                             )
        KMPlot(test_data, time='time', event='event', label=['population']).plot(ax=axs[1],
                                                                                 colors = colors, ci_show=True,
                                                                                 ci_alpha=0.10,
                                                                                 title=f'Testing set ({cohort})')

        
        
        test_data['test_cohort'] = cohort
        test_data['model'] = model
        test_data['repeat'] = repeat
        test_data['best_epoch'] = best_epoch
        
        train_data['test_cohort'] = train_data['cohort']
        train_data['model'] = model
        train_data['repeat'] = repeat
        train_data['best_epoch'] = best_epoch
        
        test_data.to_csv(os.path.join(path, 'test_data.csv'), index = False)
        train_data.to_csv(os.path.join(path, 'train_data.csv'), index = False)
        fig.savefig(os.path.join(path, 'train_test_km.svg'))
        predicted_paths.append(os.path.join(path, 'test_data.csv'))
    
    return predicted_paths



y = pd.read_csv('../data/ITRP_clinical.csv', index_col=0)
y = y[(~y.OS_Months.isna()) & (~y.OS_Event.isna())]
y['time'] = y['OS_Months']
y['event'] = y['OS_Event']
y = y[['time', 'event','cohort', 'ICI_target', 'ICI','cancer_type','response_label', 'TMB']]
cohort_list = y.cohort.unique()


Model_NAMES = ['VanillaClinicalTransformer',
              'ssGSEA43ClinicalTransformer',
              'COMPASSClinicalTransformer',]

Model_PATHS = ['01_run_Feg29_TME',
              '02_run_ssGSEA43_TME',
              '03_run_Com44_TME',]

X_PATHS = ['ITRP_ssGSEA_Fegs29',
          'ITRP_ssGSEA_concept43',
          'ITRP_COMPASS_concept44']


save_dir = './Results/ResultsBaseline'
os.makedirs(save_dir, exist_ok=True)


# cohort_list_all = ['IMVigor210','Liu', 'SU2CLC1', 'SU2CLC2','Rose', 'Snyder',
#                 'Hugo', 'Gide', 'Riaz', 'Allen', 'MGH'] 

cohort_list = ['IMVigor210','Liu', 'SU2CLC1'] 

for i in range(3):
    model_name = Model_NAMES[i]
    pt_path = Model_PATHS[i]
    X_file_path = X_PATHS[i]

    for test_cohort in cohort_list:

        X = pd.read_csv(f'../data/{X_file_path}.csv', index_col=0)
        dataset_full = y.join(X)
        features = X.columns.tolist()

        train_data = dataset_full[dataset_full.cohort != test_cohort].reset_index()
        test_data = dataset_full[dataset_full.cohort == test_cohort].reset_index()


        pretrained_path = f'./{pt_path}/FoundationModel/fold-0_id-0/'
        best_epoch = get_best_epoch(train_data, features, pretrained_path, 
                           work_dir = f'{save_dir}/{model_name}/{test_cohort}')

        cleanup_model_checkpoints(f'{save_dir}/{model_name}/{test_cohort}')

        model_dirs = train(train_data, features, pretrained_path, best_epoch, 
                          work_dir = f'{save_dir}/{model_name}/{test_cohort}'
                          )
        pred_dirs = predict(train_data.copy(), test_data.copy(), 
                            features, best_epoch, model_dirs)
        tf.keras.backend.clear_session(); gc.collect()