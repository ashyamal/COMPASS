<a href="url"><img src="./misc/compass_logo.png" align="left" height="130" width="130" ></a>


[![Dataset](https://img.shields.io/badge/datasets-ITRP-green)](https://zitniklab.hms.harvard.edu/compass-101/data)
## Compass for cross-cancer AI modeling of immunotherapy response

-----

## 1. Installing and Importing Compass

#### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/mims-harvard/Immune-compass.git
cd Immune-compass
pip install -r requirements.txt
```

#### Adding Compass to Your Environment
Before importing compass, add it to your Python path:
```python
import sys
sys.path.insert(0, 'your_path/Immune-compass')
```

#### Importing Compass
Now, you can import compass and its key components:
```python
import compass
from compass import PreTrainer, FineTuner, loadcompass
```


## 2. Making Predictions with a Compass Model

You can download all available Compass fine-tuned models [here](https://www.immuno-compass.com/download/) for prediction.

The input `df_tpm` is gene expression tabular data. Please refer [here](https://www.immuno-compass.com/help/index.html#section1) for details on generating input data. The first column represents the cancer code, while the remaining 15,672 columns correspond to genes. Each row represents one patient. An example input file can be downloaded [here](https://www.immuno-compass.com/download/other/compass_input_example.csv).

The output `df_pred` contains two columns, where `0` indicates non-response and `1` indicates response.

```python
import pandas as pd
from compass import loadcompass

df_tpm = pd.read_csv('./data/Compass_tpm.tsv', sep='\t', index_col=0)
model = loadcompass('https://www.immuno-compass.com/download/model/LOCO/pft_leave_Gide.pt')  
# Use map_location = 'cpu' if you dont have a GPU card
_, df_pred = model.predict(df_tpm, batch_size=128)
```



## 3. Extracting Features with a Compass Model

Compass can also function as a feature extractor. The extracted gene-level, geneset-level, or cell type/pathway-level features can be used to build a logistic regression model for response prediction or a Cox regression model for survival prediction.

```python
dfgn, dfgs, dfct = model.extract(df_tpm, batch_size=128, with_gene_level=True)
```

The outputs `dfgn`, `dfgs`, and `dfct` represent gene-level (15,672), geneset-level (133), and concept-level (44) features, respectively. The extracted features are scalar scores. If you need vector features (dim=32), use the following method:

```python
dfgs, dfct = model.project(df_tpm, batch_size=128)
```


## 4. Fine-Tuning Compass on Your Own Data

If you have in-house data and would like to fine-tune a Compass model, you can use any Compass model for fine-tuning. You can either load the pre-trained Compass model or a publicly available fine-tuned Compass model.

**Important Note:** If you choose a fine-tuned model, ensure that the `load_decoder` parameter is set to `True`:
```python
load_decoder = True
```

### Example Fine-Tuning Process
```python
model = loadcompass('https://www.immuno-compass.com/download/model/finetuner_pft_all.pt')  
ft_args = {'mode': 'PFT', 'lr': 1e-3, 'batch_size': 16, 'max_epochs': 100, 'load_decoder': True}
finetuner = FineTuner(model, **ft_args)

# Load the true labels
df_cln = pd.read_csv('./data/Compass_clinical.tsv', sep='\t', index_col=0)
dfy = pd.get_dummies(df_cln.response_label)

# Fine-tune the model
finetuner.tune(df_tpm, dfy)
finetuner.save('./finetuner.pt')
```
This process fine-tunes the Compass model on your data and saves the updated model for future use.




## 5. Pre-training Compass from Scracth





## 6. Addtional information
* Compass data pre-processing: 
* Compass online web server for prediction and feature extraction: