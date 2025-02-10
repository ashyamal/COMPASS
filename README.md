<a href="url"><img src="./misc/compass_logo.png" align="left" height="130" width="130" ></a>


[![Dataset](https://img.shields.io/badge/datasets-ITRP-green)](https://zitniklab.hms.harvard.edu/compass-101/data)
## Compass for cross-cancer AI modeling of immunotherapy response

-----

# 1. Installing and Importing Compass

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/mims-harvard/Immune-compass.git
cd Immune-compass
pip install -r requirements.txt
```

## Adding Compass to Your Environment
Before importing compass, add it to your Python path:
```python
import sys
sys.path.insert(0, 'your_path/Immune-compass')
```

## Importing Compass
Now, you can import compass and its key components:
```python
import compass
from compass import PreTrainer, FineTuner, loadcompass
```


# 2) Making Predictions with a Compass Model

You can download all available Compass fine-tuned models [here](https://www.immuno-compass.com/download/) for prediction.

The input `df_tpm` is gene expression tabular data. Please refer [here](https://www.immuno-compass.com/help/index.html#section1) for details on generating input data. The first column represents the cancer code, while the remaining 15,682 columns correspond to genes. Each row represents one patient. An example input file can be downloaded [here](https://www.immuno-compass.com/download/other/compass_input_example.csv).

The output `dfpred` contains two columns, where `0` indicates non-response and `1` indicates response.

```python
import pandas as pd
from compass import loadcompass

df_tpm = pd.read_csv('./data/Compass_tpm.tsv', sep='\t', index_col=0)
model = loadcompass('https://www.immuno-compass.com/download/model/LOCO/pft_leave_Gide.pt', 
                    map_location='cpu')  # Change to torch.device('cuda:device_id') if you want to use GPU

_, dfpred = model.predict(df_tpm, batch_size=128)
```



# 3) Extracting Features with a Compass Model

Compass can also serve as a feature extractor. These features can be used for build a logistic regression for response prediction or a Cox regression for survail prediction


















# Compass-101

# About
----


* Concept-based Approach: The study employs a concept-based methodology, which involves categorizing and analyzing various TIME-concepts or features related to the tumor microenvironment, immune system interactions, and treatment responses.

* Pre-trained Models: The models used in the study are pre-trained, indicating that they have been initially trained on a large TCGA dataset to learn patterns and relationships before being fine-tuned for specific tasks related to immunotherapy response prediction.

* Enhanced Pan-Cancer Immunotherapy Response Prediction: The primary focus of the study is to improve the accuracy and effectiveness of predicting patient responses to immunotherapy across different types of cancer. This includes developing models that can predict how individual patients with varying cancer types will respond to immunotherapy treatments.

Overall, the study aims to leverage concept-based, pre-trained models to enhance the prediction of immunotherapy responses in a pan-cancer context, ultimately contributing to more personalized and effective cancer treatment strategies.

----------

![image](https://github.com/mims-harvard/mims-responder/assets/21102929/0e0916fe-e040-4870-b5ac-0e1166ad188e)





