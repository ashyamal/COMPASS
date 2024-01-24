import os
import json


cwd = os.path.dirname(__file__)

with open(os.path.join(cwd, 'cancer_code.json'), 'r') as file:
    CANCER_CODE = json.load(file)