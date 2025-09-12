import ollama
from ollama import ChatResponse
import pandas as pd
import accelerate
import torch
import time
import json
import ast
import re
from tqdm import tqdm
from utils.metrics import *
from utils.dataprocessing import ProcessInputData
from utils.postprocessing import *

def extract_icd10_codes(record: str) -> list:
    import re
    icd10_pattern = r"[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?"
    codes = re.findall(icd10_pattern, record)
    return codes if codes else ["NA"]

with open('/home/debodeep.banerjee/ClinicalLLM/icd_10/data/json_guidance_llama70_full_findings_nb.json') as file: # import test files
    test = json.load(file)

# example_keys = list(example.keys())
test_keys = list(test.keys())
system_message = f"""
You are a trained clinical coder. Your task is to analyze an assessment of a patient's record prepared by an assistant physician, and provide a final diagnosis. The record will be provided in JSON format as follows:

{{
    "hadm_id": {{
        "stay_id": <integer>,
        "subject_id": <integer>,
        "guidance": <string>
    }}
}}

The guidance is constructed based on the following:
{{
    "hadm_id": {{
        "stayID": {{
            "subject_id": <integer>,
            "temperature": <float>,
            "heartrate": <integer>,
            "resprate": <integer>,
            "o2sat": <float>,
            "sbp": <integer>,
            "dbp": <integer>,
            "pain": <integer>,
            "acuity": <integer>,
            "chiefcomplaint": <string>,
            "reports": <string>
        }}
    }}
}}

Field descriptions:
- **Triage data**:
- `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`: vital signs at ED presentation.
- `pain`: numeric rating scale (1–10).
- `acuity`: case severity (1 = most severe, 5 = least).
- `chiefcomplaint`: free-text presenting complaint.
- **Radiology report**:
- `reports`: Findings section from a chest radiograph.
### Data Description:
- Each key in the JSON represents a unique hadm ID for a patient.
- The stay id is a unique id assigned to a patient under emergency treatment. 
- The subject id is a unique id assigned to a patient.
- The `guidance` is a comprehensive reasoning-based analysis of the patient's clinical condition, carefully created by an assistant physician. This guidance was generated based on the patient's triage information, radiology image and radiology report.
Your task is to analyze the information and provide relevant ICD-10 codes. 
<INSTRUCTIONS>
- The data includes a comprehensive assessment of patient's triage information and radiology report on chest radiograph.
- Analyze the data.
- Use the following process:
    2. Determine the probable ICD-10 chapters applicable. 
    3. Within that chapter, choose the most fitting categories. 
    4. Finally, select the relevant ICD-10 codes.
- Output the icd code using the format: {{'codes': [ICD Codes]}}.
- Do not include any additional explanations, comments, or suggestions.
"""


count = 0
diagnosis_list = []
for i in tqdm(test_keys):
    data = test[i]
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Analyze the following patient record and return the ICD-10 codes.\n{json.dumps(data, indent=2)}"},
    ]
    
    response = ollama.chat('llama3.3:70b', messages, options={"temperature": 0.0})
    outcome = response["message"]["content"]

    try:
        results = json.loads(outcome)
        diagnosis_list.append(results['codes'])
    except:
        # Fallback method
        results = extract_icd10_codes(outcome)
        diagnosis_list.append(results)

df = pd.read_csv('/home/debodeep.banerjee/ClinicalLLM/icd_10/data/emergency_icd_10_full_findings.csv')
df = df.iloc[:,1:]

# structure for multi output
gt = []
for i in df.stay_id.unique():
    temp = df[df['stay_id']==i]
    gt.append(list(temp['icd_code']))

count_exact = 0
count_class = 0
count_top_diag_match = 0

gt = [[re.sub('\.', '', item) for item in sublist] for sublist in gt]
diagnosis_list = [[re.sub('\.', '', item) for item in sublist] for sublist in diagnosis_list]

print('diagnosis:', diagnosis_list)
print('gt:', gt)
# Exact match
_,micro, macro = calculate_metrics_set_wise(gt, diagnosis_list)
print(f'Micro: {micro}|||Macro: {macro}')
# Category
cat_gt = [[item[:3] for item in sublist] for sublist in gt]
cat_diag = [[item[:3] for item in sublist] for sublist in diagnosis_list]
_,micro, macro = calculate_metrics_set_wise(cat_gt, cat_diag)
print('diagnosis:', cat_diag)
print('gt:', cat_gt)
print(f'Micro: {micro}|||Macro: {macro}')
# Chapter
gt_chapter_lists = [[get_icd10_chapter(code) for code in codes] for codes in gt]
pred_chapter_lists = [[get_icd10_chapter(code) for code in codes] for codes in diagnosis_list]
_,micro, macro = calculate_metrics_set_wise(gt_chapter_lists, pred_chapter_lists)
print('diagnosis:', pred_chapter_lists)
print('gt:', gt_chapter_lists)
print(f'Micro: {micro}|||Macro: {macro}')       
finish=time.time()
# print('total time:', finish-start)    