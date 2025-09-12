import ollama
from ollama import ChatResponse
import pandas as pd
import accelerate
from PIL import Image
import torch
import time
import json
import ast
import re
from tqdm import tqdm
from utils.metrics import *
from utils.postprocessing import *
from utils.dataprocessing import ProcessInputData
import base64
n_gpu_use = torch.cuda.device_count()
# device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')


# model = torch.nn.DataParallel(model)
df = pd.read_csv('/home/debodeep.banerjee/ClinicalLLM/icd_10/data/emergency_icd_10_full_findings.csv')
image_paths = list(df.paths.unique())
df = df.iloc[:,1:]
# df = df.drop(['paths'], axis = 1)
print(df.shape)
print(df.head())

processes_data = ProcessInputData(df, 'all_info')
test = processes_data.generate_json()
# print(test)
test_keys = list(test.keys())

# system_message = f"""You are an expert physician assistant trained to analyze patient records and generate a **structured, evidence-weighted summary** to aid in diagnosis. Your role is to synthesize information **probabilistically**, emphasizing **prior observations (triage data) and new evidence (radiology impression)** to refine the clinical understanding of the case.

# ### Input Data:
# Patient records are provided in JSON format:

# {{"hadm_id": 
#     {{
#         "stayID": 
#         {{
#             "subject_id": <integer>,
#             "temperature": <float>,
#             "heartrate": <integer>,
#             "resprate": <integer>,
#             "o2sat": <float>,
#             "sbp": <integer>,
#             "dbp": <integer>,
#             "pain": <integer>,
#             "acuity": <integer>,
#             "chiefcomplaint": <string>,
#             "reports": <string>
#         }}
#     }}
# }}
# - Each key in the JSON represents a unique hadm ID for a patient.
# - The stay id is a unique id assigned to a patient under emergency treatment. 
# - The subject id is a unique id assigned to a patient.
# - The vital signs collected at triage include:
#   - Body temperature (`temperature`): indicates the body’s heat level; abnormal values may signal fever or hypothermia.
#   - Heart rate (`heartrate`): Number of heartbeats per minute; used to assess cardiovascular health.
#   - Respiratory rate (`resprate`): Number of breaths per minute; reflects respiratory system function.
#   - Oxygen saturation (`o2sat`): Percentage of oxygen in the blood; low levels indicate poor oxygenation.
#   - Systolic blood pressure (`sbp`): Pressure in arteries during heart contraction; measures cardiac workload.
#   - Diastolic blood pressure (`dbp`): Pressure in arteries when the heart rests; reflects vascular resistance.
#   - The `pain` field captures the patient's reported pain level on a numerical scale of 1 to 10.
#   - The `acuity` field indicates the severity of the case, where 1 is the highest severity and 5 is the lowest severity.
#   - The `chiefcomplaint` field is a free-text description of the patient's reason for visiting the emergency department.
# - Radiology Reports: Provides imaging findings that may include indications of lung, cardiac, or structural abnormalities.
# ### Bayesian-Inspired Inference:
# 1. **Prior Hypothesis (Triage Data)**  
#    - Establish an **initial clinical suspicion** based on physiological indicators (vital signs, symptoms, and patient complaints).  

# 2. **Likelihood Adjustment (Radiology impression)**  
#    - Update the prior suspicion by assessing the radiology report.  
#    - Weigh **each new piece of evidence** proportionally to its diagnostic importance.  
#    - If imaging contradicts or reinforces the initial suspicion, adjust confidence accordingly.  

# 3. **Posterior Summary (Guidance for Diagnosis)**  
#    - **Integrate both sources** (triage + radiology) into a **coherent, uncertainty-aware summary**.  
#    - Highlight **most probable clinical concerns** with confidence levels (e.g., "high likelihood of X, moderate possibility of Y").  
#    - If impression are inconclusive, **indicate potential differential diagnoses** without committing to a single one.  

# ### Instructions:
# - Use a **Bayesian-inspired approach** when synthesizing information:  
#   - Begin with an **initial assumption** based on triage data.  
#   - Adjust this assumption in light of radiology impression, emphasizing **how new evidence modifies prior expectations**.  
#   - Conclude with a **refined summary**, ensuring a logical progression of reasoning.  

# - Provide a **structured, evidence-weighted** summary of clinical observations.  
# - Identify **key abnormalities, trends, or risk factors** while maintaining **diagnostic neutrality**.  
# - Use **qualitative confidence levels** (e.g., high, moderate, low) to reflect uncertainty in the summary.  
# - **DO NOT** provide final diagnoses or ICD-10 codes—your role is to guide, not classify.  
# """


system_message = f"""
You are an expert physician assistant trained to analyze patient records and provide insightful, in-depth summary guidance to aid in diagnosis. Your goal is to interpret the provided patient data, identify patterns, and generate a reasoning-based summary that will assist the physician in diagnosing the condition.

The patient data is provided in structured JSON format as follows:

{{
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

### Data Description:
- Temperature: Indicates body heat level. Abnormal values may reflect fever, hypothermia, or inflammation.
- Heart Rate: Assesses cardiovascular health. Elevated values (tachycardia) or low values (bradycardia) can signal underlying conditions.
- Respiratory Rate: Measures breathing. High rates may suggest respiratory distress, while low rates may indicate hypoventilation.
- Oxygen Saturation (O2Sat): Indicates oxygen in the blood. Low levels may suggest hypoxia or poor lung function.
- Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP): Provide insights into vascular resistance, cardiac workload, or hypertension.
- Pain Level: A numerical scale (1–10) reflecting the patient’s reported discomfort.
- Acuity: A severity score (1 = highest severity, 5 = lowest) indicating the urgency of the condition.
- Chief Complaint: Describes the patient’s primary reason for seeking medical attention.
- Radiology Reports: Provides imaging findings that may include indications of lung, cardiac, or structural abnormalities.

---
/your task is to analyze the patient's data and provide only an insightful summary that will be helpful in deciding the diagnosis in later stages.

### Instructions:
Your task is to:
1. Carefully analyze the data provided in the JSON record, focusing on vital signs, symptoms, and radiology findings.
2. Identify relevant patterns, abnormalities, or potential risk factors that could explain the patient’s clinical presentation.
3. Provide only a easoning-based summary of the observations that will assist in understanding the patient’s physiological state. 
4. Ensure the summary integrates the clinical data cohesively and highlights any significant findings that may warrant further investigation.
5. Do NOT provide final diagnoses or assign ICD-10 codes. Your role is to guide, not diagnose.
6. Avoid providing redundant information; focus on key points for clinical consideration.
7. Be concise, focusing only on the critical details that help explain the patient’s physiological condition and can assist in later diagnosis.
"""

guidance_list = []
for i in tqdm(range(len(test_keys))):
    data = test[test_keys[i]]
    # image_path = image_paths[i]
    question = (
    f"Analyze the provided JSON data and provide an overlall analysis that will be helpful for diagnosis."
    )
    # image = Image.open(image_path).convert('RGB')
    
    messages = [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": system_message+'\n'+question+ '\n' + str(data) },
        # {"role": "assistant", "content": ""}
        ]
    response: ChatResponse = ollama.chat(
    'llama3.3:70b',
    messages, # higher value is higher randomness
    options={"temperature":0.0}
    )
    outcome = response["message"]["content"]
    print('outcome: ', outcome)
    guidance_list.append(outcome)
    
    print('-'*100)
print(len(guidance_list))
#queries_df = pd.read_csv('icd_10/data/queries_df_icd_10.csv')
details = df[['subject_id', 'stay_id', 'hadm_id']]
details = details.drop_duplicates()
details['guidance'] = guidance_list
json_guidacne_test = details.set_index('hadm_id').to_dict(orient='index')
with open('icd_10/data/json_guidance_llama70_full_findings_nb.json', 'w') as f:
    json.dump(json_guidacne_test, f)
    