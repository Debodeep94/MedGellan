import json
import ast

class ProcessInputData:
    def __init__(self,data, mode):
        '''
        data: dataframe
        mode: str
        '''
        self.data=data
        self.mode = mode
    def generate_json(self):
        if self.mode == 'all_info':
            nested_data = {}
            for _, row in self.data.iterrows():
                hadm_id = row["hadm_id"]
                stay_id = row["stay_id"]
                
                if hadm_id not in nested_data:
                    nested_data[hadm_id] = {}
            
                if stay_id not in nested_data[hadm_id]:
                    # Extract all other columns except 'icd_10' to store under stay_id
                    stay_data = row.drop(["hadm_id", "stay_id", "icd_code"]).to_dict()
                    # stay_data["diagnosis"] = []
                    nested_data[hadm_id][stay_id] = stay_data
                
                # Append the diagnosis codes
                # nested_data[hadm_id][stay_id]["diagnosis"].append(row["icd_10"])
            
            # Convert to JSON and save
            json_output = json.dumps(nested_data, indent=4)
            json_output = ast.literal_eval(json_output)
            return json_output
        elif self.mode == 'triage':
            nested_data = {}
            for _, row in self.data.iterrows():
                hadm_id = row["hadm_id"]
                stay_id = row["stay_id"]
                
                if hadm_id not in nested_data:
                    nested_data[hadm_id] = {}
            
                if stay_id not in nested_data[hadm_id]:
                    # Extract all other columns except 'icd_10' to store under stay_id
                    stay_data = row.drop(["hadm_id", "stay_id", "icd_code", "paths", "seq_num"]).to_dict()
                    # stay_data["diagnosis"] = []
                    nested_data[hadm_id][stay_id] = stay_data
                
                # Append the diagnosis codes
                # nested_data[hadm_id][stay_id]["diagnosis"].append(row["icd_10"])
            
            # Convert to JSON and save
            json_output = json.dumps(nested_data, indent=4)
            json_output = ast.literal_eval(json_output)
            return json_output

        elif self.mode == 'chiefcomplaint':
            nested_data = {}
            for _, row in self.data.iterrows():
                hadm_id = row["hadm_id"]
                stay_id = row["stay_id"]
                
                if hadm_id not in nested_data:
                    nested_data[hadm_id] = {}
            
                if stay_id not in nested_data[hadm_id]:
                    # Extract all other columns except 'icd_10' to store under stay_id
                    stay_data = row[["subject_id", "chiefcomplaint"]].to_dict()
                    # stay_data["diagnosis"] = []
                    nested_data[hadm_id][stay_id] = stay_data
                
                # Append the diagnosis codes
                # nested_data[hadm_id][stay_id]["diagnosis"].append(row["icd_10"])
            
            # Convert to JSON and save
            json_output = json.dumps(nested_data, indent=4)
            json_output = ast.literal_eval(json_output)
            return json_output
        elif self.mode == 'CCRad':
            nested_data = {}
            for _, row in self.data.iterrows():
                hadm_id = row["hadm_id"]
                stay_id = row["stay_id"]
                
                if hadm_id not in nested_data:
                    nested_data[hadm_id] = {}
            
                if stay_id not in nested_data[hadm_id]:
                    # Extract all other columns except 'icd_10' to store under stay_id
                    stay_data = row[["subject_id", "chiefcomplaint", "reports"]].to_dict()
                    # stay_data["diagnosis"] = []
                    nested_data[hadm_id][stay_id] = stay_data
                
                # Append the diagnosis codes
                # nested_data[hadm_id][stay_id]["diagnosis"].append(row["icd_10"])
            
            # Convert to JSON and save
            json_output = json.dumps(nested_data, indent=4)
            json_output = ast.literal_eval(json_output)
            return json_output
        else:
            raise ValueError ("Invalid mode. Please choose from 'all_info', 'triage', 'chiefcomplaint', 'CCRad'")