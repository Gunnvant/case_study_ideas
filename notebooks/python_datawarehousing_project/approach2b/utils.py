import pandas as pd 
from tqdm import tqdm 
import json 
import requests 
import os

class FetchData():
    def __init__(self,origin,dest):
        self.origin = origin 
        self.dest = dest 
    
    def _get_data_web(self):
        try:
            data = requests.get(self.origin).json()
        except:
                raise Exception("Check url")
        return data
    
    def _get_data_local(self,path):
        with open(path,"r", encoding="utf-8") as infile:
                data = json.loads(infile.read())
        return data

    def _set_data_local(self,json_data):
        with open("data.json","w",encoding="utf-8") as infile:
            json.dump(json_data,infile,indent=4)

    
    def get_data_dict(self):
        try: 
            os.path.exists(self.origin)
            json_data = self._get_data_local(self.origin)
        except:
            json_data = self._get_data_web()
            self._set_data_local(json_data)

        col_info = json_data['meta']['view']['columns']
        names,desc,dtype = [],[],[]
        for info in col_info:
            names.append(info.get('fieldName').replace(":","").strip())
            desc.append(info.get('description',"").strip())
            dtype.append(info.get('dataTypeName'))
        self.data_dict = pd.DataFrame({'col_names':names,'descriptions':desc,"dtypes":dtype})
        self.data_dict.to_csv(os.path.join(self.dest,"data_dict.csv"),index=False)

    def make_table(self):
        try:
             os.path.exists(self.origin)
             json_data = self._get_data_local(self.origin)
        except:
            json_data = self._get_data_local("data.json")
        data_container = {n:[] for n in self.data_dict['col_names'].unique()}
        for row in tqdm(json_data['data']):
            for idx,val in enumerate(row):
                key_name = self.data_dict['col_names'].iloc[idx]
                dtype = self.data_dict['dtypes'].iloc[idx]
                if dtype == "number" and val is not None:
                    val = float(val)
                else:
                    val = val
                data_container[key_name].append(val)
        pd.DataFrame(data_container).to_csv(os.path.join(self.dest,"table.csv"),index=False)      
            
