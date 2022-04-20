import os 
import pandas as pd 

class Aggregate():
    def __init__(self,origin,dest):
        self.origin = origin
        self.dest = dest 
        self.cats = {"outcomes.csv" : 'outcomes',
                    "street.csv" : "crimes",
                    "stop-and-search.csv" : "stop_and_search"
                    }

    def _generate_file_name_mapping(self):
        files = os.listdir(self.origin)
        self.file_cats = {}
        for file in files:
            for cat in self.cats:
                if cat in file:
                    if self.cats[cat] not in self.file_cats:
                        self.file_cats[self.cats[cat]]=[]
                    else:
                        self.file_cats[self.cats[cat]].append(file)   

    def _generate_dfs(self):
        self._generate_file_name_mapping()
        self.data = {}
        for cat in self.file_cats:
            dfs = []
            for file in self.file_cats[cat]:
                path = os.path.join(self.origin,file)
                dfs.append(pd.read_csv(path))
            self.data[cat] = pd.concat(dfs,axis=0)

    def aggregate_write(self):
        self._generate_dfs()
        for table in self.data:
            path = os.path.join(self.dest,f"{table}.csv")
            self.data[table].to_csv(path)
                