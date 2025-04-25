import pandas as pd
from pydantic import BaseModel
from datasets import load_dataset

class LoadData(BaseModel):

    def load_dataset(self):
        df = pd.read_csv(r'transformer\data\spa.txt', sep='\t', header=None)
        df = df.iloc[:200_000]
        df.columns = ['en', 'es']
        df.to_csv(r'transformer\data\spa.csv', index=None)

        raw_dataset = load_dataset('csv', data_files=r'transformer\data\spa.csv')
        return raw_dataset