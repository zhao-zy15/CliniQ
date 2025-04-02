import pandas as pd
import json
import re
from tqdm import tqdm
from langchain.text_splitter import CharacterTextSplitter


MIMIC_3_DIR = ""

notes = pd.read_csv('%s/NOTEEVENTS.csv' % MIMIC_3_DIR, dtype = {'ROW_ID': str, 'HADM_ID': str})
notes = notes[['ROW_ID', 'HADM_ID', 'CATEGORY', 'TEXT']].astype(str)
notes = notes[notes['CATEGORY'] == 'Discharge summary']
notes.drop_duplicates(subset = 'HADM_ID', keep = 'first', inplace = True)
has_discharge_summary = notes.groupby('HADM_ID')['CATEGORY'].apply(lambda x: any(x == 'Discharge summary'))
notes = notes[notes['HADM_ID'].isin(has_discharge_summary[has_discharge_summary].index)]
print(len(notes))
print(len(pd.unique(notes['HADM_ID'])))


def clean_text(text):
    # 去除日期信息
    text = re.sub(r'\[.*?\]', '', text)
    
    # 去除多余的空行和空格
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r' +', ' ', text)
    
    text = text.strip()

    return text
    
with open("benchmark/test_hadm_ids.txt") as f:
    ids = f.readlines()
ids = [i.strip() for i in ids]
print(len(ids))

text_splitter = CharacterTextSplitter(separator = " ", chunk_size = 100, chunk_overlap = 10, length_function = lambda x: len(x.split()))

with open("benchmark/corpus.jsonl", "w") as f:
    for _, row in notes.iterrows():
        if row['HADM_ID'] not in ids:
            continue
        text = clean_text(row['TEXT'])
        texts = text_splitter.split_text(text)

        for i, sub_text in enumerate(texts):
            f.write(json.dumps({"_id": f"{row['HADM_ID']}_{i}", 'text': sub_text}) + '\n')
