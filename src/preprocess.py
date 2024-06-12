import torch
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import dill

from dataclasses import dataclass
import os.path as osp
path_root = osp.dirname(osp.dirname(osp.abspath(__file__)))

@dataclass
class AQAdata:
    doc:dict
    question:dict

def read_data():
    raw_doc = json.load(open(osp.join(path_root,'data/AQA/pid_to_title_abs_new.json')))
    raw_question = {}
    with open(osp.join(path_root,'data/AQA/qa_train.txt'), 'r') as f:
        raw_question['train'] = [json.loads(x) for x in f.readlines()]    
    for name in ['test', 'valid']:
        with open(osp.join(path_root,f'data/AQA/qa_{name}_wo_ans.txt'), 'r') as f:
            raw_question[name]  = [json.loads(x) for x in f.readlines()]
    return raw_doc, raw_question

#TODO
def EDA(raw_doc, raw_question):# 重复、缺失
    [x['title'] for x in list(raw_doc.values())]


def parse(xml):
    soup = BeautifulSoup(xml, 'html.parser')
    return soup.get_text()


def process_data(raw_doc, raw_question):
    doc = {}
    question = {}
    
    for idx in raw_doc.keys():
        doc[idx] = str(raw_doc[idx]['title']) + '\n\n' + str(raw_doc[idx]['abstract'])
    
    question['test'] = [{'query':x['question'] + '\n\n' + parse(x['body'])} for x in raw_question['test']]
    question['valid'] = [{'query':x['question'] + '\n\n' + parse(x['body'])} for x in raw_question['valid']]
    question['train'] = [{'query':x['question'] + '\n\n' + parse(x['body']), 'pids':x['pids']} for x in raw_question['train']]
    
    data = AQAdata(
        doc = doc,
        question = question
    )
    with open(osp.join(path_root, 'data/process_data', f'AQAdata.dill'), 'wb') as f:
        dill.dump(data, f)


if __name__ == "__main__":
    raw_doc, raw_question = read_data()
    process_data(raw_doc, raw_question)
    with open(osp.join(path_root, 'data/process_data', f'AQAdata.dill'), 'rb') as f:
        data = dill.load(f) 