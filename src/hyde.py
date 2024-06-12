import json

import os.path as osp

from tqdm import tqdm
from model import LLM

def hyde(query, model: LLM):
    task_description = """From the perspective of a professional, please answer the following scientific questions with specificity and clarity using the line order of an academic paper abstract:\n\n[question]:"""
    task_description += query
    task_description += "\n\n[answer]:\n"

    response = model.model_prediction(task_description)
    return response


if __name__ == "__main__":
    work_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    model = LLM('chatglm3-6b')
    with open(osp.join(work_root, "AQA", "qa_valid_wo_ans.txt"), "r") as f:
        data = f.readlines()
    
    res = []
    for item in tqdm(data):
        item = json.loads(item)
        query = item["question"]
        try:
            response = hyde(query, model)
        except Exception as e:
            print(e)
            response = "ERROR"
        res.append({"question": query, "hyde": response})
    json.dump(res, open("hyde_val.json", "w"), indent=4, ensure_ascii=False)
