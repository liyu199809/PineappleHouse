import json
import time
import os.path as osp

from model import LLM
from tqdm import tqdm
from bs4 import BeautifulSoup


def rewrite_instruct(query, body):
    instruction = f"""
    根据用户qeury和query的详细解释，以json格式输出5个和用户query表达含义一致的新query。其中，id为key（从0开始），value为新的query。
    
    注意：
    1.你只需输出json格式的新query，不要输出任何其他信息。
    2.json格式请参考```[{{"0": "xxx"}}, {{"1": "xxxx"}}]```
    3.生成的新query必须和原query的含义一致，且是一个完整的问句。
    4.每个新query之间的差异应该足够大，但含义必须一致。
    
    用户query: {query}
    
    query解释信息:{body}

    新query:
    """
    return instruction


def query_rewrite(query_list):
    result_list, erro_list = [], []
    
    for query_info in tqdm(query_list):
        query_info = json.loads(query_info)
        query, body = query_info['question'], query_info['body']

        soup = BeautifulSoup(body, 'html.parser')
        instruction = rewrite_instruct(query, soup.get_text())
        try:
            result = model.model_prediction(instruction)
            result = eval(result)
        except Exception as e:
            print(e)
            result = 'error'
            erro_list.append(instruction)
        result_list.append({'old_query':query , 'result': result})
    
    with open(osp.join(work_root, 'processed_data', 'rewrite_query_val.json'), 'w') as f:
        json.dump(result_list, f, indent=4, ensure_ascii=False)

    with open(osp.join(work_root, 'processed_data', 'erro_val.json'), 'w') as f:
        json.dump(erro_list, f, indent=4, ensure_ascii=False)


def erro_handler():
    with open(osp.join(work_root, 'processed_data', 'erro_val.json'), 'r') as f:
        erro_list = json.load(f)
    
    with open(osp.join(work_root, 'processed_data', 'rewrite_query_val.json'), 'r') as f:
        result_list = json.load(f)
    
    new_erro_list, idx = [], 0
    for res_item in tqdm(result_list):
        if res_item['result'] == 'error':
            erro_input = erro_list[idx]
            try:
                assert res_item['old_query'] in erro_input 
                result = model.model_prediction(erro_input)
                result = eval(result.replace('```json', '').replace('```', ''))
                res_item['result'] = result
            except Exception as e:
                print(e)
                result = 'error'
                new_erro_list.append(erro_input)
            idx += 1
    
    with open(osp.join(work_root, 'processed_data', 'rewrite_query_val.json'), 'w') as f:
        json.dump(result_list, f, indent=4, ensure_ascii=False)

    with open(osp.join(work_root, 'processed_data', 'erro_val.json'), 'w') as f:
        json.dump(new_erro_list, f, indent=4, ensure_ascii=False)
        

if __name__ == '__main__':
    work_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    
    model = LLM('chatglm3-6b')
    with open(osp.join(work_root, 'AQA', 'qa_valid_wo_ans.txt'), 'r') as f:
        query_list = f.readlines()

    erro_handler()
