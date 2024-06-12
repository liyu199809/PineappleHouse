import json
import torch

import os.path as osp
import numpy as np

from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_input_pair(raw_res, question_info, doc_info):
    """
    Build a list of input pairs for reranking

    Parameters:
    raw_res (list): A list of raw results
    question_info (dict): A dictionary containing question information
    doc_info (dict): A dictionary containing document information

    Returns:
    list: A list of input pairs
    """
    all_input_pair = []
    for id, raw_res in enumerate(tqdm(raw_res)):
        question = eval(question_info[id])
        soup = BeautifulSoup(question.get('body', ''), 'html.parser')
        try:
            query = '# Question:'+question.get('question', '') + '\n\nDescription:'+ soup.get_text()
        except:
            query = '# Question:'+'' + '\n\nDescription:'+ soup.get_text()

        input_pair = []
        for pid in raw_res.split(','):
            pid = pid.replace('\n', '')

            title = doc_info[pid]['title']
            abstract = doc_info[pid]['abstract']
            if title is None:
                title = ''

            if abstract is None:
                abstract = ''
            
            input_pair.append(
                [query, '# ' + title + '\n\n' + abstract]
                )
        all_input_pair.append([raw_res, input_pair])
    return all_input_pair


@ torch.no_grad()
def rerank_chunk(model_name_or_path, device_id, all_input_pair, generated_dict, batch_size=32):
    """
    Rerank a chunk of data based on the model and device id

    Args:
        model_name_or_path (str): The name or path of the model to use for reranking
        device_id (int): The id of the device to use for reranking
        all_input_pair (list): A list of input pairs to rerank
        generated_dict (dict): A dictionary to store the reranked results
        batch_size (int): The size of the batch to use for reranking

    Returns:
        None
    """
    if torch.cuda.is_available():
        device = f'cuda:{device_id}'
    else:
        device = 'cpu'

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()

    all_rerank_res = ''
    batches = [all_input_pair[i:i + batch_size] for i in range(0, len(all_input_pair), batch_size)]
    if len(all_input_pair) % batch_size != 0:
        batches[-2:] = [batches[-2] + batches[-1]]

    for batch in tqdm(batches, desc=f"{device_id} running", position=device_id, leave=True):
        inputs = tokenizer.batch_encode_plus(batch[1], padding=True, truncation=True, return_tensors='pt', max_length=1024)
        inputs = inputs.to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float().cpu().numpy()

        # rerank raw res by scores
        batch_rank = np.argsort(scores, axis=-1)
        # get top20 pid
        rerank_res = []
        for rank in batch_rank:
            for r in rank[-20:]:
                rerank_res.append(batch[0].split(',')[r])
        all_rerank_res += ','.join(rerank_res) + '\n'
    generated_dict.update({f'{device_id}': all_rerank_res})


if __name__ == '__main__':
    work_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    model_path = '/mnt/bn/liyullm2/rerank/checkpoint_0'
    retrivier_ans_path = osp.join(work_root, 'result', 'gte_finetune_680_r100.txt')
    question_path = osp.join(work_root, 'AQA', 'qa_test_wo_ans_new.txt')
    doc_path = osp.join(work_root, 'AQA', 'pid_to_title_abs_update_filter.json')

    with open(retrivier_ans_path, 'r') as f:
        raw_res_list = f.readlines()

    with open(question_path, 'r') as f:
        question_info = f.readlines()

    with open(doc_path, 'r') as f:
        doc_info = json.load(f)

    all_input_pair = build_input_pair(raw_res_list, question_info, doc_info)

    num_gpus = torch.cuda.device_count()

    # split raw_res answer
    total_data = len(raw_res_list)
    chunk_size = total_data // num_gpus
    data_chunks = [raw_res_list[i:i + chunk_size] for i in range(0, total_data, chunk_size)]
    if total_data % num_gpus != 0:
        data_chunks[-2:] = [data_chunks[-2] + data_chunks[-1]]

    manager = torch.multiprocessing.Manager()
    generated_dict = manager.dict()
    ctx = torch.multiprocessing.get_context('spawn')
    processes = [ctx.Process(target=rerank_chunk, args=(model_path, i, data_chunks[i], question_info, doc_info, generated_dict)) for i in range(num_gpus)]

    for prc in processes:
        prc.start()
    for prc in processes:
        prc.join()
    rst = dict(generated_dict)
    final_results = [rst[f"{x}"] for x in range(num_gpus)]

    with open(osp.join(work_root, 'result', 'rerank_bset_now.txt'), 'w') as f:
        f.write('\n'.join(final_results))