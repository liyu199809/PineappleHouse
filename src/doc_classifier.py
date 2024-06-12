import json
import time
import asyncio
import os.path as osp

from model import VEMaaSModel
from tqdm import tqdm

def classify_instruct(paper_source):
    instruction = f"""
    You are an academic document manager, please help me determine the category to which the journal belongs based on the journal category.
    
    The categories are only the following: Computer Science, Chemistry, Mathematics, Biology, Physics, Humanities, Medicine, Chemistry, Environmental Engineering, Materials Science, Microelectronics
    
    Note: You only need to give the classification results, not any analysis!

    Journal: {paper_source}
    
    Category:
    """
    return instruction


if __name__ == '__main__':
    work_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    
    model = VEMaaSModel('skylark2-pro-32k')
    with open(osp.join(work_root, 'AQA', 'venue.txt'), 'r') as f:
        venue = f.read()

    venue = list(eval(venue))
    result_list = []
    for paper_source in tqdm(venue):
        if paper_source != '':
            instruction = classify_instruct(paper_source)
            try:
                result = asyncio.run(model.model_prediction(instruction, [], 'You are a helpful AI assistant.'))
            except Exception as e:
                print(e)
                result = 'error'
            result_list.append({'paper_source': paper_source, 'result': result})
            time.sleep(1)
    
    with open(osp.join(work_root, 'processed_data', 'venue_class.json'), 'w') as f:
        json.dump(result_list, f, indent=4, ensure_ascii=False)