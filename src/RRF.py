import dill
import sys
import os.path as osp
from tqdm import tqdm

path_root = osp.dirname(osp.dirname(osp.abspath(__file__)))

def simple_rank_score(rank):
    return 20 - rank

# we set the default k in RRF to 20
def RRF_score(rank, k=20):
    return 1.0/(k + rank)

def main(result_list, weight, save_path='result_test/merge/gte_gritlm_sfr_linqprompt_nvemb_merge5_k@20_final.txt'):
    with open(osp.join(path_root, save_path),'w') as f:
        length = len(result_list[0])
        for i in tqdm(range(length)):
            mydict = {}
            for item, w in zip(result_list, weight):
                rank = 1
                for single_item in item[i]:
                    score = RRF_score(rank, 20)*w
                    mydict[single_item] = mydict.get(single_item, 0) + score
                    rank += 1
            res_list = []
            for key in mydict.keys():
                res_list.append([key, mydict[key]])
            res_list.sort(key=lambda x:-float(x[1]))
            
            f_res_list = []
            for i in range(20):
                f_res_list.append(res_list[i][0])
            
            f.write(','.join(f_res_list))
            f.write('\r\n')

if __name__ == "__main__":
    result_list = []
    
    # the result files to merge
    file_list = ['result_test/gritlm_1536_r100.txt', 'result_test/gte_finetune_680_r100.txt', 'result_test/sfr_4_test.txt', 'result_test/linq_0.txt', 'result_test/nvemb.txt']
    
    # the weight you hope to use for each result file
    weight = [1, 1, 1, 1.25, 1]
    
    for file_name in file_list:
        with open(osp.join(path_root, file_name), 'r') as f:
            lines = f.readlines()
        result_list.append([line.strip().split(',') for line in lines])
    
    main(result_list, weight)