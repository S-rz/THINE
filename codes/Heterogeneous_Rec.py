from __future__ import division
import random
import warnings
import functools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

warnings.filterwarnings('ignore')
random.seed(203)

def my_cmp(x, y):
    if x[1] > y[1]:
        return -1
    if x[1] < y[1]:
        return 1
    return 0

class H_Evaluation:
    def __init__(self, s_embeddings, t_embeddings):
        self.s_embeddings = s_embeddings
        self.t_embeddings = t_embeddings

    def get_rec_node(self, s_node, top_num):
        s_emb = np.array(self.s_embeddings[int(s_node) - 1])
        out_nodes = []
        for i in range(len(self.t_embeddings)):
            t_emb = np.array(self.t_embeddings[i])
            similarity = -((s_emb - t_emb) ** 2).sum()
            if len(out_nodes) < top_num:
                out_nodes.append([i+1, similarity])
            else:
                if similarity > out_nodes[-1][1]:
                    del out_nodes[-1]
                    out_nodes.append([i + 1, similarity])
            out_nodes = sorted(out_nodes, key=functools.cmp_to_key(my_cmp))
        # print(out_nodes)
        return [str(x[0]) for x in out_nodes]

    def get_precision_recall(self, t_pre, t_node):
        tp_num = set(t_pre)&set(t_node)
        pre = float(len(tp_num) / len(t_pre))
        rec = float(len(tp_num) / len(t_node))
        return pre,rec

    def link_recommendation(self, top_num):
        link_rec_file = './input/Yelp/u_b_rec.txt'
        precision_list = []
        recall_list = []
        with open(link_rec_file, 'r', encoding='utf-8') as f:
            data_line = f.readline()
            while(data_line):
                data_line = data_line.split()
                s_node = data_line[0]
                t_node = data_line[1].split(',')
                t_pre = self.get_rec_node(s_node, top_num)
                pre_temp,rec_temp = self.get_precision_recall(t_pre, t_node)
                precision_list.append(pre_temp)
                recall_list.append(rec_temp)
                data_line = f.readline()
        precision_np = np.array(precision_list)
        recall_np = np.array(recall_list)
        print('Top@',top_num)
        print('precision : ', precision_np.mean())
        print('recall : ', recall_np.mean())

def H_read_embedding(input_file):
    embeddings = []
    with open(input_file, encoding='utf-8') as f:
        data_line = f.readline()
        while (data_line):
            data_line = data_line.split()
            if len(data_line) > 2:
                embeddings.append(np.array(data_line, dtype=float))
            else:
                author_num = int(data_line[0])
            data_line = f.readline()
    embeddings = np.array(embeddings, dtype=float)
    return embeddings

if __name__ =='__main__':
    s_emb = H_read_embedding('./output/author.emb')
    t_emb = H_read_embedding('./output/conference.emb')
    evalu_temp = H_Evaluation(s_emb, t_emb)
    evalu_temp.link_recommendation(2)
    evalu_temp.link_recommendation(4)