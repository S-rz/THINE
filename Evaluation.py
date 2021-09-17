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

class Evaluation:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.map_data_type = {"Pos": 1, "Neg": 0}

    def calculate_sim(self, u, v, sum_flag):
        if sum_flag:
            return sum(np.abs(np.array(u) - np.array(v)))
        else:
            return - ((np.array(u) - np.array(v)) ** 2)
            # return np.abs(np.array(u) - np.array(v))

    def binary_classification_aa(self, x_train, y_train, x_test, y_test):
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, classifier.predict(x_test))
        acc = accuracy_score(y_test, classifier.predict(x_test))
        with open("./output/result_link_pre.txt", mode='a+', encoding='UTF-8') as fw:
            fw.write('auc: {}'.format(auc_score) + "\n")
            fw.write('f1: {}'.format(f1) + "\n")
            fw.write('acc: {}'.format(acc) + "\n\n")

    def pre_4_link_prediction(self, data_type):
        x = []
        y = []
        with open("./input/Yelp/couser_" + data_type + ".txt", 'r') as f:
            for line in f:
                tokens = line.strip("\n").split(' ')
                if int(tokens[0]) <= len(self.embeddings) and int(tokens[1]) <= len(self.embeddings):
                    pos_1_emb = self.embeddings[int(tokens[0]) - 1]
                    pos_2_emb = self.embeddings[int(tokens[1]) - 1]
                    sim_pos = self.calculate_sim(pos_1_emb, pos_2_emb, sum_flag=False)
                    x.append(sim_pos)
                    y.append(self.map_data_type[data_type])
        return x, y

    def link_prediction_with_auc(self):
        # pos_x, pos_y = self.pre_4_link_prediction("Pos")
        # neg_x, neg_y = self.pre_4_link_prediction("Neg")
        # neg_x_sample = random.sample(neg_x, len(pos_x))
        # neg_y_sample = random.sample(neg_y, len(pos_y))
        #
        # train_x = pos_x[:int(0.2 * len(pos_x))] + neg_x_sample[:int(0.2 * len(neg_x_sample))]
        # train_y = pos_y[:int(0.2 * len(pos_y))] + neg_y_sample[:int(0.2 * len(neg_y_sample))]
        # test_x = pos_x[int(0.2 * len(pos_x)):] + neg_x_sample[int(0.2 * len(neg_x_sample)):]
        # test_y = pos_y[int(0.2 * len(pos_y)):] + neg_y_sample[int(0.2 * len(neg_y_sample)):]
        pos_x, pos_y = self.pre_4_link_prediction("Pos")
        neg_x, neg_y = self.pre_4_link_prediction("Neg")

        train_x = pos_x[:int(0.2 * len(pos_x))] + neg_x[:int(0.2 * len(neg_x))]
        train_y = pos_y[:int(0.2 * len(pos_y))] + neg_y[:int(0.2 * len(neg_y))]
        test_x = pos_x[int(0.2 * len(pos_x)):] + neg_x[int(0.2 * len(neg_x)):]
        test_y = pos_y[int(0.2 * len(pos_y)):] + neg_y[int(0.2 * len(neg_y)):]

        train_data = list(zip(train_x, train_y))
        random.shuffle(train_data)
        train_x[:], train_y[:] = zip(*train_data)

        test_data = list(zip(test_x, test_y))
        random.shuffle(test_data)
        test_x[:], test_y[:] = zip(*test_data)
        # print('link prediction with auc...')
        # print(len(train_x), len(test_x))
        self.binary_classification_aa(train_x, train_y, test_x, test_y)

    def get_rec_node(self, s_node, top_num):
        s_emb = np.array(self.embeddings[int(s_node) - 1])
        out_nodes = []
        for i in range(len(self.embeddings)):
            if i != int(s_node) - 1:
                t_emb = np.array(self.embeddings[i])
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
        link_rec_file = './input/user_rec.txt'
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

def read_embedding():
    input_file = './output/'
    author_file = 'user.emb'
    embeddings = []

    with open(input_file + author_file, encoding='utf-8') as f:
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
    emb = read_embedding()
    evalu_temp = Evaluation(emb)
    # evalu_temp.link_recommendation(10)
    # evalu_temp.link_recommendation(15)
    evalu_temp.link_prediction_with_auc()
