# -*-coding:utf-8-*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
from classify import Classifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

def label():
    X = []
    Y = []
    input_path = './input/Aminer/'
    label_file = 'author_label.txt'
    with open(input_path + label_file, encoding='utf-8') as f:
        data_line = f.readline()
        while(data_line):
            data_line = data_line.split()
            X.append(str(int(data_line[0]) - 1))
            Y.append(data_line[1])
            data_line = f.readline()

    return X, Y

def evaluate_embeddings(embeddings):
    X, Y = label()
    tr_frac = 0.6
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embedding=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)

def get_embedding():
    input_file = './output/'
    b_file = 'author.emb'

    all_node = 0
    embeddings = {}
    with open(input_file + b_file, encoding='utf-8') as f:
        data_line = f.readline()
        while(data_line):
            data_line = data_line.split()
            if len(data_line) > 2:
                embeddings[str(all_node)] = np.array(data_line, dtype=float)
                all_node = all_node + 1
            data_line = f.readline()

    return embeddings

def do_node_classification():
    warnings.filterwarnings('ignore')

    embeddings = get_embedding()
    # plot_embeddings(embeddings, p_num, a_num, c_num)
    # evaluate_embeddings(embeddings)
    classification(embeddings, 0.6)
    classification(embeddings, 0.8)

def classification(embeddings, train_size):
    x = []
    y = []
    with open('./input/Aminer/author_label.txt', mode='r', encoding="UTF-8") as fpl:
        for line in fpl:
            if line != "\n":
                s = line.split(' ')
                x.append(embeddings[str(int(s[0]) - 1)])
                y.append(int(s[1]))

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=1 - train_size, random_state=9)
    # print ('train_size: {}'.format(train_sicv ze))
    lr = LogisticRegression()
    print(len(x))
    lr.fit(np.asarray(x_train).astype('float64'), y_train)
    y_valid_pred = lr.predict(np.asarray(x_valid).astype('float64'))

    micro_f1 = f1_score(y_valid, y_valid_pred, average='micro')
    macro_f1 = f1_score(y_valid, y_valid_pred, average='macro')
    print('Macro_F1_score:{}'.format(macro_f1))
    print('Micro_F1_score:{}'.format(micro_f1))
    # with open("data/result.txt", mode="a+", encoding="UTF-8") as f:
    #     f.write('Macro_F1_score:{}, Micro_F1_score:{}\n'.format(macro_f1, micro_f1))
    return macro_f1, micro_f1

if __name__ == '__main__':
    do_node_classification()