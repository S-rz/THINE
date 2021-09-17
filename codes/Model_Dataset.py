from torch.utils.data import Dataset
import functools
import numpy as np
import sys
import random

class mtne_metapath_dataset(Dataset):
    def __init__(self, args, metapath_data, train_edge):
        self.args = args
        self.metapath_data = metapath_data
        self.train_edge = train_edge     # [[node, type, node, type, time], ...]
        random.shuffle(self.train_edge)
        self.data_size = len(self.train_edge)
        self.closest_metapath = self.args.closest_metapath
        self.neg_sample = self.args.negative_sample
        self.neg_method = self.args.negative_sample_method
        self.node_type = self.args.node_type
        self.node_dim = self.args.node_dim # {node type: , ...}
        self.metapath_type = self.args.metapath_type

        self.max_time = -sys.maxsize

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)
        self.metapath_to_node_dict()

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        s_node = self.train_edge[item][0]
        s_type = self.train_edge[item][1]
        t_node = self.train_edge[item][2]
        t_type = self.train_edge[item][3]
        s_t_time = self.train_edge[item][4]

        metapath_s = self.choice_metapath(s_node, s_type, t_node, t_type, s_t_time, self.closest_metapath)
        negative_s_node = self.fun_negative_sample([s_node, s_type], [t_node, t_type])
        negative_s_metapath = {}
        for negative_node_temp in negative_s_node:
            negative_s_metapath[negative_node_temp[1] + negative_node_temp[0]] = self.choice_metapath(s_node, s_type, negative_node_temp[0],
                                                                              negative_node_temp[1], s_t_time, self.closest_metapath)

        # source_node:         [id, type]
        # target_node:         [id, type]
        # metapath_s:          [{type:, edge:, node type:, time: }, ...]
        # metapath_t:          [{type:, edge:, node type:, time: }, ...]
        # negative_s_node:     [[id, type], ...]
        # negative_s_metapath: { type + id: [{type:, edge:, node type:, time: }, ...], ...}
        # negative_t_node:     [[id, type], ...]
        # negative_t_metapath: { id: [{type:, edge:, node type:, time: }, ...], ...}
        sample = {
            'source_node':[s_node, s_type],
            'target_node':[t_node, t_type],
            'train_time':s_t_time,
            'metapath_s':metapath_s,
            'negative_s_node':negative_s_node,
            'negative_s_metapath':negative_s_metapath
        }

        # print(sample)
        return sample

    def choice_metapath(self, s_node, s_node_type, t_node, t_node_type, time, closest_metapath):
        dict_key = s_node_type + s_node
        node_metapath = []
        output_metapath = []
        for id_metapath_temp in self.node_metapath_data[dict_key]:
            # print(id_metapath_temp)
            if int(id_metapath_temp[2]) <= int(time) and id_metapath_temp[4] == t_node and id_metapath_temp[5] == t_node_type:
                node_metapath.append(id_metapath_temp)
            if len(node_metapath) > closest_metapath:
                node_metapath.sort(key=functools.cmp_to_key(self.cmp))
                node_metapath.pop(-1)

        for node_metapath_temp in node_metapath:
            type_temp = node_metapath_temp[0]
            id_temp = node_metapath_temp[1]
            for metapath_temp in self.metapath_data[type_temp]:
                if metapath_temp['id'] == id_temp:
                    metapath_temp['type'] = type_temp
                    output_metapath.append(metapath_temp)
                    break

        return output_metapath  # [{type:, edge:, node type:, time:, id:}, ...]

    def cmp(self, x, y):
        # index 2 : max_time, index 3 : avg_time
        if x[2] > y[2]:
            return -1
        elif x[2] == y[2] and x[3] > y[3]:
            return -1
        elif x[2] < y[2]:
            return 1
        elif x[2] == y[2] and x[3] < y[3]:
            return 1
        else:
            return 0

    def metapath_to_node_dict(self):
        # {node_type + node: [[metapath type, metapath id, max time, avg time, next node, next node type], ...], ...}
        self.node_metapath_data = {}
        for metapath_type_temp in self.metapath_data.keys():  # {type: [metapath, ...], ...}, metapath = {id: , edge: [], node_type: [],time: []}
            for metapath_temp in self.metapath_data[metapath_type_temp]:
                max_time, avg_time = self.max_average_time(metapath_temp['time'])
                for index in range(len(metapath_temp['edge']) - 1):
                    dict_key = metapath_temp['node_type'][index] + metapath_temp['edge'][index]
                    if dict_key not in self.node_metapath_data.keys():
                        self.node_metapath_data[dict_key] = []
                    self.node_metapath_data[dict_key].append([metapath_type_temp, metapath_temp['id'], max_time,
                                                              avg_time, metapath_temp['edge'][index + 1], metapath_temp['node_type'][index + 1]])

    def max_average_time(self, time_list):
        max_time = -sys.maxsize
        total_time = 0
        for time_temp in time_list:
            total_time = total_time + int(time_temp)
            max_time = max(max_time, int(time_temp))

        return str(max_time), str(int(total_time / len(time_list)))

    def fun_negative_sample(self, s_node, t_node):
        negative_node = []
        if self.neg_method:
            # metapath
            for i in range(self.neg_sample):
                node_type_index = np.random.randint(0, len(self.node_type) - 1, 1)
                node_type_index = node_type_index[0]
                node_id = np.random.randint(1, self.node_dim[np.array(self.node_type)[node_type_index]], 1)
                while (
                        (str(node_id[0]) == s_node[0] and np.array(self.node_type)[node_type_index] == s_node[1])
                        or(str(node_id[0]) == t_node[0] and np.array(self.node_type)[node_type_index] == t_node[1])
                ):
                    node_type_index = np.random.randint(0, len(self.node_type) - 1, 1)
                    node_type_index = node_type_index[0]
                    node_id = np.random.randint(1, self.node_dim[np.array(self.node_type)[node_type_index]], 1)
                negative_node.append([str(node_id[0]), np.array(self.node_type)[node_type_index]])
        # error error error error
        else:
            # metapath++
            node_type = t_node[1]
            node_id = np.random.randint(1, self.node_dim[node_type], self.neg_sample)
            for node_id_temp in node_id:
                negative_node.append([str(node_id_temp) ,node_type])

        return negative_node



