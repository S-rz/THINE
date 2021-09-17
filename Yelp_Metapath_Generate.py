# business - star - business
# business - star - user - star - business
# user - business - star - business - user
# user - star - user
# business - user - business

import random
import sys

num_walks_per_node = 10
walk_length = 8

class yelp_metapath:
    def __init__(self, args):
        self.args = args
        self.data_path = './input/Yelp/'
        self.input_file = 'user_business_star_time.txt'

        self.metapath_type = dict()
        self.metapath_type['1'] = ['business_star', 'star_business']
        self.metapath_type['2'] = ['business_star', 'star_user', 'user_star', 'star_business']
        self.metapath_type['3'] = ['user_business', 'business_star', 'star_business', 'business_user']
        self.metapath_type['4'] = ['user_star', 'star_user']
        self.metapath_type['5'] = ['business_user', 'user_business']
        self.node_type = ['business', 'user', 'star']

        self.max_t = -sys.maxsize
        self.min_t = sys.maxsize
        self.te_rate = self.args.train_edges_rate

        self.business_num = 0
        self.user_num = 0
        self.star_num = 0
        self.node_dim = {}
        # print(self.metapath_type)

        self.train_edges = {}  # {node id: {node id: time, ...}, ...}
        self.output_metapath = dict() # {type: [metapath, ...], ...}, metapath = {id: , edge: [], node_type: [],time: []}
        self.output_metapath['1'] = []
        self.output_metapath['2'] = []
        self.output_metapath['3'] = []
        self.output_metapath['4'] = []
        self.output_metapath['5'] = []

    def to_train_edge(self, s_node, s_type, t_node, t_type, time):
        if (s_type + ' ' + s_node) not in self.train_edges.keys():
            self.train_edges[(s_type + ' ' + s_node)] = dict()
        if (t_type + ' ' + t_node) not in self.train_edges[(s_type + ' ' + s_node)].keys():
            self.train_edges[(s_type + ' ' + s_node)][(t_type + ' ' + t_node)] = str(-sys.maxsize)
        if int(self.train_edges[(s_type + ' ' + s_node)][(t_type + ' ' + t_node)]) < int(time):
            self.train_edges[(s_type + ' ' + s_node)][(t_type + ' ' + t_node)] = time

    def train_edge_format(self):
        result_tr_ed = []
        for key1 in self.train_edges.keys():
            for key2 in self.train_edges[key1].keys():
                # print(key1.split())
                temp_edge = []
                temp_edge.extend(list(reversed(key1.split())))
                temp_edge.extend(list(reversed(key2.split())))
                temp_edge.append(self.train_edges[key1][key2])
                result_tr_ed.append(temp_edge)

        random.shuffle(result_tr_ed)
        self.train_edges = result_tr_ed

        random.shuffle(result_tr_ed)
        self.train_edges = result_tr_ed

    def reading_data(self):
        self.business_star_time_dict = {}
        self.star_business_time_dict = {}
        self.business_user_time_dict = {}
        self.user_business_time_dict = {}
        self.user_star_time_dict = {}
        self.star_user_time_dict = {}

        with open(self.data_path + self.input_file, encoding='utf-8') as f:
            data_line = f.readline()
            while (data_line):
                data_line = data_line.split()
                u_id = data_line[0]
                b_id = data_line[1]
                star = data_line[2][0]
                time = self.to_time(data_line[3])
                # print(time)

                if b_id not in self.business_star_time_dict.keys():
                    self.business_star_time_dict[b_id] = []
                if star not in self.star_business_time_dict.keys():
                    self.star_business_time_dict[star] = []
                self.business_star_time_dict[b_id].append([star, time])
                self.star_business_time_dict[star].append([b_id, time])
                self.to_train_edge(b_id, 'business', star, 'star', time)
                self.to_train_edge(star, 'star', b_id, 'business', time)

                if b_id not in self.business_user_time_dict.keys():
                    self.business_user_time_dict[b_id] = []
                if u_id not in self.user_business_time_dict.keys():
                    self.user_business_time_dict[u_id] = []
                self.business_user_time_dict[b_id].append([u_id, time])
                self.user_business_time_dict[u_id].append([b_id, time])
                self.to_train_edge(b_id, 'business', u_id, 'user', time)
                self.to_train_edge(u_id, 'user', b_id, 'business', time)

                if u_id not in self.user_star_time_dict.keys():
                    self.user_star_time_dict[u_id] = []
                if star not in self.star_user_time_dict.keys():
                    self.star_user_time_dict[star] = []
                self.user_star_time_dict[u_id].append([star, time])
                self.star_user_time_dict[star].append([u_id, time])
                self.to_train_edge(u_id, 'user', star, 'star', time)
                self.to_train_edge(star, 'star', u_id, 'user', time)

                data_line = f.readline()

        self.node_dim['business'] = len(self.business_star_time_dict.keys())
        self.node_dim['star'] = len(self.star_business_time_dict.keys())
        self.node_dim['user'] = 24586

    def to_time(self, str_time):
        str_time = str_time.split('-')
        time = (int(str_time[0]))
        # time = time + (int(str_time[1]) - 1) / 100
        # time = time + int(str_time[2]) / 10000
        return str(time)

    def random_walk(self, node_start, edge_type):
        object_start = 'self.'
        object_end = '_time_dict'
        metapath_temp = dict()
        metapath_temp['edge'] = []
        metapath_temp['node_type'] = []
        metapath_temp['time'] = []
        metapath_temp['edge'].append(node_start)
        metapath_temp['node_type'].append((edge_type[0].split('_'))[0])
        if metapath_temp['node_type'][0] == 'coauthor':
            metapath_temp['node_type'][0] = 'author'
        edge_type = edge_type * int(walk_length)
        for i in range(len(edge_type)):
            random_temp = random.choice(eval(object_start + edge_type[i] + object_end)[metapath_temp['edge'][-1]])
            # print(metapath_temp['edge'][-1], random_temp, '\n')
            if i != len(edge_type) - 1:
                num_break = 0
                while (random_temp[0] not in (eval(object_start + edge_type[i + 1] + object_end)).keys()):
                    random_temp = random.choice(eval(object_start + edge_type[i] + object_end)[metapath_temp['edge'][-1]])
                    num_break = num_break + 1
                    if num_break >= 10:
                        random_temp = 'NONE'
                        break

            if random_temp == 'NONE':
                metapath_temp['edge'][0] = 'NONE'
                break
            node_type = edge_type[i].split('_')
            metapath_temp['edge'].append(random_temp[0])
            if node_type[1] != 'coauthor':
                metapath_temp['node_type'].append(node_type[1])
            else:
                metapath_temp['node_type'].append('author')
            metapath_temp['time'].append(random_temp[1])

        return metapath_temp

    def metapath_generate(self):
        object_start = 'self.'
        object_end = '_time_dict'
        for metapath_type_key in self.metapath_type.keys():
            metapath_id = 1
            first_edge_type = self.metapath_type[metapath_type_key][0]
            object_name = object_start + first_edge_type + object_end
            for temp_key in eval(object_name).keys():
                for num in range(num_walks_per_node):
                    metapath_temp = self.random_walk(temp_key, self.metapath_type[metapath_type_key])
                    if metapath_temp['edge'][0] == 'NONE':
                        break
                    metapath_temp['id'] = str(metapath_id)
                    metapath_id = metapath_id + 1
                    self.output_metapath[metapath_type_key].append(metapath_temp)

    def get_edge_type(self):
        edge_type = []
        for mp_type_temp in self.metapath_type.keys():
            for edge_type_temp in self.metapath_type[mp_type_temp]:
                temp = edge_type_temp.replace('coauthor', 'author').split('_')
                temp.sort()
                if temp not in edge_type:
                    edge_type.append(temp)

        return edge_type  # sorted edge []

    def to_args(self):
        self.args.dataset_name = 'Yelp'
        self.args.node_type = self.node_type
        self.args.node_dim = self.node_dim
        self.args.metapath_type = len(self.metapath_type)
        self.args.edge_type = self.get_edge_type()

    def data_generate(self):
        # print('data generate\n')
        self.reading_data()
        print(self.node_dim)
        self.metapath_generate()
        self.train_edge_format()
        print(len(self.train_edges))
        self.to_args()

