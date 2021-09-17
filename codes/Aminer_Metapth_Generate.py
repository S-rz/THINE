# author - paper - author
# author - paper - paper - author
# author - paper - conference - paper - author
# conference - paper - author - paper - conference

import random
import sys

num_walks_per_node = 10
walk_length = 8

class aminer_metapath:
    def __init__(self, args):
        self.args = args
        self.data_path = './input/Aminer/'
        self.coauthor_time_path = 'Coauthor_Time_temp.txt'
        self.paper_information_path = 'Paper_Information_temp.txt'
        self.conference_id_path = 'Venue_Id_temp.txt'
        self.author_paper_path = 'Author_Paper_Time_temp.txt'

        self.metapath_type = dict()
        self.metapath_type['1'] = ['author_paper', 'paper_author']
        self.metapath_type['2'] = ['author_paper', 'paper_paper', 'paper_author']
        self.metapath_type['3'] = ['author_paper', 'paper_conference', 'conference_paper', 'paper_author']
        self.metapath_type['4'] = ['conference_paper', 'paper_author', 'author_paper', 'paper_conference']
        self.node_type = ['author', 'paper', 'conference']

        self.paper_num = 0
        self.author_num = 0
        self.conference_num = 0
        self.node_dim = {}

        self.train_edges = {}  # {node id: {node id: time, ...}, ...}
        self.output_metapath = dict() # {type: [metapath, ...], ...}, metapath = {id: , edge: [], node_type: [],time: []}
        self.output_metapath['1'] = []
        self.output_metapath['2'] = []
        self.output_metapath['3'] = []
        self.output_metapath['4'] = []

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
                temp_edge = []
                temp_edge.extend(list(reversed(key1.split())))
                temp_edge.extend(list(reversed(key2.split())))
                temp_edge.append(self.train_edges[key1][key2])
                result_tr_ed.append(temp_edge)

        random.shuffle(result_tr_ed)
        self.train_edges = result_tr_ed

    def reading_data(self):
        self.conference_id_dict = {}   # dict : {conforence : id , ... }
        with open(self.data_path + self.conference_id_path, encoding='utf-8') as f:
            data_line = f.readline()
            while(data_line):
                self.conference_num = self.conference_num + 1
                data_line = data_line.split()
                self.conference_id_dict[data_line[0]] = data_line[1]
                data_line = f.readline()
            self.node_dim['conference'] = 2584

        self.paper_author_time_dict = {}  # { paper id : [author id, time], ...}
        self.author_paper_time_dict = {}  # { author id : [paper id, time], ...}
        with open(self.data_path + self.author_paper_path, encoding='utf-8') as f:
            data_line = f.readline()
            while (data_line):
                data_line = data_line.split()
                if len(data_line) >= 3:
                    paper_id = data_line[1]
                    author_id = data_line[0]
                    time = data_line[2]

                    paper_author_time_temp = []
                    author_paper_time_temp = []
                    paper_author_time_temp.extend([author_id, time])
                    author_paper_time_temp.extend([paper_id, time])
                    if paper_id not in self.paper_author_time_dict.keys():
                        self.paper_author_time_dict[paper_id] = []
                    if author_id not in self.author_paper_time_dict.keys():
                        self.author_paper_time_dict[author_id] = []
                    self.paper_author_time_dict[paper_id].append(paper_author_time_temp)
                    self.author_paper_time_dict[author_id].append(author_paper_time_temp)
                    self.to_train_edge(paper_id, 'paper', author_id, 'author', time)
                    self.to_train_edge(author_id, 'author', paper_id, 'paper', time)

                data_line = f.readline()
            self.node_dim['author'] = 10206

        self.paper_conference_time_dict = {}  # [ [paper id, conference id, time], ...]
        self.conference_paper_time_dict = {}  # [ [conference id, paper id, time], ...]
        self.paper_paper_time_dict = {}   # [ [paper id, paper id, time], ...]
        with open(self.data_path + self.paper_information_path, encoding='utf-8') as f:
            data_line = f.readline()
            while(data_line):
                self.paper_num = self.paper_num + 1
                data_line = data_line.split()
                paper_id = data_line[0]
                if len(data_line) >= 4:
                    time = data_line[1]
                    conference = data_line[2]
                    reference = data_line[3].split(',')
                elif len(data_line) == 3:
                    time = data_line[1]
                    conference = 'NONE'
                    reference = data_line[2].split(',')
                else:
                    time = 'NONE'
                    conference = 'NONE'
                    reference = data_line[1].split(',')

                paper_conference_temp = []
                conference_paper_temp = []
                if conference != 'NONE' and time != 'NONE':
                    conference = self.conference_id_dict[conference]
                    if paper_id not in self.paper_conference_time_dict.keys():
                        self.paper_conference_time_dict[paper_id] = []
                    if conference not in self.conference_paper_time_dict.keys():
                        self.conference_paper_time_dict[conference] = []
                    paper_conference_temp.extend([conference, time])
                    conference_paper_temp.extend([paper_id, time])
                    self.paper_conference_time_dict[paper_id].append(paper_conference_temp)
                    self.conference_paper_time_dict[conference].append(conference_paper_temp)
                    self.to_train_edge(paper_id, 'paper', conference, 'conference', time)
                    self.to_train_edge(conference, 'conference', paper_id, 'paper', time)

                if reference[0] != 'NONE' and time != 'NONE':
                    for i in reference:
                        paper_reference_temp = []
                        if paper_id not in self.paper_paper_time_dict.keys():
                            self.paper_paper_time_dict[paper_id] = []
                        paper_reference_temp.extend([i, time])
                        self.paper_paper_time_dict[paper_id].append(paper_reference_temp)
                        self.to_train_edge(paper_id, 'paper', i, 'paper', time)

                data_line = f.readline()

            # self.node_dim['paper'] = self.paper_num
            self.node_dim['paper'] = 10457

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
        self.metapath_type['1'] = ['author_paper', 'paper_author']
        self.args.dataset_name = 'Aminer'
        self.args.node_type = self.node_type
        self.args.node_dim = self.node_dim
        self.args.metapath_type = len(self.metapath_type)
        self.args.edge_type = self.get_edge_type()

    def data_generate(self):
        self.reading_data()
        print(self.node_dim)
        self.metapath_generate()
        self.train_edge_format()
        print(len(self.train_edges))
        self.to_args()
