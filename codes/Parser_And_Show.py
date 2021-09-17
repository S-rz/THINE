import argparse
# from texttable import Texttable

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run Model.")

    parser.add_argument('--output_file',
                        default='./output',
                        type=str,
                        help='output_file')
    parser.add_argument('--dim',
                        default=128,
                        type=int,
                        help="embedding dimensions")
    parser.add_argument('--batch_size',
                        default=500,
                        type=int,
                        help="batch size")
    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help="epochs num")
    parser.add_argument('--initial_lr',
                        default=0.003,
                        type=float,
                        help="learning rate")

    parser.add_argument('--negative_sample',
                        default=5,
                        type=int,
                        help="number of negative sample")

    parser.add_argument('--negative_sample_method',
                        default=False,
                        type=bool,
                        help="True : metapath, False : metapath++")

    parser.add_argument('--node_similarity_method',
                        default=False,
                        type=bool,
                        help="True : matrix similarity, False : Euclidean distance")

    parser.add_argument('--metapath_influence_method',
                        default=True,
                        type=bool,
                        help="True : add, False : multiply")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=0.01,
                        help="weight-decay. Default is 0.01.")

    parser.add_argument("--decay-epoch",
                        type=int,
                        default=1,
                        help="decay-epoch. Default is 1.")

    parser.add_argument('--closest_metapath',
                        default=20,
                        type=int,
                        help="number of metapath to calculate the probability of connection")

    parser.add_argument('--closest_edge',
                        default=4,
                        type=int,
                        help="number of edge to calculate the strength of metapath")

    parser.add_argument('--DID',
                        default=1,
                        type=int,
                        help="cuda num")

    parser.add_argument('--train_edges_rate',
                        default=1 / 3,
                        type=float,
                        help="train edges rate")

    return parser.parse_args()
