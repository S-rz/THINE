import torch
from Parser_And_Show import parameter_parser
from Aminer_Metapth_Generate import aminer_metapath
from Yelp_Metapath_Generate import yelp_metapath
from THINE import metapath_mtne_Trainer
from Model_Dataset import mtne_metapath_dataset

if __name__ == '__main__':
    args = parameter_parser()
    data_temp = aminer_metapath(args)
    # data_temp = yelp_metapath(args)
    data_temp.data_generate()
    data = mtne_metapath_dataset(data_temp.args, data_temp.output_metapath, data_temp.train_edges)
    model = metapath_mtne_Trainer(data_temp.args, data_temp.output_metapath, data_temp.train_edges)
    model.fit()

