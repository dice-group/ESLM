import torch

class Config:
    def __init__(self, args):
        self.model_name = args.model
        self.max_length = args.max_length
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.ds_name = ["dbpedia", "lmdb", "faces"]
        self.topk = [5, 10]
        self.k_fold = 5
        self.do_train = args.train
        self.do_test = args.test
        self.enrichment = args.enrichment
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")