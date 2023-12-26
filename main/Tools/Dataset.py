import torch_geometric.datasets as td

class Dataset:
    def __init__(self):
        self.dataset = None
        self.data_name = None


    def load_data(self, name):
        self.data_name = name
        if name=='Cora':
            return td.Planetoid(root='/tmp/Cora', name='Cora')[0]

        elif name=='Corafull':
            return td.CoraFull(root='/tmp/Corafull')[0]

        elif name=='Actor':
            return td.CoraFull(root='/tmp/Actor')[0]

        else:
            raise 'There is no ' + name + ' dataset in graph data.'

