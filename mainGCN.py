from Models.GCN import GCN
from Tools.Dataset import Dataset
from Tools.Training import Training
import argparse

def get_argparse():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--dataset',default='Cora', type=str)
    parser.add_argument('--hidden_layer',default=20, type=int)
    parser.add_argument('--epochs',default=100, type=int)
    parser.add_argument('--lr',default=0.01, type=float)

    return parser
def main(args=None):
    dataset = Dataset().load_data(args.dataset)
    model = GCN(input_layer=dataset.num_node_features, hidden_layer=args.hidden_layer, output_layer=max(dataset['y'])+1)
    training = Training(model,dataset)
    training.training(lr=args.lr,epochs=args.epochs)

if __name__ == '__main__':
    args = get_argparse().parse_args()
    main(args)


