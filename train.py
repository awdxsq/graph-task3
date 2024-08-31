import time
from dataset import load_dataset
from models import GCN, GAT, GraphSAGE, GIN
from torch_geometric.data import DataLoader


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    return acc


def main():
    datasets = ['TU', 'ZINC']
    poolings = ['AvgPooling', 'MaxPooling', 'MinPooling']
    models = {'GCN': GCN, 'GAT': GAT, 'GraphSAGE': GraphSAGE, 'GIN': GIN}

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        data = dataset[0]
        for name, Model in models.items():
            for pooling_name in poolings:
                model = Model(dataset.num_features, dataset.num_classes).to('cuda')
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

                # Training
                start_time = time.time()
                for epoch in range(200):
                    loss = train(model, data, optimizer)
                training_time = time.time() - start_time

                # Testing
                acc = test(model, data)
                print(f'{name} with {pooling_name} on {dataset_name}: Accuracy: {acc}, Training Time: {training_time}')


if __name__ == "__main__":
    main()