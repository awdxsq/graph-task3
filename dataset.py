from torch_geometric.datasets import TUDataset, ZINC

def load_dataset(name):
    if name == 'TU':
        dataset = TUDataset(root=f'./data/{name}', name=name)
    elif name == 'ZINC':
        dataset = ZINC(root=f'./data/{name}', name=name)
    return dataset