import json

data = dict()
data['REDDIT-BINARY'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'REDDIT-BINARY',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': True,
    'use_node_label': False
}
data['REDDIT-MULTI-5K'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'REDDIT-MULTI-5K',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': True,
    'use_node_label': False
}
data['IMDB-BINARY'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'validation_ratio': 0.1,
    'dataset': 'IMDB-BINARY',
    'use_node_degree': True,
    'set_node_degree_uninformative': False,
    'use_node_label': False
}
data['IMDB-MULTI'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'IMDB-MULTI',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': False,
    'use_node_label': False
}
data['PROTEINS'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'PROTEINS',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': False,
    'use_node_label': False

}
data['NCI1'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'NCI1',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': False,
    'use_node_label': False
}
data['PROTEINS_2'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'PROTEINS',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': False,
    'use_node_label': True

}
data['NCI1_2'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'NCI1',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': False,
    'use_node_label': True
}
data['DD'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'DD',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': True,
    'use_node_label': False
}
data['ENZYMES'] = {
    'lr': 1e-2,
    'lmda': 1e-4,
    'num_epochs': 100,
    'batch_size': 32,
    'use_pers': 0,
    'gin_num': 1,
    'gin_dim': 64,
    'num_lin_layers': 2,
    'dataset': 'ENZYMES',
    'validation_ratio': 0.1,
    'use_node_degree': True,
    'set_node_degree_uninformative': True,
    'use_node_label': False
}
with open('training_cfgs.txt', 'w') as outfile:
    json.dump(data, outfile)
