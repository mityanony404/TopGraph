import numpy as np
import torch
import os
from data_graph import *
from utils_graph import *
from model_graph import *
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import matplotlib.pyplot as plt
import json


def model_factory(model_name):
    if model_name == 'TopGraph-PI':
        model = GIN(dataset,
                    use_node_degree=training_cfg['use_node_degree'],
                    set_node_degree_uninformative=training_cfg['set_node_degree_uninformative'],
                    use_node_label=training_cfg['use_node_label'],
                    gin_number=training_cfg['gin_num'],
                    gin_dimension=training_cfg['gin_dim'],
                    gin_mlp_type='lin_bn_lrelu_lin',
                    cls_hidden_dimension=64,
                    drop_out=0.5,
                    pooling_strategy='sum')
        persloss = PersLoss(dataset,
                            input_dim=training_cfg['gin_dim'],
                            cls_type='PI',
                            dropout=0.5)
    elif model_name == 'TopGraph-RH':
        model = GIN(dataset,
                    use_node_degree=training_cfg['use_node_degree'],
                    set_node_degree_uninformative=training_cfg['set_node_degree_uninformative'],
                    use_node_label=training_cfg['use_node_label'],
                    gin_number=training_cfg['gin_num'],
                    gin_dimension=training_cfg['gin_dim'],
                    gin_mlp_type='lin_bn_lrelu_lin',
                    cls_hidden_dimension=64,
                    drop_out=0.5,
                    pooling_strategy='sum')
        persloss = PersLoss(dataset,
                            input_dim=training_cfg['gin_dim'],
                            cls_type='Rational-hat',
                            dropout=0.5)
    elif model_name == 'GIN':
        model = GIN(dataset,
                    use_node_degree=training_cfg['use_node_degree'],
                    set_node_degree_uninformative=training_cfg['set_node_degree_uninformative'],
                    use_node_label=training_cfg['use_node_label'],
                    gin_number=training_cfg['gin_num'],
                    gin_dimension=training_cfg['gin_dim'],
                    gin_mlp_type='lin_bn_lrelu_lin',
                    classifier_type='Rational-hat',
                    cls_hidden_dimension=64,
                    drop_out=0.5,
                    pooling_strategy='sum')
        persloss = None
    elif model_name == 'GFL':
        model = PershomLearnedFilt(dataset,
                                   use_node_degree=training_cfg['use_node_degree'],
                                   set_node_degree_uninformative=training_cfg['set_node_degree_uninformative'],
                                   use_node_label=training_cfg['use_node_label'],
                                   gin_number=training_cfg['gin_num'],
                                   gin_dimension=training_cfg['gin_dim'],
                                   gin_mlp_type='lin_bn_lrelu_lin',
                                   classifier_type='Rational-hat',
                                   cls_hidden_dimension=64,
                                   drop_out=0.5)
        persloss = None
    else:
        raise KeyError('Model not found for testing')
    return model, persloss


def read_training_cfgs(dataset_name):
    with open('training_cfgs.txt') as json_file:
        data = json.load(json_file)
        training_cfgs = data[dataset_name]
    return training_cfgs


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)
    verbose = True
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IMDB-BINARY')
    parser.add_argument('--model', type=str, default='GIN', choices=['GIN', 'GFL', 'TopGraph-PI', 'TopGraph-RH'])
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.model
    training_cfg = read_training_cfgs(dataset_name)

    BATCH_SIZE = training_cfg['batch_size']
    dataset = dataset_factory(training_cfg['dataset'], verbose=verbose)
    split_ds, split_i = train_test_val_split(
        dataset,
        validation_ratio=training_cfg['validation_ratio'],
        verbose=verbose)
    fold_tr_acc = []
    fold_test_acc = []
    for fold_i, (train_dataset, test_dataset, validation_dataset) in enumerate(split_ds, start=1):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=my_collate,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=(len(train_dataset) % BATCH_SIZE == 1)
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=my_collate,
            batch_size=64,
            shuffle=False
        )

        dl_val = None
        if training_cfg['validation_ratio'] > 0:
            dl_val = torch.utils.data.DataLoader(
                validation_dataset,
                collate_fn=my_collate,
                batch_size=64,
                shuffle=False
            )
        model, persloss = model_factory(model_name)

        model = model.to(device)
        opt = optim.Adam(
            model.parameters(),
            lr=training_cfg['lr'],
            weight_decay=training_cfg['lmda']
        )
        save_dir = 'Checkpoints'
        lr_sched = StepLR(opt, step_size=100, gamma=0.1)
        PATH = os.path.join(save_dir, f'{model_name}_{dataset_name}_{fold_i}.pt')
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        test_acc = evaluate(test_loader, model, device)
        fold_test_acc.append(test_acc)
        print(f'Fold: {fold_i} Fold test accuracy: {test_acc:.4f}', flush=True)
    print(f'Overall test accuracy: {np.mean(fold_test_acc):.4f}  Test std: {np.std(fold_test_acc):.4f}', flush=True)
