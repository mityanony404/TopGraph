import numpy as np
import torch
import os
import torch_geometric
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from data_graph import *
from utils_graph import *
# import torch.utils.data.DataLoader as DataLoader
from model import *
from model_graph import *
import argparse
import matplotlib.pyplot as plt
import json


def read_training_cfgs(dataset_name):
    with open('training_cfgs.txt') as json_file:
        data = json.load(json_file)
        training_cfgs = data[dataset_name]
    return training_cfgs


def plot_acc_curve(dataset_name, fold, train_accuracy, test_accuracy):
    fig = plt.figure()
    epochs = np.arange(1, len(train_accuracy) + 1)
    plt.plot(epochs, train_accuracy, label='Train_acc')
    plt.plot(epochs, test_accuracy, label='Test_acc')
    plt.legend()
    save_path = f'{dataset_name}_fold_{fold}.png'
    if use_pers == 1:
        save_path = f'{dataset_name}_pers_fold_{fold}.png'
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)
    verbose = True

    parser = argparse.ArgumentParser()
    # parser.add_argument('--lr', type=float, default=1e-2)
    # parser.add_argument('--lmda', type=float, default=1e-4)
    # parser.add_argument('--num_epochs', type=int, default=200)
    # parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        choices=['REDDIT-BINARY', 'REDDIT-MULTI-5K', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1', 'PROTEINS',
                                 'NCI1_2', 'PROTEINS_2'])
    parser.add_argument('--use_pers', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    # data_dir = os.path.join('datasets', args.dataset)
    use_pers = args.use_pers
    print(f'Persloss: {use_pers}')
    training_cfg = read_training_cfgs(args.dataset)
    training_cfg['use_pers'] = use_pers

    BATCH_SIZE = training_cfg['batch_size']
    dataset = dataset_factory(training_cfg['dataset'], verbose=verbose)
    # input_dim = dataset.data[0].x.size(1)
    split_ds, split_i = train_test_val_split(
        dataset,
        validation_ratio=0,
        verbose=verbose)
    HIDDEN_DIM = 64
    use_pers = args.use_pers
    fold_tr_acc = []
    fold_test_acc = []
    for fold_i, (train_dataset, test_dataset, validation_dataset) in enumerate(split_ds, start=1):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=my_collate,
            batch_size=BATCH_SIZE,
            shuffle=True,
            # if last batch would have size 1 we have to drop it ...
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

        model = BaseLine(dataset,
                         use_node_degree=training_cfg['use_node_degree'],
                         set_node_degree_uninformative=training_cfg['set_node_degree_uninformative'],
                         use_node_label=training_cfg['use_node_label'],
                         gin_number=training_cfg['gin_num'],
                         gin_dimension=training_cfg['gin_dim'],
                         gin_mlp_type='lin_bn_lrelu_lin',
                         num_lin_layers=training_cfg['num_lin_layers'],
                         dropout=0.1
                         )
        model = model.to(device)
        persloss = PersLoss(input_dim=training_cfg['gin_dim'], dataset=dataset)
        persloss = persloss.to(device)
        opt = optim.Adam(
            model.parameters(),
            lr=training_cfg['lr'],
            weight_decay=training_cfg['lmda']
        )
        opt_2 = optim.Adam(
            persloss.parameters(),
            lr=training_cfg['lr'],
            weight_decay=training_cfg['lmda']
        )
        lr_sched = StepLR(opt, step_size=100, gamma=0.1)
        crit = torch.nn.CrossEntropyLoss()
        max_test_acc = -np.inf
        save_dir = 'saved_models_baseline_' + args.dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for epoch_i in range(1, training_cfg['num_epochs'] + 1):
            epoch_loss = []
            for batch_i, batch in enumerate(train_loader, start=1):
                if not hasattr(batch, 'node_lab'):
                    batch.node_lab = None
                batch = batch.to(device)
                y_hat, embed = model(batch)
                loss = crit(y_hat, batch.y)
                if use_pers == 1:
                    y_hat_2 = persloss(batch, embed)
                    loss_2 = crit(y_hat_2, batch.y)
                    loss = 0.5 * loss + 0.5 * loss_2
                    opt_2.zero_grad()
                opt.zero_grad()
                loss.backward()
                epoch_loss.append(loss.item())
                opt.step()
                if use_pers == 1:
                    opt_2.step()
            lr_sched.step()

            # if verbose:
            #     print(f"Epoch {epoch_i}/{training_cfg['num_epochs']}, Batch {batch_i}/{len(train_loader)}", end='\n')
            # print(f'Loss: {np.mean(epoch_loss):.4f}')
            train_acc = evaluate(train_loader, model, device)
            test_acc = evaluate(test_loader, model, device)
            if verbose:
                print(
                    f"Epoch {epoch_i}/{training_cfg['num_epochs']} Loss: {np.mean(epoch_loss):.4f} Train acc: {train_acc:.4f} Test acc: {test_acc:.4f}",
                    flush=True)
        train_acc = evaluate(train_loader, model, device)
        test_acc = evaluate(test_loader, model, device)
        fold_tr_acc.append(train_acc)
        fold_test_acc.append(test_acc)
        print(f'Fold: {fold_i} Fold test accuracy: {test_acc:.4f}, Fold train accuracy: {train_acc:.4f}', flush=True)
    print(f'Overall train accuracy: {np.mean(fold_tr_acc):.4f}, test accuracy: {np.mean(fold_test_acc):.4f} Train std:'
          f' {np.std(fold_tr_acc)}, Test std: {fold_test_acc}', flush=True)
