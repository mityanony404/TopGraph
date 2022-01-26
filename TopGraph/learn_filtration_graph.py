import numpy as np
import torch
import os
import torch_geometric
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data_graph import *
from utils_graph import *
from model_graph import *
import argparse
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
                            drop_out=0.5)
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
                            drop_out=0.5)
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
                                   num_struct_elements=100,
                                   use_super_level_set_filtration=True,
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
    fold_val_acc = []
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
        opt_2 = None
        model = model.to(device)
        use_pers = False

        if persloss is not None:
            persloss = persloss.to(device)
            opt_2 = optim.Adam(
                persloss.parameters(),
                lr=training_cfg['lr'],
                weight_decay=training_cfg['lmda']
            )
            use_pers = True

        opt = optim.Adam(
            model.parameters(),
            lr=training_cfg['lr'],
            weight_decay=training_cfg['lmda']
        )
        save_dir = 'Checkpoints'

        lr_sched = MultiStepLR(opt,
                               milestones=list(range(0, 100, 20))[1:],
                               gamma=0.5)
        lr_sched_2 = MultiStepLR(opt_2,
                                 milestones=list(range(0, 100, 20))[1:],
                                 gamma=0.5) if use_pers else None

        crit = torch.nn.CrossEntropyLoss()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for epoch_i in range(1, training_cfg['num_epochs'] + 1):
            epoch_loss = []
            for batch_i, batch in enumerate(train_loader, start=1):
                if not hasattr(batch, 'node_lab'):
                    batch.node_lab = None
                batch = batch.to(device)
                y_hat = model(batch)
                loss = crit(y_hat, batch.y)
                if use_pers:
                    y_hat_2, _ = persloss(batch, batch.x)
                    loss_2 = crit(y_hat_2, batch.y)
                    loss = 0.5 * loss + 0.5 * loss_2
                    opt_2.zero_grad()
                opt.zero_grad()
                loss.backward()
                epoch_loss.append(loss.item())
                opt.step()
                if use_pers:
                    opt_2.step()
            lr_sched.step()
            if use_pers:
                lr_sched_2.step()
            train_acc = evaluate(train_loader, model, device)
            test_acc = evaluate(test_loader, model, device)
            if verbose:
                print(
                    f"Epoch {epoch_i}/{training_cfg['num_epochs']} Loss: {np.mean(epoch_loss):.4f} Train acc: {train_acc:.4f} Test acc: {test_acc:.4f}",
                    flush=True)
        train_acc = evaluate(train_loader, model, device)
        test_acc = evaluate(test_loader, model, device)
        val_acc = None
        fold_tr_acc.append(train_acc)
        fold_test_acc.append(test_acc)
        if training_cfg['validation_ratio'] > 0.0:
            val_acc = evaluate(dl_val, model, device)
            fold_val_acc.append(val_acc * 100.0)
            print(
                f'Fold: {fold_i} Fold train accuracy: {test_acc:.4f}, Fold test accuracy: {train_acc:.4f} Fold val accuracy: {val_acc}',
                flush=True)
        else:
            print(f'Fold: {fold_i} Fold train accuracy: {train_acc:.4f}, Fold test accuracy: {test_acc:.4f}',
                  flush=True)

        checkpoint_path = os.path.join(save_dir, f'{model_name}_{dataset_name}_{fold_i}.pt')
        if use_pers:
            torch.save({
                'epoch': training_cfg['num_epochs'] + 1,
                'model_state_dict': model.state_dict(),
                'persloss_state_dict': persloss.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'optimizer_2_state_dict': opt_2.state_dict(),
                'loss': loss,
            }, checkpoint_path)
        else:
            torch.save({
                'epoch': training_cfg['num_epochs'] + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss,
            }, checkpoint_path)

    if training_cfg['validation_ratio'] > 0.0:
        print(
            f'Overall train accuracy: {np.mean(fold_tr_acc):.4f}, test accuracy: {np.mean(fold_test_acc):.4f} Train std:'
            f' {np.std(fold_tr_acc):.4f}, Test std: {np.std(fold_test_acc):.4f}, val accuracy: {np.mean(fold_val_acc):.4f}, val std: {np.std(fold_val_acc):.4f}',
            flush=True)
    else:
        print(f'Overall train accuracy: {np.mean(fold_tr_acc):.4f}, test accuracy: {np.mean(fold_test_acc):.4f} Train '
              f'std: {np.std(fold_tr_acc):.4f}, Test std: {np.std(fold_test_acc):.4f}', flush=True)
