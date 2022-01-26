import torch
import torch_geometric
import torch.optim as optim
from data import *
from utils import *
from torch.utils.data import DataLoader
from model import *
import argparse

data_path = {'shrec_16': 'datasets/shrec_16',
             'ModelNet10': 'datasets/ModelNet10'}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)
    verbose = True
    save_interval = 200
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lmda', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='shrec_16')
    args = parser.parse_args()
    data_dir = data_path[args.dataset]
    BATCH_SIZE = args.batch_size
    train_dataset = MyOwnDataset(data_dir, verbose=True, phase='train')
    test_dataset = MyOwnDataset(data_dir, verbose=True, phase='test')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=False)

    training_cfg = {'lr': args.lr, 'weight_decay': args.lmda, 'num_epochs': args.num_epochs}

    INPUT_DIM = 6
    model = GCN(num_layers=2, input_dim=INPUT_DIM, hidden_dim=32, num_lin_layers=3,
                out_classes=train_dataset.num_classes)
    model = model.to(device)
    persloss = PersLoss(input_dim=32, pers_im_size=100, pers_im_var=0.1, num_classes=train_dataset.num_classes)
    # persloss = persloss.to(device)
    opt = optim.Adam(
        model.parameters(),
        lr=training_cfg['lr'],
        weight_decay=training_cfg['weight_decay']
    )
    opt_2 = optim.Adam(
        persloss.parameters(),
        lr=training_cfg['lr'],
        weight_decay=training_cfg['weight_decay']
    )
    crit = torch.nn.CrossEntropyLoss()
    max_test_acc = -np.inf
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch_i in range(1, training_cfg['num_epochs']//2 + 1):
        epoch_loss = []
        for batch_i, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            y_hat, embed = model(batch)
            loss_1 = crit(y_hat, batch.y)
            loss = loss_1
            opt.zero_grad()
            loss.backward()
            batch_loss = loss.item()
            epoch_loss.append(batch_loss)
            opt.step()

            if verbose:
                print(
                    f"Epoch {epoch_i}/{training_cfg['num_epochs']}, Batch {batch_i}/{len(train_loader)} Loss: {batch_loss:.4f}",
                    end='\n')
        train_acc = evaluate(train_loader, model, device)
        test_acc = evaluate(test_loader, model, device)
        if verbose:
            print(
                f"Epoch {epoch_i}/{training_cfg['num_epochs']} Loss: {np.mean(epoch_loss):.4f} Train acc: {train_acc:.4f} Test acc: {test_acc:.4f}")

        if epoch_i % save_interval == 0:
                model_name = f'{args.dataset}_gcn_{epoch_i}.pt'
                persloss_name = f'{args.dataset}_persloss_{epoch_i}.pt'
                save_path = os.path.join(save_dir, model_name)
                # pers_path = os.path.join(save_dir, persloss_name)
                torch.save(model, save_path)
                # torch.save(persloss, pers_path)
    for epoch_i in range(training_cfg['num_epochs'] // 2 + 1, training_cfg['num_epochs'] + 1):
        epoch_loss = []
        for batch_i, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            y_hat, embed = model(batch)
            loss_1 = crit(y_hat, batch.y)
            y_hat_2, _ = persloss(batch, embed)
            loss_2 = crit(y_hat_2, batch.y)
            loss = 0.75 * loss_1 + 0.25 * loss_2
            opt.zero_grad()
            opt_2.zero_grad()
            loss.backward()
            # loss_2.backward()
            batch_loss = loss.item()
            epoch_loss.append(loss.item())
            opt.step()
            opt_2.step()

            if verbose:
                print(
                    f"Epoch {epoch_i}/{training_cfg['num_epochs']}, Batch {batch_i}/{len(train_loader)} Loss: {np.mean(epoch_loss):.4f}",
                    end='\n')
        train_acc = evaluate(train_loader, model, device)
        test_acc = evaluate(test_loader, model, device)
        if verbose:
            print(
                f"Epoch {epoch_i}/{training_cfg['num_epochs']} Loss: {np.mean(epoch_loss):.4f} Train acc: {train_acc:.4f} Test acc: {test_acc:.4f}")

        if epoch_i % save_interval == 0:
            model_name = f'{args.dataset}_model_{epoch_i}.pt'
            persloss_name = f'{args.dataset}_persloss_{epoch_i}.pt'
            save_path = os.path.join(save_dir, model_name)
            pers_path = os.path.join(save_dir, persloss_name)
            torch.save(model, save_path)
            torch.save(persloss, pers_path)
