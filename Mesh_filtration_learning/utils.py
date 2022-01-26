import torch_geometric
import torch


def my_collate(data_list):
    ret = torch_geometric.data.Batch().from_data_list(data_list)

    boundary_edges = []
    sample_pos = [0]
    for d in data_list:
        boundary_edges.append(d.boundary_edges)
        sample_pos.append(d.num_nodes)

    ret.sample_pos = torch.tensor(sample_pos).cumsum(0)
    ret.boundary_edges = boundary_edges

    return ret


def evaluate(dataloader, model, device):
    num_samples = 0
    correct = 0

    model = model.eval().to(device)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if not hasattr(batch, 'node_lab'):
                batch.node_lab = None
            # batch.boundary_edges = [e.to(device) for e in batch.boundary_edges]
            # batch.boundary_triangles = [f.to(device) for f in batch.boundary_triangles]

            y_hat, embed = model(batch)

            y_pred = y_hat.max(dim=1)[1]

            correct += (y_pred == batch.y).sum().item()
            num_samples += batch.y.size(0)

    return float(correct) / float(num_samples)
