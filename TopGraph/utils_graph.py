import torch_geometric
import torch


def my_collate(data_list):
    ret = torch_geometric.data.Batch().from_data_list(data_list)

    boundary_edges = []
    sample_pos = [0]
    max_edge_size = - float('inf')
    for d in data_list:
        boundary_edges.append(d.boundary_edges)
        sample_pos.append(d.num_nodes)
        max_edge_size = max((max_edge_size, d.boundary_edges.size(0)))

    ret.sample_pos = torch.tensor(sample_pos).cumsum(0)
    ret.boundary_edges = boundary_edges
    ret.num_max_node = max(sample_pos)
    ret.max_edge_size = max_edge_size
    if not hasattr(ret, 'node_lab'):
                    ret.node_lab = None

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
            batch.boundary_edges = [e.to(device) for e in batch.boundary_edges]
            # batch.boundary_triangles = [f.to(device) for f in batch.boundary_triangles]

            y_hat = model(batch)

            y_pred = y_hat.max(dim=1)[1]

            correct += (y_pred == batch.y).sum().item()
            num_samples += batch.y.size(0)

    return float(correct) / float(num_samples)


def evaluate_graph(dataloader, model, device):
    num_samples = 0
    correct = 0

    model = model.eval().to(device)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if not hasattr(batch, 'node_lab'):
                batch.node_lab = None
            batch.boundary_edges = [e.to(device) for e in batch.boundary_edges]
            y_hat = model(batch)

            y_pred = y_hat.max(dim=1)[1]

            correct += (y_pred == batch.y).sum().item()
            num_samples += batch.y.size(0)

    return float(correct) / float(num_samples)
