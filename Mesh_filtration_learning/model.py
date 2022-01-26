import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric
import torch_geometric.nn as geonn
from torch_geometric.nn import GINConv, GCNConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import global_sort_pool
import functools
import operator
from torch_geometric.nn import GINConv, global_add_pool, global_sort_pool

import extendedpersistence
ph = extendedpersistence.extended_pers.vertex_persistence_batch

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, barcode_dim=0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [250, 100]
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], output_dim)
        self.barcode_dim = barcode_dim

    def forward(self, x, bar):
        x = x.view((x.size(0), -1))
        bar = bar.view((bar.size(0), -1))
        if self.barcode_dim > 0:
            x = torch.cat((x, bar), dim=1)
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        return out


class CNN(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output num classes
        self.out = nn.Linear(32 * 25 * 25, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        # print(output.shape)
        return output


def gin_mlp_factory(gin_mlp_type: str, dim_in: int, dim_out: int, dim_hidden: int = None):
    if gin_mlp_type == 'lin':
        return nn.Linear(dim_in, dim_out)

    elif gin_mlp_type == 'lin_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )

    elif gin_mlp_type == 'lin_bn_lrelu_lin':
        return nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_out)
        )
    # elif gin_mlp_type == 'gcn_bn_lrelu_lin':
    #     assert dim_hidden is not None
    #     return nn.Sequential(
    #         GCNConv(dim_in, dim_hidden),
    #         nn.BatchNorm1d(dim_hidden),
    #         nn.LeakyReLU(),
    #         nn.Linear(dim_hidden, dim_out)
    #     )
    else:
        raise ValueError("Unknown gin_mlp_type!")


def ClassifierHead(
        dataset,
        dim_in: int = None,
        hidden_dim: int = None,
        drop_out: float = None):
    assert (0.0 <= drop_out) and (drop_out < 1.0)
    assert dim_in is not None
    assert drop_out is not None
    assert hidden_dim is not None

    tmp = [
        nn.Linear(dim_in, hidden_dim),
        nn.LeakyReLU(),
    ]

    if drop_out > 0:
        tmp += [nn.Dropout(p=drop_out)]

    tmp += [nn.Linear(hidden_dim, dataset.num_classes)]

    return nn.Sequential(*tmp)


class Filtration(torch.nn.Module):
    def __init__(self,
                 dataset,
                 input_dim=3,
                 use_node_degree=None,
                 set_node_degree_uninformative=None,
                 use_node_label=None,
                 gin_number=None,
                 gin_dimension=None,
                 gin_mlp_type=None,
                 **kwargs
                 ):
        super().__init__()

        dim = gin_dimension

        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab

        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = nn.Embedding(max_node_deg + 1, dim)
        else:
            self.embed_deg = None

        self.embed_lab = nn.Embedding(num_node_lab, dim) if use_node_label else None

        dim_input = dim * ((self.embed_deg is not None) + (self.embed_lab is not None))

        dims = [input_dim] + (gin_number) * [dim]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = torch.nn.functional.leaky_relu

        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))

        self.fc = nn.Sequential(
            nn.Linear(dim + gin_number * gin_dimension, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):

        node_deg = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index

        tmp = [e(x) for e, x in
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None]

        tmp = torch.cat(tmp, dim=1)

        z = [tmp]
        x = batch.pos
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.act(x)
        z.append(x)

        x = torch.cat(z, dim=1)
        ret = self.fc(x).squeeze()
        return ret


class UniformativeDummyEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        b = torch.ones(1, dim, dtype=torch.float)
        self.register_buffer('ones', b)

    def forward(self, batch):
        assert batch.dtype == torch.long
        return self.ones.expand(batch.size(0), -1)

    @property
    def dim(self):
        return self.ones.size(1)


class PershomBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.use_super_level_set_filtration = None
        self.use_as_feature_extractor = False
        self.fil = None
        self.cls = None

    def forward(self, batch, x=None):
        assert self.use_super_level_set_filtration is not None

        node_filt = self.fil(batch)

        ph_input = []
        ph_input_sup = []
        for i, j, e in zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_edges):
            v = node_filt[i:j]
            ph_input.append((v, [e]))
            ph_input_sup.append((-v, [e]))

        pers = ph(ph_input)
        pers_sup = ph(ph_input_sup)

        h_0 = [x[0][0] for x in pers]
        # h_0_ess = [x[1][0].unsqueeze(1) for x in pers]
        h_1 = [torch.cat([-x[0][0][:, 1], -x[0][0][:, 0]], dim=0).view((-1, 2)) for x in pers_sup]
        # h_1_ess = [x[1][1].unsqueeze(1) for x in pers]
        # print(f'h_0: {len(h_0)} pers: {pers.shape}')
        y_hat = self.cls(h_0, h_1)

        return y_hat

    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features

    @property
    def use_as_feature_extractor(self):
        return self.use_as_feature_extractor

    @use_as_feature_extractor.setter
    def use_as_feature_extractor(self, val):
        if hasattr(self, 'cls'):
            self.cls.use_as_feature_extractor = val

    def init_weights(self):
        def init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.apply(init)


class PershomLearnedFilt(PershomBase):
    def __init__(self,
                 dataset,
                 use_super_level_set_filtration: bool = None,
                 use_node_degree: bool = None,
                 set_node_degree_uninformative: bool = None,
                 use_node_label: bool = None,
                 gin_number: int = None,
                 gin_dimension: int = None,
                 gin_mlp_type: str = None,
                 num_struct_elements: int = None,
                 cls_hidden_dimension: int = None,
                 drop_out: float = None,
                 **kwargs,
                 ):
        super().__init__()

        self.use_super_level_set_filtration = use_super_level_set_filtration

        self.fil = Filtration(
            dataset,
            use_node_degree=use_node_degree,
            set_node_degree_uninformative=set_node_degree_uninformative,
            use_node_label=use_node_label,
            gin_number=gin_number,
            gin_dimension=gin_dimension,
            gin_mlp_type=gin_mlp_type,
        )
        self.cls = PersImClassifier(dataset,
                                    pers_im_dim=100,
                                    pers_im_var=0.5,
                                    cls_hidden_dimension=cls_hidden_dimension,
                                    drop_out=drop_out)

        # self.cls = PershomClassifier(
        #     dataset,
        #     num_struct_elements=num_struct_elements,
        #     cls_hidden_dimension=cls_hidden_dimension,
        #     drop_out=drop_out
        # )

        self.init_weights()


class PershomClassifier(nn.Module):
    def __init__(self,
                 dataset,
                 num_struct_elements=None,
                 cls_hidden_dimension=None,
                 drop_out=None,
                 ):
        super().__init__()
        assert isinstance(num_struct_elements, int)
        self.use_as_feature_extractor = False

        self.ldgm_0 = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        # self.ldgm_0_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        self.ldgm_1 = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        # self.ldgm_1_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        # self.ldgm_2 = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        # self.ldgm_2_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        fc_in_feat = 2 * num_struct_elements

        self.cls_head = ClassifierHead(
            dataset,
            dim_in=fc_in_feat,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    def forward(self, h_0, h_1, ):
        tmp = [self.ldgm_0(h_0), self.ldgm_1(h_1)]
        # tmp.append(self.ldgm_0(h_0))
        # tmp.append(self.ldgm_1(h_1))
        x = torch.cat(tmp, dim=1)
        if not self.use_as_feature_extractor:
            x = self.cls_head(x)

        return x


class PersImage(nn.Module):
    def __init__(self, im_dim, im_var):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(self.device)
        self.im_dim = torch.tensor(im_dim, dtype=torch.float, device=self.device)
        self.im_var = torch.tensor(im_var, dtype=torch.float, device=self.device)
        self.bdpt_to_bdps = torch.tensor([[1., -1.], [0., 1.]], dtype=torch.float, device=self.device)

    def compute_image(self, inp):
        bp_inp = torch.einsum("ijk,kl->ijl", inp, self.bdpt_to_bdps)
        dimension_before, num_pts = inp.shape[2], inp.shape[1]
        coords = [torch.arange(start=0, end=0.999, step=(1 / self.im_dim)) for _ in range(dimension_before)]
        # print(coords[0][99], coords[0][100])
        M = torch.meshgrid(*coords)
        mu = torch.cat([tens.unsqueeze(0) for tens in M], dim=0).to(self.device)
        bc_inp = torch.reshape(bp_inp, [-1, num_pts, dimension_before] + [1 for _ in range(dimension_before)])
        gaussian = -torch.square(bc_inp - mu) / (2 * self.im_var)
        im = torch.exp(gaussian.sum(axis=2)) / (2 * torch.tensor(np.pi) * self.im_var)
        im = im.sum(axis=1)
        im = im / im.max()
        return im

    def forward(self, h_0, h_1):
        pers_im = []
        for inp_0, inp_1 in zip(h_0, h_1):
            # print(f'inp_0: {inp_0.shape} inp_1: {inp_1.shape}')
            inp_0 = inp_0.reshape(1, -1, 2)
            inp_1 = inp_1.reshape(1, -1, 2)
            im_0 = self.compute_image(inp_0)
            im_1 = self.compute_image(inp_1)
            im = torch.cat([im_0, im_1], axis=0).unsqueeze(0)
            # print(im.shape)
            pers_im.append(im)

        return torch.cat(pers_im, dim=0)


class PersImClassifier(nn.Module):
    def __init__(self,
                 num_classes,
                 pers_im_dim=100,
                 pers_im_var=0.5,
                 cls_hidden_dimension=None,
                 drop_out=None,
                 ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pers_im_dim = pers_im_dim
        self.pers_im_var = pers_im_var

        fc_in_feat = self.pers_im_dim * self.pers_im_dim
        self.pers_im = PersImage(pers_im_dim, pers_im_var)
        # self.cls_head = ClassifierHead(
        #     dataset,
        #     dim_in=fc_in_feat,
        #     hidden_dim=cls_hidden_dimension,
        #     drop_out=drop_out
        # )
        self.cls_head = CNN(num_classes, 2)

    def forward(self, h_0, h_1):
        x = self.pers_im(h_0, h_1)
        # x = x.view((x.shape[0], -1))
        # print(x.shape)
        if not self.use_as_feature_extractor:
            x = self.cls_head(x)
        return x


class BaseLine(torch.nn.Module):
    def __init__(self,
                 dataset,
                 input_dim=6,
                 gin_number=None,
                 gin_dimension=None,
                 gin_mlp_type=None,
                 num_lin_layers=None,
                 **kwargs
                 ):
        super(BaseLine, self).__init__()
        # torch.manual_seed(12345)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = torch.nn.functional.leaky_relu
        dims = [input_dim] + (gin_number + 1) * [gin_dimension]

        self.lins = nn.ModuleList()

        for n_1, n_2, n_h in zip(dims[:-2], dims[1:], dims[2:]):
            l_nn = gin_mlp_factory(gin_mlp_type, n_1, n_2, n_h)
            self.convs.append(GINConv(l_nn, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))

        for _ in range(num_lin_layers-1):
            self.lins.append(nn.Linear(gin_dimension, gin_dimension))
        self.lins.append(nn.Linear(gin_dimension, dataset.num_classes))

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.act(x)

        out = x

        # 2. Readout layer
        x = global_mean_pool(out, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.act(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[-1](x)
        return x, out


class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, num_lin_layers, out_classes):
        super(GCN, self).__init__()
        self.transform = nn.Linear(input_dim, input_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # dims = [input_dim] * num_layers * [hidden_dim]
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.act = nn.ReLU()
        for n in range(num_layers-1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lins = nn.ModuleList()
        for n in range(num_lin_layers-1):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, out_classes))

    def forward(self, batch):
        x = self.transform(batch.pos)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, batch.edge_index)
            x = bn(x)
            x = self.act(x)

        out = x

        # 2. Readout layer
        x = global_mean_pool(out, batch.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.act(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[-1](x)
        return x, out


class Linear_fil(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Linear_fil, self).__init__()
        self.lin_1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.lin_2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        x = self.sigmoid(x).squeeze()
        return x


class PersLoss(PershomBase):
    def __init__(self, input_dim, pers_im_size, pers_im_var, num_classes):
        super().__init__()
        self.fil = Linear_fil(input_dim, input_dim//2)
        self.cls = PersImClassifier(num_classes,
                                    pers_im_dim=pers_im_size,
                                    pers_im_var=pers_im_var,
                                    cls_hidden_dimension=32,
                                    drop_out=0.2)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(self, batch, x=None):
        node_filt = self.fil(x)
        ph_input = []
        ph_input_sup = []
        for i, j, e in zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_edges):
            v = node_filt[i:j]
            ph_input.append((v, [e]))

        pers = ph(ph_input)
        h_0 = [x[0] for x in pers]
        h_1 = [x[1] for x in pers]

        y_hat = self.cls(h_0, h_1)
        # loss = self.loss_fn(y_hat, batch.y)
        return y_hat, node_filt




