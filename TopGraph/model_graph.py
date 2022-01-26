import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from chofer_torchex.nn import SLayerRationalHat
from torch_geometric.nn import GINConv, global_add_pool, global_sort_pool
from chofer_torchex import pershom
import extendedpersistence

ph = extendedpersistence.extended_pers.extended_persistence_batch
ph_hofer = pershom.pershom_backend.__C.VertFiltCompCuda__vert_filt_persistence_batch
eps = 1e-6


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


def gin_mlp_factory(gin_mlp_type: str, dim_in: int, dim_out: int):
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
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.LeakyReLU(),
            nn.Linear(dim_in, dim_out)
        )
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

        dims = [dim_input] + (gin_number) * [dim]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = torch.nn.functional.leaky_relu

        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))

        self.fc = nn.Sequential(
            nn.Linear(sum(dims), dim),
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

        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
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
        for i, j, e in zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_edges):
            v = node_filt[i:j]
            ph_input.append((v, [e]))

        pers = ph_hofer(ph_input)
        if not self.use_super_level_set_filtration:
            h_0 = [x[0][0] for x in pers]
            h_0_ess = [x[1][0].unsqueeze(1) for x in pers]
            h_1_ess = [x[1][1].unsqueeze(1) for x in pers]

        else:
            ph_sup_input = [(-v, e) for v, e in ph_input]
            pers_sup = ph_hofer(ph_sup_input)

            h_0 = [torch.cat([x[0][0], -(y[0][0])], dim=0) for x, y in zip(pers, pers_sup)]
            h_0_ess = [torch.cat([x[1][0], -(y[1][0])], dim=0).unsqueeze(1) for x, y in zip(pers, pers_sup)]
            h_1_ess = [torch.cat([x[1][1], -(y[1][1])], dim=0).unsqueeze(1) for x, y in zip(pers, pers_sup)]

        y_hat = self.cls(h_0, h_0_ess, h_1_ess)

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
        self.cls = PershomClassifier(dataset,
                                     num_struct_elements=num_struct_elements,
                                     cls_hidden_dimension=cls_hidden_dimension,
                                     drop_out=drop_out)

        self.init_weights()


class PersImClassifier(nn.Module):
    def __init__(self,
                 dataset,
                 pers_im_dim=100,
                 pers_im_var=0.01,
                 cls_hidden_dimension=None,
                 drop_out=None,
                 ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pers_im_dim = pers_im_dim
        self.pers_im_var = pers_im_var

        fc_in_feat = self.pers_im_dim * self.pers_im_dim
        self.pers_im = PersImage(pers_im_dim, pers_im_var)

        self.cls_head = CNN(dataset.num_classes, 4)

    def forward(self, h_0, h_0_ess, h_1, h_1_ext):
        x = self.pers_im(h_0, h_0_ess, h_1, h_1_ext)
        if not self.use_as_feature_extractor:
            x = self.cls_head(x)
        return x


def process_bd_pairs(inp):
    id_swap = torch.tensor([1, 0], dtype=torch.long, device=inp.device)
    # print(f'Inp shape {inp.shape}')
    # print(inp)
    mask = inp[:, 1] < inp[:, 0]
    res = inp
    if mask.sum().item() > 0:
        inp_1 = inp[mask]
        res = torch.index_select(inp_1, 1, id_swap)
        res = torch.cat([res, inp[~mask]], dim=0)
    return res.reshape(1, -1, 2)


class PersImage(nn.Module):
    def __init__(self, im_dim, im_var):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(self.device)
        self.im_dim = torch.tensor(im_dim, dtype=torch.float, device=self.device)
        self.im_var = torch.tensor(im_var, dtype=torch.float, device=self.device)
        self.bdpt_to_bdps = torch.tensor([[1., -1.], [0., 1.]], dtype=torch.float, device=self.device)

    def compute_image(self, inp):
        bp_inp = torch.abs(torch.einsum("ijk,kl->ijl", inp, self.bdpt_to_bdps))
        # bp_inp = torch.nn.functional.pad(bp_inp, (0, 0, 0, 1), value=0.001)
        bp_inp = bp_inp.reshape(1, -1, 2)
        dimension_before, num_pts = bp_inp.shape[2], bp_inp.shape[1]
        coords = [torch.arange(start=0, end=0.999, step=(1 / self.im_dim)) for _ in range(dimension_before)]
        # print(coords[0][99], coords[0][100])
        M = torch.meshgrid(*coords)
        mu = torch.cat([tens.unsqueeze(0) for tens in M], dim=0).to(self.device)
        bc_inp = torch.reshape(bp_inp, [-1, num_pts, dimension_before] + [1 for _ in range(dimension_before)])
        gaussian = -torch.square(bc_inp - mu) / (2 * self.im_var)
        im = torch.exp(gaussian.sum(axis=2)) / (2 * torch.tensor(np.pi) * self.im_var)
        im = im.sum(axis=1)
        im = im / (im.max() + eps)
        return im

    def forward(self, h_0, h_0_ess, h_1, h_1_ext):
        pers_im = []
        for inp_0, inp_1, inp_2, inp_3 in zip(h_0, h_0_ess, h_1, h_1_ext):
            inp_0 = process_bd_pairs(inp_0) if inp_0.size(0) != 0 else inp_0
            inp_1 = process_bd_pairs(inp_1) if inp_1.size(0) != 0 else inp_1
            inp_2 = process_bd_pairs(inp_2) if inp_2.size(0) != 0 else inp_2
            inp_3 = process_bd_pairs(inp_3) if inp_3.size(0) != 0 else inp_3
            im_0 = self.compute_image(inp_0)
            im_1 = self.compute_image(inp_1)
            im_2 = self.compute_image(inp_2)
            im_3 = self.compute_image(inp_3)
            im = torch.cat([im_0, im_1, im_2, im_3], axis=0).unsqueeze(0)
            # print(im.shape)
            pers_im.append(im)

        return torch.cat(pers_im, dim=0)


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


class PersLoss(nn.Module):
    def __init__(self, dataset,
                 input_dim,
                 cls_hidden_dimension=32,
                 cls_type='PI',
                 drop_out=0.1):
        super().__init__()
        self.cls = None
        self.fil = Linear_fil(input_dim, input_dim // 2)
        if cls_type == 'PI':
            self.cls = PersImClassifier(dataset,
                                        pers_im_dim=100,
                                        pers_im_var=0.01,
                                        cls_hidden_dimension=cls_hidden_dimension,
                                        drop_out=drop_out)
        elif cls_type == 'Rational-hat':
            self.cls = ExtendedPershomClassifier(dataset,
                                                 num_struct_elements=100,
                                                 cls_hidden_dimension=cls_hidden_dimension,
                                                 drop_out=drop_out)

        else:
            raise KeyError('Classifier type not found')

    def forward(self, batch, x=None):
        node_filt = self.fil(batch.x)
        ph_input = []
        for i, j, e in zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_edges):
            v = node_filt[i:j]
            ph_input.append((v, [e]))

        pers = ph(ph_input)

        h_0 = [x[0] for x in pers]
        h_0_ess = [x[1] for x in pers]
        h_1_ess = [x[2] for x in pers]
        h_1_ext = [x[3] for x in pers]
        y_hat = self.cls(h_0, h_0_ess, h_1_ess, h_1_ext)
        return y_hat


class ExtendedPershomClassifier(nn.Module):
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
        self.ldgm_0_ess = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        self.ldgm_1_ess = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        self.ldgm_1_ext = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        fc_in_feat = 4 * num_struct_elements

        self.cls_head = ClassifierHead(
            dataset,
            dim_in=fc_in_feat,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    def forward(self, h_0, h_0_ess, h_1_ess, h_1_ext):
        tmp = []
        tmp.append(self.ldgm_0(h_0))
        tmp.append(self.ldgm_0_ess(h_0_ess))
        tmp.append(self.ldgm_1_ess(h_1_ess))
        tmp.append(self.ldgm_1_ext(h_1_ext))

        x = torch.cat(tmp, dim=1)

        if not self.use_as_feature_extractor:
            x = self.cls_head(x)

        return x


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
        self.ldgm_0_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        self.ldgm_1_ess = SLayerRationalHat(num_struct_elements, 1, radius_init=0.1)
        fc_in_feat = 3 * num_struct_elements

        self.cls_head = ClassifierHead(
            dataset,
            dim_in=fc_in_feat,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    def forward(self, h_0, h_0_ess, h_1_ess):
        tmp = []

        tmp.append(self.ldgm_0(h_0))
        tmp.append(self.ldgm_0_ess(h_0_ess))
        tmp.append(self.ldgm_1_ess(h_1_ess))

        x = torch.cat(tmp, dim=1)

        if not self.use_as_feature_extractor:
            x = self.cls_head(x)

        return x


class OneHotEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        eye = torch.eye(dim, dtype=torch.float)

        self.register_buffer('eye', eye)

    def forward(self, batch):
        assert batch.dtype == torch.long

        return self.eye.index_select(0, batch)

    @property
    def dim(self):
        return self.eye.size(1)


class GIN(nn.Module):
    def __init__(self,
                 dataset,
                 use_node_degree: bool = None,
                 use_node_label: bool = None,
                 gin_number: int = None,
                 gin_dimension: int = None,
                 gin_mlp_type: str = None,
                 cls_hidden_dimension: int = None,
                 drop_out: float = None,
                 set_node_degree_uninformative: bool = None,
                 pooling_strategy: str = None,
                 **kwargs,
                 ):
        super().__init__()
        self.use_as_feature_extractor = False
        self.pooling_strategy = pooling_strategy
        self.gin_dimension = gin_dimension

        dim = gin_dimension

        max_node_deg = dataset.max_node_deg
        num_node_lab = dataset.num_node_lab

        if set_node_degree_uninformative and use_node_degree:
            self.embed_deg = UniformativeDummyEmbedding(gin_dimension)
        elif use_node_degree:
            self.embed_deg = OneHotEmbedding(max_node_deg + 1)
        else:
            self.embed_deg = None

        self.embed_lab = OneHotEmbedding(num_node_lab) if use_node_label else None

        dim_input = 0
        dim_input += self.embed_deg.dim if use_node_degree else 0
        dim_input += self.embed_lab.dim if use_node_label else 0
        assert dim_input > 0

        dims = [dim_input] + (gin_number) * [dim]
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.act = torch.nn.functional.leaky_relu

        for n_1, n_2 in zip(dims[:-1], dims[1:]):
            l = gin_mlp_factory(gin_mlp_type, n_1, n_2)
            self.convs.append(GINConv(l, train_eps=True))
            self.bns.append(nn.BatchNorm1d(n_2))

        if pooling_strategy == 'sum':
            self.global_pool_fn = global_add_pool
        elif pooling_strategy == 'sort':
            self.k = int(np.percentile([d.num_nodes for d in dataset], 10))
            self.global_pool_fn = functools.partial(global_sort_pool, k=self.k)
            self.sort_pool_nn = nn.Linear(self.k * gin_dimension, gin_dimension)
        else:
            raise ValueError

        self.cls = ClassifierHead(
            dataset,
            dim_in=gin_dimension,
            hidden_dim=cls_hidden_dimension,
            drop_out=drop_out
        )

    @property
    def feature_dimension(self):
        return self.cls.cls_head[0].in_features

    def forward(self, batch):

        node_deg = batch.node_deg
        node_lab = batch.node_lab if hasattr(batch, 'node_lab') else None

        edge_index = batch.edge_index

        tmp = [e(x) for e, x in
               zip([self.embed_deg, self.embed_lab], [node_deg, node_lab])
               if e is not None]

        tmp = torch.cat(tmp, dim=1)

        z = [tmp]

        for conv, bn in zip(self.convs, self.bns):
            x = conv(z[-1], edge_index)
            x = bn(x)
            x = self.act(x)
            z.append(x)

        # x = torch.cat(z, dim=1)
        x = z[-1]
        x = self.global_pool_fn(x, batch.batch)

        if self.pooling_strategy == 'sort':
            # x = x.view(x.size(0), self.gin_dimension * self.k)
            x = self.sort_pool_nn(x)
            x = x.squeeze()

        if not self.use_as_feature_extractor:
            x = self.cls(x)

        return x
