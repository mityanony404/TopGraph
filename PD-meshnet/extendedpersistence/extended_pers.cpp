#include <iostream>
#include <utility>
#include <algorithm>
#include <vector>
#include <future>
#include <torch/extension.h>
//link cut tree:
#include "./data_structure/link_cut_tree.hpp"
#include "./monoids/max_index.hpp"

#define EPS 1e-20

void print_graph(link_cut_tree<max_index_monoid<double>> lct, int n){
    for(int i=0; i<n ; i++){
        std::cout<<i <<" has parent: " <<lct.get_parent(i)<<std::endl;
        std::cout<<i << " has value: "<<lct.vertex_get(i)<<std::endl;
    }
    //std::cout<<"root of n-1 is: "<<lct.get_root(n-1)<<std::endl;
    //std::cout<<"LCA of 1 and n-1 is: "<< lct.get_lowest_common_ancestor(1,n-1)<<std::endl;
}

class union_find{
public:
    union_find(int64_t n){
        this->count = n;
        this->parent.resize(n);
        for(auto i=0; i<n; i++){
            this->parent[i] = i;
        }
        this->rank.resize(n,0);
    }

    int64_t uf_depth(int64_t x){
        return rank[x];
    }
    int64_t find(int64_t x){
        int64_t tmp= x;
        while(parent[tmp] != tmp){
            parent[tmp] = parent[parent[tmp]];
            tmp = parent[tmp];
        }
        return tmp;
    }
    void link(int64_t x, int64_t y){
        x= find(x);
        y= find(y);
        if(x==y){
            return;
        }
        if(rank[x]<rank[y]){
            parent[x]= y;
        }else if(rank[x]>rank[y]){
            parent[y]= x;
        }else{
            parent[y]= x;
            rank[x]= rank[x]+1;
        }
        count--;
    }
    int64_t num_connected_component() const{
        return this->count;
    }



private:
    int64_t count;
    std::vector<int64_t> parent;
    std::vector<int64_t> rank;
};
using std::vector;
using torch::Tensor;
typedef std::pair<long, long> Edge;
typedef std::pair<int64_t, Edge> Pers;
using namespace torch::indexing;



void print64_t_pairs(vector<Pers> &ed) {
    for (auto e: ed) {
        std::cout << e.first << " (" << e.second.first << "," << e.second.second << ")" << std::endl;
    }
}
vector<Tensor> compute_pd0(const Tensor & vertex_filtration,
                           const vector<Tensor> & boundary_info, const bool mirror = false){
    auto num_nodes = vertex_filtration.size(0);
    union_find uf = union_find(num_nodes);
    Tensor tensor_edges = boundary_info[0];
    Tensor edge_val = std::get<0>(torch::max(vertex_filtration.index({tensor_edges}), 1));
    Tensor sorted_edge_indices = edge_val.argsort(-1, false);
    const Tensor sorted_edges = tensor_edges.index({sorted_edge_indices});
    edge_val = edge_val.index({sorted_edge_indices});
    auto num_edges = sorted_edges.size(0);
    vector<Tensor> pd_0;
    pd_0.reserve(num_nodes-1);

    for(auto i = 0; i < num_edges; i++){
        auto e = sorted_edges[i];
        auto e_val = edge_val[i];
        auto u = e[0].item<int64_t>();
        auto v = e[1].item<int64_t>();
        int64_t root_u = uf.find(u);
        int64_t root_v = uf.find(v);
        if(root_u == root_v){
            continue;
        }
        int64_t root = root_u;
        int64_t merged = root_v;
        if (vertex_filtration[root].item<double>() > vertex_filtration[merged].item<double>())
            std::swap(root, merged);
        else if (std::abs(vertex_filtration[root].item<double>() - vertex_filtration[merged].item<double>()) < EPS) {
            if (root > merged)
                std::swap(root, merged);
        }
        auto merged_val = vertex_filtration[merged];
        if (std::abs(e_val.item<float>() - merged_val.item<float>()) > 0) {
            Tensor pd_pair;
            if(mirror) {
                 pd_pair = torch::stack({-e_val, -merged_val});
            }
            else{
                pd_pair = torch::stack({merged_val, e_val});
            }
            pd_0.emplace_back(pd_pair);
        }
        uf.link(root, merged);
    }

    return pd_0;

}

vector<Tensor> extended_filt_persistence_single(const Tensor & vertex_filtration,
                                                        const vector<Tensor> & boundary_info){
    vector<Tensor> pd;
    auto options = torch::TensorOptions().dtype(at::kLong);
    auto num_nodes = vertex_filtration.size(0);
    torch::Tensor dummy_ind = torch::randint(num_nodes, {1, }, options);
    const Tensor dummy_pair = torch::stack({vertex_filtration.index({dummy_ind}), vertex_filtration.index({dummy_ind})}, 1);
    union_find uf = union_find(num_nodes);
    link_cut_tree<max_index_monoid<double> > lct(num_nodes);
    for(auto i =0; i<num_nodes; i++){
        lct.vertex_set(i,{vertex_filtration[i].item<double>(),i});
    }
    vector<size_t> pos_edge_index;
    vector<Tensor> pd_0_up = compute_pd0(vertex_filtration, boundary_info);
    vector<Tensor> pd_0_down, pd_0_ext_plus, pd_1_ext;
    Tensor tensor_edges = boundary_info[0];
    Tensor edge_val = std::get<0>(torch::min(vertex_filtration.index({tensor_edges}), 1));
    Tensor sorted_edge_indices = edge_val.argsort(-1, true);
    const Tensor sorted_edges = tensor_edges.index({sorted_edge_indices});
    edge_val = edge_val.index({sorted_edge_indices});
    auto num_edges = sorted_edges.size(0);
    for(auto i = 0; i < num_edges; i++){
        auto e = sorted_edges[i];
        auto e_val = edge_val[i];
        int64_t u = e[0].item<int64_t>();
        int64_t v = e[1].item<int64_t>();
        int64_t root_u = uf.find(u);
        int64_t root_v = uf.find(v);
        if(root_u == root_v){
            pos_edge_index.push_back(i);
            continue;
        }
        auto u_rank= uf.uf_depth(u);//depth of union find connected component is in the union find data structure
        auto v_rank= uf.uf_depth(v);//depth of union find connected component is in the union find data structure

        if (u_rank < v_rank) {
            lct.evert(u);
            lct.link(u,v);
        }else{
            lct.evert(v);
            lct.link(v,u);
        }

        int64_t root = root_u;
        int64_t merged = root_v;
        if (vertex_filtration[root].item<double>() < vertex_filtration[merged].item<double>())
            std::swap(root, merged);
        else if (std::abs(vertex_filtration[root].item<double>() - vertex_filtration[merged].item<double>()) < EPS) {
            if (root < merged)
                std::swap(root, merged);
        }
        auto merged_val = vertex_filtration[merged];
        Tensor pd_pair = torch::stack({merged_val, e_val});
        pd_0_down.emplace_back(pd_pair);
        uf.link(root, merged);
    }
    //this is wrong, it assumes one connected component


    //count the min and max per connected component: this is pd_0_ext_plus
    std::vector<int64_t> connected_components_min(num_nodes, -1);
    std::vector<int64_t> connected_components_max(num_nodes, -1);
    for(auto v=0; v<num_nodes; v++) {
        auto v_root = uf.find(v);
        if(uf.uf_depth(v_root)>0) {
            if (connected_components_max[v_root] == -1 && connected_components_min[v_root] == -1) {
                connected_components_min[v_root] = v;
                connected_components_max[v_root] = v;
            } else {
                if (vertex_filtration[v].item<double>() >
                    vertex_filtration[connected_components_max[v_root]].item<double>()) {
                    connected_components_max[v_root] = v;
                }
                if (vertex_filtration[v].item<double>() <
                    vertex_filtration[connected_components_min[v_root]].item<double>()) {
                    connected_components_min[v_root] = v;
                }
            }
        }
    }

    for(auto v=0; v<num_nodes; v++) {
        auto min_i= connected_components_min[v];
        auto max_i= connected_components_max[v];
        if(min_i!=-1 && max_i!=-1) {
            pd_0_ext_plus.push_back(torch::stack({vertex_filtration[min_i], vertex_filtration[max_i]}));
        }
    }


    for(auto ii : pos_edge_index){
        auto pos_edge = sorted_edges[ii];
        auto pos_edge_val = edge_val[ii];
        int64_t u = pos_edge[0].item<int64_t>();
        int64_t v = pos_edge[1].item<int64_t>();

        auto lca= lct.get_lowest_common_ancestor(u,v);
        assert(lca!=-1);
        auto p1= lct.path_get(u,lca);
        auto p2= lct.path_get(v,lca);

        int64_t critical_vertex;//the maximum on the loop
        int64_t deletion_vertex;//the node to delete whose parent is the critical vertex
        int64_t r;
        if(p1.first>p2.first) {
            critical_vertex = p1.second;
            r= lct.get_root(v);
        }else {
            critical_vertex = p2.second;
            r= lct.get_root(u);
        }
        deletion_vertex= critical_vertex;

        auto search_node= u;
        bool find_child_of_critical= false;
        while(search_node!=critical_vertex && search_node!=lca){
            auto paren= lct.get_parent(search_node);
            if(paren==critical_vertex){
                deletion_vertex= search_node;
                //std::cout<<"FOUND DELETION VERTEX: "<<deletion_vertex<<std::endl;
                break;
            }
            search_node= paren;
        }
        if(search_node==lca){
            search_node= v;
            while(search_node!=critical_vertex && search_node!=lca){
                auto paren= lct.get_parent(search_node);
                if(paren==critical_vertex){
                    deletion_vertex= search_node;
//                    std::cout<<"FOUND DELETION VERTEX: "<<deletion_vertex<<std::endl;
                    break;
                }
                search_node= paren;
            }
        }
        lct.cut(deletion_vertex);


        auto u_rank= uf.uf_depth(u);//depth of union find connected component is in the union find data structure
        auto v_rank= uf.uf_depth(v);//depth of union find connected component is in the union find data structure

        if (u_rank < v_rank) {
            lct.evert(u);
            lct.link(u,v);
        }else{
            lct.evert(v);
            lct.link(v,u);
        }
        auto cut_edge_val= vertex_filtration[critical_vertex];
        //std::cout << "CE " << cut_edge_val.item<double>() << " AE " << pos_edge_val.item<double>() << std::endl;
        auto pers_pair = torch::stack({cut_edge_val, pos_edge_val});
        pd_1_ext.push_back(pers_pair);
    }

    try{
        auto pd_0_up_t = torch::stack(pd_0_up);
        pd.push_back(pd_0_up_t);
        }
    catch(const std::exception& e){
         //std::cout << e.what() << std::endl;
         pd.push_back(dummy_pair);
        }
    try{
        auto pd_0_down_t = torch::stack(pd_0_down);
        pd.push_back(pd_0_down_t);
        }
    catch(const std::exception& e){
         //std::cout << e.what() << std::endl;
         pd.push_back(dummy_pair);
    }
    try{
        auto pd_0_ext_plus_t = torch::stack(pd_0_ext_plus);
        pd.push_back(pd_0_ext_plus_t);
        }
    catch(const std::exception& e){
         //::cout << e.what() << std::endl;
         pd.push_back(dummy_pair);
    }
    try{
        auto pd_1_ext_t = torch::stack(pd_1_ext);
        pd.push_back(pd_1_ext_t);
        }
    catch(const std::exception& e){
            pd.push_back(dummy_pair);
    }

    return pd;

}

vector<Tensor> vertex_filt_persistence_single(const Tensor & vertex_filtration,
                                              const vector<Tensor> & boundary_info){
    vector<Tensor> pd;
    auto options = torch::TensorOptions().dtype(at::kLong);
    auto num_nodes = vertex_filtration.size(0);
    torch::Tensor dummy_ind = torch::randint(num_nodes, {1, }, options);
    const Tensor dummy_pair = torch::stack({vertex_filtration.index({dummy_ind}), vertex_filtration.index({dummy_ind})}, 1);
    union_find uf = union_find(num_nodes);


    vector<Tensor> pd_0_up = compute_pd0(vertex_filtration, boundary_info, false);
    Tensor vertex_filtration_inv = - vertex_filtration;
    vector<Tensor> pd_0_down = compute_pd0(vertex_filtration_inv, boundary_info, true);


    try{
        auto pd_0_up_t = torch::stack(pd_0_up);
        pd.push_back(pd_0_up_t);
    }
    catch(const std::exception& e){
        pd.push_back(dummy_pair);
    }
    try{
        auto pd_0_down_t = torch::stack(pd_0_down);
        pd.push_back(pd_0_down_t);
    }
    catch(const std::exception& e){

        pd.push_back(dummy_pair);
    }
    return pd;

}
vector<vector<Tensor>> extended_filt_persistence_batch(const vector<std::tuple<Tensor, vector<Tensor>>> & batch){
auto futures = vector<std::future<vector<Tensor>>>();
for (auto & arg: batch){

futures.push_back(
        async(std::launch::async,[=]{
                  return extended_filt_persistence_single(
                          std::get<0>(arg),
                          std::get<1>(arg)
                  );
              }
)
);
}
auto ret = vector<vector<Tensor>>();
for (auto & fut: futures){
ret.push_back(
        fut.get()
);
}
return ret;
}

vector<vector<Tensor>> vertex_filt_persistence_batch(const vector<std::tuple<Tensor, vector<Tensor>>> & batch){
auto futures = vector<std::future<vector<Tensor>>>();
for (auto & arg: batch){

futures.push_back(
        async(std::launch::async,[=]{
                  return vertex_filt_persistence_single(
                          std::get<0>(arg),
                          std::get<1>(arg)
                  );
              }
)
);
}
auto ret = vector<vector<Tensor>>();
for (auto & fut: futures){
ret.push_back(
        fut.get()
);
}
return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("extended_persistence_batch", &extended_filt_persistence_batch, "A function to compute extended_persistence in batches as C-hofer");
m.def("extended_persistence_single", &extended_filt_persistence_single, "A function to compute extended_persistence with (v, [e]) format");
m.def("vertex_persistence_single", &vertex_filt_persistence_single, "A function to compute extended_persistence with (v, [e]) format");
m.def("vertex_persistence_batch", &vertex_filt_persistence_batch, "A function to compute extended_persistence with (v, [e]) format");
}

