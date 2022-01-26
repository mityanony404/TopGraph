#include <iostream>
#include <utility>
#include <algorithm>
#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/transform_value_property_map.hpp>
#include <boost/property_map/function_property_map.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <future>
#include <torch/extension.h>
#define EPS 1e-20

class union_find{
public:
    union_find(int n){
        this->count = n;
        for(auto i=0; i<n; i++){
            parent[i] = i;
        }
    }

    void make_set(int x){
        this->parent[x] = x;
        count++;
    }
    int find(int x){
        int tmp = x;
        while(parent[tmp] != tmp){
            parent[tmp] = parent[parent[tmp]];
            tmp = parent[tmp];
        }
        return tmp;
    }
    // For now set parent[y] = x. Union by rank may come later. In that case we need to make sure the root is
    // always lower f value
    void link(int x, int y){
        parent[y] = x;
        count--;
    }
    int num_connected_component() const{
        return this->count;
    }
private:
    std::map<int, int> parent;
    int count;

};
using std::vector;
using torch::Tensor;
typedef std::pair<long, long> Edge;
typedef std::pair<int, Edge> Pers;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int>> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor VertexDesc;
typedef boost::graph_traits<Graph>::edge_descriptor EdgeDesc;
typedef boost::graph_traits<Graph>::vertex_iterator VertexIter;
typedef boost::graph_traits<Graph>::edge_iterator EdgeIter;
using PathType = vector<vector<VertexDesc>>;
using namespace torch::indexing;


PathType find_path(int start, VertexDesc goal, const Graph& _graph)  {
    vector<VertexDesc> p(num_vertices(_graph));
    vector<int> d(num_vertices(_graph));
    VertexDesc s = vertex(start, _graph);

    //auto idmap = boost::get(boost::vertex_index, _graph);

    //vector<VertexDesc> predecessors(boost::num_vertices(_graph), Graph::null_vertex());
    //vector<int> distances(boost::num_vertices(_graph));
    //boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, _graph);
    dijkstra_shortest_paths(_graph, s, boost::predecessor_map(&p[0]).distance_map(&d[0]));

    // extract path
    VertexDesc current = goal;
    PathType path { };
    //PathType edge_path;

    do {
        auto const pred = p.at(current);

        //std::cout << "extract path: " << current << " " << _graph[current].coord << " <- " << pred << std::endl;

        if(current == pred)
            break;
        path.push_back({current, pred});
        current = pred;


    } while(current != start);

    //std::reverse(path.begin(), path.end());

    return path;
}


bool mycmp(Edge a, Edge b){
    float a_val = std::max(a.first, a.second);
    float b_val = std::max(b.first, b.second);
    if (a_val < b_val)
        return true;
    else if(std::abs(a_val - b_val) < EPS)
        return a.second - b.second;
    return false;
}

bool vcmp(const vector<VertexDesc>& a, const vector<VertexDesc>& b){
    float a_val = std::max(a[0], a[1]);
    float b_val = std::max(b[0], b[1]);
    if (a_val < b_val)
        return true;
    else if(std::abs(a_val - b_val) < EPS)
        return a[1] - b[1];
    return false;
}

void print_pairs(vector<Pers> &ed) {
    for (auto e: ed) {
        std::cout << e.first << " (" << e.second.first << "," << e.second.second << ")" << std::endl;
    }
}

void print_path(const PathType& p){
    for(auto v : p){
        std::cout << "(" << v[0] << "," << v[1] << ")" << std::endl;
    }
}

struct CustomEdgeCompare {
    Tensor vert_fil;
    CustomEdgeCompare(const Tensor &vertex_filtration){
        this->vert_fil = vertex_filtration;
    }

    bool operator()(const vector<VertexDesc>& a, const vector<VertexDesc>& b) const {
        double a_val = std::max(vert_fil[a[0]].item<double>(), vert_fil[a[1]].item<double>());
        double b_val = std::max(vert_fil[b[0]].item<double>(), vert_fil[b[1]].item<double>());
        if (a_val < b_val)
            return true;
        else if(std::abs(a_val - b_val) < EPS)
            return (vert_fil[a[1]] - vert_fil[b[1]]).item<int>();
        return false;
    }
};
vector<Tensor> compute_pd0(const Tensor & vertex_filtration,
                           const vector<Tensor> & boundary_info){
    auto num_nodes = vertex_filtration.size(0);
    union_find uf = union_find(num_nodes);
    Tensor tensor_edges = boundary_info[0];
    Tensor edge_val = std::get<0>(torch::max(vertex_filtration.index({tensor_edges}), 1));
    Tensor sorted_edge_indices = edge_val.argsort(-1,false);
    const Tensor sorted_edges = tensor_edges.index({sorted_edge_indices});
    edge_val = edge_val.index({sorted_edge_indices});
    auto num_edges = sorted_edges.size(0);
    vector<Tensor> pd_0;

    for(auto i = 0; i < num_edges; i++){
        auto e = sorted_edges[i];
        auto e_val = edge_val[i];
        int u = e[0].item<int>();
        int v = e[1].item<int>();
        int root_u = uf.find(u);
        int root_v = uf.find(v);
        if(root_u == root_v){
            continue;
        }
        int root = root_u;
        int merged = root_v;
        if (vertex_filtration[root].item<double>() > vertex_filtration[merged].item<double>())
            std::swap(root, merged);
        else if (std::abs(vertex_filtration[root].item<double>() - vertex_filtration[merged].item<double>()) < EPS) {
            if (root > merged)
                std::swap(root, merged);
        }
        auto merged_val = vertex_filtration[merged];
        //std::cout << "M: " << merged_val.item<double>() << " E: " << e_val.item<double>()<< std::endl;
        Tensor pd_pair = torch::stack({merged_val, e_val});
        pd_0.emplace_back(pd_pair);
        uf.link(root, merged);
    }
    return pd_0;

}

vector<vector<Tensor>> extended_filt_persistence_single(const Tensor & vertex_filtration,
                                                        const vector<Tensor> & boundary_info){
    vector<vector<Tensor>> pd;
    auto num_nodes = vertex_filtration.size(0);
    union_find uf = union_find(num_nodes);
    Graph g;
    vector<size_t> pos_edge_index;
    vector<Tensor> pd_0_up = compute_pd0(vertex_filtration, boundary_info);
    vector<Tensor> pd_0_down, pd_1_rel, pd_1_ext;
    Tensor tensor_edges = boundary_info[0];
    Tensor edge_val = std::get<0>(torch::min(vertex_filtration.index({tensor_edges}), 1));
    Tensor sorted_edge_indices = edge_val.argsort(-1, true);
    const Tensor sorted_edges = tensor_edges.index({sorted_edge_indices});
    edge_val = edge_val.index({sorted_edge_indices});
    auto num_edges = sorted_edges.size(0);
    for(auto i = 0; i < num_edges; i++){
        auto e = sorted_edges[i];
        auto e_val = edge_val[i];
        int u = e[0].item<int>();
        int v = e[1].item<int>();
        int root_u = uf.find(u);
        int root_v = uf.find(v);
        if(root_u == root_v){
            pos_edge_index.push_back(i);
            continue;
        }
        boost::add_edge(u, v, 1, g);
        int root = root_u;
        int merged = root_v;
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
    //pd.push_back(pd_0);
    Tensor min_max = torch::stack({vertex_filtration.min(), vertex_filtration.max()});
    pd_1_rel.push_back(min_max);
    CustomEdgeCompare cmp = CustomEdgeCompare(vertex_filtration);
    for(auto ii : pos_edge_index){
        auto pos_edge = sorted_edges[ii];
        auto pos_edge_val = edge_val[ii];
        int u = pos_edge[0].item<int>();
        int v = pos_edge[1].item<int>();
        PathType p = find_path(u, v, g);
        //std::cout << "Edge: " << u << " " << v << std::endl;
        //print_path(p);
        auto result = *std::max_element(p.begin(), p.end(), cmp);
        boost::remove_edge(result[0], result[1], g);
        //std::cout << "Removed edge: " << result[0] << " " << result[1] << std::endl;
        boost::add_edge(u, v, 1, g);
        auto cut_edge_tensor = torch::from_blob(result.data(), {2}, torch::TensorOptions().dtype(at::kLong));
        auto cut_edge_val = vertex_filtration.index({cut_edge_tensor}).max();
        //std::cout << "CE " << cut_edge_val.item<double>() << " AE " << pos_edge_val.item<double>() << std::endl;
        auto pers_pair = torch::stack({cut_edge_val, pos_edge_val});
        pd_1_ext.push_back(pers_pair);
    }
    //pd.push_back(pd_1);
    pd.push_back(pd_0_up);
    pd.push_back(pd_0_down);
    pd.push_back(pd_1_rel);
    pd.push_back(pd_1_ext);
    return pd;

}
vector<vector<vector<Tensor>>> extended_filt_persistence_batch(const vector<std::tuple<Tensor, vector<Tensor>>> & batch){
auto futures = vector<std::future<vector<vector<Tensor>>>>();
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
auto ret = vector<vector<vector<Tensor>>>();
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
}

