import os

from torch.utils.cpp_extension import load
from typing import Tuple, List
from torch import Tensor
import os

__module_file_dir = os.path.dirname(os.path.realpath(__file__))
__cpp_src = os.path.join(__module_file_dir, "extended_pers.cpp")
try:
    extended_pers = load(name="extended_cpp", sources=[__cpp_src])
except Exception as ex:
    print(f"Error in {__file__}. Failed jit compilation. Maybe your CUDA environment is messed up?")
    print(f"Error was {ex}")


def extended_persistence(ph_inp: List[Tuple[Tensor, List[Tensor]]]):
    return extended_pers.extended_persistence_batch(ph_inp)


# v = torch.tensor([0, 1, 2, 3], dtype=torch.float)
# e = torch.tensor([[0, 1], [0, 2], [1, 3], [1, 2], [2, 3], [0, 3]], dtype=torch.long)
# ph_inp = [(v, [e])]
# ex_bars = extended_pers.extended_persistence_batch(ph_inp)
# print("------------TEST 1-----------------")
# for x in ex_bars:
#     print(ex_bars)
#
# st= gudhi.SimplexTree()
# st.insert([0, 1])
# st.insert([0, 2])
# st.insert([1, 3])
# st.insert([1, 2])
# st.insert([2, 3])
# st.insert([0, 3])
# st.assign_filtration([0], 0)
# st.assign_filtration([1], 1)
# st.assign_filtration([2], 2)
# st.assign_filtration([3], 3)
# _= st.make_filtration_non_decreasing()
# st.extend_filtration()
# dgms= st.extended_persistence(min_persistence=-1)
#
# print("gudhi dgms: ", dgms)
# print("num barcodes link-cut-tree: ", len(ex_bars[0][0])+len(ex_bars[0][1])+len(ex_bars[0][2])+len(ex_bars[0][3]))
# print("num barcodes gudhi: ", len(dgms[0])+len(dgms[1])+len(dgms[2])+len(dgms[3]))
#
# print("------------TEST 2-----------------")
#
#
# extended_pers = load(name="extended_cpp", sources=["extended_pers.cpp"])
# v = torch.tensor([0, 1, 2, 3, 4, 5, 0.5, 4.5, 8, 9], dtype=torch.float)
# e = torch.tensor([[0, 1], [1, 2], [1, 3], [2, 4], [3, 4], [4, 5], [2, 6], [3, 7], [8,9]], dtype=torch.long)
# ph_inp = [(v, [e])]
# ex_bars = extended_pers.extended_persistence_batch(ph_inp)
# print(ex_bars)
# st= gudhi.SimplexTree()
# st.insert([0, 1])
# st.insert([1, 2])
# st.insert([1, 3])
# st.insert([2, 4])
# st.insert([3, 4])
# st.insert([4, 5])
# st.insert([2, 6])
# st.insert([3, 7])
# #added new:
# st.insert([8,9])
#
# st.assign_filtration([0], 0)
# st.assign_filtration([1], 1)
# st.assign_filtration([2], 2)
# st.assign_filtration([3], 3)
# st.assign_filtration([4], 4)
# st.assign_filtration([5], 5)
# st.assign_filtration([6], 0.5)
# st.assign_filtration([7], 4.5)
# st.assign_filtration([8], 8)
# st.assign_filtration([9], 9)
#
# _ = st.make_filtration_non_decreasing()
# st.extend_filtration()
# dgms = st.extended_persistence(min_persistence=-1)
# print("gudhi dgms: ", dgms)
# print("num barcodes link-cut-tree: ", len(ex_bars[0][0])+len(ex_bars[0][1])+len(ex_bars[0][2])+len(ex_bars[0][3]))
# print("num barcodes gudhi: ", len(dgms[0])+len(dgms[1])+len(dgms[2])+len(dgms[3]))
#
# print("------------TEST 3-----------------")
#
#
# extended_pers = load(name="extended_cpp", sources=["extended_pers.cpp"])
# v = torch.tensor([0, 1, 2, 3, 4, 5, 0.5, 4.5, 8, 9], dtype=torch.float)
# e = torch.tensor([[0, 1], [1, 2], [2,6], [8,9]], dtype=torch.long)
# ph_inp = [(v, [e])]
# ex_bars = extended_pers.extended_persistence_batch(ph_inp)
# print(ex_bars)
# st= gudhi.SimplexTree()
# st.insert([0, 1])
# st.insert([1, 2])
# st.insert([2, 6])
# #added new:
# st.insert([8,9])
#
# st.assign_filtration([0], 0)
# st.assign_filtration([1], 1)
# st.assign_filtration([2], 2)
# # st.assign_filtration([3], 3)
# # st.assign_filtration([4], 4)
# # st.assign_filtration([5], 5)
# st.assign_filtration([6], 0.5)
# # st.assign_filtration([7], 4.5)
# st.assign_filtration([8], 8)
# st.assign_filtration([9], 9)
#
# _ = st.make_filtration_non_decreasing()
# st.extend_filtration()
# dgms = st.extended_persistence(min_persistence=-1)
# print("gudhi dgms: ", dgms)
# print("num barcodes link-cut-tree: ", len(ex_bars[0][0])+len(ex_bars[0][1])+len(ex_bars[0][2])+len(ex_bars[0][3]))
# print("num barcodes gudhi: ", len(dgms[0])+len(dgms[1])+len(dgms[2])+len(dgms[3]))
#
# print("------------BIG TEST-----------------")
#
# extended_pers = load(name="extended_cpp", sources=["extended_pers.cpp"])#_11-28-2021SOHAM
# clique_vertices= range(300)
# v= torch.tensor(clique_vertices, dtype=torch.float)
# clique_edges= [[a,b] for a in clique_vertices for b in clique_vertices if a<b]
# e= torch.tensor(clique_edges, dtype= torch.long)
# ph_inp = [(v, [e])]
# t1= time.time();
# ex_bars = extended_pers.extended_persistence_batch(ph_inp)
# t2=time.time();
#
# print(ex_bars)
# t3=time.time()
# st= gudhi.SimplexTree()
# for (a,b) in clique_edges:
#     st.insert([a,b])
# for i in range(300):
#     st.assign_filtration([i], i)
#
#
# _ = st.make_filtration_non_decreasing()
# st.extend_filtration()
# dgms = st.extended_persistence(min_persistence=-1)
# t4=time.time()
# print("gudhi dgms: ", dgms)
# print("num barcodes link-cut-tree: ", len(ex_bars[0][0])+len(ex_bars[0][1])+len(ex_bars[0][2])+len(ex_bars[0][3]))
# print("num barcodes gudhi: ", len(dgms[0])+len(dgms[1])+len(dgms[2])+len(dgms[3]))
# print("TIME FOR EXTENDED PERSISTENCE: ", t2-t1);
# print("TIME FOR GUDHI: ", t4-t3);
#
#
#
