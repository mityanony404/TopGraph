# TopGraph

This is the repository for the manuscript titled "**TopGraph**: A Topological Graph Neural Network Layer". This manuscript is submitted to ICML 2022 and under review.

## TopGraph as a regularizer

### Dependencies
Install anaconda and create a virtual-env
1. conda install -n venv python=3.8
2. conda activate venv
3. conda install pytorch=1.7.1 cudatoolkit=11.0 -c pytorch
4. Install pytorch-geometric and its dependencies


	torch-cluster 1.5.9
	
	torch-scatter 2.0.7 
	
	torch-sparse  0.6.9 
	
	torch-spline-conv 1.2.1

	torch-geometric 1.4.1

5. Install ninja. 
    ninja 1.10.2
    

### Run Experiments
1. Run <code>python write_training_cfgs.py </code>to generate training_cfgs.txt file.
2. <code>python learn_filtration_graph.py --dataset IMDB-BINARY --model TopGraph-PI</code>

Availlable options for --dataset flag are:

<code>IMDB-BINARY,IMDB-MULTI,REDDIT-BINARY,DD,ENZYMES</code>

--model options are:

TopGraph-PI: Persistence Image based classifier

TopGraph-RH: Rational-hat based claassifier

GIN: 1-GIN graph classifier

GFL: Model based on "Graph Filtration Learning". This code is from https://github.com/c-hofer/graph_filtration_learning. **Note**: To run GFL you need to follow instructions of the above repository. GFL code is from the authors of https://github.com/c-hofer/graph_filtration_learning. *I do not claim any authorship*. 


## Mesh filtration 

### Dependencies
1. Create a new conda environment.
<code>
	conda create -n meshfil python=3.9
	conda activate meshfil
	</code>
2. Install pytorch<=1.9.0, pytorch-geometric and its dependencies. I used the following specifications. 
<code>
pytorch 1.9.0 
pytorch-cluster 1.5.9 
pytorch-scatter 2.0.9 
pytorch-sparse  0.6.12 
pytorch-spline-conv 1.2.1
</code>

3. Create <code>datasets</code> dir inside Mesh_filtration_learning. Download the dataset from this [link](https://www.dropbox.com/sh/0qa1qiwx41bwubc/AAB6tECbx-bNuA4m6T7fu4rZa) . Unzip the zip. Datasets are SHREC_16 and ModelNet10.

### Run Experiments 
1. <code> python learn_filtration.py --lr 1e-3 --num_epochs 100 --dataset shrec_16 </code>
Options for <code> --dataset </code> are <code> shrec_16 </code> and <code> ModelNet10</code>.

### PD-MeshNet
1. PD-MeshNet is authored by authors of this repo https://github.com/MIT-SPARK/PD-MeshNet. *I do not claim any authorship/ownership*. 
2. Install dependencies as mentioned in the [PD-MeshNet repo](https://github.com/MIT-SPARK/PD-MeshNet). 
3. Clone PD-MeshNet repo. Suppose the root folder is <code>root</code>. Now copy the *contents that is inside of the PD-meshnet* folder to the source folder. 

## Some Notes
1. The code ran successfully with <code>cuda 10.2</code> and <code>gcc/6.5.0</code>.
2. The code is <code>cuda </code> compatible. But the extended persistence is computed in cpu. So it may be slower for huge datasets.

3. If you like the repo please follow [https://github.com/c-hofer/graph_filtration_learning](https://github.com/c-hofer/graph_filtration_learning) and [https://github.com/MIT-SPARK/PD-MeshNet](https://github.com/MIT-SPARK/PD-MeshNet). The repos contain awesome works.

![alt text][id]  
  
[id]: /url/to/img.jpg "Title"