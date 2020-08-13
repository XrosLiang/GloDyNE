# About dataset
Here we provide three relatively small datasets used in the experiments, due to the space limit in Github. 
For other three datasets, one can follow the paper "GloDyNE: Global Topology Preserving Dynamic Network Embedding" to construct their dynamic networks.

Concretely, the format of an input dynamic network is prepared as follows: <br>
1) create an empty dynamic network as a empty python list, called DynG; <br>
2) use Networkx to build the graph based on the edge steams from time step t-k (we set t-k=0 in this work) to t, called Gt; <br>
3) append the current snapshot Gt to the dynamic network DynG; <br>
4) repeat 2) and 3) as time goes on... <br>
5) finally, use Pickle to store DynG as a .pkl file <br>
After 1-5, we further take out the largest connected component based on the initial snapshot, and then develop it to the last snapshot. <br>
We will release our preprocessing files later. If you need them now, please contact me via Email.

Note that, the links to the datasets as indicated in the paper are not available. Please instead refer to http://konect.cc/networks/
