## Guide for Manually-aided Clustering

This repository contains the scripts used for the two steps of our manually-aided clustering procedure (described in Section 3).
These scripts are shared to foster future research. These files allow reproducing the clustering experiment (we do not
claim repeatability).

Our approach is based on [DeepCluster](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf).

### Steps of clustering procedure
1. `Code/compute_features_and_clusters.py` takes as input the filenames of the images to cluster via the parameter `--filehashes_file`, while `--data` specifies their filesystem location. We use the 20,671 unique screenshots listed in `900Clusters/filehashes_by_idx.txt`. The parameter `--exp_dir` specifies where to store the class labels returned by DeepCluster.
2. `Code/get_clusters_filehashes.py` can be used to obtain as many files as clusters, where each cluster-file lists the filenames of the images in the cluster. An example (zipped) is `900Clusters/filehashes_per_clusters_900clusters.zip`.
    * In our work, we obtained at this stage 635 homogeneous clusters, covering 90% of the input data. We determined this result thanks to manual inspection of 9000 samples (10 screenshots for each of the 900 cluster).

3. Next, we selected the embeddings of the screenshots in non-homogeneous clusters and grouped them together via DBSCAN. The code using DBSCAN is reported in `Code/DBSCAN/run_dbscan.ipynb`. We share the pairwise distances between screenshots via an outside service as the size of the file exceeds the size allowed by GitHub ([upload-in-progress]()).
    * We used the parameters ε = 50 and a minimum number of samples per cluster of 3.
    * In our work, we further inspected the results of DBSCAN before obtaining the 80 clusters. 
    

### Setup
The code runs in a Conda environment.
 1. You can download and install Anaconda from official sources (e.g., [link](https://www.anaconda.com/download)) and follow the official documentation.
 2. Once conda is installed, create an ad-hoc environment and install the necessary dependencies: `conda env create -f Code/environment.yaml`
 3. The training was performed on a server with multiple GPUs available. On the other hand, DBSCAN can be run locally.
