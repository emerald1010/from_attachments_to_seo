# From Attachments to SEO: Click Here to Learn More about Clickbait PDFs!
This repository contains the code and documentation for our ACSAC 2023 paper _From Attachments to SEO: Click Here to Learn More about Clickbait PDFs!_.
Due to file size constraints, part of the dataset and  the code are uploaded on Kaggle:

* [Data](https://www.kaggle.com/datasets/emerald101/from-attachments-to-seo/data)
* [Code](https://www.kaggle.com/code/emerald101/artifact-code-for-paper-from-attachments-to-seo)

The data we share ("_artifact_") is made of multiple files and proves most of the results presented in our paper. We recommend inspecting the data and the code (a Jupyter notebook) on the Kaggle platform, as this allows to run the scripts without any setup nor file download.

### Setup (to run locally)
All the code in the notebook can be executed locally. We ran the experiments on a Linux Ubuntu 18.0 laptop using Python 3.6.9.
We also use a minimal set of libraries:
- Pandas 1.3.3 or higher
- Numpy 1.21.2
- Matplotlib 3.4.3

This repository contains additional code not shared on Kaggle, which we used to train the deep learning model used in our paper. This procedure, presented in Section 3 of the paper has a more complex setup and relies on specialized hardware. We detail it below.

## Guide for Manually-aided Clustering

This repository contains the scripts used for the two steps of our manually-aided clustering procedure (described in Section 3).
These scripts are shared to foster future research. These files allow reproducing the clustering experiment (we do not
claim repeatability).

Our approach is based on [DeepCluster](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf).
### Training 

### Steps of clustering procedure
1. `Code/compute_features_and_clusters.py` takes as input the filenames of the images to cluster via the parameter `--filehashes_file`, while `--data` specifies their filesystem location. We use the 20,671 unique screenshots listed in `900Clusters/filehashes_by_idx.txt`. The parameter `--exp_dir` specifies where to store the class labels returned by DeepCluster.
2. `Code/get_clusters_filehashes.py` can be used to obtain as many files as clusters, where each cluster-file lists the filenames of the images in the cluster. An example (zipped) is `900Clusters/filehashes_per_clusters_900clusters.zip`.
    * In our work, we obtained at this stage 635 homogeneous clusters, covering 90% of the input data. We determined this result thanks to manual inspection of 9000 samples (10 screenshots for each of the 900 cluster).

3. Next, we selected the embeddings of the screenshots in non-homogeneous clusters and grouped them together via DBSCAN. The code using DBSCAN is reported in `Code/DBSCAN/run_dbscan.ipynb`. We share the pairwise distances between screenshots via an outside service as the size of the file exceeds the size allowed by GitHub ([link](https://www.kaggle.com/datasets/emerald101/from-attachments-to-seo?select=pairwise_distances_ConvNet.npy)).
    * We used the parameters ε = 50 and a minimum number of samples per cluster of 3.
    * In our work, we further inspected the results of DBSCAN before obtaining the 80 clusters. 
    

### Setup
We used Ubuntu 19.01 for the development of this code, and a TITAN RTX GPU to train the model.

The code runs in a Conda environment.
 1. You can download and install Anaconda from official sources (e.g., [link](https://www.anaconda.com/download)) and follow the official documentation.
 2. Once conda is installed, create an ad-hoc environment and install the necessary dependencies using: `conda create -n deepcluster python=3.7 faiss-gpu numpy scipy pillow torchvision=0.8.2 pytorch=1.7.1 cudatoolkit=10.1 imagehash scikit-learn -c pytorch -c conda-forge`
 3. The training was performed on a GPU server. On the other hand, DBSCAN can be run locally.
