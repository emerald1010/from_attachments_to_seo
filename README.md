# From Attachments to SEO: Click Here to Learn More about Clickbait PDFs!
This repository contains the code and documentation for our ACSAC 2023 paper _From Attachments to SEO: Click Here to Learn More about Clickbait PDFs!_.
Due to file size constraints, part of the dataset and  the code are uploaded on Kaggle:

* [Data](https://www.kaggle.com/datasets/emerald101/from-attachments-to-seo/data)
* [Code](https://www.kaggle.com/code/emerald101/artifact-code-for-paper-from-attachments-to-seo)

The data we share ("_artifact_") is made of multiple files and proves most of the results presented in our paper. We recommend inspecting the data and the code (a Jupyter notebook) on the Kaggle platform, as this allows to run the scripts without any setup nor file download.

### Setup (run Kaggle code locally)
All the code in the notebook can be executed locally. We ran the experiments on a Linux Ubuntu 18.0 laptop using Python 3.6.9.
We also use a minimal set of libraries:
- Pandas 1.3.3 or higher
- Numpy 1.21.2
- Matplotlib 3.4.3


## Guide for Manually-aided Clustering
This repository contains the scripts used for the two steps of our manually-aided clustering procedure (described in Section 3). The first step uses a deep learning model based on [DeepCluster](https://openaccess.thecvf.com/content_ECCV_2018/papers/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.pdf), while the second step uses the clustering algorithm [DBSCAN](https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf).
These scripts are shared to foster future research. These files allow reproducing the clustering experiment (we do not
claim repeatability).

### Setup
 
#### Necessary hardware 
We used a TITAN RTX GPU to train the model. On the other hand, DBSCAN can be run locally.

#### Software dependencies
We used Ubuntu 19.01 for the development of the deep learning model. The dependencies for this code are managed via Conda.
 1. You can download and install Anaconda from official sources (e.g., [link](https://www.anaconda.com/download)) and follow the official documentation.
 2. Once conda is installed, create an ad-hoc environment and install the necessary dependencies using: `conda create -n deepcluster python=3.7 faiss-gpu numpy scipy pillow torchvision=0.8.2 pytorch=1.7.1 cudatoolkit=10.1 imagehash scikit-learn -c pytorch -c conda-forge`
 3. DBSCAN is implemented within the sklearn library. We used version 1.0.2.

### Training
- We used the hyperparameters specified in `Code/main.sh`.
- Edit `Code/main.sh` to change the paths.
- To train, run: `Code/main.sh`.
- The images that will be used to train should be placed as flattened files under `<DATASET_ROOT_DIR>/<DATASET_DIR>`. The training scripts takes the path `<DATASET_ROOT_DIR>'. This is to facilitate data loading by pytorch dataloader. 

### Computing features
To compute and save ConvNet features after training given the last checkpoint, use:

`python Code/compute_features_and_clusters.py --data <DATA_DIR> --exp_dir <OUTPUT_DIR>`

This will also save processed features after applying PCA and normalization. It also computes pairwise L2 distances between features. Optionally, you can get more information about the k-means clustering done during training (you can safely ignore this step, set `--get_kmeans_info` to 0).

### Get clusters 
Once the training is completed, to output which data point belong to which cluster, run:

`python Code/get_clusters_filehashes.py --clusters_file <OUTPUT_DIR>/clusters --saved_filehashes_dir <OUTPUT_DIR>/filehashes_per_clusters/ --filehashes_file Code/filehashes_by_idx.txt`

This uses the clustering information saved in the output dir `--clusters_file` and saves the filehashes (data points) per cluster in `--saved_filehashes_dir`. It also needs a list of `--filehashes_file` (text file with all images' names saved with the same order it is read by pytorch data loader).  

- `Code/get_clusters_filehashes.py` can be used to obtain as many files as clusters, where each cluster-file lists the filenames of the images in the cluster. An example (zipped) is `900Clusters/filehashes_per_clusters_900clusters.zip`.
* In our work, we obtained at this stage 635 homogeneous clusters, covering 90% of the input data. We determined this result thanks to manual inspection of 9000 samples (10 screenshots for each of the 900 cluster).

- All results from training and clustering can be found under `900clusters`. The subset of files we used is in `filehashes_by_idx.txt`.
  
### DBSCAN
Finally, we selected the embeddings of the screenshots in non-homogeneous clusters and grouped them together via DBSCAN. The code using DBSCAN is reported in `Code/DBSCAN/run_dbscan.ipynb`. We share the pairwise distances between screenshots via an outside service as the size of the file exceeds the size allowed by GitHub ([link](https://www.kaggle.com/datasets/emerald101/from-attachments-to-seo?select=pairwise_distances_ConvNet.npy)).
* We used the parameters ε = 50 and a minimum number of samples per cluster of 3.
* In our work, we further inspected the results of DBSCAN before obtaining the 80 clusters. 
    


