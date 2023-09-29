## Training
- We used the hyperparameters specified in `main.sh`
- Edit `main.sh` to change the paths
- To train, run: `./main.sh`
## Computing features
- To compute and save ConvNet features after training given the last checkpoint, use: `python compute_features_and_clusters.py --data <DATA_DIR> --exp_dir <OUTPUT_DIR>`. This will also save processed features after applying PCA and normalization. It also computes pairwise L2 distances between features. Optionally, you can get more information about the k-means clustering done during training (you can safely ignore this step, set `--get_kmeans_info` to 0).
## Get clusters 
- To output which data point belong to which cluster, run: 
  - `python get_clusters_filehashes.py --clusters_file <OUTPUT_DIR>/clusters --saved_filehashes_dir <OUTPUT_DIR>/filehashes_per_clusters/ --filehashes_file filehashes_by_idx.txt`
- This uses the clustering information saved in the output dir `--clusters_file` and saves the filehashes (data points) per cluster in `--saved_filehashes_dir`. It also needs a precomputed `--filehashes_file` (text file with all images' names saved with the same order it is read by pytorch data loader).  
