## Training
- We used the hyperparameters specified in `main.sh`
- Edit `main.sh` to change the paths
- To train, run: `./main.sh`
## Computing features
- To compute and save ConvNet features after training given the last checkpoint, use: `python compute_features_and_clusters.py --data <DATA_DIR> --exp_dir <OUTPUT_DIR>`. This will also save processed features after applying PCA and normalization. It also computes pairwise L2 distances between features. Optionally, you can get more information about the k-means clustering done during training (you can safely ignore this step, set `--get_kmeans_info` to 0).
