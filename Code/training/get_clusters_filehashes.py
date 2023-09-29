import argparse
import pickle 
import os

#log the filehashes of each cluster (the K-means produced by training)

parser = argparse.ArgumentParser(description='Inspect clusters')
#file containing the filehashes of the training subset, line i is the filehash of the image having index i 
parser.add_argument('--filehashes_file', default='filehashes_by_idx.txt', type=str, metavar='PATH',
                        help='path to filehashes')
#last cluster assignment 
parser.add_argument('--clusters_file', default='../exp10_folder/clusters', type=str, metavar='PATH',
                        help='path to the clusters file')
#dir path to save the filenames
parser.add_argument('--saved_filehashes_dir', default='../exp10_folder/filehashes_per_clusters/', type=str, metavar='PATH',
                        help='path to the dir to save file hashes')                        
                        
args = parser.parse_args()
             
clustering_file_path = args.clusters_file
#read filehashes of all dataset 
with open(args.filehashes_file, 'r') as f:
    all_filehashes = f.readlines()

if not os.path.exists(args.saved_filehashes_dir):
    os.makedirs(args.saved_filehashes_dir)

#given cluster i, get and save filehashes of all instances in the cluster 
def print_filenames_of_oneCluster(all_filehashes, final_epoch_cluster_assignment, cluster_idx):
    oneCluster = final_epoch_cluster_assignment[cluster_idx]
    with open(args.saved_filehashes_dir+str(cluster_idx)+'.txt', 'w') as f:
        for i in range(0,len(oneCluster)):
            filehash = all_filehashes[oneCluster[i]].rstrip()
            f.write(filehash+'\n')
            
with open(clustering_file_path, 'rb') as pickle_file:
    clusters = pickle.load(pickle_file)
final_epoch_cluster_assignment = clusters[-1]

for i in range(0,len(final_epoch_cluster_assignment)):
    print_filenames_of_oneCluster(all_filehashes,final_epoch_cluster_assignment,i)
