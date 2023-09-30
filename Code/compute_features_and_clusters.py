import argparse
import os
import pickle
import time
import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import clustering2
import models
from util import AverageMeter

#compute features of the trained ConvNet .. compute pairwise l2 distances between the features of the convNet
#if args.get_kmeans_info is 1: it will compute the k-means centroids and labels. 
    ##given the k-means clustering, compute and save information about the clusters (centroids, labels as array).

parser = argparse.ArgumentParser(description='Inspect clusters')
#where the checkpoints are stored and where to save the computed features 
parser.add_argument('--exp_dir', default='../exp10_folder', type=str, metavar='PATH',
                        help='path to the clusters file')
#cluster assignment produced by k-means, should be saved in args.exp_dir
parser.add_argument('--clusters_file', default='clusters', type=str, metavar='PATH',
                        help='name of the clusters file')
#last checkpoint, should be saved in args.exp_dir                        
parser.add_argument('--checkpt', default='checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to the model checkpt')
# the dataset folder should be a root folder (inside it there should be a subfolder, containing the images                        
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
#we don't need sobel filtering 
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
#currently we only have alexnet
parser.add_argument('--arch', '-a', type=str, metavar='ARCH', choices=['alexnet'], default='alexnet', help='CNN architecture (default: alexnet)')
parser.add_argument('--batch', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--verbose', action='store_true', help='chatty')
                        
#if it is not needed, set to 0
parser.add_argument('--get_kmeans_info', default=1, type=int, help='if 1, compute and save info about the k-means clusters')
args = parser.parse_args()

### load checkpoint ###
model = models.__dict__[args.arch](sobel=args.sobel)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
fd = int(model.top_layer.weight.size()[1])

#remove last classification layer
model.top_layer = None
model.features = torch.nn.DataParallel(model.features)
model.cuda()
cudnn.benchmark = True
    
checkpoint = torch.load(os.path.join(args.exp_dir,args.checkpt))
delete_keys = [key for key in checkpoint['state_dict'] if 'top_layer' in key]
for key in delete_keys:
    del checkpoint['state_dict'][key]
model.load_state_dict(checkpoint['state_dict'])

### load cluster assignment ###
with open(os.path.join(args.exp_dir,args.clusters_file), 'rb') as pickle_file:
    clusters = pickle.load(pickle_file)
final_epoch_cluster_assignment = clusters[-1]

### load dataset ###
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tra = [transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize]
dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, pin_memory=True)

def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            aux = model(input_var).data.cpu().numpy()
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')
        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.verbose and (i % 100) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features

# these are the raw features of the VGG 
features = compute_features(dataloader, model, len(dataset))

#the processed feature are with applying PCA and normalization 
processed_features = clustering2.preprocess_features(features)
np.save(os.path.join(args.exp_dir,'features_ConvNet'),features)
np.save(os.path.join(args.exp_dir,'processed_features'),processed_features)

## compute pairwise distances between features 
pairwise_feature_distances = np.zeros((features.shape[0],features.shape[0]))
for i in range(0,features.shape[0]):
    for j in range(i+1,features.shape[0]):
        pairwise_feature_distances[i,j] = np.linalg.norm(features[i]-features[j],ord=2)
        pairwise_feature_distances[j,i] = pairwise_feature_distances[i,j]
np.save(os.path.join(args.exp_dir,'pairwise_distances_ConvNet'),pairwise_feature_distances)

#if you want to compute the features only (e.g. for more data other than the training data), pass 0
if args.get_kmeans_info == 1:
    # should contain the cluster assignments 
    labels = np.zeros(features.shape[0],)

    # save more information about the clustering, centroids, centroid l2 distances.
    centroids_arr = np.zeros((len(final_epoch_cluster_assignment),processed_features.shape[1]))
    centroid_distance = np.zeros((len(final_epoch_cluster_assignment),len(final_epoch_cluster_assignment)))

    #loop through clusters, get the features of the data points belonging to each cluster 
    #the centroids are the mean of the features 
    for i in range(0,len(final_epoch_cluster_assignment)):
        cluster_features = processed_features[final_epoch_cluster_assignment[i]]
        centroids_arr[i,:] = np.mean(cluster_features,axis=0)
    
    #compute l2 distances between centroids
    for i in range(0,len(final_epoch_cluster_assignment)):
        for j in range(i+1,len(final_epoch_cluster_assignment)):
            centroid_distance[i,j] = np.linalg.norm(centroids_arr[i]-centroids_arr[j],ord=2)
            centroid_distance[j,i] = centroid_distance[i,j]
        
    #save centroids, and l2 distances 
    np.save(os.path.join(args.exp_dir,'centroids'),centroids_arr)
    np.save(os.path.join(args.exp_dir,'centroid_distances'),centroid_distance)

    #save labels 
    for i in range(0,len(final_epoch_cluster_assignment)):
        labels[final_epoch_cluster_assignment[i]] = int(i)
    np.save(os.path.join(args.exp_dir,'labels'),labels)
