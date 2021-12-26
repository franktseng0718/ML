import numpy as np
import cv2
from scipy.spatial.distance import pdist,squareform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from array2gif import write_gif

colormap= np.random.choice(range(256),size=(100,3))

def openImage(path):
	image = cv2.imread(path)
	H, W, C = image.shape
	image_flat = np.zeros((W * H, C))
	for h in range(H):
		image_flat[h * W:(h + 1) * W] = image[h]

	return image_flat,H,W
def precomputed_kernel(X, gamma_s, gamma_c):
    n=len(X)
    # S(x) spacial information
    S=np.zeros((n,2))
    for i in range(n):
        S[i]=[i//100,i%100]
    print(pdist(S,'sqeuclidean').shape)
    K=squareform(np.exp(-gamma_s*pdist(S,'sqeuclidean')))*squareform(np.exp(-gamma_c*pdist(X,'sqeuclidean')))
    print(K.shape)
    return K


def visualize(C,k,H,W):
    '''
    @param C: (10000) belonging classes ndarray
    @param k: #clusters
    @param H: image_H
    @param W: image_W
    @return : (H,W,3) ndarray
    '''
    colors= colormap[:k,:]
    res=np.zeros((H,W,3))
    for h in range(H):
        for w in range(W):
            res[h,w,:] = colors[C[h*W+w]]

    return res.astype(np.uint8)
def kmeans(CLUSTER_NUM, Gram, H, W):
    # kmeans++ init
    Cluster = np.zeros((CLUSTER_NUM, Gram.shape[1]))
    Cluster[0]=Gram[np.random.randint(low=0,high=Gram.shape[0],size=1),:]
    for c in range(1,CLUSTER_NUM):
            Dist=np.zeros((len(Gram),c))
            for i in range(len(Gram)):
                for j in range(c):
                    Dist[i,j]=np.sqrt(np.sum((Gram[i]-Cluster[j])**2))
            Dist_min=np.min(Dist,axis=1)
            sum=np.sum(Dist_min)*np.random.rand()
            for i in range(len(Gram)):
                sum-=Dist_min[i]
                if sum<=0:
                    Cluster[c]=Gram[i]
                    break
    # kmeans++
    diff = 1e9
    eps = 1e-9
    count = 1
    # Classes of each Xi
    C=np.zeros(len(Gram),dtype=np.uint8)
    segments=[]
    while diff > eps:
        # E-step
        for i in range(len(Gram)):
            dist=[]
            for j in range(CLUSTER_NUM):
                dist.append(np.sqrt(np.sum((Gram[i]-Cluster[j])**2)))
            C[i]=np.argmin(dist)
        
        #M-step
        New_Mean=np.zeros(Cluster.shape)
        for i in range(CLUSTER_NUM):
            belong=np.argwhere(C==i).reshape(-1)
            for j in belong:
                New_Mean[i]=New_Mean[i]+Gram[j]
            if len(belong)>0:
                New_Mean[i]=New_Mean[i]/len(belong)

        diff = np.sum((New_Mean - Cluster)**2)
        Cluster=New_Mean
        # visualize
        segment = visualize(C, CLUSTER_NUM, H, W)
        segments.append(segment)
        print('iteration {}'.format(count))
        for i in range(CLUSTER_NUM):
            print('k={}: {}'.format(i + 1, np.count_nonzero(C == i)))
        print('diff {}'.format(diff))
        print('-------------------')
        cv2.imshow('', segment)
        cv2.waitKey(1)
    return C, segments

def plot_eigenvector(xs,ys,zs,C):
    '''
    only for 3-dim datas
    @param xs: (#datapoint) ndarray
    @param ys: (#datapoint) ndarray
    @param zs: (#datapoint) ndarray
    @param C: (#datapoint) ndarray, belonging class
    '''
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    markers=['o','^','s']
    for marker,i in zip(markers,np.arange(3)):
        ax.scatter(xs[C==i],ys[C==i],zs[C==i],marker=marker)

    ax.set_xlabel('eigenvector 1st dim')
    ax.set_ylabel('eigenvector 2nd dim')
    ax.set_zlabel('eigenvector 3rd dim')
    plt.show()