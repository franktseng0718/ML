# ML hw6 Report
註：pdf檔的GIF不會動

Please ref https://hackmd.io/DjBqaeTDTr-yNy7xOZev_A

---

## Code with detailed explanations
### Part 1
### Initialize
```
CLUSTER_NUM = k
GIF_path = './GIF'
colormap= np.random.choice(range(256),size=(100,3))
```
k值即number of cluster，以下均用k表示
### read image
![](https://i.imgur.com/Tib7Fy5.png)

讓image flat方便計算

### precomputed kernel

![](https://i.imgur.com/JKtO4oR.png)
按規定的kernel建立(10000, 10000)的K
### kmeans++ init
![](https://i.imgur.com/JWanAg3.png)
Kmeans++ init步驟
![](https://i.imgur.com/HUOx0D3.png)

註：D(x)即為Dist_min[x]
### kmeans
![](https://i.imgur.com/YvPFMej.png)
E-step : 根據最短距離算出每個點的Cluster belonging
![](https://i.imgur.com/bEpXpmw.png)
M-step : 根據E-step assign好的cluster belonging計算新的mean

![](https://i.imgur.com/5AQf8kd.png)
為此刻的每個分群填上不同顏色

### Spectral clustering
#### Compute L
```
D=np.diag(np.sum(W,axis=1))
L=D-W
```
計算L

#### Compute eigenvalue, eigenvector

```
if cut == 'ratio':
    eigenvalue,eigenvector=np.linalg.eig(L)
    np.save('{}_eigenvalue_{:.3f}_{:.3f}_unnormalized'.format(path.split('.')[0],gamma_s,gamma_c),eigenvalue)
    np.save('{}_eigenvector_{:.3f}_{:.3f}_unnormalized'.format(path.split('.')[0],gamma_s,gamma_c),eigenvector)
```

取L的eigenvalue, eigenvector並存起來（避免重複計算）

        
```
if cut == 'normalized':
    D_inverse_square_root=np.diag(1/np.diag(np.sqrt(D)))
    L_sym=D_inverse_square_root@L@D_inverse_square_root
    eigenvalue,eigenvector=np.linalg.eig(L_sym)
    np.save('{}_eigenvalue_{:.3f}_{:.3f}_normalized'.format(path.split('.')[0],gamma_s,gamma_c),eigenvalue)
    np.save('{}_eigenvector_{:.3f}_{:.3f}_normalized'.format(path.split('.')[0],gamma_s,gamma_c),eigenvector)      
```
若為normalized cut，則取L_sym的eigenvalue, eigenvector

```
if cut == 'ratio':
    eigenvalue=np.load('{}_eigenvalue_{:.3f}_{:.3f}_unnormalized.npy'.format(path.split('.')[0],gamma_s,gamma_c))
    eigenvector=np.load('{}_eigenvector_{:.3f}_{:.3f}_unnormalized.npy'.format(path.split('.')[0],gamma_s,gamma_c))
if cut == 'normalized':
    D_inverse_square_root=np.diag(1/np.diag(np.sqrt(D)))
    L_sym=D_inverse_square_root@L@D_inverse_square_root
    eigenvalue=np.load('{}_eigenvalue_{:.3f}_{:.3f}_normalized.npy'.format(path.split('.')[0],gamma_s,gamma_c))
    eigenvector=np.load('{}_eigenvector_{:.3f}_{:.3f}_normalized.npy'.format(path.split('.')[0],gamma_s,gamma_c))
```
若已經有存檔的eigenvalue及eigenvector，可以直接將它們讀進來。

#### Compute U from eigenvector

```
sort_index = np.argsort(eigenvalue)
U = eigenvector[:, sort_index[1:1+k]]
if cut == 'normalized':
    sums=np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1)
    U=U/sums
C, segments = kmeans(k, U, H, W)
```
按大小排序eigenvalue，取除第二後最小的k個建立U，若為normalized cut則必須normalize row to norm 1，最後將U做kmeans得最終的Cluster belonging

### Part 2

只需修改k為3, 4即可

### Part 3
除了Part 1提到的kmeans++外，我還用了random及gaussian來init

```
if init == 'random':
    random_pick=np.random.randint(low=0,high=Gram.shape[0],size=CLUSTER_NUM)
    Cluster = Gram[random_pick,:]
if init == 'gaussian':
    X_mean=np.mean(Gram,axis=0)
    X_std=np.std(Gram,axis=0)
    for c in range(Gram.shape[1]):
        Cluster[:,c]=np.random.normal(X_mean[c],X_std[c],size=CLUSTER_NUM)
```
random即隨機選擇，gaussian即用所有data的mean及variance建構gaussian distribution並sample

### Part 4

```
def plot_eigenvector(xs,ys,zs,C):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    markers=['o','^','s']
    for marker,i in zip(markers,np.arange(3)):
        ax.scatter(xs[C==i],ys[C==i],zs[C==i],marker=marker)

    ax.set_xlabel('eigenvector 1st dim')
    ax.set_ylabel('eigenvector 2nd dim')
    ax.set_zlabel('eigenvector 3rd dim')
    plt.show()
```
在k=3的情況下做出3維的示意圖，分別代表3個不同的eigenvector，並將3個不同的cluster用不同的marker表示





---

## Experiments settings and results (20%) & discussion

I use gamma_s = 0.001 and gamma_c = 0.001 below.



### input image

![](https://i.imgur.com/qeioMC5.png)
![](https://i.imgur.com/2tEe5DY.png)

### Part 1&2

###    kernel-kmeans++



#### k=2
![](https://i.imgur.com/3t6bq38.gif)
![](https://i.imgur.com/BIMV1Mw.gif)

#### k=3
![](https://i.imgur.com/kUZ05ZX.gif)
![](https://i.imgur.com/niMe34R.gif)

#### k=4
![](https://i.imgur.com/h5IzZSV.gif)
![](https://i.imgur.com/NXRKf5W.gif)



### ratio cut

#### k=2
![](https://i.imgur.com/0LVnIiv.gif)
![](https://i.imgur.com/4PlJdpd.gif)


#### k=3
![](https://i.imgur.com/N3fnQVy.gif)
![](https://i.imgur.com/rpkH8E6.gif)


#### k=4
![](https://i.imgur.com/bTx0YZP.gif)
![](https://i.imgur.com/zHEdBKy.gif)
#### Discussion
對image1而言，ratio cut的點都較集中，然而對image2而言，在樹上還是有一些交錯的點，推測是因為樹上白的和綠的顏色相差太多所致，但以分群結果來看，在現實世界中，同一物體都是連著的，ratio cut的效果看起來較kernel kmeans好。
### normalized cut


#### k=2
![](https://i.imgur.com/TIXL48T.gif)
![](https://i.imgur.com/Dgjtwgo.gif)

#### k=3
![](https://i.imgur.com/iDOxCZX.gif)
![](https://i.imgur.com/w5QkxOv.gif)


#### k=4
![](https://i.imgur.com/TbpQono.gif)
![](https://i.imgur.com/rVNreHC.gif)

#### Discussion
除k=2外，結果大致和ratio cut相似，但在這k=2的情形下，normalized cut似乎較ratio cut有更好的表現。

### Part 3 different initialization method
### kernel kmeans
#### k=2 random
![](https://i.imgur.com/k8mXRvW.gif)
![](https://i.imgur.com/Ph7Xaks.gif)

看起來跟kmeans++的差不多，可能是k只有2的關係

#### k=2 gaussian
![](https://i.imgur.com/tZSV83T.gif)
![](https://i.imgur.com/wMmoOtT.gif)

image1上看起來完全不同，由此可知，不同initialize方式確實會產生不同的結果。

#### k=3 random
![](https://i.imgur.com/IVmVxUT.gif)
![](https://i.imgur.com/F2q9jd8.gif)
#### k=3 gaussian
![](https://i.imgur.com/Hv3bsaW.gif)
![](https://i.imgur.com/jWwvV5N.gif)

gaussian的結果看起來比較好，推測是init在較中間（接近mean）的位置，所以比較不會因為init太偏而收斂到local minimum
#### k=4 random
![](https://i.imgur.com/Sh5kzTH.gif)
![](https://i.imgur.com/CMesreW.gif)

在image2的表現很明顯不如kmeans++
#### k=4 gaussian
![](https://i.imgur.com/968yLVF.gif)
![](https://i.imgur.com/Ch4FRmW.gif)

gaussian還是較random好，在image1甚至較kmeans++好

### ratio cut
#### k=4 random
![](https://i.imgur.com/YEbJHKx.gif)
![](https://i.imgur.com/9UrKLpq.gif)

#### k=4 gaussian
![](https://i.imgur.com/EKpPhP5.gif)
![](https://i.imgur.com/FnXRR78.gif)

這裡看起來都差不多（和kmeans++）
### normalized cut
#### k=4 random
![](https://i.imgur.com/mtFwYcO.gif)
![](https://i.imgur.com/LTaYfhQ.gif)

#### k=4 gaussian
![](https://i.imgur.com/S2C9MXu.gif)
![](https://i.imgur.com/58VBI5D.gif)

image1明顯gaussian分的比random好

### Discussion
以這兩張圖的情形看起來，以kmeans++和gaussian init幾乎效果都較隨機init好。

### Part 4 eigenspace visualization(k=3)

### ratio cut
#### image 1
![](https://i.imgur.com/hyiD7LA.png)

#### image 2
![](https://i.imgur.com/NTg5dRP.png)

#### Discussion
確實不同cluster在graph Laplacian的eigenspace上的coordinates上是分開的

### normalized cut
#### image 1
![](https://i.imgur.com/SmFugGr.png)

#### image 2
![](https://i.imgur.com/piBwi0Z.png)
#### Discussion
不同cluster在normalized graph Laplacian的eigenspace上的coordinates上也是分開的
Note : All use Gaussian init


---
## Observations and discussion
這次的作業跑了大概有好幾十次的clustering，各種clustering方法，各種initialization方法，各種k，如果要做個summary的話，我認為，如果要做圖片的clustering，且用的是同一種kernel，initialization選gaussian或kmeans++應該比較適合，clustering方法選ratio cut或normalized cut應該會較用普通kmeans好（點分佈的較集中，和我們肉眼的分群應該較接近），至於k的話，其實k是最重要的，像image1，k=3最適合，但像image2，k=4或5才是最佳的，至於要怎麼找k呢，並不在這次作業的範圍內，在data science的領域中，我們會以手肘法(elbow method)或輪廓係數法（Silhouette Coefficient）來挑選適合的k值。
### Elbow method
其概念是基於 SSE（sum of the squared errors，誤差平方和）作為指標，去計算每一個群中的每一個點，到群中心的距離。算法如下：

![](https://i.imgur.com/frkRHEY.png)

根據 K 與 SSE 作圖，可以從中觀察到使 SSE 的下降幅度由「快速轉為平緩」的點，一般稱這個點為拐點（Inflection point），我們會將他挑選為 K。因為該點可以確保 K 值由小逐漸遞增時的一個集群效益，因此適合作為分群的標準。

### Silhouette method
輪廓係數法的概念是「找出相同群凝聚度越小、不同群分離度越高」的值，也就是滿足 Cluster 一開始的目標。其算法如下：

![](https://i.imgur.com/rTAnNqp.png)

其中，凝聚度（a）是指與相同群內的其他點的平均距離；分離度（b）是指與不同群的其他點的平均距離。S 是指以一個點作為計算的值，輪廓係數法則是將所有的點都計算 S 後再總和。S 值越大，表示效果越好，適合作為 K。

Ref : https://blog.v123582.tw/2019/01/20/K-means-%E6%80%8E%E9%BA%BC%E9%81%B8-K/

因為逼近期末時間緊迫，Elbow method和Silhouette method的實作之後會補在github

https://github.com/franktseng0718/ML
### Note
To see full report : https://hackmd.io/DjBqaeTDTr-yNy7xOZev_A

註：pdf檔的GIF不會動