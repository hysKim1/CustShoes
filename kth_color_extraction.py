

import numpy as np
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans




path='img/aa.jpeg'
n_clusters=5
def knn(n_clusters,path ):
    image = cv2.imread(path)
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_flatten = img_RGB.reshape((img_RGB.shape[0] * img_RGB.shape[1],3))
    KNN = KMeans(n_clusters = n_clusters).fit(image_flatten)
    return KNN



def centroid_histogram(KNN):
  
    numLabels = np.arange(0, len(np.unique(KNN.labels_)) +1)
    (hist,_) = np.histogram(KNN.labels_, bins = numLabels)
    
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

\

def plot_colors(hist,centroids):
   
    bar = np.zeros((50,300,3), dtype = "uint8") #이 사각형에 주요 색 넣음
    startX = 0
    
    for (percent,color) in zip(hist,centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar,(int(startX),0), (int(endX),50), color.astype("uint8").tolist(),-1)
        startX = endX
        print(round(percent,3),color)
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
        
    return bar





def plot_knn(k,path):
    uploaded = cv2.imread(path)
    Z = np.float32(uploaded.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    ret,label,center=cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((uploaded.shape))

    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    img_KNN=plt.imshow(res2)
    plt.show()
    return 

\




