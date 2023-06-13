import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# enter yout path 
path=input("what is your path ?")
 # Reading the image
img1 = cv2.imread(path)
cv2.imshow("i",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#reseize
print(img1.shape)
tab=img1.reshape(-1,3)
print("new tab ",tab.shape)
#Standardization of data 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
data = sc_X.fit_transform(tab)
#quatification
from sklearn.cluster import  KMeans
n_clusters=int(input("what is the number of color u need to extract ?"))
kmeans = KMeans(n_clusters).fit(data)
#get unique values and counts of each value
unique, counts = np.unique(kmeans.labels_, return_counts=True)
#display unique values and counts side by side
print(np.asarray((unique, counts)).T)
print(kmeans.cluster_centers_)
#convert data from standardization to normal data
centercolor=sc_X.inverse_transform(kmeans.cluster_centers_)
print(centercolor)
#show pie chart
print(np.asarray((counts)).T)
y=np.asarray((counts)).T
#Percentage 
r=np.round(y/tab.shape[0]*100) 
print("le pourcentage",r)
#convert rgb color to hex
def rgb_to_hex(r, g, b):
   return '#{:02x}{:02x}{:02x}'.format((r),(g),(b))
table=[]
p=[]
for i in range(0,n_clusters):
    table.append(rgb_to_hex(int(centercolor[i][2]),int(centercolor[i][1]),int(centercolor[i][0])))
    p.append(rgb_to_hex(int(centercolor[i][2]),int(centercolor[i][1]),int(centercolor[i][0]))+' '+' | '+str(r[i])+ ' '+'%')
print(table)
plt.pie(y,labels=p ,colors=table)
plt.show()
