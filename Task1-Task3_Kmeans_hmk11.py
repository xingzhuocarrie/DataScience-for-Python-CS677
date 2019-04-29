# K-means


##Task 1
"""--------------------------------Task1--------------------------------"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



df=pd.read_csv("C:\\Users\\carri\\Documents\\2019 Spring\\CS677\\Assignments\\XON_weekly_return_volatility.csv")
df=df.rename(columns={'volatility':'Std','mean_return':'Mean'})
df['Label']=np.where(df['Mean']>=0,'g','r')
X=df[['Mean','Std']].values

# Plotting to see what it originally looks like:

print('\n','\n','\n')
figure=plt.figure(1)
plt.title('Visualization of raw data');
plt.scatter(X[:,0],X[:,1], label='True Position') 
plt.xlabel('Mean')
plt.ylabel("Standard Deviation")
plt.grid()
plt.tight_layout() 
figure.show()


km = KMeans(n_clusters=5)
y_km = km.fit_predict(X)
y_km

print('\n','\n','\n')
print("Cluster:")
print(km.cluster_centers_)  
print("Here the first row contains values for the coordinates of the first centroid i.e. (0.022 , 0.00254) ")
print("And the second row contains values for the coordinates of the other centroid i.e. (0.04986, 0.01840)")

#In this case we are also passing km.labels_ as value for the c parameter that corresponds to labels. 
print('\n','\n','\n')
f=plt.figure(2)
plt.scatter(X[:,0],X[:,1], c=km.labels_, cmap='viridis',s=80) 
plt.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1], color='red',marker='*',s=240,label='Centroids')
plt.legend() 
plt.grid()
plt.xlabel('Mean')
plt.ylabel("Standard Deviation")
plt.tight_layout()
f.show()
 






   
   
# Task2
"""--------------------------------Task2-------------------------------"""

print('\n','\n','\n')
print('Distortion: %.2f' % km.inertia_)
distortions = []
for i in range(1, 9,1):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
print('\n','\n','\n')
dis=plt.figure(3)
plt.plot(range(1, 9,1), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
#plt.savefig('./figures/elbow.png', dpi=300)
dis.show()
# Based on the plot, when k=8, we can see distortion seems to flatten, but we can't find a obvious L-shape. 



#Task 3

"""--------------------------------Task3-------------------------------"""


km8 = KMeans(n_clusters=8)
df['y_km'] = km.fit_predict(X)


def count_red(x):
    red=[]
    gre=[]
    for i in range(len(df)):
        if df['y_km'].iloc[i]==x and df['Label'].iloc[i]=='r':
          red.append(df['y_km'].iloc[i])
        if df['y_km'].iloc[i]==x and df['Label'].iloc[i]=='g':
         gre.append(df['y_km'].iloc[i])
    return(len(red),len(gre))
#xx=count_red(df,1)
cluster_df=pd.DataFrame({'0':count_red(0),'1':count_red(1),'2':count_red(2),
                         '3':count_red(3),'4':count_red(4),'5':count_red(5),'6':count_red(6),'7':count_red(7)})
j=1
cluster_df.loc[2]=cluster_df.loc[j]+cluster_df.loc[j-1]
cluster_df.loc[3]=round(100*(cluster_df.loc[j-1]/cluster_df.loc[2]),2)#%red weeks
cluster_df.loc[4]=round(100*(cluster_df.loc[j]/cluster_df.loc[2]),2)#%green weeks

# in Cluster_df dataframe, the first row is the number of red points in each cluster and the second row is the number of green ones.
# We sum up the first row and the second one in the third row.
    
cluster_result=pd.DataFrame({'Cluster':np.arange(0,8,1), '%red_weeks':cluster_df.loc[3],'%green_weeks':cluster_df.loc[4] })

print('\n','\n','\n')

print("Table for Task 3")

print(cluster_result)








