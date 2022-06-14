# Analysis 
I used dataset from [UCI Machine learning Repository: Wholesale customers Dataset](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers) for this lab work.  
This dataset has 440 customer datas, which compose of ordered amount of food by each customers annualy.  
## Preprosessing  
As first step, I scaled the dataset with Standard scaler. The reason why I used this scaler is because the data does　not have limits such as mimimum and maximum. They are spontanious and some outliers might be important features. I would choose to use Min-Max scaler if the data has mimimum and maximum value.   
## Algorithm and parameters
I tried with k-means and DBSCAN  
To choose suitable parameters for k-means, I used elbow method.  
As a result, I chose 18 clusters for the parameter setting.  
For DBSCAN, I brootforce the parameters (Number of cluser points, radious) and compare each graphs.  
Unfortunately, DBSCAN did not show any significant difference in various settings, except DBSCAN spot more numbers of outliers than the k-means.   
## Best algorithm for this dataset  
If you would like to grasp features of each crusters, k-means might be better option, since k-means shows various types of crusters in this dataset.  
On the other hand, DBSCAN seems that it only shows fewer cruster, which are mostly outliers.  
Therefore, k-means is better options for general business analysis such as analysing variety of customers, what might be popular, unpopular, etc.  Or, DBSCAN is better if you are looking for anomaly detection.