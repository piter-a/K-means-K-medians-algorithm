#!/usr/bin/env python
# coding: utf-8

# In[18]:


#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

#reading data using pandas
animals = pd.read_csv(r"C:\Users\piotr\Downloads\CA2data\animals.csv", header = None)
countries = pd.read_csv(r"C:\Users\piotr\Downloads\CA2data\countries.csv", header = None)
fruits = pd.read_csv(r"C:\Users\piotr\Downloads\CA2data\fruits.csv", header = None)
veggies = pd.read_csv(r"C:\Users\piotr\Downloads\CA2data\veggies.csv", header = None)

# merging data together (ignoring index) on axis 0 into asc numb
data = pd.concat([animals, countries, fruits, veggies],axis=0, ignore_index = True)
#array of data values excluding 1st column with words (the names) 
x = data.iloc[:,1:301].values 

# K-means algorithm:
# k -number of cluster (integer)
# x - data used for clustering in the form of an array
def kmeans_clustering(x, k, distance): 

    # Initialization of centroids (random)
    centroids = []
    tmp = np.random.randint(x.shape[0], size = k)
    while (len(tmp) > len(set(tmp))):
        tmp = np.random.randint(x.shape[0], size = k)
    for i in tmp:
        centroids.append(x[i])
    # Creating centroids copies used for updating
    centroids_initial = np.zeros(np.shape(centroids))
    centroids_updated = deepcopy(centroids)

    # Creating object with blank distances and assignment of clusters to store results
    clusters = np.zeros(x.shape[0])
    # Error object used for updating results
    error = np.linalg.norm(centroids_updated - centroids_initial)
    errors_number = 0

    # Checking error value and adding 1 to it (used for updating results) :
    while error != 0:
        dist = np.zeros([x.shape[0], k])
        errors_number += 1
        # Calculating Euclidean distance from each data point to each centroid
        if distance == "Euclidean":
            for j in range(len(centroids)):
                dist[:, j] = np.linalg.norm(x - centroids_updated[j], axis=1)
  

        # Calculating clusters assignment
        clusters = np.argmin(dist, axis = 1)

        # Assigning updated copy of centroids to the initial centroids
        centroids_initial = deepcopy(centroids_updated)

        # Calculating mean in order to update clusters centroids
        for m in range(k):
            centroids_updated[m] = np.mean(x[clusters == m], axis = 0)

        # Calculate error with updated centroids
        error = np.linalg.norm(np.array(centroids_updated) - np.array(centroids_initial))

    #Assigning updated clusters and centroids to new objects
    clusters_prediction = clusters
    actual_centroids = np.array(centroids_updated)

    print("K-means clustering results:")
    print("Clusters number:", k)
    print("Updates number:", errors_number)
    #print ("Clusters:", clusters_prediction)
    #print ("Centroid Locations:", actual_centroids)
    return clusters_prediction   

def kmedians_clustering(x, k, distance): 

    # Initialization of centroids (random)
    centroids = []
    tmp = np.random.randint(x.shape[0], size = k)
    while (len(tmp) > len(set(tmp))):
        tmp = np.random.randint(x.shape[0], size = k)
    for i in tmp:
        centroids.append(x[i])
    # Creating centroids copies used for updating
    centroids_initial = np.zeros(np.shape(centroids))
    centroids_updated = deepcopy(centroids)

    # Creating object with blank distances and assignment of clusters to store results
    clusters = np.zeros(x.shape[0])
    # Error object used for updating results
    error = np.linalg.norm(centroids_updated - centroids_initial)
    errors_number = 0

    # Checking error value and adding 1 to it (used for updating results) :
    while error != 0:
        dist = np.zeros([x.shape[0], k])
        errors_number += 1
        # Calculating Manhattan distance from each data point to each centroid
        if distance == "Manhattan":
            for j in range(len(centroids)):
                dist[:, j] = np.linalg.norm(x - centroids_updated[j], axis=1)
  

        # Calculating clusters assignment
        clusters = np.argmin(dist, axis = 1)

        # Assigning updated copy of centroids to the initial centroids
        centroids_initial = deepcopy(centroids_updated)

        # Calculating median in order to update clusters centroids
        for m in range(k):
            centroids_updated[m] = np.median(x[clusters == m], axis = 0)

        # Calculate error with updated centroids
        error = np.linalg.norm(np.array(centroids_updated) - np.array(centroids_initial))

    #Assigning updated clusters and centroids to new objects
    clusters_prediction = clusters
    actual_centroids = np.array(centroids_updated)

    print("K-medians clustering results:")
    print("Clusters number:", k)
    print("Updates number:", errors_number)
    #print ("Clusters:", clusters_prediction)
    #print ("Centroid Locations:", actual_centroids)
    return clusters_prediction


#L2 normalization of data    
def l2norm(x):
    x = x / np.linalg.norm(x)
    return x
 

#Data preparation for Using B-Cubed evaluation     
def data_prep(clusters_prediction):
    # counting how many items in each category by getting highest index in each category
    index_animal=len(animals.index) 
    index_countries=len(countries.index)
    index_fruits=len(fruits.index)
    index_veggies=len(veggies.index)

    #finding last index of each category of merged data
    last_animal = index_animal
    last_country = index_animal+index_countries
    last_fruit = index_animal+index_countries+index_fruits
    last_veggie = index_animal+index_countries+index_fruits+index_veggies 
    
    #Creating objects of each category index positioning
    a_pos = clusters_prediction[:last_animal]
    print(a_pos)
    c_pos = clusters_prediction[last_animal:last_country]
    f_pos = clusters_prediction[last_country:last_fruit]
    v_pos = clusters_prediction[last_fruit:last_veggie]
   
    return a_pos, c_pos, f_pos, v_pos  

# assigning True Positive and False Negative to clustered data
def TP_FN(a_pos, c_pos, f_pos, v_pos):
    # True Positives
    TP = 0
    # False Negatives
    FN = 0
    #animals
    for i in range(len(a_pos)):
        for j in range(len(a_pos)):
          # If i and j are not equal, and j >i 
            if (i != j & j>i):
                # If i=j then add 1 to TP
                if(a_pos[i] == a_pos[j]):
                    TP += 1
                    # els add 1 to FN
                else:
                    FN += 1
        
    #countries
    for i in range(len(c_pos)): 
        for j in range(len(c_pos)):
            if (i != j & j>i):
                if(c_pos[i] == c_pos[j]):
                    TP += 1
                else:
                    FN += 1    
        
     #fruits
    for i in range(len(f_pos)):
        for j in range(len(f_pos)):
            if (i != j & j>i):
                if(f_pos[i] == f_pos[j]):
                    TP += 1
                else:
                    FN += 1   
        
    #veggies
    for i in range(len(v_pos)):       
        for j in range(len(v_pos)):
            if (i != j & j>i):
                if(v_pos[i] == v_pos[j]):
                    TP += 1
                else:
                    FN += 1    
    return TP, FN      
        
# assigning False Positive and True Negative to clustered data         
def FP_TN(a_pos, c_pos, f_pos, v_pos): 
     # True Negatives
    TN = 0
    # False Positives
    FP = 0
    #animals
    for i in range(len(a_pos)):                   
        for j in range(len(c_pos)):
            # If i =j then add 1 to FP
            if(a_pos[i] == c_pos[j]):
                FP += 1
                # else add 1 to TN
            else:
                    TN += 1
        #fruit
        for j in range(len(f_pos)):
            if(a_pos[i]==f_pos[j]):
                FP += 1
            else:
                    TN += 1
        #veggies
        for j in range(len(v_pos)):
            if(a_pos[i] == v_pos[j]):
                FP += 1
            else:
                TN += 1
    #countries
    for i in range(len(c_pos)):  
        for j in range(len(f_pos)):
            if(c_pos[i] == f_pos[j]):
                FP += 1
            else:
                 TN += 1
          
        for j in range(len(v_pos)):
            if(c_pos[i] == v_pos[j]):
                FP += 1
            else:
                TN += 1     
    #fruits
    for i in range(len(f_pos)):
        for j in range(len(v_pos)):
            if(f_pos[i] == v_pos[j]):
                FP += 1
            else:
                TN += 1    
    return FP, TN
             
def B_cubed_score(TP, FN, FP, TN):
    #precision= Tp/TP+FP, recall= TP/TP+FN, F-Score= 2*(precision*recall/precision+recall)
    # to 3 decimal places
        precision = round(TP / (TP + FP),3)
        recall = round((TP / (TP + FN)), 3)
        F_score = round((2 * (precision * recall) / (precision + recall)), 3)
        return precision, recall, F_score

def plot(k_lst, precision_lst, f_score_lst, recall_lst):
    # line 1 points
    x1 = k_lst
    y1 = precision_lst
    # plotting precision points 
    plt.plot(x1, y1, label = "precision")
  
    # line 2 points
    x2 = k_lst
    y2 = f_score_lst

    # plotting the F-score points 
    plt.plot(x2, y2, label = "F-score")

    # line 3 points
    x3 = k_lst
    y3 = recall_lst

    # plotting the recall points 
    plt.plot(x3, y3, label = "recall")
  
    # naming the x axis
    plt.xlabel('k- values')
    # naming the y axis
    plt.ylabel('Score value')
    # giving a title to my graph
    plt.title('B-Cubed')
  
    # show a legend on the plot
    plt.legend()
  
    # function to show the plot
    plt.show()

#########################################################################################################  
#Question 1  
"""
clusters_prediction=kmeans_clustering(x, 4, 'Euclidean')
"""
#Question 2
"""
clusters_prediction=kmedians_clustering(x, 4, 'Manhattan')
"""
#Question 3
"""
precision_lst= []
recall_lst= []
f_score_lst=[]
k_lst= []

def scores_kmeans(x):
    for k in range(1,10):
        k_lst.append(k)
        clusters_prediction=kmeans_clustering(x, k, 'Euclidean')
        a_pos, c_pos, f_pos, v_pos= data_prep(clusters_prediction)
        TP, FN= TP_FN(a_pos, c_pos, f_pos, v_pos)
        FP, TN= FP_TN(a_pos, c_pos, f_pos, v_pos)
        precision, recall, F_score= B_cubed_score(TP, FN, FP, TN)
        precision_lst.append(precision)
        recall_lst.append(recall)
        f_score_lst.append(F_score)
    #to print final list with values from all iterations
        if k==9:
            print(k_lst)
            print(f"the precision list is: {precision_lst}")
            print(f"the F-score list is: {f_score_lst}")
            print(f"the recall list is: {recall_lst}")
    return k_lst, precision_lst, f_score_lst, recall_lst

k_lst, precision_lst, f_score_lst, recall_lst=scores_kmeans(x)

#Plotting graph

plot(k_lst, precision_lst, f_score_lst, recall_lst)
"""
#Question 4
# !!!!!!!IMPORTANT!!!!!!: 
# Data normalization seems to cause an issue with scores function and
# returns lists of scores that repeat themselves)
# hence for plotting purposes only first 9 values from each list were taken 
"""
x1 = l2norm(x)
scores_kmeans(x1)
k_lst1, precision_lst1, f_score_lst1, recall_lst1 =scores_kmeans(x1)
plot(k_lst1[:9], precision_lst1[:9], f_score_lst1[:9], recall_lst1[:9])
"""

#Question 5

"""
precision_lst= []
recall_lst= []
f_score_lst=[]
k_lst= []

def scores_kmedians(x):
    for k in range(1,10):
        k_lst.append(k)
        clusters_prediction=kmedians_clustering(x, k, 'Manhattan')
        a_pos, c_pos, f_pos, v_pos= data_prep(clusters_prediction)
        TP, FN= TP_FN(a_pos, c_pos, f_pos, v_pos)
        FP, TN= FP_TN(a_pos, c_pos, f_pos, v_pos)
        precision, recall, F_score= B_cubed_score(TP, FN, FP, TN)
        precision_lst.append(precision)
        recall_lst.append(recall)
        f_score_lst.append(F_score)
    #to print final list with values from all iterations
        if k==9:
            print(k_lst)
            print(f"the precision list is: {precision_lst}")
            print(f"the F-score list is: {f_score_lst}")
            print(f"the recall list is: {recall_lst}")
    return k_lst, precision_lst, f_score_lst, recall_lst

k_lst, precision_lst, f_score_lst, recall_lst=scores_kmedians(x)

plot(k_lst, precision_lst, f_score_lst, recall_lst)
"""

#Question 6
# !!!!!!!IMPORTANT!!!!!!: 
# Data normalization seems to cause an issue with scores function and
# returns lists of scores that repeat themselves)
# hence for plotting purposes only first 9 values from each list were taken
"""
x1 = l2norm(x)
scores_kmedians(x1)
k_lst1, precision_lst1, f_score_lst1, recall_lst1 =scores_kmedians(x1)
plot(k_lst1[:9], precision_lst1[:9], f_score_lst1[:9], recall_lst1[:9])
"""


# In[ ]:





# In[ ]:




