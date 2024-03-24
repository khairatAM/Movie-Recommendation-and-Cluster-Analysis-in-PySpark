from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

spark = SparkSession.builder \
        .master("local[15]") \
        .appName("Assignment Q3") \
        .config("spark.local.dir","/fastdata/acp22kam") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

print("\nTask A\n")
#read ratings.csv and order by timestamp
ratings = spark.read.load\
        ('Data/ml-20m/ratings.csv', format = 'csv', inferSchema = "true", header = "true")\
        .sort('timestamp').cache()
ratings.show(20,False)
myseed=28 #set seed

'''
Task A Time-split Recommendation: 

Perform time-split recommendation using ALS-based matrix factorisation on the rating data.

First, sort all data in ascending order of timestamp and split so that earlier entries
are used for training and later entries are used for testing.
Three such splits are considered with the following sizes for the training set: 40%, 60% and 80%.

For each split above, we consider two settings on which to train our models.
Then we compute and report three metrics: RMSE, MSE, and MAE (3 metrics x 3 splits x 2 ALS settings = 18 numbers). 
Lastly, we visualise these 18 numbers in ONE single figure.
'''

ratings_len = ratings.count()

# 40% split
train1 = ratings.limit(int(ratings_len*0.4)).cache()
test1 = ratings.exceptAll(train1).cache()

# 60% split
train2 = ratings.limit(int(ratings_len*0.6)).cache()
test2 = ratings.exceptAll(train2).cache()

# 80% split
train3 = ratings.limit(int(ratings_len*0.8)).cache()
test3 = ratings.exceptAll(train3).cache()

#Training ALS with Setting 1
print("\nALS with Setting 1\n")
als1 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")

model1 = als1.fit(train1)
model2 = als1.fit(train2)
model3 = als1.fit(train3)

predictions1 = model1.transform(test1)
predictions2 = model2.transform(test2)
predictions3 = model3.transform(test3)

#evaluating models with setting 1
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")

rmse1 = evaluator_rmse.evaluate(predictions1)
rmse2 = evaluator_rmse.evaluate(predictions2)
rmse3 = evaluator_rmse.evaluate(predictions3)
print('\nRMSE for 3 respective models: %.4f \t %.4f \t %.4f '%(rmse1,rmse2,rmse3))

mse1 = evaluator_mse.evaluate(predictions1)
mse2 = evaluator_mse.evaluate(predictions2)
mse3 = evaluator_mse.evaluate(predictions3)
print('\nMSE for 3 respective models: %.4f \t %.4f \t %.4f '%(mse1,mse2,mse3))

mae1 = evaluator_mae.evaluate(predictions1)
mae2 = evaluator_mae.evaluate(predictions2)
mae3 = evaluator_mae.evaluate(predictions3)
print('\nMAE for 3 respective models: %.4f \t %.4f \t %.4f '%(mae1,mae2,mae3))

#Training ALS with Setting 2
print("\nALS with Setting 2\n")
als2 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop", \
           rank=50, maxIter=20, implicitPrefs=True, regParam=0.01)

model4 = als2.fit(train1)
model5 = als2.fit(train2)
model6 = als2.fit(train3)

predictions4 = model4.transform(test1)
predictions5 = model5.transform(test2)
predictions6 = model6.transform(test3)

#Setting 2 model evaluation
rmse4 = evaluator_rmse.evaluate(predictions4)
rmse5 = evaluator_rmse.evaluate(predictions5)
rmse6 = evaluator_rmse.evaluate(predictions6)
print('\nRMSE for 3 respective models: %.4f \t %.4f \t %.4f '%(rmse4,rmse5,rmse6))

mse4 = evaluator_mse.evaluate(predictions4)
mse5 = evaluator_mse.evaluate(predictions5)
mse6 = evaluator_mse.evaluate(predictions6)
print('\nMSE for 3 respective models: %.4f \t %.4f \t %.4f '%(mse4,mse5,mse6))

mae4 = evaluator_mae.evaluate(predictions4)
mae5 = evaluator_mae.evaluate(predictions5)
mae6 = evaluator_mae.evaluate(predictions6)
print('\nMAE for 3 respective models: %.4f \t %.4f \t %.4f '%(mae4,mae5,mae6))

#plotting ALS performance Setting 1 vs Setting 2 as a bar char
groups=("Split 1","Split 2","Split 3")

#setting 1 metrics
rmse_arr1=[rmse1,rmse2,rmse3]
mse_arr1=[mse1,mse2,mse3]
mae_arr1=[mae1,mae2,mae3]
#setting 2 metrics
rmse_arr2=[rmse4,rmse5,rmse6]
mse_arr2=[mse4,mse5,mse6]
mae_arr2=[mae4,mae5,mae6]

X = np.arange(3)
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots()
width=0.25

ax.bar_label(ax.bar(X + 0.00, rmse_arr1, width = width, label='RMSE - Setting 1'),label_type='center', fmt='%.2f')
ax.bar_label(ax.bar(X + 0.25, mse_arr1, width = width, label='MSE - Setting 1'),label_type='center', fmt='%.2f')
ax.bar_label(ax.bar(X + 0.50, mae_arr1, width = width, label='MAE - Setting 1'),label_type='center', fmt='%.2f')

ax.bar_label(ax.bar(X + 0.00, rmse_arr2, width = width, label='RMSE - Setting 2', bottom=rmse_arr1),label_type='center', fmt='%.2f')
ax.bar_label(ax.bar(X + 0.25, mse_arr2, width = width, label='MSE - Setting 2', bottom=mse_arr1),label_type='center', fmt='%.2f')
ax.bar_label(ax.bar(X + 0.50, mae_arr2, width = width, label='MAE - Setting 3', bottom=mae_arr1),label_type='center', fmt='%.2f')

ax.legend(loc='upper left', ncol=3)
ax.set_xticks(X + width, groups)
ax.set_title('ALS Evaluation Metrics')
plt.savefig("Output/TaskA_ALS_Evaluation_Metrics.png")


'''
Task B User Analysis: 

After ALS, each user is modelled by some factors. For each of the three time-splits, we use k-means in with k=25 
to cluster all the users based on the user factors learned with the ALS Setting 2 above. Then we find 
the top five largest user clusters and report the size of (i.e. the number of users in) each of the top five clusters
in one Table (total 3 splits x 5 clusters = 15 numbers). Next we visualise these 15 numbers in ONE single figure.
'''


print("\nTask B\n")
#task B: user factors learned from setting 2 ALS model
user_factors1 = model4.userFactors
user_factors2 = model5.userFactors
user_factors3 = model6.userFactors

#K-means clustering with 25 clusters
k=25
kmeans = KMeans().setK(k).setSeed(myseed)
kmeans_model1 = kmeans.fit(user_factors1.select('features')) 
kmeans_model2 = kmeans.fit(user_factors2.select('features')) 
kmeans_model3 = kmeans.fit(user_factors3.select('features')) 

kmeans_pred1 = kmeans_model1.transform(user_factors1)
kmeans_pred2 = kmeans_model2.transform(user_factors2)
kmeans_pred3 = kmeans_model3.transform(user_factors3)

#top 5 clusters
def topFive(model):
    sizes = model.summary.clusterSizes
    top = sorted(sizes, reverse=True)[:5]
    return top  

top_clusters_sizes1=topFive(kmeans_model1)
top_clusters_sizes2=topFive(kmeans_model2)
top_clusters_sizes3=topFive(kmeans_model3)

print('Largest clustes in Group 1: {}, {}, {}, {} and {}'.format(*top_clusters_sizes1))
print('Largest clustes in Group 2: {}, {}, {}, {} and {}'.format(*top_clusters_sizes2))
print('Largest clustes in Group 3: {}, {}, {}, {} and {}'.format(*top_clusters_sizes3))

#plot cluster sizes for 3 models
plt.clf()
plt.rcParams.update({'font.size': 8})
fig,ax=plt.subplots()
width=0.15
data=np.array([top_clusters_sizes1,top_clusters_sizes2,top_clusters_sizes3]).T
for i in range(5):
    ax.bar_label(ax.bar(X+width*i, data[i], width = width, label='Rank{}'.format(i+1)),label_type='center')
ax.legend(loc='upper left', ncol=2)
ax.set_xticks(X + width*1.75, groups)
ax.set_title('Top Cluster Sizes')
plt.savefig("Output/TaskB_ALS_Largest_Clusters.png")

'''
Task B User Analysis continued: 

For each of the three splits in Task A, find the largest cluster. Find all the movies 
that have been rated by the users in each of the largest clusters and the respective average ratings for the movies.
Then, of those movies find those with an average rating >=4.0.

Lastly, we use another file 'movies.csv' to find the genres for all the movies previously identified
(i.e. with avg rating >=4.0) and report the top ten genres with the most appearances.
'''
#task B2
#users in the largest cluster
users_largest1=kmeans_pred1.filter(kmeans_pred1.prediction==kmeans_model1.summary.clusterSizes.index(top_clusters_sizes1[0]))
users_largest2=kmeans_pred2.filter(kmeans_pred2.prediction==kmeans_model2.summary.clusterSizes.index(top_clusters_sizes2[0]))
users_largest3=kmeans_pred3.filter(kmeans_pred3.prediction==kmeans_model3.summary.clusterSizes.index(top_clusters_sizes3[0]))

#Movies rated by users in largest cluster
#split 1
print("\nMovies in the Largest Cluster Split 1\n")
movies_largest_cluster1=ratings.\
    join(users_largest1, ratings.userId == users_largest1.id, "inner").\
    groupBy('movieId').avg('rating')
movies_largest_cluster1.show(5, False)

#split 2
print("\nMovies in the Largest Cluster Split 2\n")
movies_largest_cluster2=ratings.\
    join(users_largest2, ratings.userId == users_largest2.id, "inner").\
    groupBy('movieId').avg('rating')
movies_largest_cluster2.show(5, False)

#split 3
print("\nMovies in the Largest Cluster Split 3\n")
movies_largest_cluster3=ratings.\
    join(users_largest3, ratings.userId == users_largest3.id, "inner").\
    groupBy('movieId').avg('rating')
movies_largest_cluster3.show(5, False)

#movies rated >=4
#split 1
print("\nTop Movies with Rating >= 4.0 Split 1\n")
top_movies1 = movies_largest_cluster1.where(F.col("avg(rating)")>=4.0)
top_movies1.show(20, False)

#split 2
print("\nTop Movies with Rating >= 4.0 Split 2\n")
top_movies2 = movies_largest_cluster2.where(F.col("avg(rating)")>=4.0)
top_movies2.show(20, False)

#split 3
print("\nTop Movies with Rating >= 4.0 Split 3\n")
top_movies3 = movies_largest_cluster3.where(F.col("avg(rating)")>=4.0)
top_movies3.show(20, False)

#genres of top rated movies
#read movies.csv
movies = spark.read.load\
        ('Data/ml-20m/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()

#genres of top movies 
genres1 = movies.join(top_movies1, movies.movieId == top_movies1.movieId, "inner").select('genres')
genres2 = movies.join(top_movies2, movies.movieId == top_movies2.movieId, "inner").select('genres')
genres3 = movies.join(top_movies3, movies.movieId == top_movies3.movieId, "inner").select('genres')

#extract top 10 genres
from collections import Counter
def top_ten(df):
    split = df.rdd.map(lambda row: row.genres.split("|")).collect()
    flat_split = [item for sublist in split for item in sublist]
    top=Counter(flat_split)
    return top.most_common(10)

top1=top_ten(genres1)
top2=top_ten(genres2)
top3=top_ten(genres3)

#print results
def print_c(c):
    for gen, count in c:
        print(gen,":", count)

print("\nTop Genres Split 1\n")
print_c(top1)
print("\nTop Genres Split 2\n")
print_c(top2)
print("\nTop Genres Split 3\n")
print_c(top3)
