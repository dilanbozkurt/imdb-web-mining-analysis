# -*- coding: utf-8 -*-
"""
Created on Wed May 11 22:38:57 2022

@author: Çilem Emre & Dilan Bozkurt
"""

#Import libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

#Reading the Data 
movie_df=pd.read_csv("movie_metadata.csv")

print("------------------------------------------------------")

#Displaying the first 10 records
print(movie_df.head(10))

print("------------------------------------------------------")

#Printing the attribute's datatypes
print(movie_df.dtypes)

print("------------------------------------------------------")

#categorical columns: Color,Director name, actor name,genres,movie_title,language,country,content_rating
#numerical columns: num_critic_for_reviews,duration,director_facebook_likes ,actor_3_facebook_likes,actor_1_facebook_likes ,gross,num_voted_users,cast_total_facebook_likes,facenumber_in_poster,num_user_for_reviews ,budget,title_year, actor_2_facebook_likes ,imdb_score,aspect_ratio,movie_facebook_likes

#------------------PREPROCESSING---------------------------------

#Dropping the Imdb link from the dataset
movie_df.drop('movie_imdb_link', axis=1, inplace=True)

#Removing the color section as most of the movies is colored
movie_df["color"].value_counts()
movie_df.drop('color',axis=1,inplace=True)

#Checking for the missing values in the dataset
movie_df.isna().any()

#No of the missing values in the dataset
movie_df.isna().sum()

# We can remove the null values from the dataset where the count is less . so that we don't loose much data 
movie_df.dropna(axis=0,subset=['director_name', 'num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_2_name','actor_1_facebook_likes','actor_1_name','actor_3_name','facenumber_in_poster','num_user_for_reviews','language','country','actor_2_facebook_likes','plot_keywords'],inplace=True)

#Replacing the content rating with Value R as it has highest frequency
movie_df["content_rating"].fillna("R", inplace = True) 

#Replacing the aspect_ratio with the median of the value as the graph is right skewed 
movie_df["aspect_ratio"].fillna(movie_df["aspect_ratio"].median(),inplace=True)

#We need to replace the value in budget with the median of the value
movie_df["budget"].fillna(movie_df["budget"].median(),inplace=True)

# We need to replace the value in gross with the median of the value 
movie_df['gross'].fillna(movie_df['gross'].median(),inplace=True)

# Recheck that all the null values are removed
movie_df.isna().sum()


# Graphical presentaion 
#Language counts
plt.figure(figsize=(40,10))
sns.countplot(movie_df["language"],palette="Set3")
plt.show()

#Content ratings
sns.factorplot('content_rating',kind='count',data=movie_df, size=8)
plt.xticks(rotation=45)
plt.show()

#TOP 10 MOVIES CAST BASED ON POPULARITY
top10 = movie_df['cast_total_facebook_likes'].value_counts().reset_index().sort_values(by='cast_total_facebook_likes', ascending=False).head(10)
temp = []
for i in range(10):
    temp.append(movie_df.movie_title.iloc[[top10.index[i]]].values)
movies = []
for i in range(10):
    movies.append(temp[i][0])
top10 = pd.DataFrame(zip(movies, top10.cast_total_facebook_likes), columns=['movies', 'likes'])

plt.title('Top 10 movies based on Cast popularity')
sns.barplot(data=top10, x='movies', y='likes',palette='Set2')
plt.xticks(rotation=90)
plt.ylabel('Facebook Likes')
plt.show()

#MOVIES RELEASED PER YEAR 1916-2016
plt.figure(figsize=(18,8))
sns.countplot(movie_df['title_year'].sort_values(ascending=False))
plt.title('Movies released per year 1916-2016')
plt.xlabel('Year')
plt.ylabel('Movies Released')
plt.xticks(rotation=90)
plt.show()

#Most of the values for the languages is english we can drop the english column
movie_df.drop('language',axis=1,inplace=True)

print("------------------------------------------------------")

#Creating a new column to check the net profit made by the company (Gross-Budget) 
movie_df["Profit"]=movie_df['budget'].sub(movie_df['gross'], axis = 0) 
print(movie_df[['director_name','imdb_score','budget']].head(5))

print("------------------------------------------------------")

#Creating a new column to check the profit percentage made by the company 
movie_df['Profit_Percentage']=(movie_df["Profit"]/movie_df["gross"])*100

print("-----------------------Value counts for the countries -------------------------------")

#Value counts for the countries 
value_counts=movie_df["country"].value_counts()
print(value_counts)

print("------------------------------------------------------")

#get top 2 values of index
vals = value_counts[:2].index
print (vals)

print("------------------------------------------------------")

movie_df['country'] = movie_df.country.where(movie_df.country.isin(vals), 'other')

#Successfully divided the country into three catogories 
movie_df["country"].value_counts()

print("------------------------------------------------------")

fig, axs = plt.subplots(ncols=2,figsize=(20.7,16.27))

# print(len(movie_df[(movie_df['country'] == 'USA') & (movie_df['genres'] == 'Action|Thriller')]))

genres = [] 

# country_list=['USA','UK','France','Canada','Germany','Australia']

new_movie_df=movie_df.loc[movie_df['country'] == 'USA']

for i in new_movie_df['genres']:
        genres.append(i.split('|'))

all_genres = sum(genres,[])
print(len(set(all_genres)))


all_genres = nltk.FreqDist(all_genres) 

# create dataframe
all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()), 
                              'Count': list(all_genres.values())})

g = all_genres_df.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Genre",ax=axs[0]) 
ax.set(xlabel='Count',ylabel = 'Genres') 
ax.set(title='The most produced genres in USA')

print("------------------------------------------------------")
uk_genres = [] 
new_movie_uk=movie_df.loc[movie_df['country'] == 'UK']

for i in new_movie_uk['genres']:
        uk_genres.append(i.split('|'))

all_uk_genres = sum(uk_genres,[])
print(len(set(all_uk_genres)))


all_uk_genres = nltk.FreqDist(all_uk_genres) 

# create dataframe
all_uk_genres = pd.DataFrame({'Genre': list(all_uk_genres.keys()), 
                              'Count': list(all_uk_genres.values())})

g = all_uk_genres.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Genre",ax=axs[1]) 
ax.set(xlabel='Count',ylabel = 'Genres') 
ax.set(title='The most produced genres in UK')
plt.show()

print("------------------------------------------------------")


#DATA VISUALIZATION

#COUNTRY-IMDB
sns.boxplot(x="country", y="imdb_score",
              palette='Set2',
            data=movie_df)
plt.ylim(0, 10)
plt.show()

#IMDB & FACEBOOK LIKES
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=movie_df, x="imdb_score", y="movie_facebook_likes",palette="Set3")
plt.show()

#PROFIT & BUDGET
plt.figure(figsize=(10,8))
movie_df= movie_df.sort_values(by ='Profit' , ascending=False)
movie_df_new=movie_df.head(20)
ax=sns.pointplot(movie_df_new['Profit'], movie_df_new['budget'], hue=movie_df_new['movie_title'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# Top 20 movies based on the profit percentage
plt.figure(figsize=(10,8))
movie_df= movie_df.sort_values(by ='Profit_Percentage' , ascending=False)
movie_df_new=movie_df.head(20)
ax=sns.pointplot(movie_df_new['Profit_Percentage'], movie_df_new['budget'], hue=movie_df_new['movie_title'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

#Top 20 actors of movies based on the commerical success
plt.figure(figsize=(10,8))
movie_df= movie_df.sort_values(by ='Profit_Percentage' , ascending=False)
movie_df_new=movie_df.head(20)
ax=sns.pointplot(movie_df_new['actor_1_name'], movie_df_new['Profit_Percentage'], hue=movie_df_new['movie_title'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

#Top 20 directors based on the IMDB ratings
plt.figure(figsize=(10,8))
movie_df= movie_df.sort_values(by ='imdb_score' , ascending=False)
movie_df_new=movie_df.head(20)
ax=sns.pointplot(movie_df_new['director_name'], movie_df_new['imdb_score'], hue=movie_df_new['movie_title'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

#Commercial success vs critial acclaim
movie_df= movie_df.sort_values(by ='Profit_Percentage' , ascending=False)
movie_df_new=movie_df.head(20)
sns.scatterplot(data=movie_df_new, x="imdb_score", y="gross", hue="content_rating", style="content_rating")
plt.ylim(0, 500000)
plt.show()

#Data Preparation for the models

#Removing the director name column
movie_df.drop('director_name', axis=1, inplace=True)
#Removing the actor1 ,actor 2 and actor 3 names 
movie_df.drop('actor_1_name',axis=1,inplace=True)
movie_df.drop('actor_2_name',axis=1,inplace=True)
movie_df.drop('actor_3_name',axis=1,inplace=True)
#Dropping the movie title 
movie_df.drop('movie_title',axis=1,inplace=True)
# Dropping the plot keywords
movie_df.drop('plot_keywords',axis=1,inplace=True)

print("------------------------------------------------------")
#Value count of genres
print(movie_df['genres'].value_counts())
print("------------------------------------------------------")
#Most of the values are equally distributed in genres column ,so we can remove the genres column
movie_df.drop('genres',axis=1,inplace =True)

#Remove the linear dependent variables

# Dropping the profit column from the dataset
movie_df.drop('Profit',axis=1,inplace=True)
#Dropping the profit percentage column from the dataset
movie_df.drop('Profit_Percentage',axis=1,inplace=True)


## Correlation with heat map
corr = movie_df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)

#Adding the facebook likes of actor 2 and actor 3 together 
movie_df['Other_actor_facebbok_likes']=movie_df["actor_2_facebook_likes"] + movie_df['actor_3_facebook_likes']
#Dropping the actor 2 and actor 3 facebook likes columns as they have been added together 
movie_df.drop('actor_2_facebook_likes',axis=1,inplace=True)
movie_df.drop('actor_3_facebook_likes',axis=1,inplace=True)
movie_df.drop('cast_total_facebook_likes',axis=1,inplace=True)

#Ratio of the ratio of num_user_for_reviews and num_critic_for_reviews.
movie_df['critic_review_ratio']=movie_df['num_critic_for_reviews']/movie_df['num_user_for_reviews']
#Dropping the num_critic_for_review
movie_df.drop('num_critic_for_reviews',axis=1,inplace=True)
movie_df.drop('num_user_for_reviews',axis=1,inplace=True)

# New Correlation matrix shown in the figure 
corr = movie_df.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


# We need to categorize the imdb values in the range of 0-4,4-6,6-8 and 8-10 to mark them as the bad,average,good and excellent movies respectively
movie_df["imdb_binned_score"]=pd.cut(movie_df['imdb_score'], bins=[0,4,6,8,10], right=True, labels=False)+1
#Dropping the imdb_score column as it is being replaced with the imdb_binned_score values 
movie_df.drop('imdb_score',axis=1,inplace=True)

#Handling the categorical data
movie_df = pd.get_dummies(data = movie_df, columns = ['country'] , prefix = ['country'] , drop_first = True)
movie_df = pd.get_dummies(data = movie_df, columns = ['content_rating'] , prefix = ['content_rating'] , drop_first = True)


#Splitting the data into training and test data

X=pd.DataFrame(columns=['duration','director_facebook_likes','actor_1_facebook_likes','gross','num_voted_users','facenumber_in_poster','budget','title_year','aspect_ratio','movie_facebook_likes','Other_actor_facebbok_likes','critic_review_ratio','country_USA','country_other','content_rating_G','content_rating_GP','content_rating_M','content_rating_NC-17','content_rating_Not Rated','content_rating_PG','content_rating_PG-13','content_rating_Passed','content_rating_R','content_rating_TV-14','content_rating_TV-G','content_rating_TV-PG','content_rating_Unrated','content_rating_X'],data=movie_df)
y=pd.DataFrame(columns=['imdb_binned_score'],data=movie_df)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=100)

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Classification Model Selection

#Logistic Regression
logit =LogisticRegression()
logit.fit(X_train,np.ravel(y_train,order='C'))
y_pred=logit.predict(X_test)

#Confusion matrix for logistic regression**
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

print("Accuracy for logistic regression:",metrics.accuracy_score(y_test, y_pred))

#KNN 
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, np.ravel(y_train,order='C'))
knnpred = knn.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, knnpred)
print(cnf_matrix)
print("Accuracy for k-nearest neighbor:",metrics.accuracy_score(y_test, knnpred))

#SVC
svc= SVC(kernel = 'sigmoid')
svc.fit(X_train, np.ravel(y_train,order='C'))
svcpred = svc.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, svcpred)
print(cnf_matrix)
print("Accuracy for SVC:",metrics.accuracy_score(y_test, svcpred))

#Naive bayes
gaussiannb= GaussianNB()
gaussiannb.fit(X_train, np.ravel(y_train,order='C'))
gaussiannbpred = gaussiannb.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, gaussiannbpred)
print(cnf_matrix)
print("Accuracy for Naive Bayes:",metrics.accuracy_score(y_test, gaussiannbpred))

#Decision Tree
dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtree.fit(X_train, np.ravel(y_train,order='C'))
dtreepred = dtree.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, dtreepred)
print(cnf_matrix)
print("Accuracy for decision tree:",metrics.accuracy_score(y_test, dtreepred))

#Ada Boosting
abcl = AdaBoostClassifier(base_estimator=dtree, n_estimators=60)
abcl=abcl.fit(X_train,np.ravel(y_train,order='C'))
abcl_pred=abcl.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, abcl_pred)
print(cnf_matrix)
print("Accuracy for Ada Boosting:",metrics.accuracy_score(y_test, abcl_pred))

#Random Forest
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(X_train, np.ravel(y_train,order='C'))
rfcpred = rfc.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, rfcpred)
print(cnf_matrix)
print("Accuracy for Random Forest:",metrics.accuracy_score(y_test, rfcpred))


new_movie_df=movie_df.pop("imdb_binned_score")

#Bagging classfier
bgcl = BaggingClassifier(n_estimators=60, max_samples=.7 , oob_score=True)
bgcl = bgcl.fit(movie_df, new_movie_df)
print("Accuracy for Bagging Classifier:",bgcl.oob_score_)

#Gradient boosting
gbcl = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.09, max_depth=5)
gbcl = gbcl.fit(X_train,np.ravel(y_train,order='C'))
test_pred = gbcl.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, test_pred)
print(cnf_matrix)
print("Accuracy for Gradient Boosting:",metrics.accuracy_score(y_test, test_pred))

#MODEL COMPARISON
print('Logistic  Reports\n',classification_report(y_test, y_pred))
print('KNN Reports\n',classification_report(y_test, knnpred))
print('SVC Reports\n',classification_report(y_test, svcpred))
print('Naive Bayes Reports\n',classification_report(y_test, gaussiannbpred))
print('Decision Tree Reports\n',classification_report(y_test, dtreepred))
print('Ada Boosting\n',classification_report(y_test, abcl_pred))
print('Random Forests Reports\n',classification_report(y_test, rfcpred))
print('Bagging Clasifier',bgcl.oob_score_) 
print('Gradient Boosting',classification_report(y_test, test_pred))

