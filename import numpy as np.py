import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df = pd.read_csv("heart.csv")
# First 5 rows of our data
df.head()
df.target.value_counts()
sns.countplot(x="target", data=df, palette="bwr")
plt.show()
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
sns.countplot(x='sex', data=df, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()
df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()
y = df.target.values
x_data = df.drop(['target'], axis = 1)
# Normalize
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
#initialize
def initialize(dimension):
    
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias
 def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction
# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
# try ro find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
print("KNN Accuracy is {:.2f}%".format(acc))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)

acc = nb.score(x_test.T,y_test.T)*100
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
# Predicted values

knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(x_train.T, y_train.T)
y_head_knn = knn3.predict(x_test.T)
y_head_nb = nb.predict(x_test.T)
y_head_nb[0:10]
y_head_knn[0:10]
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test,y_head_knn)
cm_nb = confusion_matrix(y_test,y_head_nb)
plt.figure(figsize=(24,12))
plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)
plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.show()
import numpy as np
from sklearn.metrics import accuracy_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[27, 34])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[21, 6, 7, 27])


accuracy_score(actual, pred)
import numpy as np
from sklearn.metrics import f1_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[27, 34])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[21, 6, 7, 27])

#calculate F1 score
f1_score(actual, pred)
import numpy as np
from sklearn.metrics import precision_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[27, 34])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[21, 6, 7, 27])


precision_score(actual, pred)
import numpy as np
from sklearn.metrics import recall_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[27, 34])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[21, 6, 7, 27])


recall_score(actual, pred)
import numpy as np
from sklearn.metrics import accuracy_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[27, 34])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[19, 8, 6, 28])


accuracy_score(actual, pred)
import numpy as np
from sklearn.metrics import recall_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[27, 34])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[19, 8, 6, 28])


recall_score(actual, pred)
import numpy as np
from sklearn.metrics import precision_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[27, 34])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[19, 8, 6, 28])


precision_score(actual, pred)
import numpy as np
from sklearn.metrics import f1_score

#define array of actual classes
actual = np.repeat([1, 0], repeats=[27, 34])

#define array of predicted classes
pred = np.repeat([1, 0, 1, 0], repeats=[19, 8, 6, 28])

#calculate F1 score
f1_score(actual, pred)   