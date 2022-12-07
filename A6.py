import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, log_loss

# https://archive.ics.uci.edu/ml/datasets/banknote+authentication

# Read data from .csv
x = pd.read_csv("data_banknote_authentication.csv", usecols = ['variance','skewness','curtosis','entropy'])
y = pd.read_csv("data_banknote_authentication.csv", usecols = ['class'])

# Split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

def printCVS():
    print("Train Input")
    print(X_train)
    print("Test Input")
    print(X_test)
    print("Train Output")
    print(y_train)
    print("Test Output")
    print(y_test)

def runKNN(numOfNeighbors, inputData, outputData):
    print("KNN with",numOfNeighbors,"num of neighbors")
    # Train Model
    knn = KNeighborsClassifier(n_neighbors=numOfNeighbors)
    knn.fit(X_train, y_train.values.ravel())
    #Predict
    y_pred = knn.predict(inputData)
    tn, fp, fn, tp = confusion_matrix(outputData, y_pred).ravel()
    # Print metrics
    print("Accuracy: ", end = '')
    print((tn+tp)/(tn+fp+fn+tp))
    print("Sensitivity: ", end = '')
    print((tp)/(tp+fn))
    print("F1 Score: ", end = '')
    print((2)*((precision_score(outputData, y_pred)*recall_score(outputData, y_pred))/(precision_score(outputData, y_pred)+recall_score(outputData, y_pred))))
    print("Specificity: ", end = '')
    print((tn)/(tn+fp))
    print("Log Loss: ", end = '')
    print(log_loss(y_pred, outputData),"\n")


#printCVS()
print("\n\033[4mTesting Set\n\033[0m", end = '')
runKNN(1, X_test, y_test)
runKNN(2, X_test, y_test)
runKNN(10, X_test, y_test)
print("\n")

print("\n\033[4mTraining Set\n\033[0m", end = '')
runKNN(1, X_train, y_train)
runKNN(2, X_train, y_train)
runKNN(10, X_train, y_train)
