import numpy as np

def createData():
    X_train = np.array([[1, 1], [5, 1], [4, 4]])
    X = np.array([[0, 0], [1, 0], [5, 0], [5, 5]])
    y = np.array([1, 2, 0])
    return X_train, y, X;

#Lp距离
def LpDistance(X, X_train, p):
    numTest = X.shape[0]
    numTrain = X_train.shape[0]
    res = np.zeros((numTest, numTrain))
    for i in range(numTest):
        tmp = (X[i, :] - X_train)
        tmp = tmp ** p
        res[i, :] = np.sum(tmp, axis = 1)
        res[i, :] = res[i] ** (1.0 / p)
    return res

#计算曼哈顿距离
def ManhattanDistance(X, X_train):
    numTest = X.shape[0]
    numTrain = X_train.shape[0]
    res = np.zeros((numTest, numTrain))
    for i in range(numTest):
        res[i, :] = np.sum(abs(X[i, :] - X_train), axis = 1)
    return res

#为每一个数据做标记
def predictLabels(dists, k, y):
    numTest = dists.shape[0]
    numClass = max(y) + 1
    label = np.zeros(numTest)
    for i in range(numTest):
        rank = np.argsort(dists[i, :])[:k]
        #print(rank)
        fre = np.zeros(numClass)
        for j in rank:
            fre[y[j]] = fre[y[j]] + 1;
        label[i] = np.argmax(fre)
    return label;

if __name__ == "__main__":
    X_train, y, X = createData();
    dists = LpDistance(X, X_train, 2)
    #print(dists)
    label = predictLabels(dists, 1, y)
    print(label)
