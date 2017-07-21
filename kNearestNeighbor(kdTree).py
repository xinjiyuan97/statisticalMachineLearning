import numpy as np


def createData():
    x = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    return x;

def createTree(dataSet, k, j):
    if len(dataSet) == 0:
        return
    if len(dataSet) == 1:
        print(dataSet[0, :], j)
        return
        #return dataSet[0, :]
    num = dataSet.shape[0]
    rank = np.argsort(dataSet[:, j % k])
    #print(rank)
    mid = int(num / 2);
    #print(mid)
    left = np.array([dataSet[x, :] for x in rank[ : mid]])
    #print(left)
    right = np.array([dataSet[x, :] for x in rank[mid + 1:]])
    #print(right)
    #print(dataSet[mid])
    root = dataSet[rank[mid], :]
    print(root, j)
    #myTree = { root.tostring() : {}}
    #myTree[root.tostring()][0] = createTree(left, k, j + 1)
    #myTree[root.tostring()][1] = createTree(right, k, j + 1)
    createTree(left, k, j + 1)
    createTree(right, k, j + 1)
    return 
    #return myTree

if __name__ ==  "__main__":
    X = createData()
    k = X.shape[1]
    #print(X)
    createTree(X, k, 0)
    #print(myTree)
    