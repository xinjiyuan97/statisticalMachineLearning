import numpy as np
import pandas as pd

global epsilon

def createData():
    X = pd.DataFrame([['youth', False, False, 0], ['youth', False, False, 1], ['youth', True, False, 1], ['youth', True, True, 0], ['youth', False, False, 0], ['mid', False, False, 0], ['mid', False, False, 1], ['mid', True, True, 1], ['mid', False, True, 2], ['mid', False, True, 2], ['eld', False, True, 2], ['eld', False, True, 1], ['eld', True, False, 1], ['eld', True, False, 2], ['eld', False, False, 0]])
    X.columns = ['age', 'employ', 'family', 'cridet']
    #y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    y = pd.Series ([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    return X, y

def entropy(p):  
    return -np.dot(p , np.log(p) / np.log(2))

def countEntropy(X, y): #计算经验熵和经验条件熵
    num = X.shape[0]
    classes = pd.Series(y).value_counts()
    num_labels = classes.shape[0]  #统计共有类别数
    labels = classes.index
    classes = np.array(classes) / num
    E = entropy(classes)  #经验熵计算
    return E

#算法5.1 信息增益算法
def informationGain(X, y): #计算信息增益
    #print(X, y)
    num = X.shape[0]
    num_features = X.shape[1]
    E = countEntropy(X, y)
    res = np.zeros(num_features, dtype = float)
    index = 0;
    for i in X.columns:
        for j in X.groupby(i):
            x = pd.DataFrame(j[1])
            sub_num = x.shape[0]  #计算子类数据量
            #print(x.index)
            tmp = y[x.index] #获取对应数据的标签
            res[index] += countEntropy(x, tmp) * (sub_num / num)
        index += 1
            #print("this entropy is %f" % res[i])
    res = E - res
    return res

def createTrees(X, y):
    C = pd.Series(y).value_counts()
    
    if C.shape[0] == 1 or X.shape[1] == 0:  #(1) && (2)
        y = list(y)
        return y[0]
    g = informationGain(X, y)
    index_num = np.argmax(g)  #取最大信息增益值
    index_str = X.columns[index_num]
    myTree = {index_str : {}}
    if g[index_num] <= epsilon:  #(4)
        return C.argmax()
    Group = X.groupby(index_str)
    for i in Group:
        x = pd.DataFrame(i[1])
        del(x[index_str])
        
        #print(index)
        #print(myTree)
        myTree[index_str][str(i[0])] = createTrees(x, y[x.index])
    return myTree

#算法5.2 ID3算法
def ID3(X, y, epsilon):
    myTree = createTrees(X, y)
    return myTree

if __name__ == "__main__":
    X, y = createData()
    #print(X, y)
    epsilon = 0
    dTree = ID3(X, y, epsilon)
    print(dTree)
    