import numpy as np

def creatData():
    x = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    return x, y

def getGram(x):
    return np.dot(x, x.T)

def check(w, b):
    res = np.dot(w, x.T) + b
    res[res < 0]  = -1
    res[res > 0] = 1
    if (res == y).all():
        return True
    else:
        return False

def cal(i):
    res = np.dot(alpha * y , g[i, :]) + b;
    if (res < 0):
        return -1
    else:
        return 1

def update(i):
    global alpha, b
    alpha[i] += eite
    b += y[i]

if __name__ == "__main__":
    x, y = creatData()
    g = getGram(x)
    alpha = np.zeros(len(y))
    b = 0
    eite = 1
    w = np.zeros(len(x[1]), np.float)
    print(w)
    #迭代循环
    while not check(w, b):
        w = np.dot(alpha * y, x);
        for i in range(len(x)):
            #print(alpha, b)
            if cal(i) != y[i]:
                update(i)
    
    print(w, b)
