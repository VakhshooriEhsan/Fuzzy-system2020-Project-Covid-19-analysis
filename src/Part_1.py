import pandas as pd
import numpy as np
import math

def readData(addr):
    _data = pd.read_csv(addr)
    _data -= _data.mean()
    _data /= np.sqrt(_data.var())
    _data = _data.values
    return _data

def correlation(_x, _y):

    res = np.zeros((len(_x[0]), 2))

    x1 = _x[_y[:, 0]<0]
    x_avg = pd.DataFrame(x1).mean().values
    for j in range(len(x_avg)):
        tmp1 = 0
        tmp2 = 0
        for i in range(len(x1)):
            if(not math.isnan(x1[i][j])):
                tmp1 += (x1[i][j]-x_avg[j])
                tmp2 += (x1[i][j]-x_avg[j])**2
        res[j][0] = tmp1

    x2 = _x[_y[:, 0]>0]
    x_avg = pd.DataFrame(x2).mean().values
    for j in range(len(x_avg)):
        tmp1 = 0
        tmp2 = 0
        for i in range(len(x2)):
            if(not math.isnan(x2[i][j])):
                tmp1 += (x2[i][j]-x_avg[j])
                tmp2 += (x2[i][j]-x_avg[j])**2
        res[j][1] = tmp1
    
    return res

def minmax(a, b):
    res = np.zeros((len(a), len(b[0])))
    for i in range(len(a)):
        for j in range(len(b[0])):
            mx = -1000
            for k in range(len(a[0])):
                mn = b[k][j]
                if(a[i][k]!=np.NaN and a[i][k]<b[k][j]):
                    mn = a[i][k]
                if(mn>mx):
                    mx = mn
            res[i][j] = mx
    return res

# ------------------------ Read and normalize data -------------------------

_data = readData('Datas/data.csv')
_xd = _data[:, 1:15]
_yd = _data[:, [15]]

# -------------------------- Train and test data --------------------------

md = len(_xd)
mt = int(md/2)
m = md-mt

_x = _xd[:int(m), :]
_y = _yd[:int(m), :]
_xt = _xd[:int(mt), :]
_yt = _yd[:int(mt), :]

j = 0
for i in np.random.permutation(md).tolist():
    if(j<m):
        _x[j] = _xd[i]
        _y[j] = _yd[i]
    else:
        _xt[md-j-1] = _xd[i]
        _yt[md-j-1] = _yd[i]
    j += 1

# ------------------------------- Relation -------------------------------

# s = correlation(_x, _y)

s1 = pd.DataFrame(_x[_y[:, 0]<0]).mean().values
s1 = s1[..., None]
s2 = pd.DataFrame(_x[_y[:, 0]>0]).mean().values
s2 = s2[..., None]
s = np.append(s1, s2, axis=1)

print("Relation:")
print(s)

# -------------------------------- Test --------------------------------

res = minmax(_xt, s)

k = 0.0
for i in range(len(res)):
    if((res[i][0] > res[i][1] and _yt[i]<0) or (res[i][0] < res[i][1] and _yt[i]>0)):
        k += 1
k = k/len(res)

print("Accuracy: " + str(k))
