import pandas as pd
import numpy as np

def readData(addr):
    _data = pd.read_csv(addr)
    _data -= _data.mean()
    _data /= np.sqrt(_data.var())
    _data = _data.values
    return _data

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

# ------------------------------ Read Datas ------------------------------

_data = readData('Datas/data.csv')
_xd = _data[:, 1:15]
_yd = _data[:, [15]]

# ------------------------ make main and test data ------------------------

md = len(_xd)
mt = int(md/2)
m = md-mt

_x = _xd[:int(m), :]
_y = _yd[:int(m), :]
_xt = _xd[:int(mt), :]
_yt = _yd[:int(mt), :]

# ------------------------------- make alpha -------------------------------

j = 0
for i in np.random.permutation(md).tolist():
    if(j<m):
        _x[j] = _xd[i]
        _y[j] = _yd[i]
    else:
        _xt[md-j-1] = _xd[i]
        _yt[md-j-1] = _yd[i]
    j += 1

s1 = pd.DataFrame(_x[_y[:, 0]<0]).mean().values
s1 = s1[..., None]
s2 = pd.DataFrame(_x[_y[:, 0]>0]).mean().values
s2 = s2[..., None]
s = np.append(s1, s2, axis=1)

print("Alpha:")
print(s)

# --------------------------------- Test ---------------------------------

res = minmax(_xt, s)

k = 0.0
for i in range(len(res)):
    if((res[i][0] < res[i][1] and _yt[i]<0) or (res[i][0] > res[i][1] and _yt[i]>0)):
        k += 1
k = k/len(res)

print("Accuracy: " + str(k))
