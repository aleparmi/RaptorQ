from numpy import shape
from scipy import sparse as sp

def oct_sum(u, v): return u^v #octet sum 5.7.2
def oct_diff(u, v): return u^v #octet difference 5.7.2

def oct_mult(u, v, OCT_EXP, OCT_LOG):
    #octet multiplication 5.7.2
    if u==0 or v ==0: return 0
    return OCT_EXP[OCT_LOG[int(u) - 1] + OCT_LOG[int(v) - 1]]

def oct_div(u, v, OCT_EXP, OCT_LOG):
    #octet division 5.7.2
    if u==0: return 0
    return OCT_EXP[OCT_LOG[u - 1] - OCT_LOG[v - 1] + 255]

def alphai(i, OCT_EXP):
    # octet alpha**i 5.7.2
    assert (i>=0 and i<256), "alpha**i not valid, i out of range"
    return OCT_EXP[i]

def matrix_oct_mult(U, V, OCT_EXP, OCT_LOG):
    #matrix dot multiplication with GF(256) octet multiplication
    temp_sum = 0
    if shape(U)[1] == None and shape(V)[1] == None:
        for i in range(len(U)):
            temp_sum = oct_sum(temp_sum, oct_mult(U[i], V[i], OCT_EXP, OCT_LOG))
        return temp_sum
    assert shape(U)[1] == shape(V)[0], "Wrong size requirements for matrix dot multiplication"
    temp_sum = 0
    #W = zeros((shape(U)[0], shape(V)[1]))
    W = sp.lil_matrix((shape(U)[0], shape(V)[1]), dtype = int)
    for i in range (shape(U)[0]):
        for z in range(shape(V)[1]):
            for j in range (shape(U)[1]):
                 temp_sum = oct_sum(temp_sum, oct_mult(U[i, j], V[j, z], OCT_EXP, OCT_LOG))

            #W[i][z] = temp_sum
            W[i, z] = temp_sum
            temp_sum = 0
    return W