import pickle
from numpy import *
import scipy.sparse as sp

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def CfromD(pcr, C):
    S = pcr["S"]
    H = pcr["H"]
    L = pcr["L"]

    N = len(C)
    M = S + H + N

    c = arange(0, L)
    d = arange(0, M)

    pcr.update({'M': M, 'N': N})

def scheduler(pcr):
    A = pcr["A"]
    P = pcr["P"]
    X = A[:] #check lower triangular matrix

    i = 0
    u = P

    I = sp.identity(i)
    #fill matrix with zeros in two parts when creating A 2), 3)
    U = X[:, -u:]
    V = X[i:, i:u+1]

    nz = lambda nc: nc.count_nonzero()
    original_degrees = array(list(map(nz, V)))

    #i and u iterators until i + u = L

    assert (V.count_nonzero() != 0), "Decoding failure: all nodes are = 0"

    r = min(original_degrees) #minimum number of ONES (important!)
    rows_with_r = argwhere(original_degrees == r)

    if (r != 2):
        non_hdpc_rows = argwhere(rows_with_r < S and rows_with_r >= S + H)
        if (non_hdpc_rows != None):  row = non_hdpc_rows[0, 0]
        else : row = rows_with_r[0, 0]

    on = lambda o: V[o].count(1)
    rows_with_ones = map(on, rows_with_r)

    if (r == 2 and rows_with_ones.count(2) >= 0):
        
    









def main():
    pcr = load_obj("pcr")
    encoded_block = load_obj("encoded_block")

if __name__ == "__main__":
    main()