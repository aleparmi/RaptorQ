from oct_utlities import alphai, matrix_oct_mult, oct_sum
from invert_oct_matrix import invert_matrix

from numpy import *
from itertools import count
from scipy import sparse as sp
from scipy.sparse import linalg as sl
import time
import pickle

def sys_index(K):
    #fetches the systematic indices table 5.6
    systematic_indices = loadtxt("systematic_indices.txt", dtype='i', delimiter='\t')
    i=0
    while K>=systematic_indices[i][0]:
        i=i+1
    return systematic_indices[i]

def gen_paramaters(K, si):
     #establishes the pre-coding parameters 5.3.3.3
    systematic_indices = si
    Kp = si[0]
    J = si[1]
    S = si[2]
    H = si[3]
    W = si[4]
    L = Kp + S + H
    P = L - W

    #P1 is equal to the smallest prime number >= P
    for i in count(start=P):
        if i > 1:
            for j in range(2,i):
                if (i % j) == 0: break
            else:
                P1 = i
                break
    
    B = W - S

    OCT_EXP = loadtxt("OCT_EXP.txt", dtype='i', delimiter='\n')
    OCT_LOG = loadtxt("OCT_LOG.txt", dtype='i', delimiter='\n')

    V0 = loadtxt("V0.txt", dtype="int64", delimiter="\n")
    V1 = loadtxt("V1.txt", dtype="int64", delimiter="\n")
    V2 = loadtxt("V2.txt", dtype="int64", delimiter="\n")
    V3 = loadtxt("V3.txt", dtype="int64", delimiter="\n")

    fd = loadtxt("degree_distribution.txt", dtype="int", delimiter="\t")

    pcr = {"K": K, "Kp": Kp, "L": L, "S": S, "H": H, "B": B,
    "W": W, "P": P, "P1": P1, "J": J, "OCT_EXP": OCT_EXP, 
    "OCT_LOG": OCT_LOG, "V0": V0, "V1": V1, "V2": V2, "V3": V3,
    "fd": fd, "systematic_indices": systematic_indices}
    return pcr

def padding(Cp, K):
    #Adds the padding to the source block 5.3.1
    si = sys_index(K)
    Kp = si[0]
    for _ in range(Kp - K): Cp.extend(bytearray([0]))
    return (Cp, si)

def pre_coding_relationships(pcr):
    S = pcr["S"]
    H = pcr["H"]
    B = pcr["B"]
    Kp = pcr["Kp"]
    P = pcr["P"]
    OCT_EXP = pcr["OCT_EXP"]
    V0 = pcr["V0"]
    V1 = pcr["V1"]
    V2 = pcr["V2"]
    V3 = pcr["V3"]
   
    '''LDPC1: creates an LDPC cyclic matrix by assigning ones at columns
    0, i+1 and 2*(i+1) and down-shifting them. Not described in the RFC. See:
    https://fenrirproject.org/Luker/libRaptorQ/-/blob/master/src/RaptorQ/v1/Precode_Matrix_Init.hpp'''
    
    vals = array([1, 1, 1, 1, 1])
    offsets = array([-S + 1, -S + 2, 0, 1, 2])
    G_LDPC11 = sp.diags(vals, offsets, shape = (S, S))
    vals = array([1, 1, 1])
    offsets = array([0, 2, 4])
    G_LDPC12 = sp.diags(vals, offsets, shape = (B-S, S))
    G_LDPC1 = sp.vstack([G_LDPC11, G_LDPC12])
    del(G_LDPC11)
    del(G_LDPC12)
    G_LDPC1 = sp.coo_matrix(G_LDPC1, dtype = int)
    G_LDPC1 = sp.coo_matrix.transpose(G_LDPC1)

    '''LDPC2: creates an LDPC cyclic matrix by assigning two consecutive
    ones in the first rows, first two columns and then right-shifting.
    Not described in the RFC. See:
    https://fenrirproject.org/Luker/libRaptorQ/-/blob/master/src/RaptorQ/v1/Precode_Matrix_Init.hpp'''
    z = ones(S)
    G_LDPC2 = sp.diags([z, z], [0, 1], 
    shape = (S, P), format = 'coo', dtype = int)

    MT = sp.dok_matrix((H, Kp + S), dtype = int)
    for i in range(0,H):
        for j in range(0,Kp+S-1):
            temp = Rand(j+1, 6, H, V0, V1, V2, V3)
            if (i == temp or i == (temp + Rand(j+1,7,H-1, V0, V1, V2, V3) + 1) % H):
                MT[i, j] = 1

        MT[i, j + 1] = alphai(i, OCT_EXP)

    MT = MT.tocoo()
    
    temp = array([])
    for i in range(Kp+S):
        temp = concatenate((temp, array([OCT_EXP[i%255]])))
    GAMMA = sp.diags(temp[::-1], arange(-(Kp+S)+1, 1), 
    shape = (Kp + S, Kp + S), format = 'coo', dtype = int)

    pcr.update({"G_LDPC1": G_LDPC1, 
    "G_LDPC2": G_LDPC2, "GAMMA": GAMMA, "MT": MT})
    return pcr

def intermediate_symbols(Cp, pcr):
    #intermediate symbol generation 5.3.3.4.2
    L = pcr["L"]
    S = pcr["S"]
    H = pcr["H"]
    Kp = pcr["Kp"]
    G_LDPC1 = pcr["G_LDPC1"]
    G_LDPC2 = pcr["G_LDPC2"]
    GAMMA = pcr["GAMMA"]
    MT = pcr["MT"]
    P1 = pcr["P1"]
    W = pcr["W"]
    P = pcr["P"]
    OCT_EXP = pcr["OCT_EXP"]
    OCT_LOG = pcr["OCT_LOG"]

    D = sp.csr_matrix((Cp, (arange(S + H, S + H + Kp), 
    zeros(Kp))), shape = (S + H + Kp, 1), dtype = int)
    start1 = time.time()
    G_HDPC = matrix_oct_mult(MT.todok(), GAMMA.todok(), OCT_EXP, OCT_LOG)
    print("matrix oct mult: ", time.time() - start1)
    I_S = sp.identity(S, format = 'csr')
    I_H = sp.identity(H, format = 'csr')
    G_ENC = sp.dok_matrix((Kp, L))
    for i in range(Kp): #G_ENC generation
        (d, a, b, d1, a1, b1) = TupleGen(i, pcr)
        assert (d >=0 and a >=1 and a <= W-1 and b >=0 and b <= W-1 and \
        (d1 == 2 or d1 == 3) and a1 >= 1 and a1 <= P1-1 and b1 >=0 and \
        b1 <= P1-1), "Wrong constraints for TupleGen"
        G_ENC[i, b] = 1
        for _ in range(1, d):
            b = (b+a) % W
            G_ENC[i, b] = 1
        while(b1 >= P): b1 = (b1 + a1) % P1
        G_ENC[i, W + b1] = 1
        for _ in range(1, d1):
            b1 = (b1 + a1) % P1
            while(b1 >= P): b1 = (b1 + a1) % P1
            G_ENC[i, W + b1] = 1
    
    G_ENC = sp.csr_matrix(G_ENC)
    B = sp.hstack([G_LDPC1, I_S, G_LDPC2])
    C = sp.hstack([G_HDPC, I_H])
    A = sp.vstack([B, C, G_ENC], dtype = int)
    A = A.tolil()
    start2 = time.time()
    C = matrix_oct_mult(invert_matrix(A, OCT_EXP, OCT_LOG), D, OCT_EXP, OCT_LOG)
    print(time.time() - start2)
    pcr.update({"A": A})
    return (C.astype(int), pcr)

def pre_coding_conditions(C, pcr):
    G_LDPC1 = pcr["G_LDPC1"]
    G_LDPC2 = pcr["G_LDPC2"]
    GAMMA = pcr["GAMMA"]
    MT = pcr["MT"]
    B = pcr["B"]
    W = pcr["W"]
    P = pcr["P"]
    Kp = pcr["Kp"]
    S = pcr["S"]
    OCT_EXP = pcr["OCT_EXP"]
    OCT_LOG = pcr["OCT_LOG"]

    mul1 = matrix_oct_mult(G_LDPC1.tocsr(), C[0:B].tocsr(), OCT_EXP, OCT_LOG)
    mul2 = matrix_oct_mult(G_LDPC2.tocsr(), C[W:W+P].tocsr(), OCT_EXP, OCT_LOG)
    sum1 = sp.dok_matrix((mul1.shape[0], 1), dtype = int)
    #vectors component wise sum
    for i in range(mul1.shape[0]): sum1[i, 0] = oct_sum(oct_sum(mul1[i, 0], mul2[i, 0]), C[B+i, 0])

    assert sum1.count_nonzero() == 0, "Pre Coding Condition 1 not satisfied"

    mul1 = matrix_oct_mult(MT.tocsr(), GAMMA.tocsr(), OCT_EXP, OCT_LOG)
    mul2 = matrix_oct_mult(mul1.tocsr(), C[0:Kp+S].tocsr(), OCT_EXP, OCT_LOG)
    sum1 = sp.dok_matrix((mul2.shape[0], 1))
    #vectors component wise sum
    for i in range(mul2.shape[0]): sum1[i, 0] = oct_sum(mul2[i, 0], C[Kp+S+i, 0])
    
    assert sum1.count_nonzero() == 0, "Pre Coding Condition 2 not satisfied"

def Rand(y, i, m, V0, V1, V2, V3):
    #random number generator 5.3.5.1
    x0 = (y + i) % 2**8
    x1 = int((floor(y / 2**8) + i) % 2**8)
    x2 = int((floor(y / 2**16) + i) % 2**8)
    x3 = int((floor(y / 2**24) + i) % 2**8)

    return (V0[x0] ^ V1[x1] ^ V2[x2] ^ V3[x3]) % m

def TupleGen(X, pcr):
    #generate the tuple for the encoding process 5.3.5.4
    J = pcr["J"]
    W = pcr["W"]
    P1 = pcr["P1"]
    fd = pcr["fd"]
    V0 = pcr["V0"]
    V1 = pcr["V1"]
    V2 = pcr["V2"]
    V3 = pcr["V3"]

    A = 53591 + J*997
    if A % 2 == 0: A = A + 1
    B = 10267*(J+1)
    y = (B + X*A) % 2**32
    v = Rand(y, 0, 2**20, V0, V1, V2, V3)
    d = Deg(v, W, fd)
    a = 1 + Rand(y, 1, W-1, V0, V1, V2, V3)
    b = Rand(y, 2, W, V0, V1, V2, V3)
    if d < 4: d1 = 2 + Rand(X, 3, 2, V0, V1, V2, V3)
    else: d1 = 2
    a1 = 1 + Rand(X, 4, P1-1, V0, V1, V2, V3)
    b1 = Rand(X, 5, P1, V0, V1, V2, V3)

    return (d, a, b, d1, a1, b1)

def Deg(v, W, fd):
    #Degree Generator 5.3.5.2
    if v >= 2**20: raise NameError("v in Degree Generator is >= 2**20")
    d=1
    for d in range(len(fd)):
        if v >= fd[d-1][1] and v < fd[d][1]: return min(fd[d][0], W-2)

def encoder(C, ISI, pcr):
    #encoding symbol generator 5.3.5.3
    P = pcr["P"]
    P1 = pcr["P1"]
    W = pcr["W"]

    (d, a, b, d1, a1, b1) = TupleGen(ISI, pcr)
    assert (d >=0 and a >=1 and a <= W-1 and b >=0 and b <= W-1 and \
    (d1 == 2 or d1 == 3) and a1 >= 1 and a1 <= P1-1 and b1 >=0 and \
    b1 <= P1-1), "Wrong constraints for TupleGen"

    result = C[b, 0]
    for _ in range(1, d):
        b = (b+a) % W
        result = oct_sum(result, C[b,0])
    while(b1 >= P): b1 = (b1 + a1) % P1
    result = oct_sum(result, C[(W + b1),0])
    for _ in range(1, d1):
        b1 = (b1 + a1) % P1
        while(b1 >= P): b1 = (b1 + a1) % P1
        result = oct_sum(result, C[(W + b1), 0])

    return result

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def main():
    start = time.time()
    #filename = '/Users/Andrea/Pictures/Album Cover/download.jpg'
    #f = open(filename, 'rb')
    #Cp = bytearray(f.read())
    #f.close()
    Cp = bytearray([79, 6, 99, 250, 45, 63, 12, 0, 136, 200, 100])
    ISI = 0
    K = len(Cp)
    (Cp, si) = padding(Cp, K)
    pcr = gen_paramaters(K, si)
    pcr = pre_coding_relationships(pcr)
    (C, pcr) = intermediate_symbols(Cp, pcr)
    pre_coding_conditions(C, pcr)
    encoded_block = bytearray([])
    for i in range(C.shape[0]):
        ISI = i
        encoded_symbol = encoder(C, ISI, pcr)
        encoded_block = append(encoded_block, encoded_symbol)
    
    save_obj(pcr, "pcr")
    save_obj(encoded_block, "encoded_block")

    print("total: ", time.time() - start)

    print(encoded_block)
    return encoded_block

if __name__ == "__main__":
    main()