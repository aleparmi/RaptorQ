from numpy import *
from itertools import count

pcr = {}

def padding(Cp, K):
    #Adds the padding to the source block 5.3.1
    si = sys_index(K)
    Kp = si[0]
    Cp = append(Cp, zeros(Kp-K))
    return Cp

def sys_index(K):
    #fetches the systematic indices table 5.6
    systematic_indices = loadtxt("systematic_indices.txt", dtype='i', delimiter='\t')
    i=0
    while K>=systematic_indices[i][0]:
        i=i+1
    return systematic_indices[i]

def oct_sum(u, v): return u^v #octet sum 5.7.2

def oct_mult(u, v):
    #octet multiplication 5.7.2
    if u==0 or v ==0: return 0
    OCT_EXP = loadtxt("OCT_EXP.txt", dtype='i', delimiter='\n')
    OCT_LOG = loadtxt("OCT_LOG.txt", dtype='i', delimiter='\n')
    return OCT_EXP[OCT_LOG[u-1] + OCT_LOG[v-1]]

def matrix_oct_mult(U, V):
    #matrix dot multiplication with GF(256) octet multiplication
    temp_sum = 0
    if shape(U)[1] == None and shape(V)[1] == None:
        for i in range(len(U)):
            temp_sum = oct_sum(temp_sum, oct_mult(U[i], V[i]))
        return temp_sum
    assert shape(U)[1] == shape(V)[0], "Wrong size requirements for matrix dot multiplication"
    temp_sum = 0
    W = zeros((shape(U)[0], shape(V)[1]))
    for i in range (shape(U)[0]):
        for z in range(shape(V)[1]):
            for j in range (shape(U)[1]):
                temp_sum = oct_sum(temp_sum, oct_mult(U[i][j], V[j][z])) 

            W[i][z] = temp_sum
            temp_sum = 0x00
    return W

def oct_div(u, v):
    #octet division 5.7.2
    if u==0: return 0
    OCT_EXP = loadtxt("OCT_EXP.txt", dtype='i', delimiter='\n')
    OCT_LOG = loadtxt("OCT_LOG.txt", dtype='i', delimiter='\n')
    return OCT_EXP[OCT_LOG[u] - OCT_LOG[v] + 255]

def alphai(i):
    # octet alpha**i 5.7.2
    assert (i>=0 and i<256), "alpha**i not valid, i out of range"
    OCT_EXP = loadtxt("OCT_EXP.txt", dtype='i', delimiter='\n')
    return OCT_EXP[i]

def gen_paramaters(K):
     #establishes the pre-coding parameters 5.3.3.3
    si = sys_index(K)
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

    pcr.update({"K": K, "Kp": Kp, "L": L, "S": S, "H": H, "B": B,
    "W": W, "P": P, "P1": P1, "J": J})

def pre_coding_relationships():
    S = pcr["S"]
    H = pcr["H"]
    B = pcr["B"]
    Kp = pcr["Kp"]
    W = pcr["W"]
    P = pcr["P"]
   
    D = C[B:B+S]

    for i in range(B-1):
        a = 1+floor(i/S)
        b = i % S
        D[b] = D[b] ^ C[i]
        b = (b + a) % S
        D[b] = D[b] ^ C[i]
        b = (b + a) % S
        D[b] = D[b] ^ C[i]
    
    G_LDPC1 = D
    #del(D)
    #D = C[B:B+S]

    for i in range(S-1):
        a = i%P
        b = (i+1)%P
        D[i] = D[i] ^ C[W+a] ^ C[W+b]
    
    G_LDPC2 = D

    MT = zeros((H, Kp+S))
    for j in range(0,Kp+S-2):
        for i in range(0,H-1):
            if (i == Rand(j+1, 6, H) or 
            i == (Rand(j+1,6,H) + Rand(j+1,7,H-1) + 1) % H):
                MT[i, j] = 0x01

            if j == Kp+S-1: MT[i, j] = alphai(i)
            else: MT[i, j] = 0x00
        
    GAMMA = zeros((Kp+S, Kp+S))
    for j in range(Kp+S):
        for i in range(Kp+S):
                if i>=j: GAMMA[i, j] = alphai(i-j)
                else: GAMMA[i, j] = 0x00

    pcr.update({"G_LDPC1": G_LDPC1, 
    "G_LDPC2": G_LDPC2, "GAMMA": GAMMA, "MT": MT})

def pre_coding_conditions(C):
    G_LDPC1 = pcr["G_LDPC1"]
    G_LDPC2 = pcr["G_LDPC2"]
    GAMMA = pcr["GAMMA"]
    MT = pcr["MT"]
    B = pcr["B"]
    W = pcr["W"]
    P = pcr["P"]
    Kp = pcr["Kp"]
    S = pcr["S"]

    mul1 = matrix_oct_mult(G_LDPC1, transpose(C[0:B-1]))
    mul2 = matrix_oct_mult(G_LDPC2, transpose(C[W:W+P-1]))
    sum1 = []
    #vectors component wise sum
    for i in range(len(mul1)): sum1[i] = mul1[i] ^ mul2[i] ^ C[B+i]

    assert not any(sum1), "Pre Coding Condition 1 not satisfied"

    mul1 = matrix_oct_mult(MT, GAMMA)
    mul2 = matrix_oct_mult(mul1, transpose(C[0:Kp+S-1]))
    sum1 = []
    #vectors component wise sum
    for i in range(len(mul2)): sum1[i] = mul2[i] ^ C[Kp+S+i]
    
    assert not any(sum1), "Pre Coding Condition 2 not satisfied"

def Rand(y, i, m):
    #random number generator 5.3.5.1
    x0 = (y + i) % 2**8
    x1 = int((floor(y / 2**8) + i) % 2**8)
    x2 = int((floor(y / 2**16) + i) % 2**8)
    x3 = int((floor(y / 2**24) + i) % 2**8)

    V0 = loadtxt("V0.txt", dtype="int64", delimiter="\n")
    V1 = loadtxt("V1.txt", dtype="int64", delimiter="\n")
    V2 = loadtxt("V2.txt", dtype="int64", delimiter="\n")
    V3 = loadtxt("V3.txt", dtype="int64", delimiter="\n")

    return (V0[x0] ^ V1[x1] ^ V2[x2] ^ V3[x3]) % m

def TupleGen(Kp, X):
    #generate the tuple for the encoding process 5.3.5.4
    J = pcr["J"]
    W = pcr["W"]
    P1 = pcr["P1"]

    A = 53591 + J*997
    if A % 2 == 0: A = A + 1
    B = 10267*(J+1)
    y = (B + X*A) % 2**32
    v = Rand(y, 0, 2**20)
    d = Deg(v, W)
    a = 1 + Rand(y, 1, W-1)
    b = Rand(y, 2, W)
    if d < 4: d1 = 2 + Rand(X, 3, 2)
    else: d1 = 2
    a1 = 1 + Rand(X, 4, P1-1)
    b1 = Rand(X, 5, P1)

    return (d, a, b, d1, a1, b1)

def Deg(v, W):
    #Degree Generator 5.3.5.2
    if v >= 2**20: raise NameError("v in Degree Generator is >= 2**20")
    fd = loadtxt("degree_distribution.txt", dtype="int", delimiter="\t")
    d=1
    for d in range(len(fd)):
        if v >= fd[d-1][1] and v < fd[d][1]: return min(fd[d][0], W-2)

def intermediate_symbols(Cp):
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

    D = zeros((S + H + Kp, 1))
    D[S+H:S+H+Kp] = Cp
    G_LDPC1 = pcr["G_LDPC1"]
    G_LDPC2 = pcr["G_LDPC2"]
    G_HDPC = matrix_oct_mult(GAMMA, MT)
    I_S = identity(S)
    I_H = identity(H)
    G_ENC = zeros((Kp, L))
    for i in range(Kp): #G_ENC generation
        (d, a, b, d1, a1, b1) = TupleGen(Kp, i)
        assert ((d >=0 and a >=1 and a <= W-1 and b >=0 and b <= W-1 and
        (d1 == 2 or d1 == 3) and a1 >= 1 and a1 <= P1-1 and b1 >=0 and
        b1 <= P1-1), "Wrong constraints for TupleGen")
        G_ENC[i, b] = 1
        for j in range(1, d):
            b = (b+a) % W
            G_ENC[i, b] = 1
        while(b1 >= P): b1 = (b1 + a1) % P1
        G_ENC[i, W + b1] = 1
        for j in range(1, d1):
            b1 = (b1 + a1) % P1
            while(b1 >= P): b1 = (b1 + a1) % P1
            G_ENC[i, W + b1] = 1
    
    A = bmat([[G_LDPC1, I_S, G_LDPC2], [G_HDPC, I_H], [G_ENC]]) #figure 5
    C = matrix_oct_mult(invert(A), D)
    return C

def encoder(Kp, C):
    #encoding symbol generator 5.3.5.3
    W = pcr["W"]
    P = pcr["P"]
    P1 = pcr["P1"]
    (d, a, b, d1, a1, b1) = TupleGen(Kp, i)
    assert ((d >=0 and a >=1 and a <= W-1 and b >=0 and b <= W-1 and
    (d1 == 2 or d1 == 3) and a1 >= 1 and a1 <= P1-1 and b1 >=0 and
    b1 <= P1-1), "Wrong constraints for TupleGen")
    result = C[b]
    for j in range(1, d):
        b = (b+a) % W
        result = result + C[b]
    while(b1 >= P): b1 = (b1 + a1) % P1
    result = result + C[W + b1]
    for j in range(1, d1):
        b1 = (b1 + a1) % P1
        while(b1 >= P): b1 = (b1 + a1) % P1
        result = result + C[W + b1]

    return result

def main():
    #pre_coding_relationships #todo
    '''Encoding flow: Padding, Intermediate Symbol Generation,
        pre-coding relationship, encoding'''
    Cp = array([[2], [1], [0], [2], [1], [0], [2], [1], [0], [2], [1], [0]])
    K = len(Cp)
    Cp = padding(Cp, K)
    gen_paramaters(K)
    pre_coding_relationships()
    C = intermediate_symbols(Cp)
    encoder(Kp, C)
    pre_coding_conditions(C)

    print(pcr)

if __name__ == "__main__":
    main()