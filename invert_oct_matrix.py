from oct_utlities import oct_sum, oct_diff, oct_mult, oct_div, matrix_oct_mult

from scipy.sparse import eye, lil_matrix, csr_matrix
from numpy import ones, array, loadtxt, append, shape
import time

# Linear Regression - Library Free, i.e. no numpy or scipy 

def check_squareness(A):
    """
    Makes sure that a matrix is square
        :param A: The matrix to be checked.
    """
    if A.shape[0] != A.shape[1]:
        raise ArithmeticError("Matrix must be square to inverse.")

'''
def determinant(A, OCT_EXP, OCT_LOG, total=0):
    indices = list(range(A.shape[0]))
    
    if A.shape[1] == 2 and A.shape[0] == 2:
        val = oct_diff(oct_mult(A[0,0], A[1,1], OCT_EXP, OCT_LOG), 
        oct_mult(A[1,0], A[0,1], OCT_EXP, OCT_LOG))
        return val

    for fc in indices:
        As = A
        As = As[1:]
        height = As.shape[0]
        Am = lil_matrix((height, len(indices) -1))

        for i in range(height):
            Am[i] = append((As[i,0:fc].toarray()), (As[i,fc+1:].toarray()))

        sub_det = determinant(Am, OCT_EXP, OCT_LOG)
        total = oct_sum(total, oct_mult(A[0,fc], sub_det, OCT_EXP, OCT_LOG))

    return total
'''
'''
def check_non_singular(A, OCT_EXP, OCT_LOG):
    det = determinant(A, OCT_EXP, OCT_LOG)
    if det != 0:
        return det
    else:
        raise ArithmeticError("Singular Matrix!")
'''

def invert_matrix(AM, OCT_EXP, OCT_LOG, tol=None):
    """
    Returns the inverse of the passed in matrix.
        :param A: The matrix to be inversed
        :return: The inverse of the matrix A
    """
    start3 = time.time()
    # Section 1: Make sure A can be inverted.
    check_squareness(AM)
    #check_non_singular(AM, OCT_EXP, OCT_LOG)

    # Section 2: Make copies of A & I, AM & IM, to use for row operations
    n = AM.shape[0]
    IM = eye(n, dtype = int).todok()

    # Section 3: Perform row operations
    indices = list(range(n)) # to allow flexible row referencing ***
    for fd in range(n): # fd stands for focus diagonal
        fdScaler = oct_div(1, AM[fd,fd], OCT_EXP, OCT_LOG)
        # FIRST: scale fd row with fd inverse. 
        for j in range(n): # Use j to indicate column looping.
            AM[fd,j] = oct_mult(AM[fd,j], fdScaler, OCT_EXP, OCT_LOG)
            IM[fd,j] = oct_mult(IM[fd,j], fdScaler, OCT_EXP, OCT_LOG)
        # SECOND: operate on all rows except fd row as follows:
        for i in indices[0:fd] + indices[fd+1:]: # *** skip row with fd in it.
            crScaler = AM[i,fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM[i,j] = oct_diff(AM[i,j], oct_mult(crScaler, AM[fd,j], OCT_EXP, OCT_LOG))
                IM[i,j] = oct_diff(IM[i,j], oct_mult(crScaler, IM[fd,j], OCT_EXP, OCT_LOG))
    
    print("inverse time: ", time.time() - start3)
    return IM

from pyfinite.genericmatrix import DotProduct

if __name__ == "__main__":
    
    A = [[255, 0, 2], [1, 0, 0], [0, 1, 2]]
    B = [[1, 2, 255], [1, 0, 0], [0, 1, 2]]
    OCT_EXP = loadtxt("OCT_EXP.txt", dtype='i', delimiter='\n')
    OCT_LOG = loadtxt("OCT_LOG.txt", dtype='i', delimiter='\n')
    mul = lambda u, v: oct_mult(u, v, OCT_EXP, OCT_LOG)
    add = lambda u, v: oct_sum(u, v)

    C = DotProduct(mul, add, A, B)

    print(C)