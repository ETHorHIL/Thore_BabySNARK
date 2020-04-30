from finitefield.finitefield import FiniteField
from finitefield.polynomial import polynomialsOver
from ssbls120 import Fp, Poly, Group
import numpy as np
import random


def _polydemo():
    p1 = Poly([1, 2, 3, 4, 5])
    print(p1)


# _polydemo()


# Example: Fp(53)
# Fp = FiniteField(53, 1)
# Poly = polynomialsOver(Fp)

G = Group.G
GT = Group.GT


# this is a global value
# initializing it for small values 1 - 128 is enough for small examples
# We'll overwrite it in the "babysnark_setup" when a larger constraint system
# is needed. And in "babysnark_opt.py" well define a different policy for
# choosing roots that leads to FFT optimization
# Fp is the integers mod P
# Fp(1)

ROOTS = [Fp(i) for i in range(128)]


def vanishing_poly(S):
    """
    args:
        S (m vector)
    returns:
        p(X) = (X -S1)*(X-S2)* ... * (X-Sm)

    """
    p = Poly([Fp(1)])
    for s in S:
        p *= Poly([-s, Fp(1)])
    return p

# p is a polynomial over Fp
# print(vanishing_poly([0, -1, -2]))
# Generate random isinstance


def random_fp():
    return Fp(random.randint(0, Fp.p-1))


def random_matrix(m, n):
    return np.array([[random_fp() for _ in range(n)] for _ in range(m)])


def generate_solved_instance(m, n):
    """
    Generates a random  circuit and satisfying witness
    U, (stmt, wit)
    """
    # Generate a, U
    a = np.array([random_fp() for i in range(n)])
    U = random_matrix(m, n)

    # Normalize U to satisfy constraints
    Ua2 = U.dot(a) * U.dot(a)
    for i in range(m):
        U[i, :] /= Ua2[i].sqrt()
    assert((U.dot(a) * U.dot(a) == 1).all())
    return U, a


U, a = generate_solved_instance(10, 12)


# Evaluate a polynomial in exponent
# returns sum(g^tau^power(i)^polycoefficinet(i))
def evaluate_in_exponent(powers_of_tau, poly):
    # powers_of_tau:
    #     [G*0, G*tau, ..., G*(Tau**m)]
    # poly:
    #    degree m-bound polynomial in coefficient form
    print("P.degree: " + poly.degree())
    print("Powers of tau: " + len(powers_of_tau))
    assert poly.degree() + 1 < len(powers_of_tau)
    return sum([powers_of_tau[i] * poly.coefficients[i] for i in
                range(poly.degree()+1)], G*0)


# setup

def babysnark_setup(U, n_stmt):
    """
    U: the matrix representing the problem equations
    n_stmt: the first l entries of the solution vectore representing the
    statement
    """
    (m, n) = U.shape
    assert n_stmt < n

    # Generate roots for each gate
    # Make sure there are roots for each row (equation)
    # Doesnt matter what values the roots have
    # Roots are public
    global ROOTS
    if len(ROOTS) < m:
        ROOTS = tuple(range(m))

    # Generate polynomials for columns of U
    # intrerpolate for points (x,y) where x's are the roots and y's are the
    # values of the k-th row
    # This is public
    Us = [Poly.interpolate(ROOTS[:y], U[:, k]) for k in range(n)]

    # Trapdoors
    # These are only known to the trusted party that generates the setup
    global tau, beta, gamma
    tau = random_fp()
    beta = random_fp()
    gamma = random_fp()

    # CRS elements
    # CRS:= [G^tau^i, G^gamma, G^(beta*gamma),
    #        G^beta()*(non_statement_polys(tau))]
    CRS = [G * (tau ** i) for i in range(m+1)] + \
          [G * gamma, G * (beta * gamma)] + \
          [G * (beta * Ui(tau)) for Ui in Us[n_stmt:]]

    # Precomputation
    # This is not considered to be part of the trusted setup, since it
    # could be computed directly from the G^tau^i since U is public

    # Compute the target poly term
    t = vanishing_poly(ROOTS[:m])
    T = G * t(tau)

    # Evaluate the Ui's corresponding to statement values
    Uis = [G * Ui(tau) for Ui in Us]
    precomp = Uis, T

    return CRS, precomp


# Prover
def babysnark_prover(U, n_stmt, CRS, precomp, a):
    """
    U: the matrix m*n representing the problem equations
    n_stmt: the first l entries of the solution vectore representing the stmt
    CRS: the common reference string, babysnark_setup()[0]
    Precomp: precomputation provided by babysnark_setup()[1]
    a: the vector [solution + witness]
    """
    (m, n) = U.shape
    assert n == len(a)
    assert len(CRS) == (m+1) + 2 + (n - n_stmt)
    assert len(ROOTS) >= m

    # parse the CRS
    taus = CRS[:m+1]
    bUis = CRS[-n-n_stmt:]

    Uis, T = precomp

    # Target is the vanishing polynomial
    t = vanishing_poly(ROOTS[:m])

    # 1. Find the polynomial p(x) this is the prover polynomials p = t * h
    # Convert the basis polynomials Us to coefficient form by interpolating
    # This is to make sure we can evaluate with the powers of tau
    Us = [Poly.interpolate([ROOTS[:m]], Uis[:, k]) for k in range(n)]

    # First compute v
    # thes are the polynomials calculated out of the basis polynomials and the
    # solution
    v = Poly([])
    for k in range(n):
        v += Us[k] * a[k]

    # Finally p
    p = v * v - 1

    # compute the H term, i.e. cofactor H so that P = T * H
    h = p/t
    assert p == h*t

    # first part of the proof
    H = evaluate_in_exponent(taus, h)

    # 3. compute the Vw terms using precomputed Uis
    # Provers solution polynomials evaluated at tau
    Vw = sum([Uis[k] * a[k] for k in range(n_stmt, n)], G*0)
    # assert G * vw(tau) == Vw

    # 4. Compute the Bw terms, i.e. the shifted evaluations to make sure P can
    # not fiddle around with the polynomials
    Bw = sum([bUis[k-n_stmt] * a[k] for k in range(n_stmt, n)], G * 0)
    # assert G * beta *vw(tau) ==Bw

    # V = G * v(tau)
    # assert H.pair(T) * GT == V.pair(V)
    # print('H:', H)
    # print('Bw:', Bw)
    # print('Vw:', Vw)
    return H, Bw, Vw
