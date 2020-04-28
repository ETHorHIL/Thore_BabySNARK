from finitefield.finitefield import FiniteField
from finitefield.polynomial import polynomialsOver
from ssbls120 import Fp, Poly, Group
import numpy as np
import random


def _polydemo():
    p1 = Poly([1, 2, 3, 4, 5])
    print(p1)


_polydemo()


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
# print(U)


# Evaluate a polynomial in exponent

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
