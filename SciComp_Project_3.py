import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot
from mpl_toolkits.mplot3d import Axes3D
import timeit
import scipy
from scipy import optimize

Arg_lines = np.load("/Users/BareMarcP/Desktop/Mads_stuff/Projects/Mads_Projects/Project 3/Ar-lines.npy 2")
r = np.random
r.seed(42)

# Plotting the atoms in system
def show3d(data):
    data.shape = 3, int( len(data) / 3 )
    fig = figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[0], data[1], data[2] )
    data.shape = data.size
    # fig.savefig("Arg_pot.eps")

print("Initial configuration of Argon atoms")
print(show3d((Arg_lines)))

# Constants for Argon atoms
sigma = 3.401
epsilon = 0.997
A = 4*epsilon*(sigma**12)
B = 4*epsilon*(sigma**6)

# potential function
def Lennard(r):
    V = A/r**12 - B/r**6
    return V

# plot of the potential between to Argon atoms
Two_argon = np.linspace(1.65,10,1000)

fig, ax  = plt.subplots()
plt.plot(Two_argon, Lennard(Two_argon), label = "LJ-Potential")
# plt.plot(np.ones(1000)*Bisection(3,4)[0],np.linspace(-1,3,1000), label = "Bisection-root")
ax.set_xlim(3,8)
ax.set_ylim(-1.2,3)
ax.set( xlabel = 'Relative Distance [r]', ylabel = 'Potential [V]')
ax.legend()
ax.grid(True)
plt.show()
# fig.savefig("Bisection.eps")

# Bisection method for finding minimum

def Bisection(a,b):
    tol = 1e-10
    diff = 1
    n_iter = 0
    while diff > tol:
        m = a + (b-a)/2
        if np.sign(Lennard(a)) == np.sign(Lennard(m)):
            a = m
        else:
            b = m
        diff = abs(a-b)
        n_iter +=1

    return m , n_iter


Two_atoms = Bisection(3,4)
print(f"The first root of the potential function between two argon atoms are at distance {Two_atoms[0]}, \n calculated in {Two_atoms[1]} iterations")

def prime(r):
    V_p = (- 12*A)/r**13 + (6*B)/r**7
    return V_p

def double_prime(r):
    V_pp = (156*A)/r**14 - (42*B)/r**8
    return V_pp

# Newton method for root finding
def Newton(r):
    x = r
    tol = 1e-30
    diff = 1
    n_iter = 0
    while diff > tol:
        x_new = x - (prime(x) / double_prime(x))
        diff =  Lennard (x_new)
        x = x_new
        n_iter +=1
    return x, Lennard(x), n_iter

Two_atoms_newton = Newton(2)
print(f"Using newton method the root is: {Two_atoms_newton[0]} and the iterations are:{Two_atoms_newton[2]}")
print("not as accurated, but noticeably fewer iterations")


# Calculates the Total Potential in a system of n-particles
def LJ_potential(A):
    Arg = np.reshape(A,(3,len(A)//3))
    X = Arg[0,:]
    Y = Arg[1,:]
    Z = Arg[2,:]

    Arg_new = np.sqrt((X[:,None] - X[None,:])**2 +
                      (Y[:,None] - Y[None,:])**2 +
                      (Z[:,None] - Z[None,:])**2)
    Arg_new = np.triu(Arg_new)
    Arg_new = Arg_new[np.nonzero(Arg_new)]
    Total_Potential = np.sum(Lennard(Arg_new))

    return Total_Potential

#Total_Potential via for loops, only used for comparing energies
def Tot_loop(A):

    Arg = np.reshape(A,(3,len(A)//3))
    Arg_new = np.zeros((40,40))

    for i in range(40):
        for j in range(40):
            Arg_new[i,j] = LA.norm(Arg[:,i] - Arg[:,j])


    Arg_new = np.triu(Arg_new)
    Arg_new = Arg_new[np.nonzero(Arg_new)]
    Total_Potential = np.sum(Lennard(Arg_new))

    return Total_Potential, Arg_new

#Time test of optimize.minimize
# Outcommented due to long running times

# % timeit optimize.minimize(LJ_potential,Arg_lines, method='CG')
# 12.7 s ± 442 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) for the test
#My annealing for finding minimum potential
# % timeit annealing(Arg_lines)
#11.6 s ± 542 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) for my annealing method

scipy_min = optimize.minimize(LJ_potential,Arg_lines, method='CG').fun
print("Using the Scipy optimize function yields that the absolute minimum to be around")
print(scipy_min) # True result

## Failed attempt to implement Conjugated gradient method

def Grad(A,eps):
    GG = np.zeros(len(A))
    for i in range(len(A)):
        G = np.copy(A)
        G[i] += eps
        GG[i] = (LJ_potential(G) - LJ_potential(A))/ eps

    return GG
#
#optimize.approx_fprime(Arg_lines,LJ_potential,1e-13) == Grad(Arg_lines,1e-13) # checking if gradient function works
# def Conjugated(A,eps,ite):
#     x = A
#     g = Grad(x)
#     s = -g
#     n_iter = 0
#     n_max  = ite
#     print(LJ_potential(x))
#     while n_max > n_iter:
#         alpha = optimize.line_search(LJ_potential,Grad,x,s, amax = 1e100)[0]
#         if alpha == None:
#             g = Grad(x)
#             s = -g
#             alpha = optimize.line_search(LJ_potential,Grad,x,s, amax = 1e100)[0]
#         print(alpha)
#         x_new = x + np.dot(alpha , s)
#         g_new = Grad(x_new,eps)
#         beta = np.dot(g_new.T, g_new) / np.dot(g.T , g)
#         s_new = - g_new + np.dot(beta, s)
#         s = s_new
#         g = g_new
#         x = x_new
#         n_iter += 1
#         # print(show3d(x))
#         print(LJ_potential(x))
#     return x, LJ_potential(x), n_iter


# Function for initializing movement of an atom
def move(A,T):
    x = A
    x_new = np.copy(A)
    randint = r.randint(2)
    if randint == 0:
        x_new[r.randint(len(x))] += 1e-1
    else:
        x_new[r.randint(len(x))] -= 1e-1
    dE = LJ_potential(x_new) - LJ_potential(x)
    if dE < 0 or r.rand() < np.exp(-dE/T):
        return x_new
    else:
        return x

# function for minimizing the energy of the system
def annealing(A):
    start = timeit.default_timer()
    k = 0
    x = A
    T = 13.7
    x_new = np.copy(x)
    while T > 0:
        x = move(x,T)
        if LJ_potential(x) < LJ_potential(x_new):
            x_new = x

        T -=0.001
        k += 1
    stop = timeit.default_timer()
    return LJ_potential(x_new), k, x_new, (stop - start)


Atoms = annealing(Arg_lines)

print(f"The initial potential is: {LJ_potential(Arg_lines)}")
print("Using the annealing method:")
print(f"Final potential of the system is: {Atoms[0]}")
print(f"Script was executed in {Atoms[3]} seconds")
print(f"Giving {Atoms [1] / Atoms[3]} iterations pr second")
print(f"\n Configuration of atoms is:")
print(show3d(Atoms[2]))



# % timeit optimize.minimize(LJ_potential,Arg_lines, method='CG')
# 12.7 s ± 442 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) for the test
#My annealing for finding minimum potential
# % timeit annealing(Arg_lines)
#11.6 s ± 542 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) for my annealing method

# Tests the statistics of the ouput in annealing method
#
# N = 50
#
# Vecs = np.zeros(N)
# for i in range(N):
#     Vecs[i] = annealing(Arg_lines)[0]
#
#
# np.mean(Vecs)
# np.min(Vecs)
# np.max(Vecs)
