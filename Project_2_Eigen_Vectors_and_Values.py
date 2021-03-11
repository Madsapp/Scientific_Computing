import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Chladni_basis import *


Chladni_mat = np.load("Chladni-Kmat.npy") # The K-Matrix data import

# A1-A3 should work with any implementation
A1   = np.array([[1,3],[3,1]]);

eigvals1 = [4,-2];

A2   = np.array([[3,1],[1,3]]);
eigvals2 = [4,2];

A3   = np.array([[1,2,3],[4,3.141592653589793,6],[7,8,2.718281828459045]])
eigvals3 = [12.298958390970709, -4.4805737703355,  -0.9585101385863923];

# A4-A5 require the method to be robust for singular matrices
A4   = np.array([[1,2,3],[4,5,6],[7,8,9]]);
eigvals4 = [16.1168439698070429897759172023, -1.11684396980704298977591720233, 0]

A5   = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]);
eigvals5 = [68.6420807370024007587203237318, -3.64208073700240075872032373182, 0, 0, 0];

# A6 has eigenvalue with multiplicity and is singular
A6  = np.array(
    [[1.962138439537238,0.03219117137713706,0.083862817159563,-0.155700691654753,0.0707033370776169],
       [0.03219117137713706, 0.8407278248542023, 0.689810816078236, 0.23401692081963357, -0.6655765501236198],
       [0.0838628171595628, 0.689810816078236,   1.3024568091833602, 0.2765334214968566, 0.25051808693319155],
       [-0.1557006916547532, 0.23401692081963357, 0.2765334214968566, 1.3505754332321778, 0.3451234157557794],
       [0.07070333707761689, -0.6655765501236198, 0.25051808693319155, 0.3451234157557794, 1.5441014931930226]]);

eigvals6 = [2,2,2,1,0]


# Extract Gersh_gorin radii and centers from an input Matrix
def Gersh_gorin(A):
    m,n = A.shape
    centers = np.zeros(n)
    radii = np.zeros(n)
    for i in range(len(A)):
        centers[i] = np.diag(A)[i]
        radii[i] = np.sum(np.abs(A-np.diag(A)*np.eye(len(A)))[:,i])

    return centers, radii

Disc_center, Disc_radii = Gersh_gorin(Chladni_mat) # Assigning the Center and radii values to print

print(f"The centers of the 15 Gershgorin are:{Disc_center.astype(int)} \n while their respective radii are:{Disc_radii.astype(int)}\n")


# Plot function for visualizing the Gersh_gorin discs
def plot_GG(A,xsize, ysize):
    colors = cm.rainbow(np.linspace(0,1,len(A)))
    fig, ax = plt.subplots(figsize = (xsize,ysize))
    for i in range(len(A)):
        ax.add_patch(plt.Circle((Gersh_gorin(A)[0][i], 0), Gersh_gorin(A)[1][i], color = colors[i], alpha=0.5))
        plt.scatter(Gersh_gorin(A)[0][i], 0, alpha=1, marker = 'o', s = 40, label = f'Center {i}')

    #Use adjustable='box-forced' to make the plot area square-shaped as well.
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot()
    ax.set_xlabel('Re', fontsize = '18')
    ax.set_ylabel('Im', fontsize = '18')
    ax.grid(True)
    # ax.legend(fontsize=15)
    plt.show()
    # fig.savefig("GG_center_4_5.PNG", bpi = 400)





# Power iteration method
def power(A): # Return Vector, iterations and Eigenvalue
    x = np.random.rand(len(A),1)
    n_iter = 0
    temp = 1
    y = x
    epsilon = 1e-14
    while temp > epsilon:

        old = y
        y = np.dot(A,x)
        new = y
        x = y / np.linalg.norm(new,2)
        temp = np.linalg.norm((new - old),2)/ np.linalg.norm(old,2)
        n_iter += 1

    return x, n_iter, np.linalg.norm(new,2)


print (f"Largest eigenvalue of A6: {power(A6)[2]} and number of iterations of the Power method: {power(A6)[1]}\n")

#Rayleigh Quotient
def rayleigh_qt(A,x):
    sigma = np.linalg.multi_dot((x.T,A,x))/np.dot(x.T,x)
    return sigma

# Residual of the relative error via Power method
def residual(A):
    x0 = power(A)[0]
    lambda_x = rayleigh_qt(A,x0)
    Ax = np.dot(A,x0)
    residual = np.linalg.norm(Ax - lambda_x * x0 ,2)

    return residual

# Residual of the relative error of the Rayleigh shift method
def Rayleigh_residual(A,shift0):
        x0 = Rayleigh_interate_wShift(A,shift0)[0]
        lambda_x = rayleigh_qt(A,x0)
        Ax = np.dot(A,x0)
        residual = np.linalg.norm(Ax - lambda_x * x0 ,2)
        return residual


K_max = power(Chladni_mat) # Power method on the K-Matrix


print(f" The largest Eigenvalue of the K-Matrix using the power method is: {np.round(K_max[2],2)}\n")
print(f" Number of iterations: {K_max[1]}, and Residual:{residual(Chladni_mat)}")

print(f" The wavenodes for the largest eigenvector looks like:")

print(show_nodes(power(Chladni_mat)[0].flatten()))
print("\n")


## Home made "State of the art" solvers for x in Ax = b
def reflection_vector(a): # updates v and a in for loop
    anorm = -np.copysign(np.sqrt(np.dot(a,a)), a[0])
    v    = -a
    v[0] += anorm
    return v/np.sqrt(np.dot(v,v))

def apply_reflection(v,x): # inverts the Matrix
    x -= 2*v*(np.dot(v,x))

def House_holder(A): # Takes a NxM Matrix as input and give R and Qt as output
    N,M = A.shape
    R = np.copy(A)
    Qt = np.eye(N)
    for k in range(M):
        v = reflection_vector(R[k:N,k])
        # print('V:',v)
        for j in range(k,M):
            apply_reflection(v,R[k:N,j])
        # c[k:N,j] = apply_reflection(v,R[k:N,j])
        # print(R)
        for j in range(N):
            apply_reflection(v,Qt[k:N,j])
        # print(Qt)
    return Qt, R

def least_squares(A,b): # input A is a NxM matrix and b is thee Nx1 vector

    Qt, R = House_holder(A) # new MxM matrix
    B_T = np.dot(Qt, b) #  new Mx1 vector

    x_tilde = backward_sub(R,B_T) # Backwards substitution to find x_tilde
    # b_tilde = np.dot(A,x_tilde) # Approximated solution vector

    return x_tilde

def backward_sub(U,y):# Backwards Substitution
    m,n = U.shape
    x = np.zeros((n))
    for i in range(n-1,-1,-1):

        s = sum(U[i,j]*x[j]for j in range(i,n))

        x[i] = (y[i]-s)/U[i,i]
    return x


## Rayleigh_interation w/ shift

# Functions using inbuilt numpy functions
def Rayleigh_interate_wShift(A,shift0):
    x = np.random.rand(len(A),1)
    I = np.eye(len(A))
    err = 1.0
    n_iter = 0
    epsilon = 1e-10
    y = x
    while (err > epsilon):
        x = y/np.linalg.norm(y)
        y = np.linalg.solve(A-shift0*I,x)
        theta = np.dot(x.T,y)
        err = np.linalg.norm(y-theta*x)/abs(theta)
        n_iter +=1
    return x, shift0+1/theta, n_iter

# Functions using home build state of the art least_squares to solve eigenvector
def Rayleigh_Manual(A,shift0):
    x = np.random.rand(len(A),1)
    I = np.eye(len(A))
    err = 1.0
    n_iter = 0
    epsilon = 1e-10
    y = x
    while (err > epsilon):
        x = y/np.linalg.norm(y)
        y = least_squares(A-shift0*I,x)
        theta = np.dot(x.T,y)
        err = np.linalg.norm(y-theta*x)/abs(theta)
        n_iter +=1
    return x, shift0+1/theta, n_iter


print(f" Using Rayleigh_Manual, The Eigenvalue for A6 is {Rayleigh_Manual(A6,2)[1]} and the number of iterations is {Rayleigh_Manual(A6,3)[2]}")
print (f"The residual is {Rayleigh_residual(A6,3)} \n" )



## Testing Library- and home made solvers and comparing them.
Builtin_Solve = Rayleigh_interate_wShift(Chladni_mat,10000)[0].flatten()

Home_made_Solve = Rayleigh_Manual(Chladni_mat,10000)[0].flatten()

print(f"Comparing the Builtin_Solve with my own functions, the output vectors yields:\n{np.isclose(Builtin_Solve,Home_made_Solve)}\n")


def Trans_Matrix_test(A): # Function generating Transformation matrix and Eigen_Values

    Cents = Gersh_gorin(A)[0]

    E_va = np.zeros((len(A),1))

    E_ve = np.zeros((len(A),len(A)))

    for i in range(len(A)):

        E_ve[i,:] = Rayleigh_Manual(A,Cents[i])[0].flatten()
        E_ve[4,:] = Rayleigh_Manual(A,32000)[0].flatten() # Manually located guess for 5th eigenvalue
        E_va[i] = Rayleigh_Manual(A,Cents[i])[1].flatten()
        E_va[4] = Rayleigh_Manual(A,32000)[1].flatten()

    K_Matrix = np.linalg.multi_dot((E_ve.T,E_va.flatten()*np.eye(len(A)),np.linalg.inv(E_ve.T)))

    return K_Matrix, E_ve, E_va

K_Matrix_guess, Eigen_Vecs, Eigen_Vals = Trans_Matrix_test(Chladni_mat)

print(f" Using np.isclose to test if My Transformation matrix can give K_mat, to check np.isclose is used, yielding:\n {np.isclose(Chladni_mat, K_Matrix_guess)}")



# Using centers in the Gersh_gorin_disc as shifts to find the eigenvalues and eigenvectors
def Show_dopeness(vec, lambdas):

    E_vecs = vec
    E_vals = lambdas

    show_all_wavefunction_nodes(E_vecs.T,E_vals.flatten(), basis_set)

# prints the Gersh_gorin_discs for the Chladni matrix
print("Gershgorin discs of the K-Matrix are shown below:\n")
print(plot_GG(Chladni_mat, 15 , 10))



# print all the wavenodes for the chladni Matrix
print("Wave Function nodes for all the eignevalues are shown below:\n")
print(Show_dopeness(Eigen_Vecs,Eigen_Vals))
