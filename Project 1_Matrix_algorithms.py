import numpy as np
import matplotlib.pyplot as plt

## PART I
Mx = np.array([[1,2],[3,4]])

def max_norm(Mx):
    norm = max(np.sum(abs(Mx), axis = 1))
    return norm

def cond(Mx):
    inv = np.linalg.inv(Mx)
    cond = max_norm(Mx)*max_norm(inv)
    return cond

def forward_error(x):
    error = -np.log10(x)
    return error

# Calculate the 3 condition numbers for the matrix (E-omega*S) with the 3 frequencies

Amat = np.array([
    [22.13831203, 0.16279204, 0.02353879, 0.02507880,-0.02243145,-0.02951967,-0.02401863],
    [0.16279204, 29.41831006, 0.02191543,-0.06341569, 0.02192010, 0.03284020, 0.03014052],
    [0.02353879,  0.02191543, 1.60947260,-0.01788177, 0.07075279, 0.03659182, 0.06105488],
    [0.02507880, -0.06341569,-0.01788177, 9.36187184,-0.07751218, 0.00541094,-0.10660903],
    [-0.02243145, 0.02192010, 0.07075279,-0.07751218, 0.71033323, 0.10958126, 0.12061597],
    [-0.02951967, 0.03284020, 0.03659182, 0.00541094, 0.10958126, 8.38326265, 0.06673979],
    [-0.02401863, 0.03014052, 0.06105488,-0.10660903, 0.12061597, 0.06673979, 1.15733569]]);
Bmat = np.array([
    [-0.03423002, 0.09822473,-0.00832308,-0.02524951,-0.00015116, 0.05321264, 0.01834117],
    [ 0.09822473,-0.51929354,-0.02050445, 0.10769768,-0.02394699,-0.04550922,-0.02907560],
    [-0.00832308,-0.02050445,-0.11285991, 0.04843759,-0.06732213,-0.08106876,-0.13042524],
    [-0.02524951, 0.10769768, 0.04843759,-0.10760461, 0.09008724, 0.05284246, 0.10728227],
    [-0.00015116,-0.02394699,-0.06732213, 0.09008724,-0.07596617,-0.02290627,-0.12421902],
    [ 0.05321264,-0.04550922,-0.08106876, 0.05284246,-0.02290627,-0.07399581,-0.07509467],
    [ 0.01834117,-0.02907560,-0.13042524, 0.10728227,-0.12421902,-0.07509467,-0.16777868]]);
yvec= np.array([-0.05677315,-0.00902581, 0.16002152, 0.07001784, 0.67801388,-0.10904168, 0.90505180]);

omega = np.array([1.300,1.607,3.000,1.300,1.607,3.000,1.300,1.607,3.000])
delta_omega = 0.5*10**-3
omega[3:6]-=delta_omega
omega[6:]+=delta_omega
E = np.block([[Amat,Bmat],[Bmat, Amat]])
S = np.block([[np.identity(7),np.zeros(shape = (7,7))],[np.zeros(shape = (7,7)),-1*np.identity(7)]])
z = np.block([yvec,-yvec])


# condition number for the matrix with the three condition numbers
m = np.zeros(len(omega))
for i in range(len(omega)):
    m[i]= cond(E-omega[i]*S)*((max_norm(S*delta_omega))/(max_norm(E-omega[i]*S)))
print("The relative forward error for all omega values", np.round(m,2))






print("The first digit expected to have error for omegas are", np.round(forward_error(m),2))



# LU-factorization

A_lu = np.array([2,1,1,4,1,4,-8,-5,3]).reshape(3,3) # A Matrix
b = np.array([4,11,4]) # b vector

def LU_decomposition(A): # Function for LU-decomposition
    n = len (A)# Define the range to loop over in matrix
    L = np.eye(n) # Creates identity matrix with same size as input
    U = np.zeros((n, n)) # Zero matrix: Same size as input

    for j in range(n):

        for i in range(j+1):
            U[i,j] = A[i,j] - np.sum(U[:j+1, j] * L[i, :j+1])

        for i in range(j+1, n):
            L[i,j] = (A[i,j] - np.sum(U[:j+1, j] * L[i, :j+1])) / U[j,j]

    return U, L

def forward_sub(L,b): #forward Substitution
    m,n = L.shape
    y = np.zeros((n))
    for i in range(n):
        s = 0.0
        for j in range(0,i):
            s += L[i,j]*y[j]
        y[i] = (b[i]-s)/L[i,i]
    return y

def backward_sub(U,y):# Backwards Substitution
    m,n = U.shape
    x = np.zeros((n))
    for i in range(n-1,-1,-1):

        s = sum(U[i,j]*x[j]for j in range(i,n))

        x[i] = (y[i]-s)/U[i,i]
    return x

U, L = LU_decomposition(A_lu)
print("The upper and lower triangle matrices from LU decomposition \n", U,L)

y = forward_sub(L,b)

print("The y-vector from forward substitution came out as \n", y.T)

x = backward_sub(U,y)

print("The output x-vector from backwards substitution:",x, "and if x is dotted with the original matrix A the result is:\n", np.dot(A_lu,x), "the orignal b-vector")

def solve_alpha(omega): # Solve the polizability for alpha with respect to omega

    n = omega.size
    alpha = np.zeros(n)
    for i in range(n):
        U, L = LU_decomposition((E-omega[i]*S))
        y = forward_sub(L,z)
        x = backward_sub(U,y)
        alpha[i] = np.dot(z.T,x)

    return alpha

print("The alpha values in the usual omega order:",np.round(solve_alpha(omega)), "\n Again again, The values spikes around Omega = 1.607")

# evenly space array of frequencies
omega_lin = np.linspace(1.2,4,1000) # frequency

alpha = solve_alpha(omega_lin)

# Plot of alpha(Omega)

fig, ax = plt.subplots(figsize = (12,8))

plt.plot(omega_lin,alpha, label = 'Alpha_w')
ax.grid(True)
ax.set_xlim(1.2,2)
ax.set_xticks(np.arange(1.2,2,0.05))
ax.set(xlabel = 'Omega', ylabel = 'Alpha')
ax.legend()

plt.show()
# fig.savefig("alpha_omega.eps")


print(omega_lin[(np.argmax(alpha))], " is around the value where alpha divergences, pretty close to the ")
## PART II

A_test = np.vstack([np.eye(3),np.array([-1,1,0,-1,0,1,0,-1,1]).reshape(3,3)])
b_test = np.array([1237,1941,2417,711,1177,475]).T

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

Q, R  = House_holder(A_test) # Assigns Q and R

QQ = np.dot(Q,Q.T) # Q.TQ test

Test_Q = np.allclose(np.eye(6),QQ) # Checks if Q are similar to identity matrix

print("The dot product of Q and Q.T compared via np.allclose to identity matrix yields:",Test_Q)

Norm_Q = np.linalg.norm(Q,2) # Calculates the 2-norm to the Q-Vector
print("The norm of the Q vector is: %.2f which implies that Q is reasonable orthognonal" % Norm_Q)

Test_R = np.allclose(np.dot(Q.T,R), A_test)
print("R is an upper triangle matrix that holds Q.TR = A, using np.allclose again yields:",Test_R)

def least_squares(A,b): # input A is a NxM matrix and b is thee Nx1 vector

    Qt, R = House_holder(A) # new MxM matrix
    B_T = np.dot(Qt, b) #  new Mx1 vector

    x_tilde = backward_sub(R,B_T) # Backwards substitution to find x_tilde
    # b_tilde = np.dot(A,x_tilde) # Approximated solution vector

    return x_tilde


X_approx = least_squares(A_test,b_test) # Value for the approximated X-vector
b_test
b_tilde = np.dot(A_test,X_approx)
print(" The approximated x vector is:", X_approx)

print("The intitial b-vector is:", b_test, '\nAnd b_tilde is:',b_tilde)


def P_omega(n,omega_p):# Approximation fucntion 1 for alpha
    omega_range = np.linspace(1.2,omega_p,1000)
    P = solve_alpha(omega_range)
    p = np.zeros((omega_range.size, n+1))
    for j in range(n+1):
        p[:,j] = omega_range**(2*j)
    a_j = least_squares(p,P.T)
    P_omg = a_j*p
    P_w = np.sum(P_omg, axis = 1)

    return a_j, omega_range, P_w, P_omg

a_j4, o_4 , p_omg4, P4 = P_omega(4,1.7)
a_j6, o_6, p_omg6, P6 = P_omega(6,1.7)
a_j8, o_8, p_omg8, P8 = P_omega(8,1.7)
alpha_P = solve_alpha(o_4)

print("In P_w the coefficients for n = 4 and 6 are respectively", a_j4, a_j6)

fig3, ax3 = plt.subplots(figsize = (12,8))
plt.plot(o_4,alpha_P - p_omg4 ,label = 'P_4')
plt.plot(o_6,alpha_P - p_omg6  ,label = 'P_6')
plt.plot(o_8,alpha_P - p_omg8  ,label = 'P_8')
plt.plot(o_4, alpha_P, label = " Alpha")
ax3.legend()
plt.yscale('log')
ax3.set_xticks(np.arange(1.48,1.608,0.01))
ax3.set(xlabel = 'Omega', ylabel = 'Alpha')
ax3.set_xlim(1.48,1.608)
ax3.grid(True)
plt.show()
# fig3.savefig("P_omega_1.7.eps")



def Q_omega(n):
    omega_range = np.linspace(1.2,4,1000)
    Alp = solve_alpha(omega_range)
    p_a = np.zeros((omega_range.size,n+1))
    p_b = np.zeros((omega_range.size,n))

    for j in range(n+1): # Assigns omega values to the A_array
        p_a[:,j] = omega_range**j

    for j in range(1,n+1): # Assigns omega values to The B_array
        p_b[:,j-1] = omega_range**j

    p_b_alpha = (p_b.T * -Alp).T # Matrix for the approximation of B

    p_c = np.hstack((p_a, p_b_alpha)) # Combines matrices to Calculate approximated 1xn*2+1 matrix
    c = least_squares(p_c,Alp) #least_squares of the combined matrix to find a_j and b_j

    A = p_a*c[0:n+1] # multiply the respective a and b values to vectors
    B = p_b*c[n+1:]

    Q = np.sum(A, axis = 1 )/(1 + np.sum(B, axis = 1 )) # Final product of aplha approximation

    return Q

Q_2 = Q_omega(2) # Q_w with n = 2
Q_4 = Q_omega(4) #Q_w with n = 4
Q_6 = Q_omega(6) # Q_w with n = 6

omeg_range = np.linspace(1.2,4,1000)

fig2, ax2 = plt.subplots(figsize = (12,8))
plt.plot(omeg_range,Q_2,label = 'Q_w, n=2')
plt.plot(omeg_range,Q_4,label = 'Q_w, n=4')
plt.plot(omeg_range,Q_6,label = 'Q_w, n=6')
ax2.legend()
ax2.set_xlim(1.3,2)
ax2.grid(True)
plt.show()
# fig2.savefig("Q_omega.eps")

#Mean squared error dope
M_sqrt_Q2 = np.mean((alpha - Q_2)**2)
M_sqrt_Q4 = np.mean((alpha - Q_4)**2)
M_sqrt_Q6 = np.mean((alpha - Q_6)**2)

print ( "The mean squared error on the Q_w function are:",M_sqrt_Q2,M_sqrt_Q4,M_sqrt_Q6, "respectively for n = 2,4 and 6")
