import numpy as np
import matplotlib.pyplot as plt
import timeit
C = 4.5
Dp = 1
Dq = 8

def stencil(A):
    M = np.insert(A,  0, A[0,:], axis = 0)
    M = np.insert(M, -1, M[-1,:], axis = 0)
    M = np.insert(M,  0, M[:,0], axis = 1)
    M = np.insert(M,  -1, M[:,-1], axis = 1)
    return M


def Laplace(A):
    Lp_matrix = np.zeros((len(A)-2, len(A)-2))
    for i in range(1,len(A) - 1):
        for j in range(1, len(A) - 1):
            D_1 = A[i + 1,j]
            D_2 = A[i - 1,j]
            D_3 = A[i,j + 1]
            D_4 = A[i,j - 1]
            D_5 = - 4 * A[i,j]
            Lp_matrix[i - 1,j - 1] = (D_1 + D_2 + D_3 + D_4 + D_5)
    return Lp_matrix

def forward_euler(A,dP,D,h):
    new_mat = np.zeros((len(A), len(A)))
    dt = (h * h) / (16 * D)
    dx = (D * dt)/(h * h)
    for i in range(1,len(A) + 1):
        for j in range(1, len(A) + 1):
            new_mat[i - 1,j - 1] = A[i - 1,j - 1] + dx * (dP[i + 1,j] + dP[i - 1,j] + dP[i,j + 1] + dP[i,j - 1] - 4 * dP[i,j])
    return new_mat

def d_P(A,B,K,D,h):
    P = stencil(A) # extended p matrix
    Q = stencil(B) # extended q matrix
    Lp = Laplace(P)
    dP = D * stencil(Lp) + P * P * Q + C - (K + 1) * P
    Euler = forward_euler(A,dP,D,h)

    return Euler


def d_Q(A,B,K,D,h):
    P = stencil(A) # extended p matrix
    Q = stencil(B) # extended q matrix
    Lp = Laplace(Q)
    dQ = D * stencil(Lp) - P * P * Q + K * P
    Euler = forward_euler(B, dQ, D, h)

    return Euler

pi = np.zeros((40,40))
pi[10:30,10:30] += C + 0.1
qi = np.zeros((40,40))
qi[10:30,10:30] += 7/C + 0.2

pnew = d_P(pi,qi,7,1,1e-10)
qnew = d_Q(pi,qi,7,8,1e-10)
print(np.sum(pnew))
print(np.sum(qnew))
print("==========")
print(np.sum(Laplace(pnew)))
print(np.sum(Laplace(qnew)))
print("==========")
print(np.sum(pi))
print(np.sum(qi))
pi = pnew
qi = qnew
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10))
ax[0].contour(pi)
ax[1].contour(qi)
# ax.grid(True)
plt.show()



def reaction_squared(K,h,steps):

    p0 = np.zeros((40,40))
    p0[10:30,10:30] += C + 0.1
    q0 = np.zeros((40,40))
    q0[10:30,10:30] += K/C + 0.2

    for i in range(steps):
        p_new = d_P(p0, q0, K, Dp, h)
        q_new = d_P(p0, q0, K, Dq, h)
        p0 = p_new
        q0 = q_new

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10))
    ax[0].contour(p_new)
    ax[1].contour(q_new)
    # ax.grid(True)
    plt.show()

    return p_new, q_new

reaction_squared(7,100,5);

#
# def d_PQ(A,B,K,h,mode): # If mode is true then calculate for dp, else calculate dq
#     new_mat = np.zeros((len(A), len(A)))
#     Laplace = np.zeros((len(A), len(A)))
#     M = stencil(A) # extended p matrix
#     N = stencil(B) # extended q matrix
#     if mode:
#         L = M
#     else:
#         L = N
#     for i in range(1,len(A) + 1):
#         for j in range(1, len(A) + 1):
#             D_1 = L[i + 1,j]
#             D_2 = L[i - 1,j]
#             D_3 = L[i,j + 1]
#             D_4 = L[i,j - 1]
#             D_5 = - 4 * L[i,j]
#             Laplace[i - 1,j - 1] = (D_1 + D_2 + D_3 + D_4 + D_5) / (h * h)
#             if mode:
#                 D = 1
#                 dt = (h * h) / (4 * D)
#                 dPQ = D * stencil(Laplace) + M * M * N + C - (K + 1) * M
#                 new_mat[i - 1,j - 1] = M[i,j] + ((D * dt) / h * h) *  (dPQ[i + 1,j] + dPQ[i - 1,j] + dPQ[i,j + 1] + dPQ[i,j - 1] - 4 * dPQ[i,j])
#             else:
#                 D = 8
#                 dt = (h * h) / (4 * D)
#                 dPQ = D * stencil(Laplace) - M * M * N + K * M
#                 new_mat[i - 1,j - 1] = N[i,j] + ((D * dt) / h * h) *  (dPQ[i + 1,j] + dPQ[i - 1,j] + dPQ[i,j + 1] + dPQ[i,j - 1] - 4 * dPQ[i,j])
#
#     return new_mat



# def forw_euler(p0,p1,h,D): #  forward euler for PDE
    new_mat = np.zeros((len(p1), len(p1)))
    M = np.insert(p1,  0, p1[0,:], axis = 0)
    M = np.insert(M, -1, M[-1,:], axis = 0)
    M = np.insert(M,  0, M[:,0], axis = 1)
    M = np.insert(M,  -1, M[:,-1], axis = 1)
    dt = (h * h) / (4 * D)
    for i in range(1,len(p1)):
        for j in range(1, len(p1)):
            D_1 = M[i + 1,j]
            D_2 = M[i - 1,j]
            D_3 = M[i,j + 1]
            D_4 = M[i,j - 1]
            D_5 = - 4 * M[i,j]
            new_mat[i,j] = p0[i,j] + ((D * dt) / h * h) *  (D_1 + D_2 + D_3 + D_4 + D_5)


    return new_mat

# def reaction_square(K,h, steps):
#
#     p0 = np.zeros((40,40))
#     p0[10:30,10:30] += C + 0.1
#
#     q0 = np.zeros((40,40))
#     q0[10:30,10:30] += K/C + 0.2
#
#     p1 = d_PQ(p0, q0, K, h, True)
#     q1 = d_PQ(p0, q0, K, h, False)
#
#     for i in range(steps):
#         p_new = forw_euler(p0, p1, h, Dp)
#         q_new = forw_euler(q0, p1, h, Dq)
#         p0 = p1
#         p1 = p_new
#         q0 = q1
#         q1 = q_new
#
#     fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20,10))
#     ax[0].contour(p_new)
#     ax[1].contour(q_new)
#     # ax.grid(True)
#     plt.show()
#
#     return p_new, q_new

# Try K = [7, 8, 9, 10, 11, 12]

# reaction_square(7,0.07,2000);

# reaction_square(12,0.07,2000);


# laplace calculate x and y seperately then add as a usual laplacian
# Soak matrix under the main matrix to use avoid error
# læg lag udover kanten som er den samme som er det samme som kanten derved er gradienten 0
