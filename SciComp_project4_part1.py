import numpy as np
import matplotlib.pyplot as plt
import timeit
# Rate constants
a1 = 10
a2 = 5
b1 = 5
b2 = 1
b3 = 1
c1 = 1
c2 = 1
d1 = 1

## Euler method section method.
def Homo(x1,x2,p1):
    return a1 * x1 * (p1 - x1) + a2 * x2 * (p1 - x1)

def Bi(x2,x1,y,p2):
    return b1 * x1 * (p2 - x2) + b2 * x2 * (p2 - x2) + b3 * y * (p2 - x2)

def Het_fem(y,x2,z,q):
    return c1 * x2 * (q - y) + c2 * z * (q - y)

def Het_men(z,y,r):
    return d1 * y * (r - z)

def Blood_transfusion(e,x1,p1):
    return e * (p1 - x1)

def forw_Euler(x,dx,dt):
    return x + dx * dt

# Standard function
def Sygdomdomdoooooom(n_steps,dt,mode):
    x1 = 0.01
    x2 = 0
    y = 0
    z = 0
    p1 = 5
    p2 = 5
    q = 100
    r = 100
    RC = np.zeros((n_steps,4)) # Rate change

    Labels = ["Homo-sexuals","Bi-sexuals", "Hetero-females", "Hetero-males"]

    for i in range(n_steps):
        if i == 0:
            RC[i,0] += forw_Euler(x1,Homo(x1,x2,p1),dt)
            RC[i,1] += forw_Euler(x2,Bi(x1,x2,p2,y),dt)
        else:

            dx1 = Homo(RC[i-1,0],RC[i-1,1],p1)
            RC[i,0] = forw_Euler(RC[i-1,0],dx1,dt)

            dx2 = Bi(RC[i-1,1],RC[i-1,0],RC[i-1,2],p2)
            RC[i,1] = forw_Euler(RC[i-1,1],dx2,dt)


            dy = Het_fem(RC[i-1,2],RC[i-1,1],RC[i-1,3],q)
            RC[i,2] = forw_Euler(RC[i-1,2],dy,dt)


            dz = Het_men(RC[i-1,3],RC[i-1,2],r)
            RC[i,3] = forw_Euler(RC[i-1,3],dz,dt)

    fig,ax = plt.subplots(figsize = (8,6))
    for i in range(4):
        if i < 2:
            plt.plot(np.arange(n_steps),RC[:,i]/5,label = Labels[i] )
        else:
            plt.plot(np.arange(n_steps),RC[:,i]/100,label = Labels[i] )
    ax.legend()
    ax.set(xlabel = f'Time steps of dt = {dt}', ylabel = f'% of population')
    ax.grid(True)
    ax.set_title("Forward Euler")
    plt.show()
    if mode:
        fig.savefig("Disease_Euler.PNG", bpi = 400)
    return RC

# Blood fusion introduced

def Disease_spread_featblood(n_steps,dt,e,e_upper,mode):
    x1 = 0.01
    x2 = 0
    y = 0
    z = 0
    p1 = 5
    p2 = 5
    q = 100
    r = 100
    RC = np.zeros((n_steps,4)) # Rate change

    Labels = ["Homo-sexuals","Bi-sexuals", "Hetero-females", "Hetero-males"]

    for i in range(n_steps):
        if i == 0:
            RC[i,0] += forw_Euler(x1,Homo(x1,x2,p1),dt)
            RC[i,1] += forw_Euler(x2,Bi(x1,x2,p2,y),dt)
        else:

            dx1 = Homo(RC[i-1,0],RC[i-1,1],p1)
            RC[i,0] = forw_Euler(RC[i-1,0],dx1,dt)

            dx2 = Bi(RC[i-1,1],RC[i-1,0],RC[i-1,2],p2)
            RC[i,1] = forw_Euler(RC[i-1,1],dx2,dt)


            dy = Het_fem(RC[i-1,2],RC[i-1,1],RC[i-1,3],q)
            RC[i,2] = forw_Euler(RC[i-1,2],dy,dt)


            dz = Het_men(RC[i-1,3],RC[i-1,2],r) + Blood_transfusion(e,RC[i-1,0],p1)
            RC[i,3] = forw_Euler(RC[i-1,3],dz,dt)

            e = np.linspace(e,e_upper,n_steps)[i]

    fig,ax = plt.subplots(figsize = (8,6))
    for i in range(4):
        if i < 2:
            plt.plot(np.arange(n_steps),RC[:,i]/5,label = Labels[i] )
        else:
            plt.plot(np.arange(n_steps),RC[:,i]/100,label = Labels[i] )
    ax.legend()
    ax.set(xlabel = f'Time steps of dt = {dt}', ylabel = f'% of population')
    ax.grid(True)
    ax.set_title("Forward Euler")
    plt.show()
    if mode:
        fig.savefig("Disease_Blood_Euler.PNG", bpi = 400)
    return RC


## Section with introduction of death aka removal of people

def Disease_spread_feat_blood_death(n_steps,dt,e,e_upper,r1,r1_upper,mode):
    x1 = 0.01
    x2 = 0
    y = 0
    z = 0
    p1 = 5
    p2 = 5
    q = 100
    r = 100
    ee = np.linspace(e,e_upper,n_steps)
    rr = np.linspace(r1,r1_upper,n_steps)
    RC = np.zeros((n_steps,4)) # Rate change

    Labels = ["Homo-sexuals","Bi-sexuals", "Hetero-females", "Hetero-males"]

    for i in range(n_steps):
        if i == 0:
            RC[i,0] += forw_Euler(x1,Homo(x1,x2,p1),dt)
            RC[i,1] += forw_Euler(x2,Bi(x1,x2,p2,y),dt)
        else:

            dx1 = Homo(RC[i-1,0],RC[i-1,1],p1) - r1 * RC[i-1,0]
            RC[i,0] = forw_Euler(RC[i-1,0],dx1,dt)

            dx2 = Bi(RC[i-1,1],RC[i-1,0],RC[i-1,2],p2) - r1 * RC[i-1,1]
            RC[i,1] = forw_Euler(RC[i-1,1],dx2,dt)


            dy = Het_fem(RC[i-1,2],RC[i-1,1],RC[i-1,3],q) - r1 * RC[i-1,2]
            RC[i,2] = forw_Euler(RC[i-1,2],dy,dt)


            dz = Het_men(RC[i-1,3],RC[i-1,2],r) + Blood_transfusion(e,RC[i-1,0],p1) - r1 * RC[i-1,3]
            RC[i,3] = forw_Euler(RC[i-1,3],dz,dt)

            e  = ee[i]
            r1 = rr[i]
    fig,ax = plt.subplots(figsize = (8,6))
    for i in range(4):
        if i < 2:
            plt.plot(np.arange(n_steps),RC[:,i]/p1,label = Labels[i] )
        else:
            plt.plot(np.arange(n_steps),RC[:,i]/q,label = Labels[i] )
    ax.legend()
    ax.set(xlabel = f'Time steps of dt = {dt}', ylabel = f'% of population')
    ax.grid(True)
    ax.set_title("Forward Euler")
    plt.show()
    if mode:
        fig.savefig("Disease_BloodnDeath_Euler.PNG", bpi = 400)
    return RC






## 4th order Runge-Kutta method

# Runge_Kutta for only infection

def Runge_Kutta(f, dt, mode, x, y, z, p):
    # Mode defines if the function called has 3(True) or 4(False) variables
    if mode:
        k1 = f(x, y, p) * dt
        k2 = f(x + k1 * dt / 2, y, p) * dt
        k3 = f(x + k2 * dt / 2, y, p) * dt
        k4 = f(x + k3 * dt,     y, p) * dt

        dy = x + (k1 + 2 * k2 + 2 * k3 + k4 ) * dt / 6
    else:
        k1 = f(x, y, z, p) * dt
        k2 = f(x + k1 * dt / 2, y, z, p) * dt
        k3 = f(x + k2 * dt / 2, y, z, p) * dt
        k4 = f(x + k3 * dt,     y, z, p) * dt

        dy = x + (k1 + 2 * k2 + 2 * k3 + k4 ) * dt / 6

    return dy

def Disease_Kutta_kutta(n_steps, dt, mode):
    x1 = 0.01
    x2 = 0
    y = 0
    z = 0
    p1 = 5
    p2 = 5
    q = 100
    r = 100
    RC = np.zeros((n_steps,4)) # Rate change

    Labels = ["Homo-sexuals","Bi-sexuals", "Hetero-females", "Hetero-males"]

    for i in range(n_steps):
        if i == 0:
            RC[i,0] += Runge_Kutta(Homo,dt,True,x1,x2,0,p1)
            RC[i,1] += Runge_Kutta(Bi,dt,False,x2,x1,y,p2)
        else:
            dx1 = RC[i-1,0]
            dx2 = RC[i-1,1]
            dy  = RC[i-1,2]
            dz  = RC[i-1,3]

            RC[i,0] = Runge_Kutta(Homo,dt,True,dx1,dx2,0,p1)

            RC[i,1] = Runge_Kutta(Bi,dt,False,dx2,dx1,dy,p2)

            RC[i,2] = Runge_Kutta(Het_fem,dt,False,dy,dx2,dz,q)

            RC[i,3] = Runge_Kutta(Het_men,dt,True,dz,dy,0,r)

            # e = np.linspace(e,e_upper,n_steps)[i]
            # r1 = np.linspace(r1,r1_upper,n_steps)[i]
    fig,ax = plt.subplots(figsize = (8,6))
    for i in range(4):
        if i < 2:
            plt.plot(np.arange(n_steps),RC[:,i]/p1,label = Labels[i] )
        else:
            plt.plot(np.arange(n_steps),RC[:,i]/q,label = Labels[i] )
    ax.legend()
    ax.set(xlabel = f'Time steps of dt = {dt}', ylabel = f'%/100 of population')
    ax.grid(True)
    ax.set_title("Runge-Kutta")
    plt.show()
    if mode:
        fig.savefig("Disease_Kutta.PNG", bpi = 400)
    return RC


## Infection of with blood transfusion introduce

def Het_men_blood(z,y,r,x1,p1,e):
    return d1 * y * (r - z) + e * (p1 - x1)

def Runge_blood(dt, z, y,r, x1,p1, e):

    k1 = Het_men_blood(z, y, r, x1, p1, e) * dt
    k2 = Het_men_blood(z +  k1 * dt / 2, y, r, x1, p1, e) * dt
    k3 = Het_men_blood(z +  k2 * dt / 2, y, r, x1, p1, e) * dt
    k4 = Het_men_blood(z +  k3 * dt,     y, r, x1, p1, e) * dt

    dy = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return dy

def Disease_Kutta_blood(n_steps, dt, e, e_upper, mode):
    x1 = 0.01
    x2 = 0
    y = 0
    z = 0
    p1 = 5
    p2 = 5
    q = 100
    r = 100
    RC = np.zeros((n_steps,4)) # Rate change

    Labels = ["Homo-sexuals","Bi-sexuals", "Hetero-females", "Hetero-males"]

    for i in range(n_steps):
        if i == 0:
            RC[i,0] += Runge_Kutta(Homo,dt,True,x1,x2,0,p1)
            RC[i,1] += Runge_Kutta(Bi,dt,False,x2,x1,y,p2)
        else:
            dx1 = RC[i-1,0]
            dx2 = RC[i-1,1]
            dy  = RC[i-1,2]
            dz  = RC[i-1,3]

            RC[i,0] = Runge_Kutta(Homo,dt,True,dx1,dx2,0,p1)

            RC[i,1] = Runge_Kutta(Bi,dt,False,dx2,dx1,dy,p2)

            RC[i,2] = Runge_Kutta(Het_fem,dt,False,dy,dx2,dz,q)

            RC[i,3] = Runge_blood(dt,dz,dy,r,dx1,p1,e)

            e = np.linspace(e,e_upper,n_steps)[i]
            # r1 = np.linspace(r1,r1_upper,n_steps)[i]
    fig,ax = plt.subplots(figsize = (8,6))
    for i in range(4):
        if i < 2:
            plt.plot(np.arange(n_steps),RC[:,i]/p1,label = Labels[i] )
        else:
            plt.plot(np.arange(n_steps),RC[:,i]/q,label = Labels[i] )
    ax.legend()
    ax.set(xlabel = f'Time steps of dt = {dt}', ylabel = f'% of population')
    ax.grid(True)
    ax.set_title("Runge-Kutta")
    plt.show()
    if mode:
        fig.savefig("Disease_Blood_Kutta.PNG", bpi = 400)
    return RC


## Infection introduced with removal via death

def Runge_death(f, dt, mode, x, y, z, p, r1):
    if mode:
        k1 = (f(x, y, p) - x * r1) * dt
        k2 = (f(x + k1 * dt / 2, y, p) - x * r1 )* dt
        k3 = (f(x + k2 * dt / 2, y, p) - x * r1 )* dt
        k4 = (f(x + k3 * dt,     y, p) - x * r1 )* dt

        dy = x + (k1 + 2 * k2 + 2 * k3 + k4 ) * dt / 6
    else:
        k1 = (f(x, y, z, p) - x * r1) * dt
        k2 = (f(x + k1 * dt / 2, y, z, p) - x * r1) * dt
        k3 = (f(x + k2 * dt / 2, y, z, p) - x * r1) * dt
        k4 = (f(x + k3 * dt,     y, z, p) - x * r1) * dt

        dy = x + (k1 + 2 * k2 + 2 * k3 + k4 ) * dt / 6

    return dy

def Runge_bloodNdeath(dt, z, y,r, x1, p1, e, r1):

    k1 = (Het_men_blood(z,                y, r, x1, p1, e) - z * r1) * dt
    k2 = (Het_men_blood(z +  k1 * dt / 2, y, r, x1, p1, e) - z * r1) * dt
    k3 = (Het_men_blood(z +  k2 * dt / 2, y, r, x1, p1, e) - z * r1) * dt
    k4 = (Het_men_blood(z +  k3 * dt,     y, r, x1, p1, e) - z * r1) * dt

    dy = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return dy

def Disease_Kutta_death(n_steps, dt, e, e_upper, r1, r1_upper,mode):
    x1 = 0.01
    x2 = 0
    y = 0
    z = 0
    p1 = 5
    p2 = 5
    q = 100
    r = 100

    RC = np.zeros((n_steps,4)) # Rate change

    Labels = ["Homo-sexuals","Bi-sexuals", "Hetero-females", "Hetero-males"]

    for i in range(n_steps):
        if i == 0:
            RC[i,0] = Runge_death(Homo, dt, True, x1, x2, 0, p1, r1)
            RC[i,1] = Runge_death(Bi, dt, False, x2, x1, y, p2, r1)
        else:
            dx1 = RC[i-1,0]
            dx2 = RC[i-1,1]
            dy  = RC[i-1,2]
            dz  = RC[i-1,3]

            e = np.linspace(e, e_upper, n_steps)[i]
            r1 = np.linspace(r1, r1_upper, n_steps)[i]

            RC[i,0] = Runge_death(Homo,    dt, True,  dx1, dx2, 0,  p1,  r1)

            RC[i,1] = Runge_death(Bi,      dt, False, dx2, dx1, dy, p2,  r1)

            RC[i,2] = Runge_death(Het_fem, dt, False, dy,  dx2, dz, q,   r1)

            RC[i,3] = Runge_bloodNdeath(   dt,        dz,  dy,  r,  dx1, p1, e, r1)



    fig, ax = plt.subplots(figsize = (8,6))
    for i in range(4):
        if i < 2:
            plt.plot(np.arange(n_steps),RC[:,i]/p1,label = Labels[i] )
        else:
            plt.plot(np.arange(n_steps),RC[:,i]/q,label = Labels[i] )
    ax.legend()
    ax.set(xlabel = f'Time steps of dt = {dt}', ylabel = f'% of population')
    ax.grid(True)
    # ax.set(xlim = (0,200))
    ax.set_title("Runge-Kutta")
    plt.show()
    if mode:
        fig.savefig("Disease_BloodnDeath_Kutta.PNG", bpi = 400)
    return RC


# Just infection with Kutta and Euler
Sygdomdomdoooooom(1750,0.0001,True);

Disease_Kutta_kutta(1750,0.01,True);


# Infection and blood transfusion
Disease_spread_featblood(150,0.001,0.001,100,True);

Disease_Kutta_blood(1500,0.01,0.001,100,True);

# now with death

Disease_spread_feat_blood_death(3000,0.0001,0.00,0,0.005,300,True); # No blood

Disease_spread_feat_blood_death(3000,0.0001,0.001,60,0.005,300,True); # /w blood
0.05 * 3000

Disease_Kutta_death(3000, 0.01, 0.001, 60, 0.005, 800,False);
