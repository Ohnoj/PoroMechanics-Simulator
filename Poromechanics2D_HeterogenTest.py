# -*- coding: utf-8 -*-
"""
Interactive 1D poromechanics simulation
Created on Thu Jul 22 09:46:11 2021

@author: Johno van IJsseldijk (j.e.vanijsseldijk@tudelft.nl)
"""

import sys

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from twoD_Functions import *

import supython as sup
import time

import pypardiso

t0 = time.time()

Lx = 4000.0
Ly = 2000.0
Nx = 801
Ny = 401
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
dz = 1
dV = dx*dy*dz

top = 1175//5
bot = 1075//5

plot=True

startT = int(sys.argv[1])
endt = int(startT+50)*24*60*60 

imfold = 'PM_test'

tol=1e-2

Params = np.load(r"Parameters/Heterogeneous.npz")
K = Params['K']
phi0 = Params['phi0']
K0 = Params['K0']
nu = Params['nu']
E = Params['E']
mask = Params['mask']
rhoR = Params['rhoR']

b = .5
rho_a0 = 530*np.ones(Nx*Ny)
rho_b0 = 335*np.ones(Nx*Ny)
cf_a = 1e-8
cf_b = 2e-8
pref = 101325

mu_a = 2.3e-5
mu_b = 1e-5

rho_a0 = 1035*np.ones(Nx*Ny)
rho_b0 = 750*np.ones(Nx*Ny)
cf_a = 5e-10
cf_b = 5e-10
pref = 101325

mu_a = 1e-3
mu_b = 5e-4

pin = 5 * 1e7
pout = 0.5 * 1e7 
uxl = 0 * 1e-3
uxr = 0 * 1e-3

L = (E*nu/((1+nu)*(1-2*nu))) # Lame's first parameter
G = (E/(2*(1+nu))) # * np.ones(Nx) # Shear modulus
Kdr = (L+2*G/3) # Dry bulk modulus

K_ave = (Kdr[:-1,:-1]+Kdr[1:,1:] + Kdr[:-1,1:] + Kdr[1:,:-1])/4

G_ave = (G[:-1,:-1]+G[1:,1:] + G[:-1,1:] + G[1:,:-1])/4

## Time
dt=(5/8/32)*24*60*60 
maxdt = 10*60*60*24
Nt= int(endt // dt)

p0 = 1e7 

coords=[]
cells=[]

coords = [(304,200), (609,207)]
cells = [Nx*200+304, Nx*207+609]

Well = {"Coords": coords,
        "Cell"  : cells, 
        "WI"    : [1000]*len(coords),
        "Pwell" : [pin  ,  pout]*(len(coords)//2),
        "Type"  : [1   ,   0]*(len(coords)//2)}

## Discretization
xc = np.linspace(dx/2,Lx-dx/2,Nx)

## Initialization
qa = np.zeros(Nx*Ny)
qb = np.zeros(Nx*Ny)

p_n = np.ones(Nx*Ny)*p0
S_n = np.ones(Nx*Ny)*0

rho_a,drhoa = ComputeRho(cf_a,rho_a0,pref,p_n)
rho_b,drhob = ComputeRho(cf_b,rho_b0,pref,p_n)

Fx = np.zeros(((Nx+1),(Ny+1)))
Fy = np.zeros(((Nx+1),(Ny+1)))

Fy = Fy.reshape(-1,order='F')
Fx = Fx.reshape(-1,order='F')

ux0 = 0
uy0 = 0

ux_n = np.ones((Nx+1)*(Ny+1))*ux0
uy_n = np.ones((Nx+1)*(Ny+1))*uy0

BC_Flags_U = np.zeros((Nx+1,Ny+1),dtype=int)
# BC_Flags_U[:,0] = 1 # Top / North
BC_Flags_U[:,-1] = 1 # Bottom / South
BC_Flags_U[0,:] = 1 # Left / West
BC_Flags_U[-1,:] = 1 # Right / East
Flags_Ux = BC_Flags_U.reshape(-1,order='F')
BC_Flags_U = np.zeros((Nx+1,Ny+1),dtype=int)
BC_Flags_U[0,:] = 1 # Left / West
BC_Flags_U[-1,:] = 1 # Right / East
BC_Flags_U[:,0] = 1 # Top / North
BC_Flags_U[:,-1] = 1 # Bottom / South
Flags_Uy = BC_Flags_U.reshape(-1,order='F')

Flags_P = np.zeros((Nx,Ny),dtype=int)
Flags_P = np.abs(mask-1)
#Flags_P[:,top:] = 1 # Bottom / South
#Flags_P[:,:bot] = 1 # Top / North
#Flags_P[:1500//5,:] = 1 # Left / West
#Flags_P[Nx-1500//5:,:] = 1 # Right / East
Flags_P = Flags_P.reshape(-1,order='F')

Trockx,Trocky = ComputeRockTransmissibility(Nx,Ny,dx,dy,dz,K,K)

## BC FLOW:
maskx = np.zeros_like(Trockx)
maskx[:-1,:] += mask
maskx[1:,:] += mask
maskx = (maskx >= 1)

masky = np.zeros_like(Trocky)
masky[:,:-1] += mask
masky[:,1:] += mask
masky = (masky >= 1)

Trockx *= maskx
Trocky *= masky

phi=phi0
dphidp = 0
mob_a, dmob_a = ComputeApp(Nx,Ny,Trockx,Trocky,mu_a,S_n,p_n,rho_a)[2:]
mob_b, dmob_b = ComputeAsp(Nx,Ny,Trockx,Trocky,mu_b,S_n,p_n,rho_b)[2:]

## INITIAL Mechanics solution
Axx = ComputeAxx(Nx,Ny,dx,dy,L,G)
Axy = ComputeAxy(Nx,Ny,L,G)
Ayx = ComputeAyx(Nx,Ny,L,G)
Ayy = ComputeAyy(Nx,Ny,dx,dy,L,G)

Axp = ComputeAxp(Nx,Ny,dx,dy,b)
Ayp = ComputeAyp(Nx,Ny,dx,dy,b)
Axx,Axy,Ayx,Ayy,Axp,Ayp = Add_BC_U(Axx,Axy,Ayx,Ayy,Axp,Ayp,Flags_Ux,Flags_Uy)

J = sparse.csc_matrix(sparse.vstack([
        sparse.hstack([Axx, Axy]),
        sparse.hstack([Ayx, Ayy])]))

R = np.hstack([-Fx,
               -Fy])
               
ux_n = linalg.spsolve(J,-R)[:(Nx+1)*(Ny+1)]
uy_n = linalg.spsolve(J,-R)[(Nx+1)*(Ny+1):]

phi,dphidu,dphidp = DeltaPoro(Nx,Ny,phi0,ux0,uy0,p0,b,Kdr,ux_n,uy_n,p_n,dx,dy)

Kfl = ((1-S_n)*(cf_a) + S_n*(cf_b)).reshape((Nx,Ny),order='F')       
Ksat = calc_Ksat(K_ave,K0,Kfl,phi.reshape((Nx,Ny),order='F'))
rhoT0 = (1-phi.reshape((Nx,Ny),order='F'))*rhoR + mask*(phi*((1-S_n)*rho_a + S_n*rho_b)).reshape((Nx,Ny),order='F')

rho_cp = np.zeros((rhoT0.shape[0]+1,rhoT0.shape[1]+1))

rho_cp[1:,1:] += rhoT0
rho_cp[1:,:-1] += rhoT0
rho_cp[:-1,1:] += rhoT0
rho_cp[:-1,:-1] += rhoT0
rho_cp[1:-1,:] /= 2
rho_cp[:,1:-1] /= 2

rho_cp = (rho_cp[:-1,:-1]+rho_cp[1:,1:] + rho_cp[:-1,1:] + rho_cp[1:,:-1])/4

cp0 = np.sqrt((Ksat+(4/3)*G_ave)/rho_cp)

if plot:
    fig = plt.figure(1,figsize=[19.2,10])
    fig.clf()
    axes = [[]]*4
    axes[0] = fig.add_subplot(2,2,1)
    axes[1] = fig.add_subplot(2,2,2)
    axes[2] = fig.add_subplot(2,2,3)
    axes[3] = fig.add_subplot(2,2,4)
    
    
    
    axes[0].imshow(p_n.reshape((Nx,Ny),order='F').transpose(),aspect='auto',extent=[0,Lx,Ly,0])
    axes[0].set_title('Pressure')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('y [m]')
    axes[1].imshow(S_n.reshape((Nx,Ny),order='F').transpose(),aspect='auto',extent=[0,Lx,Ly,0],clim=[0,1])
    axes[1].set_title('Saturation')
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    axes[2].imshow((rhoT0-rhoT0).transpose(),aspect='auto',extent=[0,Lx,Ly,0])
    axes[2].set_title(r'Total density change')
    axes[2].set_xlabel('x [m]')
    axes[2].set_ylabel('y [m]')
    axes[3].imshow((cp0-cp0).transpose(),aspect='auto',extent=[0,Lx,Ly,0])
    axes[3].set_title(r'P-wave velocity change')
    axes[3].set_xlabel('x [m]')
    axes[3].set_ylabel('y [m]')
    plt.suptitle('Simulation running: {:.1f}%'.format(0),weight='bold')
	
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

f = open(r'logfile{:03d}.txt'.format(int(endt/60/60/24)), 'w')

plottot=20
ploti=0

Axs = ComputeAxs(Nx,Ny)
Ays = ComputeAys(Nx,Ny)

CONV = 0
if startT == 0:
    t=0
    it=0

    plt.savefig(r"{:s}/{:04d}.jpg".format(imfold,0))

    hdr = sup.makehdr(cp0.T, dx=dx, dt=dy, t0=0, f2=0, scl=-1000, gelev=0, sdepth=0)
    sup.writesu("SU_models/cp_mod_{:04d}.su".format(0),cp0.T,hdr)
    sup.writesu("SU_models/ro_mod_{:04d}.su".format(0),rhoT0.T,hdr)

else:
    fz = np.load('StateAtT{:.2f}.npz'.format(startT))
    for item in fz.files:
        globals()[item] = fz[item]

    if dt_fin > dt:
        dt = dt_fin

### MAIN TIME LOOP
while t < endt:
    Iter=0
    ux_nu = ux_n.copy()
    uy_nu = uy_n.copy()
    S_nu = S_n.copy()
    p_nu = p_n.copy()
  
    rho_a_old = rho_a
    rho_b_old = rho_b
    
    Rold = np.NaN
    
    while True:     
        ## Update densities and porosity
        rho_a,drhoa = ComputeRho(cf_a,rho_a0,pref,p_nu)
        rho_b,drhob = ComputeRho(cf_b,rho_b0,pref,p_nu)
        
        phi,dphidu,dphidp = DeltaPoro(Nx,Ny,phi0,ux0,uy0,p0,b,Kdr,ux_nu,uy_nu,p_nu,dx,dy)

        ## Construct matrices 
        App, Tpp, mob_a, dmob_a = ComputeApp(Nx,Ny,Trockx,Trocky,mu_a,S_nu,p_nu,rho_a,drhoa)
        Aps = ComputeAps(Nx,Ny,phi,rho_a,dV,dt)
        Asp, Tsp, mob_b, dmob_b = ComputeAsp(Nx,Ny,Trockx,Trocky,mu_b,S_nu,p_nu,rho_b,drhob)
        Ass = ComputeAss(Nx,Ny,phi,rho_b,dV,dt)
    
        Axx = ComputeAxx(Nx,Ny,dx,dy,L,G)
        Axy = ComputeAxy(Nx,Ny,L,G)
        Ayx = ComputeAyx(Nx,Ny,L,G)
        Ayy = ComputeAyy(Nx,Ny,dx,dy,L,G)
        
        Axp = ComputeAxp(Nx,Ny,dx,dy,b)
        Ayp = ComputeAyp(Nx,Ny,dx,dy,b)
        
        Apx = ComputeApx(Nx,Ny,dphidu,S_nu,rho_a,dt,dx,dy)
        Apy = ComputeApy(Nx,Ny,dphidu,S_nu,rho_a,dt,dx,dy)
        
        Asx = ComputeAsy(Nx,Ny,dphidu,S_nu,rho_b,dt,dx,dy)
        Asy = ComputeAsy(Nx,Ny,dphidu,S_nu,rho_b,dt,dx,dy)        

        ## Add Boundary conditions/Wells
        qa, qb, dqdpa, dqdSa, dqdpb, dqdSb = ComputeWellFluxes(Well,mob_a,mob_b,dmob_a,dmob_b,dV,p_nu,Nx,Ny,S_nu,ComputeRho(cf_a,rho_a0,pref,pin),rho_a,rho_b,mu_a,K,drhoa,drhob)  
        
        Apx,Apy = Add_BC_P(Apx,Apy,Flags_P)
        Asx,Asy = Add_BC_P(Asx,Asy,Flags_P)

        Axx,Axy,Ayx,Ayy,Axp,Ayp = Add_BC_U(Axx,Axy,Ayx,Ayy,Axp,Ayp,Flags_Ux,Flags_Uy)

        Capp = sparse.diags(dphidp*S_nu*(rho_a)*dV/dt) + sparse.diags(phi * dV * S_nu * drhoa / dt)
        Cbpp = sparse.diags(dphidp*(1-S_nu)*(rho_b)*dV/dt) + sparse.diags(phi * dV * (1-S_nu) * drhob / dt)    
    
 
        ## Calculate the residual
        AS = sparse.diags(phi.reshape(-1,order='F') * dV/dt + dphidp * dV/dt)
        R = np.hstack([-Fx + Axx@ux_nu + Axy@uy_nu + Axs@S_nu + Axp@p_nu,
                       -Fy + Ayx@ux_nu + Ayy@uy_nu + Ays@S_nu + Ayp@p_nu,
                       -qa - Apx@ux_n - Apy@uy_n + AS * (rho_a * S_nu - rho_a_old * S_n) + Tpp @ p_nu + Apx@ux_nu + Apy@uy_n,
                       -qb - Asx@ux_n - Asy@uy_n+ AS * (rho_b * (1-S_nu) - rho_b_old * (1-S_n)) + Tsp @ p_nu + Asx@ux_nu + Asy@uy_nu])
        
        ## Calculate Jacobian    
        Jxx = Axx
        Jyy = Ayy
        Jyx = Ayx
        Jxy = Axy
        Jxp = Axp
        Jxs = Axs
        Jyp = Ayp
        Jys = Ays
        Jpy = Apy
        Jsy = Asy
        Jpx = Apx
        Jsx = Asx
        Jpa = App + Tpp - dqdpa + Capp
        Jsa = Aps - dqdSa + SaturationDerivatives(Nx,Ny,Trockx,Trocky,p_nu,dmob_a,rho_a)
        Jpb = Asp + Tsp - dqdpb + Cbpp
        Jsb = Ass - dqdSb + SaturationDerivatives(Nx,Ny,Trockx,Trocky,p_nu,dmob_b,rho_b)
		
        J = sparse.bmat([[Jxx,Jxy,Jxp,Jxs],
                         [Jyx,Jyy,Jyp,Jys],
                         [Jpx,Jpy,Jpa,Jsa],
                         [Jsx,Jsy,Jpb,Jsb]],format='csc')
            
        ## Solve the system
        delta = pypardiso.spsolve(J,-R)
        
        ## Update p, S and u
        ux_nu += delta[:(Nx+1)*(Ny+1)]
        uy_nu += delta[(Nx+1)*(Ny+1):2*(Nx+1)*(Ny+1)]
        S_nu  += delta[2*(Nx+1)*(Ny+1)+Nx*Ny:]
        p_nu  += delta[2*(Nx+1)*(Ny+1):2*(Nx+1)*(Ny+1)+Nx*Ny]
        
        S_nu[S_nu>1] = 1
        S_nu[S_nu<0] = 0
        
        ## Recompute porosity and wells
        rho_a,drhoa = ComputeRho(cf_a,rho_a0,pref,p_nu)
        rho_b,drhob = ComputeRho(cf_b,rho_b0,pref,p_nu)

        App, Tpp, mob_a, dmob_a = ComputeApp(Nx,Ny,Trockx,Trocky,mu_a,S_nu,p_nu,rho_a,drhoa)
        Asp, Tsp, mob_b, dmob_b = ComputeAsp(Nx,Ny,Trockx,Trocky,mu_b,S_nu,p_nu,rho_b,drhob)
		
        qa, qb, dqdpa, dqdSa, dqdpb, dqdSb = ComputeWellFluxes(Well,mob_a,mob_b,dmob_a,dmob_b,dV,p_nu,Nx,Ny,S_nu,ComputeRho(cf_a,rho_a0,pref,pin),rho_a,rho_b,mu_a,K,drhoa,drhob) 
        
        Capp = sparse.diags(dphidp*S_nu*(rho_a)*dV/dt) + sparse.diags(phi * dV * S_nu * drhoa / dt)
        Cbpp = sparse.diags(dphidp*(1-S_nu)*(rho_b)*dV/dt) + sparse.diags(phi * dV * (1-S_nu) * drhob / dt)
        
        Apx,Apy = Add_BC_P(Apx,Apy,Flags_P)
        Asx,Asy = Add_BC_P(Asx,Asy,Flags_P)
        
        ## Recompute Residual
        AS = sparse.diags(phi.reshape(-1,order='F') * dV/dt + dphidp * dV/dt)
        R = np.hstack([-Fx + Axx@ux_nu + Axy@uy_nu + Axs@S_nu + Axp@p_nu,
                       -Fy + Ayx@ux_nu + Ayy@uy_nu + Ays@S_nu + Ayp@p_nu,
                       -qa - Apx@ux_n - Apy@uy_n + AS * (rho_a * S_nu - rho_a_old * S_n) + Tpp @ p_nu + Apx@ux_nu + Apy@uy_nu,
                       -qb - Asx@ux_n - Asy@uy_n + AS * (rho_b * (1-S_nu) - rho_b_old * (1-S_n)) + Tsp @ p_nu + Asx@ux_nu + Asy@uy_nu])
        
        ## Check for convergence or iteration limit        
        Iter += 1
        print('Mechanics NORM',np.linalg.norm(R[:2*(Nx+1)*(Ny+1)]), flush=True)
        print('Saturation NORM',np.linalg.norm(R[2*(Nx+1)*(Ny+1):2*(Nx+1)*(Ny+1)+Nx*Ny]), flush=True)
        print('Pressure NORM',np.linalg.norm(R[2*(Nx+1)*(Ny+1)+Nx*Ny:]), flush=True)
        print(np.linalg.norm(R), flush=True)

        if np.linalg.norm(R) < tol:         
            p_n = p_nu.copy() 
            S_n = S_nu.copy()
            ux_n = ux_nu.copy()
            uy_n = uy_nu.copy()
            
            Kfl = ((1-S_nu)*(cf_a) + S_nu*(cf_b)).reshape((Nx,Ny),order='F')
            
            rhoT = (1-phi.reshape((Nx,Ny),order='F'))*rhoR + mask*(phi*((1-S_nu)*rho_a + S_nu*rho_b)).reshape((Nx,Ny),order='F')
         
            rho_cp = np.zeros((rhoT.shape[0]+1,rhoT.shape[1]+1))
            rho_cp[1:,1:] += rhoT
            rho_cp[1:,:-1] += rhoT
            rho_cp[:-1,1:] += rhoT
            rho_cp[:-1,:-1] += rhoT
            rho_cp[1:-1,:] /= 2
            rho_cp[:,1:-1] /= 2
            rho_cp = (rho_cp[:-1,:-1]+rho_cp[1:,1:] + rho_cp[:-1,1:] + rho_cp[1:,:-1])/4

            Ksat = calc_Ksat(K_ave,K0,Kfl,phi.reshape((Nx,Ny),order='F'))

            cp = np.sqrt((Ksat+(4/3)*G_ave)/rho_cp)

            it += 1
            t += dt
            
            hdr = sup.makehdr(cp.T, dx=dx, dt=dy, t0=0, f2=0, scl=-1000, gelev=0, sdepth=0)
            sup.writesu("SU_models/cp_mod_{:04d}.su".format(it),cp.T,hdr)
            sup.writesu("SU_models/ro_mod_{:04d}.su".format(it),rhoT.T,hdr)
            
            infostr='Step {:d} at {:.2f} Days ({:.0f}s)'.format(it,t/(24*60*60),t)
            
            f.write(infostr+'\n')
            
            if Iter < 4 and dt < maxdt:
                dt *= 2
                infostr='     Doubling dt at time {:.2f} days (step: {:d}), new dt: {:.2f} days'.format(t/(24*60*60),it,dt/(24*60*60))
                print(infostr, flush=True)
                f.write(infostr+'\n') 
                
            if plot: 
                axes[0].imshow(p_n.reshape((Nx,Ny),order='F').transpose(),aspect='auto',extent=[0,Lx,Ly,0])
                axes[0].set_title('Pressure')
                axes[0].set_xlabel('x [m]')
                axes[0].set_ylabel('y [m]')
                axes[1].imshow(S_n.reshape((Nx,Ny),order='F').transpose(),aspect='auto',extent=[0,Lx,Ly,0],clim=[0,1])
                axes[1].set_title('Saturation')
                axes[1].set_xlabel('x [m]')
                axes[1].set_ylabel('y [m]')
                axes[2].imshow((rhoT-rhoT0).transpose(),aspect='auto',extent=[0,Lx,Ly,0])
                axes[2].set_title(r'Density change')
                axes[2].set_xlabel('x [m]')
                axes[2].set_ylabel('y [m]')
                axes[3].imshow((cp-cp0).transpose(),aspect='auto',extent=[0,Lx,Ly,0])
                axes[3].set_title(r'P-wave velocity change')
                axes[3].set_xlabel('x [m]')
                axes[3].set_ylabel('y [m]')
                ploti += 1
                plt.suptitle('Simulation running: {:.1f}%'.format(t/endt*100),weight='bold')
                plt.draw()
                plt.pause(0.001)
                plt.savefig(r"{:s}/{:04d}.jpg".format(imfold,it))
            
            if t+dt > endt and t != endt:
                dt_fin = dt
                dt = endt-t
                infostr='     Final dt at time {:.2f} days (step: {:d}), new dt: {:.2f} days'.format(t/(24*60*60),it,dt/(24*60*60))
                print(infostr, flush=True)
                f.write(infostr+'\n')
            break
        elif Iter > 15 or np.isnan(np.linalg.norm(R)) or np.linalg.norm(R) > 1e6 or (np.abs(np.linalg.norm(R)-Rold) <= 1e-3*np.abs(np.linalg.norm(R))):
            if (np.abs(np.linalg.norm(R)-Rold) <= 1e-3*np.abs(np.linalg.norm(R))):
                print("CONVERGED ABOVE TOLERANCE HALVING dt", flush=True)
                CONV += 1

            p_nu = p_n.copy() 
            S_nu = S_n.copy()
            ux_nu = ux_n.copy()
            uy_nu = uy_n.copy()

            Rold = np.NaN
                        
            rho_a,drhoa = ComputeRho(cf_a,rho_a0,pref,p_nu)
            rho_b,drhob = ComputeRho(cf_b,rho_b0,pref,p_nu)
            
            mob_a, dmob_a = ComputeApp(Nx,Ny,Trockx,Trocky,mu_a,S_nu,p_nu,rho_a)[2:]
            mob_b, dmob_b = ComputeAsp(Nx,Ny,Trockx,Trocky,mu_b,S_nu,p_nu,rho_b)[2:]            
        
            dt /= 2
            Iter=0
            
            infostr='     Halving dt at time {:.2f} days (step: {:d}), new dt: {:.2f} days'.format(t/(24*60*60),it,dt/(24*60*60))
            print(infostr, flush=True)
            f.write(infostr+'\n')

            if dt < 60*60*24/3600: print("DT is really small breaking", flush=True); break
        if dt < 60*60*24/3600: print("DT is really small breaking", flush=True); break
        Rold = np.linalg.norm(R)
infostr='     Finished simulation at t: {:.2f} days after {:d} steps'.format(t/(24*60*60),it)
print(infostr, flush=True)
f.write(infostr)
f.close()

np.savez_compressed('StateAtT{:.2f}.npz'.format(t/(24*60*60)),p_n=p_n,S_n=S_n,ux_n=ux_n,uy_n=uy_n,rho_a=rho_a,rho_b=rho_b,phi=phi,dt=dt,it=it,dt_fin=dt_fin,t=t)

if dt < 60*60*24/8:
    plt.suptitle('Simulation Aborted at t: {:.2f} days!'.format(t/(24*60*60)),weight='bold')
else:
    plt.suptitle('Simulation Finished!',weight='bold')

t1 = time.time()

print("Simulation finished after {:.2f} minutes".format((t1-t0)/60), flush=True)
