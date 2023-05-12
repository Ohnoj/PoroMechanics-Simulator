"""
Created on Wed Jul 28 19:10:22 2021

@author: Johno van IJsseldijk (j.e.vanijsseldijk@tudelft.nl)
"""

import numpy as np
from scipy import sparse

def ComputeWellFluxes(Well,mob_a,mob_b,dmob_a,dmob_b,dV,p,Nx,Ny,S,rho_ain,rho_a,rho_b,mu_a,K,drhoa=0,drhob=0):
    if np.isscalar(drhoa):
        drhoa = np.zeros(Nx*Ny)
    if np.isscalar(drhob):
        drhob = np.zeros(Nx*Ny)
    q_a = np.zeros(Nx*Ny)
    q_b = np.zeros(Nx*Ny)
    
    dqdpa = np.zeros(Nx*Ny)
    dqdSa = np.zeros(Nx*Ny)
    dqdpb = np.zeros(Nx*Ny)
    dqdSb = np.zeros(Nx*Ny)
    
    for w in range(len(Well["Pwell"])):
        i = Well["Cell"][w]
        if (Well["Type"][w]): 
            # injection well
            q_a[i] = Well["WI"][w] * (rho_ain[0][i]/mu_a)*K.reshape(-1,order='F')[i] * (Well["Pwell"][w] - p[i]) * dV
            dqdpa[i] += Well["WI"][w] * rho_ain[0][i] / mu_a * K.reshape(-1,order='F')[i] * dV * -1  
        else: 
            # production well
            q_a[i] = Well["WI"][w] * rho_a[i] * mob_a[i] * K.reshape(-1,order='F')[i] * (Well["Pwell"][w] - p[i]) * dV
            q_b[i] = Well["WI"][w] * rho_b[i] * mob_b[i] * K.reshape(-1,order='F')[i] * (Well["Pwell"][w] - p[i]) * dV
            dqdpb[i] += Well["WI"][w] * rho_b[i] * mob_b[i] * K.reshape(-1,order='F')[i] * dV * -1 + Well["WI"][w] * drhob[i] * mob_b[i] * K.reshape(-1,order='F')[i] * dV * (Well["Pwell"][w] - p[i])
            dqdSb[i] += Well["WI"][w] * rho_b[i] * dmob_b[i] * K.reshape(-1,order='F')[i] * dV * (Well["Pwell"][w] - p[i])
            
            dqdpa[i] += Well["WI"][w] * rho_a[i] * mob_a[i] * K.reshape(-1,order='F')[i] * dV * -1 + Well["WI"][w] * drhoa[i] * mob_a[i] * K.reshape(-1,order='F')[i] * dV * (Well["Pwell"][w] - p[i])
            dqdSa[i] += Well["WI"][w] * rho_a[i] * dmob_a[i] * K.reshape(-1,order='F')[i] * dV * (Well["Pwell"][w] - p[i])
            
    return q_a, q_b, sparse.diags(dqdpa), sparse.diags(dqdSa), sparse.diags(dqdpb), sparse.diags(dqdSb)

def DeltaPoro(Nx,Ny,phi0,ux0,uy0,p0,b,Kdr,ux_nu,uy_nu,p_nu,dx,dy,M=0):    
    
    ux = ux_nu.copy().reshape((Nx+1,Ny+1),order='F')
    uy = uy_nu.copy().reshape((Nx+1,Ny+1),order='F')
    p = p_nu.copy().reshape((Nx,Ny),order='F')
    
    if hasattr(p0, "__len__"):
        ux0 = ux0.copy().reshape((Nx+1,Ny+1),order='F')
        uy0 = uy0.copy().reshape((Nx+1,Ny+1),order='F')
        p0 = p0.copy().reshape((Nx,Ny),order='F')
        
        Kave = (Kdr[:-1,:-1]+Kdr[1:,1:] + Kdr[:-1,1:] + Kdr[1:,:-1])/4

        dphi = b*((ux[:-1,:-1]-ux0[:-1,:-1]+ux[1:,1:]-ux0[1:,1:])/dx/dy + (uy[:-1,:-1]-uy0[:-1,:-1]+uy[1:,1:]-uy0[1:,1:])/dy/dx) + (((b-phi0)*(1-b))/Kave) * (p[:,:]-p0)
        dphidu = b
        dphidp = ((b-phi0.reshape(-1,order='F'))*(1-b))/Kave.reshape(-1,order='F')

    else: 
        if M != 0:
            dphi = b*((ux[:-1,:-1]-ux0+ux[1:,1:]-ux0)/dx/dy + (uy[:-1,:-1]-uy0+uy[1:,1:]-uy0)/dy/dx) + 1/M * (p[:,:]-p0)  
            dphidu = b
            dphidp = 1/M
        else:
            Kave = (Kdr[:-1,:-1]+Kdr[1:,1:] + Kdr[:-1,1:] + Kdr[1:,:-1])/4
            dphi = b*((ux[:-1,:-1]-ux0+ux[1:,1:]-ux0)/dx/dy + (uy[:-1,:-1]-uy0+uy[1:,1:]-uy0)/dy/dx) + (((b-phi0)*(1-b))/Kave) * (p[:,:]-p0)     
            dphidu = b
            dphidp = ((b-phi0.reshape(-1,order='F'))*(1-b))/Kave.reshape(-1,order='F') 
    
    phi = phi0.reshape(-1,order='F') + dphi.reshape(-1,order='F')
    
    return phi, dphidu, dphidp

def CalculateUpwind(Nx,Ny,Trockx,Trocky,p):
    upx = np.zeros((Nx+1,Ny))
    upy = np.zeros((Nx,Ny+1))
    
    N = Nx * Ny
    
    upx[1:-1,:] = Trockx[1:-1,:] * (p.reshape((Nx,Ny),order='F')[:-1,:] - p.reshape((Nx,Ny),order='F')[1:,:])
    upy[:,1:-1] = Trocky[:,1:-1] * (p.reshape((Nx,Ny),order='F')[:,:-1] - p.reshape((Nx,Ny),order='F')[:,1:])
    
    x1 = (upx[1:,:] >= 0).astype(int).reshape(-1,order='F')
    x2 = (upx[:-1,:] < 0).astype(int).reshape(-1,order='F')
    
    y1 = (upy[:,1:] >= 0).astype(int).reshape(-1,order='F')
    y2 = (upy[:,:-1] < 0).astype(int).reshape(-1,order='F')
    
    Upwindx = sparse.spdiags((x1,x2),(0,1),N,N)
    Upwindy = sparse.spdiags((y1,y2),(0,Nx),N,N)
    
    return Upwindx, Upwindy, upx, upy

def ComputeRockTransmissibility(Nx,Ny,dx,dy,dz,Kx,Ky):
    Trockx = np.zeros((Nx+1,Ny))
    Trocky = np.zeros((Nx,Ny+1))
        
    
    LambdaHx = 2*Kx[:-1,:]*Kx[1:,:]/(Kx[:-1,:]+Kx[1:,:])
    LambdaHy = 2*Ky[:,:-1]*Ky[:,1:]/(Ky[:,:-1]+Ky[:,1:])
    
    Trockx[1:Nx,:] = LambdaHx*dy*dz/dx
    Trocky[:,1:Ny] = LambdaHy*dx*dz/dy
    
    
    return Trockx, Trocky

def SaturationDerivatives(Nx,Ny,Trockx,Trocky,p_nu,dmob,rho):
    ux, uy = CalculateUpwind(Nx,Ny,Trockx,Trocky,p_nu)[2:]
    
    X1 = np.min((ux[:-1,:].reshape(-1,order='F'),np.zeros(Nx*Ny)),axis=0)*dmob*rho
    X2 = np.max((ux[1:,:].reshape(-1,order='F'),np.zeros(Nx*Ny)),axis=0)*dmob*rho
    Y1 = np.min((uy[:,:-1].reshape(-1,order='F'),np.zeros(Nx*Ny)),axis=0)*dmob*rho
    Y2 = np.max((uy[:,1:].reshape(-1,order='F'),np.zeros(Nx*Ny)),axis=0)*dmob*rho
    
    Vecs = (-Y2,-X2,X2+Y2-X1-Y1,X1,Y1)
    Indices = (-Nx,-1,0,1,Nx)
    
    return sparse.spdiags(Vecs,Indices,Nx*Ny,Nx*Ny)

def ComputeRho(cf,rho0,p0,p):
    rho = rho0 * np.exp(cf * (p-p0))
    drhodp = cf * rho0 * np.exp(cf * (p-p0))
    
    return rho, drhodp

def ComputeAsp(Nx,Ny,Trockx,Trocky,mu,S,p,rho,drho=0,kralg='square'): 
    if kralg == 'linear':
        kr = 1-S
    else:
        kr = (1-S)**2
    Asp = np.zeros((Nx,Nx))

    Upwindx, Upwindy, upx, upy = CalculateUpwind(Nx,Ny,Trockx,Trocky,p)
    
    mob = kr/mu
    
    Tx = np.zeros((Nx+1,Ny))
    Ty = np.zeros((Nx,Ny+1))
    Tx[1:-1,:] = Trockx[1:-1,:]*((Upwindx@(mob*rho)).reshape((Nx,Ny),order='F'))[:-1,:] 
    Ty[:,1:-1] = Trocky[:,1:-1]*((Upwindy@(mob*rho)).reshape((Nx,Ny),order='F'))[:,:-1] 
    
    x1 = Tx[:-1,:].reshape(-1,order='F')
    x2 = Tx[1:,:].reshape(-1,order='F')
    
    x11 = (np.fmin(upx[:-1,:],0)*((Upwindx@(mob*drho)).reshape((Nx,Ny),order='F'))[:,:]).reshape(-1,order='F')
    x22 = (np.fmax(upx[1:,:],0)*((Upwindx@(mob*drho)).reshape((Nx,Ny),order='F'))[:,:]).reshape(-1,order='F')
    
    y1 = Ty[:,:-1].reshape(-1,order='F')
    y2 = Ty[:,1:].reshape(-1,order='F')
    
    y11 = (np.fmin(upy[:,:-1],0)*((Upwindy@(mob*drho)).reshape((Nx,Ny),order='F'))[:,:]).reshape(-1,order='F')
    y22 = (np.fmax(upy[:,1:],0)*((Upwindy@(mob*drho)).reshape((Nx,Ny),order='F'))[:,:]).reshape(-1,order='F')

        
    Tsp = sparse.spdiags((-y2,-x2,x1+x2+y1+y2,-x1,-y1),(-Nx,-1,0,1,Nx),Nx*Ny,Nx*Ny,format="csr")   
    Asp = sparse.spdiags((-y22,-x22,x22+y22-x11-y11,x11,y11),(-Nx,-1,0,1,Nx),Nx*Ny,Nx*Ny,format="csr")
    
    if kralg == 'linear':
        dkr = -1
    else:
        dkr = - 2 * (1 - S)
    dmob = dkr / mu
    
    return Asp, Tsp, mob, dmob

def ComputeApp(Nx,Ny,Trockx,Trocky,mu,S,p,rho,drho=0,kralg='square'): 
    if kralg == 'linear':
        kr = S
    else:
        kr = S**2
    App = np.zeros((Nx,Nx))

    Upwindx, Upwindy, upx, upy = CalculateUpwind(Nx,Ny,Trockx,Trocky,p)
    
    mob = kr/mu
    
    Tx = np.zeros((Nx+1,Ny))
    Ty = np.zeros((Nx,Ny+1))
    
    Tx[1:-1,:] = Trockx[1:-1,:]*((Upwindx@(mob*rho)).reshape((Nx,Ny),order='F'))[:-1,:] 
    Ty[:,1:-1] = Trocky[:,1:-1]*((Upwindy@(mob*rho)).reshape((Nx,Ny),order='F'))[:,:-1] 
    
    
    
    x1 = Tx[:-1,:].reshape(-1,order='F')
    x2 = Tx[1:,:].reshape(-1,order='F')
    
    x11 = (np.fmin(upx[:-1,:],0)*((Upwindx@(mob*drho)).reshape((Nx,Ny),order='F'))[:,:]).reshape(-1,order='F')
    x22 = (np.fmax(upx[1:,:],0)*((Upwindx@(mob*drho)).reshape((Nx,Ny),order='F'))[:,:]).reshape(-1,order='F')
    
    y1 = Ty[:,:-1].reshape(-1,order='F')
    y2 = Ty[:,1:].reshape(-1,order='F')
    
    y11 = (np.fmin(upy[:,:-1],0)*((Upwindy@(mob*drho)).reshape((Nx,Ny),order='F'))[:,:]).reshape(-1,order='F')
    y22 = (np.fmax(upy[:,1:],0)*((Upwindy@(mob*drho)).reshape((Nx,Ny),order='F'))[:,:]).reshape(-1,order='F')
    
    
    Tpp = sparse.spdiags((-y2,-x2,x1+x2+y1+y2,-x1,-y1),(-Nx,-1,0,1,Nx),Nx*Ny,Nx*Ny,format="csr")
    App = sparse.spdiags((-y22,-x22,x22+y22-x11-y11,x11,y11),(-Nx,-1,0,1,Nx),Nx*Ny,Nx*Ny,format="csr")
    
    if kralg == 'linear':
        dkr = 1
    else:
        dkr = 2 * S
    dmob = dkr / mu
    
    return App, Tpp, mob, dmob


def ComputeAps(Nx,Ny,phi,rho,dV,dt):
    Aps = sparse.diags(np.ones(Nx*Ny)*phi*dV/dt*rho,format="csr")
    
    return Aps

def ComputeAss(Nx,Ny,phi,rho,dV,dt):
    Ass = -sparse.diags(np.ones(Nx*Ny)*phi*dV/dt*rho,format="csr")
    
    return Ass

def ComputeAxx(Nx,Ny,dx,dy,L,G):
    N = (Nx+1)*(Ny+1)
    
    l = L.copy().reshape(-1,order='F')
    g = G.copy().reshape(-1,order='F') 
    
    
    d0 = 4 * g * (-3/8) * (dx/dy) + 4 * (l+2*g) * (-3/8) * (dy/dx)
    d0[:Nx+1] /= 2
    d0[-Nx:] /= 2
    d0[0::Nx+1] /= 2
    d0[Nx::Nx+1] /= 2
    d1u = 2 * g * (-1/8) * (dx/dy) + 2 * (l+2*g) * (3/8) * (dy/dx)
    d1u[:Nx+1] /= 2
    d1u[-Nx:] /= 2
    d1u[0::Nx+1] = 0
    d1l = 2 * g * (-1/8) * (dx/dy) + 2 * (l+2*g) * (3/8) * (dy/dx)
    d1l[:Nx+1] /= 2
    d1l[-Nx-1:] /= 2
    d1l[Nx::Nx+1] = 0
    d3 = 2 * g * (3/8) * (dx/dy) + 2 * (l+2*g) * (-1/8) * (dy/dx)
    d3[0::Nx+1] /= 2
    d3[Nx::Nx+1] /= 2
    d4 = g * (1/8) * (dx/dy) + (l+2*g) * (1/8) * (dy/dx)
    d4[0::Nx+1] = 0
    d5 = g * (1/8) * (dx/dy) + (l+2*g) * (1/8) * (dy/dx)
    d5[Nx::Nx+1] = 0
    
    Axx = sparse.spdiags((d5,d3,d4,d1l,d0,d1u,d5,d3,d4),(-Nx-2,-Nx-1,-Nx,-1,0,1,Nx,Nx+1,Nx+2),N,N,format="csr")
            
    return Axx

def ComputeAxp(Nx,Ny,dx,dy,b):
    x1 = np.ones(Nx+1) * b * dy/2
    
    A0 = sparse.spdiags((x1,-x1),(-1,0),Nx+1,Nx)
    
    A = sparse.block_diag([A0]*(Ny))

    Z = np.zeros(((Nx+1),Nx*Ny))
    
    Axp = sparse.vstack([Z,A]) + sparse.vstack([A,Z]) 
    
    return Axp.tocsr()

def ComputeAxs(Nx,Ny):
    
    Axs = sparse.csr_matrix(((Nx+1)*(Ny+1),Nx*Ny))
    
    return Axs

def ComputeApx(Nx,Ny,dphidu,S,rho_a,dt,dx,dy):
    x1 = S * dphidu * rho_a * dy/2 / dt
    
    A0 = sparse.spdiags((-x1,x1),(0,1),Nx,Nx+1)
    
    A = sparse.block_diag([A0]*(Ny))

    Z = np.zeros((Nx*Ny,(Nx+1)))
    
    Apx = sparse.hstack([Z,A]) + sparse.hstack([A,Z])
    
    return Apx.tocsr()

def ComputeApy(Nx,Ny,dphidu,S,rho_a,dt,dx,dy):
    x1 = S * dphidu * rho_a * dx/2/dt
    
    A0 = sparse.spdiags((x1,x1),(0,1),Nx,Nx+1)
    
    A = sparse.block_diag([A0]*(Ny))

    Z = np.zeros((Nx*Ny,(Nx+1)))
    
    Apy = sparse.hstack([Z,A]) + sparse.hstack([-A,Z])
    
    return Apy.tocsr()

def ComputeAsx(Nx,Ny,dphidu,S,rho_b,dt,dx,dy):   
    x1 = (1-S) * dphidu * rho_b * dy/2 / dt
    
    A0 = sparse.spdiags((-x1,x1),(0,1),Nx,Nx+1)
    
    A = sparse.block_diag([A0]*(Ny))

    Z = np.zeros((Nx*Ny,(Nx+1)))
    
    Asx = sparse.hstack([Z,A]) + sparse.hstack([A,Z])
    
    return Asx.tocsr()

def ComputeAsy(Nx,Ny,dphidu,S,rho_b,dt,dx,dy):   
    x1 = (1-S) * dphidu * rho_b * dx/2 / dt
    
    A0 = sparse.spdiags((x1,x1),(0,1),Nx,Nx+1)
    
    A = sparse.block_diag([A0]*(Ny))
    Z = np.zeros((Nx*Ny,(Nx+1)))
    
    Asy = sparse.hstack([Z,A]) + sparse.hstack([-A,Z])
    
    return Asy.tocsr()

def ComputeAyy(Nx,Ny,dx,dy,L,G):
    N = (Nx+1)*(Ny+1)
    
    l = L.copy().reshape(-1,order='F')
    g = G.copy().reshape(-1,order='F')
    
    d0 = 4 * (l+2*g) * (-3/8) * (dx/dy) + 4 * (g) * (-3/8) * (dy/dx)
    d0[:Nx+1] /= 2
    d0[-Nx-1:] /= 2
    d0[0::Nx+1] /= 2
    d0[Nx::Nx+1] /= 2
    d1u = 2 * (l+2*g) * (-1/8) * (dx/dy) + 2 * (g) * (3/8) * (dy/dx)
    d1u[:Nx+1] /= 2
    d1u[-Nx-1:] /= 2
    d1u[0::Nx+1] = 0
    d1l = 2 * (l+2*g) * (-1/8) * (dx/dy) + 2 * (g) * (3/8) * (dy/dx)
    d1l[:Nx+1] /= 2
    d1l[-Nx-1:] /= 2
    d1l[Nx::Nx+1] = 0
    d3 = 2 * (l+2*g) * (3/8) * (dx/dy) + 2 * (g) * (-1/8) * (dy/dx)
    d3[0::Nx+1] /= 2
    d3[Nx::Nx+1] /= 2
    d4 = (l+2*g) * (1/8) * (dx/dy) + (g) * (1/8) * (dy/dx)
    d4[0::Nx+1] = 0
    d5 = (l+2*g) * (1/8) * (dx/dy) + (g) * (1/8) * (dy/dx)
    d5[Nx::Nx+1] = 0    
    
    Ayy = sparse.spdiags((d5,d3,d4,d1l,d0,d1u,d5,d3,d4),(-Nx-2,-Nx-1,-Nx,-1,0,1,Nx,Nx+1,Nx+2),N,N,format="csr")
            
    return Ayy

def ComputeAyp(Nx,Ny,dx,dy,b):
    x1 = np.ones(Nx+1) * b * dx/2
    
    A0 = sparse.spdiags((x1,x1),(-1,0),Nx+1,Nx)
    
    A = sparse.block_diag([A0]*(Ny))

    Z = np.zeros(((Nx+1),Nx*Ny))
    
    Ayp = sparse.vstack([Z,A]) + sparse.vstack([-A,Z]) 
    
    return Ayp.tocsr()

def ComputeAys(Nx,Ny):
    
    Ays = sparse.csr_matrix(((Nx+1)*(Ny+1),Nx*Ny))
    
    return Ays

def ComputeAxy(Nx,Ny,L,G):
    l = L.copy().reshape(-1,order='F')
    g = G.copy().reshape(-1,order='F')
    
    y1 = .25*g+.25*l
    y1[0::Nx+1] = 0
    y3 = .25*g+.25*l
    y3[Nx::Nx+1] = 0
    y2 = np.zeros((Nx+1)*(Ny+1))
    y2[0::Nx+1] = .25*(l-g)[0::Nx+1]
    y2[Nx::Nx+1] = .25*(g-l)[Nx::Nx+1]
    x2 = np.zeros((Nx+1)*(Ny+1))
    x2[0] = (-.25*g-.25*l)[0]
    x2[-1] = (-.25*l-.25*g)[-1]
    x2[Nx] = (.25*g+.25*l)[Nx+1]
    x2[-(Nx+1)] = (.25*g+.25*l)[-(Nx+1)]
    x1 = np.zeros((Nx+1)*(Ny+1))
    x1[:Nx+1] = .25*(g-l)[:Nx+1]
    x1[-(Nx):] = .25*(l-g)[-(Nx):]
    x3 = np.zeros((Nx+1)*(Ny+1))
    x3[:Nx] = .25*(l-g)[:Nx]
    x3[-(Nx+1):] = .25*(g-l)[-(Nx+1):]
    
    Axy = sparse.spdiags((y3,-y2,-y1,x3,x2,x1,-y3,y2,y1),(-Nx-2,-Nx-1,-Nx,-1,0,1,Nx,Nx+1,Nx+2),(Nx+1)*(Ny+1),(Nx+1)*(Ny+1),format="csr")
            
    return Axy

def ComputeAyx(Nx,Ny,L,G):
    l = L.copy().reshape(-1,order='F')
    g = G.copy().reshape(-1,order='F')
    
    y1 = .25*g+.25*l
    y1[0::Nx+1] = 0
    y3 = .25*g+.25*l
    y3[Nx::Nx+1] = 0
    y2 = np.zeros((Nx+1)*(Ny+1))
    y2[0::Nx+1] = .25*(l-g)[0::Nx+1]
    y2[Nx::Nx+1] = .25*(g-l)[Nx::Nx+1]
    x2 = np.zeros((Nx+1)*(Ny+1))
    x2[0] = (-.25*g-.25*l)[0]
    x2[-1] = (-.25*l-.25*g)[-1]
    x2[Nx] = (.25*g+.25*l)[Nx+1]
    x2[-(Nx+1)] = (.25*g+.25*l)[-(Nx+1)]
    x1 = np.zeros((Nx+1)*(Ny+1))
    x1[:Nx+1] = .25*(g-l)[:Nx+1]
    x1[-(Nx):] = .25*(l-g)[-(Nx):]
    x3 = np.zeros((Nx+1)*(Ny+1))
    x3[:Nx] = .25*(l-g)[:Nx]
    x3[-(Nx+1):] = .25*(g-l)[-(Nx+1):]
        
    Ayx = np.transpose(sparse.spdiags((y3,-y2,-y1,x3,x2,x1,-y3,y2,y1),(-Nx-2,-Nx-1,-Nx,-1,0,1,Nx,Nx+1,Nx+2),(Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
            
    return Ayx.tocsr()

def Add_BC_U(Axx,Axy,Ayx,Ayy,Axp,Ayp,Flagsx,Flagsy):
       
    for i in Flagsx.nonzero()[0]:
        Axx.data[Axx.indptr[i]:Axx.indptr[i+1]] = 0
        Axp.data[Axp.indptr[i]:Axp.indptr[i+1]] = 0
        Axy.data[Axy.indptr[i]:Axy.indptr[i+1]] = 0
        
    for i in Flagsy.nonzero()[0]:    
        Ayp.data[Ayp.indptr[i]:Ayp.indptr[i+1]] = 0
        Ayx.data[Ayx.indptr[i]:Ayx.indptr[i+1]] = 0
        Ayy.data[Ayy.indptr[i]:Ayy.indptr[i+1]] = 0
        
    Axx += sparse.diags(Flagsx,format='csr')
    Ayy += sparse.diags(Flagsy,format='csr')
    
    return Axx, Axy, Ayx, Ayy, Axp, Ayp 

def Add_BC_P(Apx,Apy,Flagsp):
    
    for i in Flagsp.nonzero()[0]:
        Apx.data[Apx.indptr[i]:Apx.indptr[i+1]] = 0
        Apy.data[Apy.indptr[i]:Apy.indptr[i+1]] = 0

    return Apx,Apy
    
def calc_Ksat(Kdry,K0,Kfl,por):
    return Kdry + ((1-Kdry/K0)**2)/((por/Kfl)+((1-por)/K0)-(Kdry/(K0**2)))

