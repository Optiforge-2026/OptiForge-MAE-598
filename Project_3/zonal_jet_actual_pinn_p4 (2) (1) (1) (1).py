import numpy as np
import time
tic = time.perf_counter()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib
from scipy.integrate import solve_ivp
from pathlib import Path

OUT=Path(__file__).resolve().parent / 'zonal_jet_outputs'
OUT.mkdir(exist_ok=True)
SHOW_ANIMATION = True
SAVE_GIF = True
L=100.0; N=128; gamma=1.0; r=2-gamma**2
x=np.linspace(0,L,N,endpoint=False)
k=2*np.pi*np.fft.fftfreq(N,d=L/N)
# deterministic low-mode initial condition, zero mean
rng=np.random.default_rng(7)
u0=np.zeros(N)
for n,a in [(3,.20),(4,.18),(5,.14),(6,.10)]:
    ph=rng.uniform(0,2*np.pi); u0 += a*np.cos(2*np.pi*n*x/L+ph)
u0-=u0.mean()

def rhs(mu):
    def f(t,u):
        uh=np.fft.fft(u)
        uxx=np.fft.ifft(-(k**2)*uh).real
        uxxxx=np.fft.ifft((k**4)*uh).real
        u2xx=np.fft.ifft(-(k**2)*np.fft.fft(u*u)).real
        u3xx=np.fft.ifft(-(k**2)*np.fft.fft(u*u*u)).real
        return -mu*u-r*uxx-3*uxxxx-2*gamma*u2xx+(1/3)*u3xx
    return f

def sim(mu,tend=1200,nt=241):
    ts=np.linspace(0,tend,nt)
    sol=solve_ivp(rhs(mu),(0,tend),u0,t_eval=ts,method='BDF',rtol=2e-6,atol=1e-8,max_step=5.0)
    if not sol.success: raise RuntimeError(sol.message)
    return ts,sol.y.T

def dominant_mode(u):
    e=np.abs(np.fft.rfft(u))**2; e[0]=0
    return int(np.argmax(e))

def jet_count(u):
    # positive local maxima above 20% positive amplitude
    thr=.2*max(np.max(u),1e-9); c=0
    for i in range(N):
        if u[i]>thr and u[i]>u[i-1] and u[i]>u[(i+1)%N]: c+=1
    return c

cases={}
for mu in [0.0,0.001,0.01,0.03]:
    print('sim',mu,flush=True); cases[mu]=sim(mu)

# snapshots for benchmark mu=.01
mu=.01; ts,U=cases[mu]
fig,axs=plt.subplots(2,3,figsize=(11,6.5),sharex=True,sharey=True)
for ax,tq in zip(axs.ravel(),[0,100,250,500,800,1200]):
    j=np.argmin(abs(ts-tq)); ax.plot(x,U[j]); ax.set_title(f't = {ts[j]:.0f}'); ax.grid(alpha=.25)
for ax in axs[-1]: ax.set_xlabel('eta')
for ax in axs[:,0]: ax.set_ylabel('U')
fig.suptitle('P1: Spectral-reference evolution, gamma=1, mu=0.01')
fig.tight_layout(); fig.savefig(OUT/'p1_snapshots.png',dpi=180); plt.close(fig)

# spacetime
fig,ax=plt.subplots(figsize=(9,4.6)); im=ax.imshow(U.T,origin='lower',aspect='auto',extent=[ts[0],ts[-1],0,L],cmap='coolwarm'); fig.colorbar(im,ax=ax,label='U'); ax.set_xlabel('time'); ax.set_ylabel('eta'); ax.set_title('P1/P4: Space-time evolution of zonal velocity'); fig.tight_layout(); fig.savefig(OUT/'spacetime.png',dpi=180); plt.close(fig)

# modes and jet counts across mu
mus=[]; modes=[]; jets=[]
for m,(tt,UU) in cases.items(): mus.append(m); modes.append(dominant_mode(UU[-1])); jets.append(jet_count(UU[-1]))
fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(mus,jets,'o-'); ax.set_xlabel('Bottom drag mu'); ax.set_ylabel('Final eastward-jet count'); ax.set_title('P3: Final jet count versus bottom drag'); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT/'jet_count_vs_mu.png',dpi=180); plt.close(fig)

# spectrum final for mu=.01
freq=np.arange(N//2+1); en=np.abs(np.fft.rfft(U[-1]))**2/N**2
fig,ax=plt.subplots(figsize=(7,4.5)); ax.semilogy(freq[1:25],en[1:25]+1e-16,'o-'); ax.set_xlabel('Fourier mode k'); ax.set_ylabel('Energy'); ax.set_title('P4: Final Fourier spectrum, mu=0.01'); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT/'fourier_spectrum.png',dpi=180); plt.close(fig)


# ---------------- ACTUAL PINN P4 ----------------
# Train a real PINN on a tractable early-time window. Fourth-order long-time PINNs
# are expensive, so T_PINN=100 is used first. Increase only after this case works.
import torch
import torch.nn as nn
torch.manual_seed(7)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_PINN=100.0
mask=ts<=T_PINN+1e-12
ts_p=ts[mask]; Uref=U[mask]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.m=nn.Sequential(nn.Linear(2,64),nn.Tanh(),
                             nn.Linear(64,64),nn.Tanh(),
                             nn.Linear(64,64),nn.Tanh(),
                             nn.Linear(64,1))
    def forward(self,xv,tv):
        z=torch.cat([2*xv/L-1,2*tv/T_PINN-1],1)
        return self.m(z)
net=Net().to(device)
def D(y,z):
    return torch.autograd.grad(y,z,torch.ones_like(y),create_graph=True,retain_graph=True)[0]

NR=1800; NIC=384; NBC=192
xr=(L*torch.rand(NR,1,device=device)).requires_grad_(True)
tr=(T_PINN*torch.rand(NR,1,device=device)).requires_grad_(True)
xic=L*torch.rand(NIC,1,device=device)
tb=T_PINN*torch.rand(NBC,1,device=device)
xleft=torch.zeros_like(tb,requires_grad=True)
xright=(L*torch.ones_like(tb)).requires_grad_(True)

# Recreate the deterministic initial condition used by the spectral reference by interpolation.
# This guarantees the PINN and baseline use exactly the same initial field.
def ic_interp(xq):
    xp=torch.tensor(x,dtype=torch.float32,device=device)
    yp=torch.tensor(u0,dtype=torch.float32,device=device)
    # periodic Fourier evaluation from u0
    coeff=np.fft.rfft(u0)/N
    out=torch.zeros_like(xq)
    for n,c in enumerate(coeff):
        if n==0:
            out=out+float(c.real)
        elif n < len(coeff)-1:
            phase=2*np.pi*n*xq/L
            out=out+2*(float(c.real)*torch.cos(phase)-float(c.imag)*torch.sin(phase))
    return out

def pinn_losses():
    q=net(xr,tr)
    qt=D(q,tr); qx=D(q,xr); qxx=D(qx,xr); qxxx=D(qxx,xr); qxxxx=D(qxxx,xr)
    q2=q*q; q2xx=D(D(q2,xr),xr)
    q3=q*q*q; q3xx=D(D(q3,xr),xr)
    res=qt+mu*q+r*qxx+3*qxxxx+2*gamma*q2xx-(1/3)*q3xx
    lp=(res*res).mean()
    li=((net(xic,torch.zeros_like(xic))-ic_interp(xic))**2).mean()
    q0=net(xleft,tb); qL=net(xright,tb)
    qb=((q0-qL)**2).mean()+((D(q0,xleft)-D(qL,xright))**2).mean()
    return lp,li,qb

opt=torch.optim.Adam(net.parameters(),lr=1e-3)
hist=[]
print("Training ACTUAL PINN on t in [0,100]...",flush=True)
tic=time.perf_counter()
for it in range(3001):
    opt.zero_grad(); lp,li,lb=pinn_losses()
    total=lp+10*li+lb
    total.backward(); opt.step()
    if it%100==0:
        hist.append([it,lp.item(),li.item(),lb.item(),total.item()])
        print(f"{it:5d} pde={lp.item():.3e} ic={li.item():.3e} bc={lb.item():.3e} total={total.item():.3e}",flush=True)
train_time=time.perf_counter()-tic

net.eval(); Up=[]
with torch.no_grad():
    xx=torch.tensor(x[:,None],dtype=torch.float32,device=device)
    for tv in ts_p:
        tt=torch.full_like(xx,float(tv))
        Up.append(net(xx,tt).cpu().numpy().ravel())
Up=np.asarray(Up)
err=Up-Uref
rel=np.linalg.norm(err,axis=1)/np.maximum(np.linalg.norm(Uref,axis=1),1e-12)

def jc(v):
    th=.2*max(float(np.max(v)),1e-9); c=0
    for i in range(len(v)):
        if v[i]>th and v[i]>v[i-1] and v[i]>v[(i+1)%len(v)]: c+=1
    return c
jr=np.array([jc(v) for v in Uref]); jp=np.array([jc(v) for v in Up])

H=np.asarray(hist)
fig,ax=plt.subplots(figsize=(7,4.5))
for col,lab in [(1,"PDE"),(2,"IC"),(3,"BC"),(4,"total")]: ax.semilogy(H[:,0],H[:,col],label=lab)
ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.set_title("P2: Actual PINN losses"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"actual_pinn_loss.png",dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(7,4.5)); ax.plot(ts_p,rel)
ax.set_xlabel("time"); ax.set_ylabel("Relative L2 error"); ax.set_title("P3/P4: Actual PINN relative error")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT/"actual_pinn_error_time.png",dpi=180); plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(11,7.5)); ax1,ax2,ax3,ax4=axs.ravel()
lr,=ax1.plot([],[],lw=2,label="Spectral reference"); lpinn,=ax1.plot([],[],"--",lw=2,label="Actual PINN")
yymin=min(Uref.min(),Up.min()); yymax=max(Uref.max(),Up.max()); pd=.1*max(yymax-yymin,1e-3)
ax1.set(xlim=(0,L),ylim=(yymin-pd,yymax+pd),xlabel="eta",ylabel="U",title="Reference vs actual PINN"); ax1.legend(); ax1.grid(alpha=.25)
le,=ax2.plot([],[],lw=2); em=max(np.abs(err).max(),1e-6)
ax2.set(xlim=(0,L),ylim=(-1.1*em,1.1*em),xlabel="eta",ylabel="PINN error",title="P4: pointwise PINN error"); ax2.axhline(0,lw=.8); ax2.grid(alpha=.25)
specall=np.abs(np.fft.rfft(err,axis=1))**2/N**2
bars=ax3.bar(np.arange(1,16),np.full(15,1e-12)); ax3.set_yscale("log"); ax3.set_ylim(1e-12,max(1e-8,specall[:,1:16].max()*1.2))
ax3.set(xlabel="Fourier mode",ylabel="Error energy",title="P4: Fourier spectrum of PINN error")
lrel,=ax4.plot([],[],lw=2); mark,=ax4.plot([],[],"o")
ax4.set(xlim=(ts_p[0],ts_p[-1]),ylim=(0,max(rel.max()*1.1,1e-3)),xlabel="time",ylabel="Relative L2 error",title="P4: actual PINN error vs time"); ax4.grid(alpha=.25)
ttl=fig.suptitle(""); fig.tight_layout(rect=[0,0,1,.95])
def upd(i):
    lr.set_data(x,Uref[i]); lpinn.set_data(x,Up[i]); le.set_data(x,err[i])
    ee=np.abs(np.fft.rfft(err[i]))**2/N**2
    for b,v in zip(bars,ee[1:16]): b.set_height(max(v,1e-12))
    lrel.set_data(ts_p[:i+1],rel[:i+1]); mark.set_data([ts_p[i]],[rel[i]])
    ttl.set_text(f"ACTUAL PINN P4: gamma=1, mu=0.01, t={ts_p[i]:.1f}, relative L2={rel[i]:.3e}, jets ref/PINN={jr[i]}/{jp[i]}")
    return [lr,lpinn,le,lrel,mark,*bars]
ani=FuncAnimation(fig,upd,frames=len(ts_p),interval=100,repeat=True)
print("Saving actual PINN P4 GIF...",flush=True)
ani.save(OUT/"p4_actual_pinn_animation.gif",writer=PillowWriter(fps=10))
with open(OUT/"actual_pinn_results.txt","w") as f:
    f.write(f"device={device}\ntrain_time={train_time:.6f}\nfinal_relative_L2={rel[-1]:.8e}\n")
    f.write(f"final_jets_reference={jr[-1]}\nfinal_jets_PINN={jp[-1]}\n")
    f.write("These are actual trained-PINN results, not proxy data.\n")
print("Final actual PINN relative L2:",rel[-1])
print("Final jets reference/PINN:",jr[-1],jp[-1])
print("Opening live actual-PINN P4 animation. Close window to finish.")
plt.show()
