"""
OPTIFORGE TEAM 8 — P3: ERROR VS DIFFICULTY KNOB
Difficulty knob = time horizon T = 5, 20, 50, 90, 100.
Corrected cubic coefficient = 2/3.
"""
from pathlib import Path
import time, numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch, torch.nn as nn

SEED=7; L=100.; N=128; gamma=1.; mu=.01; r=2-gamma**2
TF=100.; NT=101; NR=1800; NIC=384; NBC=192; STEPS=3000
P3_TIMES=[5.,20.,50.,90.,100.]
OUT=Path(__file__).resolve().parent/"zonal_jet_outputs_P3"; OUT.mkdir(exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

x=np.linspace(0,L,N,endpoint=False); k=2*np.pi*np.fft.fftfreq(N,d=L/N)
rng=np.random.default_rng(SEED); phases={}; u0=np.zeros(N)
for n,a in [(3,.20),(4,.18),(5,.14),(6,.10)]:
    ph=rng.uniform(0,2*np.pi); phases[n]=ph
    u0+=a*np.cos(2*np.pi*n*x/L+ph)
u0-=u0.mean()

def rhs(t,u):
    uh=np.fft.fft(u)
    uxx=np.fft.ifft(-(k**2)*uh).real
    uxxxx=np.fft.ifft((k**4)*uh).real
    u2xx=np.fft.ifft(-(k**2)*np.fft.fft(u*u)).real
    u3xx=np.fft.ifft(-(k**2)*np.fft.fft(u*u*u)).real
    return -mu*u-r*uxx-3*uxxxx-2*gamma*u2xx+(2/3)*u3xx

ts=np.linspace(0,TF,NT); tic=time.perf_counter()
sol=solve_ivp(rhs,(0,TF),u0,t_eval=ts,method="BDF",rtol=2e-6,atol=1e-8,max_step=5)
if not sol.success: raise RuntimeError(sol.message)
Uref=sol.y.T; spectral_seconds=time.perf_counter()-tic

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.m=nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),
                             nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self,xx,tt):
        return self.m(torch.cat([2*xx/L-1,2*tt/TF-1],1))

def D(y,z):
    return torch.autograd.grad(y,z,torch.ones_like(y),create_graph=True,retain_graph=True)[0]

def ic(xx):
    v=torch.zeros_like(xx)
    for n,a in [(3,.20),(4,.18),(5,.14),(6,.10)]:
        v+=a*torch.cos(2*np.pi*n*xx/L+phases[n])
    return v

g=torch.Generator(device=device); g.manual_seed(SEED)
xr=(L*torch.rand(NR,1,device=device,generator=g)).requires_grad_(True)
tr=(TF*torch.rand(NR,1,device=device,generator=g)).requires_grad_(True)
xi=L*torch.rand(NIC,1,device=device,generator=g)
tb=TF*torch.rand(NBC,1,device=device,generator=g)
xl=torch.zeros_like(tb,requires_grad=True); xR=(L*torch.ones_like(tb)).requires_grad_(True)

torch.manual_seed(SEED); net=Net().to(device)
opt=torch.optim.Adam(net.parameters(),lr=1e-3); history=[]
tic=time.perf_counter()
for it in range(STEPS+1):
    opt.zero_grad()
    q=net(xr,tr); qt=D(q,tr); qx=D(q,xr); qxx=D(qx,xr); qxxx=D(qxx,xr); qxxxx=D(qxxx,xr)
    q2xx=D(D(q*q,xr),xr); q3xx=D(D(q*q*q,xr),xr)
    res=qt+mu*q+r*qxx+3*qxxxx+2*gamma*q2xx-(2/3)*q3xx
    lp=(res**2).mean(); li=((net(xi,torch.zeros_like(xi))-ic(xi))**2).mean()
    ql=net(xl,tb); qr=net(xR,tb)
    lb=((ql-qr)**2).mean()+((D(ql,xl)-D(qr,xR))**2).mean()
    total=lp+10*li+lb; total.backward(); opt.step()
    if it%100==0:
        history.append([it,lp.item(),li.item(),lb.item(),total.item()])
        print(f"{it:5d} PDE={lp.item():.3e} IC={li.item():.3e} BC={lb.item():.3e} total={total.item():.3e}")
train_seconds=time.perf_counter()-tic

xx=torch.tensor(x[:,None],dtype=torch.float32,device=device); Up=[]
with torch.no_grad():
    for tv in ts: Up.append(net(xx,torch.full_like(xx,float(tv))).cpu().numpy().ravel())
Up=np.asarray(Up)
rel=np.linalg.norm(Up-Uref,axis=1)/np.maximum(np.linalg.norm(Uref,axis=1),1e-12)

def jets(v):
    th=.2*max(float(np.max(v)),1e-9)
    return sum(v[i]>th and v[i]>v[i-1] and v[i]>v[(i+1)%len(v)] for i in range(len(v)))

def energy(A):
    return .5*np.trapezoid(A*A,x=x,axis=1)

Jr=np.array([jets(v) for v in Uref]); Jp=np.array([jets(v) for v in Up])
Er=energy(Uref); Ep=energy(Up)
idx=np.array([int(np.argmin(np.abs(ts-t))) for t in P3_TIMES])
Tp=ts[idx]; L2p=rel[idx]; ratio=Ep[idx]/np.maximum(Er[idx],1e-12)

print("\nP3 — ERROR VS TIME HORIZON")
print("T      Relative_L2   Jets(ref/PINN)   Energy_ratio")
for i in range(len(idx)):
    print(f"{Tp[i]:5.1f}  {L2p[i]:11.6f}      {Jr[idx[i]]}/{Jp[idx[i]]}             {ratio[i]:.6f}")

fig,ax=plt.subplots(figsize=(8,5))
ax.plot(ts,rel,lw=2,label="Standard PINN")
ax.scatter(Tp,L2p,s=55,zorder=3,label="P3 points")
for T,e in zip(Tp,L2p): ax.annotate(f"{e:.2f}",(T,e),xytext=(5,6),textcoords="offset points",fontsize=8)
ax.axhline(1,ls=":",label="Order-one error")
ax.set(xlabel="Difficulty knob: time horizon T",ylabel="Relative L2 error",title="P3 — PINN error vs time horizon")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"P3_error_vs_time_horizon.png",dpi=200)

fig,axs=plt.subplots(1,2,figsize=(12,4.6))
axs[0].plot(ts,Jr,label="Reference"); axs[0].plot(ts,Jp,"--",label="PINN")
axs[0].set(xlabel="time",ylabel="Detected eastward jets",title="Jet count vs difficulty"); axs[0].grid(alpha=.3); axs[0].legend()
axs[1].plot(ts,Ep/np.maximum(Er,1e-12)); axs[1].axhline(1,ls=":")
axs[1].set(xlabel="time",ylabel="E_PINN / E_reference",title="Energy recovery vs difficulty"); axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT/"P3_supporting_physics.png",dpi=200)

fig,axs=plt.subplots(len(idx),1,figsize=(9,13),sharex=True)
for a,j in zip(axs,idx):
    a.plot(x,Uref[j],label="Reference"); a.plot(x,Up[j],"--",label="PINN")
    a.set_ylabel("U"); a.set_title(f"T={ts[j]:.0f}, relative L2={rel[j]:.3f}"); a.grid(alpha=.25)
axs[0].legend(); axs[-1].set_xlabel("eta")
fig.tight_layout(); fig.savefig(OUT/"P3_field_snapshots.png",dpi=190)

np.savez(OUT/"P3_results.npz",x=x,ts=ts,Uref=Uref,Upinn=Up,relative_L2=rel,
         p3_time=Tp,p3_error=L2p,jets_ref=Jr,jets_pinn=Jp,energy_ref=Er,energy_pinn=Ep)
with open(OUT/"P3_results.txt","w",encoding="utf-8") as f:
    f.write("P3 — ERROR VS DIFFICULTY KNOB\nDifficulty knob = time horizon T\nCorrected cubic coefficient = 2/3\n")
    f.write(f"spectral_seconds={spectral_seconds:.8f}\npinn_training_seconds={train_seconds:.8f}\n")
    f.write("T,relative_L2,jets_ref,jets_PINN,energy_ratio\n")
    for i,j in enumerate(idx):
        f.write(f"{Tp[i]:.1f},{L2p[i]:.8f},{Jr[j]},{Jp[j]},{ratio[i]:.8f}\n")
print("\nSaved P3 outputs to:",OUT)
plt.show()
