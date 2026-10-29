# CORRECTED PAPER COEFFICIENT: cubic term uses 2/3, not 1/3.
# Spectral RHS and PINN residual are kept consistent.

import numpy as np
import time
tic = time.perf_counter()
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib
from scipy.integrate import solve_ivp
from pathlib import Path
import platform

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
        return -mu*u-r*uxx-3*uxxxx-2*gamma*u2xx+(2/3)*u3xx
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

def hardware_summary():
    info = {
        "operating_system": platform.platform(),
        "processor": platform.processor() or "Not reported by OS",
        "pytorch_device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None - CPU training",
        "torch_version": torch.__version__,
    }
    return info

HW = hardware_summary()
print("\n========== HARDWARE ==========")
for key, value in HW.items():
    print(f"{key}: {value}")

T_PINN=100.0
mask=ts<=T_PINN+1e-12
ts_p=ts[mask]; Uref=U[mask]

# P5: time one conventional forward solve over the SAME PINN horizon.
# This avoids comparing a t=1200 spectral sweep against a t=100 PINN experiment.
p5_ts = np.linspace(0, T_PINN, len(ts_p))
spectral_start = time.perf_counter()
p5_sol = solve_ivp(
    rhs(mu), (0, T_PINN), u0, t_eval=p5_ts,
    method="BDF", rtol=2e-6, atol=1e-8, max_step=5.0
)
spectral_time = time.perf_counter() - spectral_start
if not p5_sol.success:
    raise RuntimeError(p5_sol.message)
print(f"P5 spectral solve time (mu={mu}, t=0..{T_PINN:g}): {spectral_time:.6f} s")


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


# =====================================================================
# OPTIFORGE TEAM 8 — PATH B + PATH C BEFORE/AFTER EXPERIMENT
# Standard PINN vs improved Fourier-feature PINN on the SAME reference.
# The improved method targets periodic/multiscale representation.
# =====================================================================

# Training settings
NR=1800; NIC=384; NBC=192
STEPS=3000
LR=1e-3
FOURIER_MODES=8

def make_training_points(seed):
    g=torch.Generator(device=device)
    g.manual_seed(seed)
    xr=(L*torch.rand(NR,1,device=device,generator=g)).requires_grad_(True)
    tr=(T_PINN*torch.rand(NR,1,device=device,generator=g)).requires_grad_(True)
    xic=L*torch.rand(NIC,1,device=device,generator=g)
    tb=T_PINN*torch.rand(NBC,1,device=device,generator=g)
    xleft=torch.zeros_like(tb,requires_grad=True)
    xright=(L*torch.ones_like(tb)).requires_grad_(True)
    return xr,tr,xic,tb,xleft,xright

class StandardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.m=nn.Sequential(
            nn.Linear(2,64),nn.Tanh(),
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,1))
    def forward(self,xv,tv):
        z=torch.cat([2*xv/L-1,2*tv/T_PINN-1],1)
        return self.m(z)

class FourierNet(nn.Module):
    """Path C model: periodic Fourier spatial encoding + normalized time."""
    def __init__(self,K=FOURIER_MODES):
        super().__init__()
        self.K=K
        nin=1+2*K
        self.m=nn.Sequential(
            nn.Linear(nin,64),nn.Tanh(),
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,64),nn.Tanh(),
            nn.Linear(64,1))
    def forward(self,xv,tv):
        feats=[2*tv/T_PINN-1]
        for m in range(1,self.K+1):
            phase=2*np.pi*m*xv/L
            feats.extend([torch.sin(phase),torch.cos(phase)])
        return self.m(torch.cat(feats,1))

def model_losses(model,pts):
    xr,tr,xic,tb,xleft,xright=pts
    q=model(xr,tr)
    qt=D(q,tr)
    qx=D(q,xr); qxx=D(qx,xr); qxxx=D(qxx,xr); qxxxx=D(qxxx,xr)
    q2=q*q; q2xx=D(D(q2,xr),xr)
    q3=q*q*q; q3xx=D(D(q3,xr),xr)
    res=qt+mu*q+r*qxx+3*qxxxx+2*gamma*q2xx-(2/3)*q3xx
    lp=(res*res).mean()
    li=((model(xic,torch.zeros_like(xic))-ic_interp(xic))**2).mean()
    q0=model(xleft,tb); qL=model(xright,tb)
    # value + slope periodicity
    lb=((q0-qL)**2).mean()+((D(q0,xleft)-D(qL,xright))**2).mean()
    return lp,li,lb

def train_model(model,label,seed=7):
    pts=make_training_points(seed)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    hist=[]
    tic=time.perf_counter()
    print(f"\nTraining {label}...",flush=True)
    for it in range(STEPS+1):
        opt.zero_grad()
        lp,li,lb=model_losses(model,pts)
        total=lp+10*li+lb
        total.backward()
        opt.step()
        if it%100==0:
            hist.append([it,lp.item(),li.item(),lb.item(),total.item()])
            print(f"{label:18s} {it:5d} pde={lp.item():.3e} ic={li.item():.3e} bc={lb.item():.3e} total={total.item():.3e}",flush=True)
    return np.asarray(hist),time.perf_counter()-tic

def evaluate(model):
    model.eval(); pred=[]
    with torch.no_grad():
        xx=torch.tensor(x[:,None],dtype=torch.float32,device=device)
        for tv in ts_p:
            tt=torch.full_like(xx,float(tv))
            pred.append(model(xx,tt).cpu().numpy().ravel())
    pred=np.asarray(pred)
    error=pred-Uref
    relative=np.linalg.norm(error,axis=1)/np.maximum(np.linalg.norm(Uref,axis=1),1e-12)
    return pred,error,relative

def jc(v):
    th=.2*max(float(np.max(v)),1e-9); c=0
    for i in range(len(v)):
        if v[i]>th and v[i]>v[i-1] and v[i]>v[(i+1)%len(v)]:
            c+=1
    return c

# Identical initialization seed for a fair comparison.
torch.manual_seed(7)
standard=StandardNet().to(device)
H_std,time_std=train_model(standard,"Standard PINN",7)
U_std,E_std,R_std=evaluate(standard)

torch.manual_seed(7)
improved=FourierNet().to(device)
H_imp,time_imp=train_model(improved,"Path C Fourier PINN",7)
U_imp,E_imp,R_imp=evaluate(improved)

def timed_inference(model, repeats=10):
    """Mean wall time to evaluate all x points for every saved time in ts_p."""
    model.eval()
    xx=torch.tensor(x[:,None],dtype=torch.float32,device=device)

    # Warm-up, especially important on GPU.
    with torch.no_grad():
        tt=torch.full_like(xx,float(ts_p[-1]))
        _=model(xx,tt)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times=[]
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.no_grad():
            for tv in ts_p:
                tt=torch.full_like(xx,float(tv))
                _=model(xx,tt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    return float(np.mean(times)), float(np.std(times))

standard_inference_time, standard_inference_std = timed_inference(standard)
improved_inference_time, improved_inference_std = timed_inference(improved)


J_ref=np.array([jc(v) for v in Uref])
J_std=np.array([jc(v) for v in U_std])
J_imp=np.array([jc(v) for v in U_imp])

print("\n========== PATH C BEFORE / AFTER ==========")
print(f"Final relative L2: standard={R_std[-1]:.6f}, improved={R_imp[-1]:.6f}")
print(f"Final jets: reference={J_ref[-1]}, standard={J_std[-1]}, improved={J_imp[-1]}")
print(f"Training seconds: standard={time_std:.2f}, improved={time_imp:.2f}")
print("\n========== P5 COMPUTATIONAL COST ==========")
print(f"Spectral solve (same t=0..100 horizon): {spectral_time:.6f} s")
print(f"Standard PINN training:               {time_std:.6f} s")
print(f"Fourier PINN training:                {time_imp:.6f} s")
print(f"Standard PINN inference:              {standard_inference_time:.6f} +/- {standard_inference_std:.6f} s")
print(f"Fourier PINN inference:               {improved_inference_time:.6f} +/- {improved_inference_std:.6f} s")
print(f"Standard final relative L2:           {R_std[-1]:.6f}")
print(f"Fourier final relative L2:            {R_imp[-1]:.6f}")
print(f"Jets reference/standard/Fourier:      {J_ref[-1]}/{J_std[-1]}/{J_imp[-1]}")


# ---------- C1: final solution comparison ----------
fig,ax=plt.subplots(figsize=(9,5))
ax.plot(x,Uref[-1],lw=2,label="Spectral reference")
ax.plot(x,U_std[-1],"--",lw=2,label="Standard PINN")
ax.plot(x,U_imp[-1],"-.",lw=2,label="Improved Fourier PINN")
ax.set_xlabel("eta"); ax.set_ylabel("U")
ax.set_title(f"Path C: final solution at t={ts_p[-1]:.0f}")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"pathC_final_solution_before_after.png",dpi=180); plt.close(fig)

# ---------- C2: relative L2 error comparison ----------
fig,ax=plt.subplots(figsize=(8,4.8))
ax.plot(ts_p,R_std,label="Standard PINN")
ax.plot(ts_p,R_imp,label="Improved Fourier PINN")
ax.set_xlabel("time"); ax.set_ylabel("Relative L2 error")
ax.set_title("Path C: relative error before vs after")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"pathC_L2_before_after.png",dpi=180); plt.close(fig)

# ---------- C3: jet-count comparison ----------
fig,ax=plt.subplots(figsize=(8,4.8))
ax.plot(ts_p,J_ref,lw=2,label="Reference")
ax.plot(ts_p,J_std,"--",label="Standard PINN")
ax.plot(ts_p,J_imp,"-.",label="Improved Fourier PINN")
ax.set_xlabel("time"); ax.set_ylabel("Detected eastward jets")
ax.set_title("Path C: physical jet-count comparison")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"pathC_jet_count_before_after.png",dpi=180); plt.close(fig)

# ---------- C4: Fourier error at final time ----------
S_std=np.abs(np.fft.rfft(E_std[-1]))**2/N**2
S_imp=np.abs(np.fft.rfft(E_imp[-1]))**2/N**2
modes=np.arange(1,16)
fig,ax=plt.subplots(figsize=(8,4.8))
ax.semilogy(modes,S_std[1:16]+1e-16,"o-",label="Standard PINN error")
ax.semilogy(modes,S_imp[1:16]+1e-16,"s-",label="Improved PINN error")
ax.set_xlabel("Fourier mode"); ax.set_ylabel("Error energy")
ax.set_title("Path C: Fourier error before vs after")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"pathC_fourier_error_before_after.png",dpi=180); plt.close(fig)

# ---------- C5: total training loss ----------
fig,ax=plt.subplots(figsize=(8,4.8))
ax.semilogy(H_std[:,0],H_std[:,4],label="Standard total loss")
ax.semilogy(H_imp[:,0],H_imp[:,4],label="Improved total loss")
ax.set_xlabel("iteration"); ax.set_ylabel("Total loss")
ax.set_title("Path C: optimization history before vs after")
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT/"pathC_loss_before_after.png",dpi=180); plt.close(fig)

# ---------- C6: compact summary figure ----------
fig,axs=plt.subplots(2,2,figsize=(11,7.5))
ax1,ax2,ax3,ax4=axs.ravel()
ax1.plot(x,Uref[-1],label="Reference"); ax1.plot(x,U_std[-1],"--",label="Standard"); ax1.plot(x,U_imp[-1],"-.",label="Improved")
ax1.set_title("Final solution"); ax1.set_xlabel("eta"); ax1.set_ylabel("U"); ax1.grid(alpha=.25); ax1.legend()
ax2.plot(ts_p,R_std,label="Standard"); ax2.plot(ts_p,R_imp,label="Improved")
ax2.set_title("Relative L2 error"); ax2.set_xlabel("time"); ax2.grid(alpha=.25); ax2.legend()
ax3.semilogy(modes,S_std[1:16]+1e-16,"o-",label="Standard"); ax3.semilogy(modes,S_imp[1:16]+1e-16,"s-",label="Improved")
ax3.set_title("Final Fourier error"); ax3.set_xlabel("mode"); ax3.set_ylabel("error energy"); ax3.grid(alpha=.25); ax3.legend()
labels=["Reference","Standard","Improved"]; vals=[J_ref[-1],J_std[-1],J_imp[-1]]
ax4.bar(labels,vals); ax4.set_title("Jets at final time"); ax4.set_ylabel("jet count"); ax4.grid(axis="y",alpha=.25)
fig.suptitle("Optiforge Team 8 — Path C before/after evidence")
fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(OUT/"pathC_summary.png",dpi=180); plt.close(fig)

# ---------- Live animation: reference + before + after ----------
fig,(axv,axe)=plt.subplots(1,2,figsize=(11,4.5))
lref,=axv.plot([],[],lw=2,label="Reference")
lstd,=axv.plot([],[],"--",lw=2,label="Standard PINN")
limp,=axv.plot([],[],"-.",lw=2,label="Improved PINN")
vmin=min(Uref.min(),U_std.min(),U_imp.min()); vmax=max(Uref.max(),U_std.max(),U_imp.max()); pad=.1*max(vmax-vmin,1e-3)
axv.set(xlim=(0,L),ylim=(vmin-pad,vmax+pad),xlabel="eta",ylabel="U",title="Zonal velocity")
axv.grid(alpha=.25); axv.legend()
erst,=axe.plot([],[],label="Standard error")
erimp,=axe.plot([],[],label="Improved error")
axe.set(xlim=(ts_p[0],ts_p[-1]),ylim=(0,max(R_std.max(),R_imp.max())*1.1),xlabel="time",ylabel="Relative L2 error",title="Error history")
axe.grid(alpha=.25); axe.legend()
ttl=fig.suptitle("")
def animate(i):
    lref.set_data(x,Uref[i]); lstd.set_data(x,U_std[i]); limp.set_data(x,U_imp[i])
    erst.set_data(ts_p[:i+1],R_std[:i+1]); erimp.set_data(ts_p[:i+1],R_imp[:i+1])
    ttl.set_text(f"Path C t={ts_p[i]:.1f} | L2 std/imp={R_std[i]:.3f}/{R_imp[i]:.3f} | jets ref/std/imp={J_ref[i]}/{J_std[i]}/{J_imp[i]}")
    return lref,lstd,limp,erst,erimp
ani=FuncAnimation(fig,animate,frames=len(ts_p),interval=110,repeat=True)
print("Saving Path C comparison GIF...",flush=True)
ani.save(OUT/"pathC_before_after_animation.gif",writer=PillowWriter(fps=10))

# ---------- machine-readable results ----------
with open(OUT/"pathC_results.txt","w",encoding="utf-8") as f:
    f.write("OPTIFORGE TEAM 8 — PATH C RESULTS\n")
    f.write(f"device={device}\n")
    f.write(f"standard_train_seconds={time_std:.6f}\n")
    f.write(f"improved_train_seconds={time_imp:.6f}\n")
    f.write(f"standard_final_relative_L2={R_std[-1]:.8e}\n")
    f.write(f"improved_final_relative_L2={R_imp[-1]:.8e}\n")
    f.write(f"reference_final_jets={J_ref[-1]}\n")
    f.write(f"standard_final_jets={J_std[-1]}\n")
    f.write(f"improved_final_jets={J_imp[-1]}\n")
    f.write(f"standard_final_pde_loss={H_std[-1,1]:.8e}\n")
    f.write(f"improved_final_pde_loss={H_imp[-1,1]:.8e}\n")
    f.write(f"standard_final_total_loss={H_std[-1,4]:.8e}\n")
    f.write(f"improved_final_total_loss={H_imp[-1,4]:.8e}\n")
    f.write(f"spectral_same_horizon_seconds={spectral_time:.8f}\n")
    f.write(f"standard_inference_seconds_mean={standard_inference_time:.8f}\n")
    f.write(f"standard_inference_seconds_std={standard_inference_std:.8f}\n")
    f.write(f"improved_inference_seconds_mean={improved_inference_time:.8f}\n")
    f.write(f"improved_inference_seconds_std={improved_inference_std:.8f}\n")
    for key,value in HW.items():
        f.write(f"hardware_{key}={value}\n")

with open(OUT/"p5_computational_cost.txt","w",encoding="utf-8") as f:
    f.write("OPTIFORGE TEAM 8 - P5 COMPUTATIONAL COST\n")
    f.write("========================================\n")
    f.write("All methods timed on the same machine.\n")
    f.write(f"Reference horizon: t=0..{T_PINN:g}, mu={mu}\n\n")
    f.write("HARDWARE\n")
    for key,value in HW.items():
        f.write(f"{key}: {value}\n")
    f.write("\nTIMING\n")
    f.write(f"Spectral forward solve: {spectral_time:.8f} s\n")
    f.write(f"Standard PINN training: {time_std:.8f} s\n")
    f.write(f"Fourier PINN training: {time_imp:.8f} s\n")
    f.write(f"Standard PINN inference (all saved frames): {standard_inference_time:.8f} +/- {standard_inference_std:.8f} s\n")
    f.write(f"Fourier PINN inference (all saved frames): {improved_inference_time:.8f} +/- {improved_inference_std:.8f} s\n")
    f.write("\nACCURACY / PHYSICS\n")
    f.write(f"Standard final relative L2: {R_std[-1]:.8f}\n")
    f.write(f"Fourier final relative L2: {R_imp[-1]:.8f}\n")
    f.write(f"Reference final jets: {J_ref[-1]}\n")
    f.write(f"Standard final jets: {J_std[-1]}\n")
    f.write(f"Fourier final jets: {J_imp[-1]}\n")


print("Saved Path C figures/results in:",OUT)
print("Opening live before/after animation. Close the window to finish.")
plt.show()
