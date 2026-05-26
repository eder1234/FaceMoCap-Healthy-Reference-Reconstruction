#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_dynamic_motion_projector_v3.py

Train one masked dynamic healthy-motion projector per movement from healthy-train displacement trajectories.

This is the dynamic branch of AE-v3. The final healthy twin is assembled later as:
  predicted_healthy_neutral_face + predicted_healthy_displacement(t)

Input per movement from build_healthy_twin_dataset_v3.py:
  M*/X_train_healthy.npy  (S,100,105,3)
  M*/M_train_healthy.npy  (S,100,105)

Output:
  dynamic_motion_projector_v3/M1/model.pt, train_history.csv, config.json, normalization.json
  ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires PyTorch.") from exc


def ensure_dir(p: Path) -> None: p.mkdir(parents=True, exist_ok=True)
def write_json(p: Path, obj: object) -> None: p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_norm(X: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = M.astype(bool); mean=[]; std=[]
    for c in range(3):
        vals = X[..., c][valid]; vals = vals[np.isfinite(vals)]
        if len(vals)==0: mean.append(0.0); std.append(1.0)
        else:
            mean.append(float(vals.mean())); s=float(vals.std()); std.append(s if s>1e-8 else 1.0)
    return np.asarray(mean,dtype=np.float32), np.asarray(std,dtype=np.float32)

def normalize(X: np.ndarray, M: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (X.astype(np.float32)-mean.reshape(1,1,1,3))/std.reshape(1,1,1,3)
    Z = np.where(M[...,None].astype(bool), Z, 0.0)
    return np.where(np.isfinite(Z), Z, 0.0).astype(np.float32)

class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray):
        self.X=torch.from_numpy(X.astype(np.float32)); self.M=torch.from_numpy(M.astype(np.float32))
    def __len__(self): return int(self.X.shape[0])
    def __getitem__(self, idx): return self.X[idx], self.M[idx]

class DynamicMotionProjector(nn.Module):
    def __init__(self, n_markers:int=105, n_frames:int=100, hidden:int=160, latent:int=40, dropout:float=0.20):
        super().__init__(); self.n_markers=n_markers; self.n_frames=n_frames; self.in_dim=n_markers*4; self.out_dim=n_markers*3
        self.enc1=nn.Sequential(nn.Conv1d(self.in_dim,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU(),nn.Dropout(dropout))
        self.enc2=nn.Sequential(nn.Conv1d(hidden,hidden,5,stride=2,padding=2),nn.BatchNorm1d(hidden),nn.GELU(),nn.Dropout(dropout))
        self.enc3=nn.Sequential(nn.Conv1d(hidden,latent,5,stride=2,padding=2),nn.BatchNorm1d(latent),nn.GELU(),nn.Dropout(dropout))
        self.dec1=nn.Sequential(nn.Conv1d(latent,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU())
        self.dec2=nn.Sequential(nn.Conv1d(hidden,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU())
        self.out=nn.Conv1d(hidden,self.out_dim,3,padding=1)
    def forward(self,x:torch.Tensor,m:torch.Tensor)->torch.Tensor:
        B,T,N,_=x.shape
        y=torch.cat([x,m.unsqueeze(-1)],dim=-1).reshape(B,T,N*4).permute(0,2,1)
        y=self.enc1(y); y=self.enc2(y); y=self.enc3(y)
        y=F.interpolate(y,size=max(1,math.ceil(T/2)),mode='linear',align_corners=False); y=self.dec1(y)
        y=F.interpolate(y,size=T,mode='linear',align_corners=False); y=self.dec2(y); y=self.out(y)
        return y.permute(0,2,1).reshape(B,T,N,3)


def corrupt(x: torch.Tensor, m: torch.Tensor, args: argparse.Namespace):
    B,T,N,_=x.shape; keep=m>0.5
    keep = keep & (torch.rand(B,T,N,device=x.device) > args.point_mask_prob)
    marker_drop=torch.rand(B,N,device=x.device)<args.marker_mask_prob
    keep = keep & (~marker_drop[:,None,:])
    # temporal block masks
    for b in range(B):
        for _ in range(args.n_time_blocks):
            if torch.rand((),device=x.device)<args.time_block_prob:
                L=int(torch.randint(args.time_block_min,args.time_block_max+1,(1,),device=x.device).item())
                s=int(torch.randint(0,max(1,T-L+1),(1,),device=x.device).item())
                keep[b,s:s+L,:]=False
    xin=x + torch.randn_like(x)*args.noise_std
    xin=torch.where(keep.unsqueeze(-1),xin,torch.zeros_like(xin))
    return xin, keep.float()

def masked_mse(pred,target,mask):
    valid=mask.unsqueeze(-1); denom=valid.sum()*target.shape[-1]+1e-8
    return (((pred-target)**2)*valid).sum()/denom

def velocity_loss(pred,target,mask):
    if pred.shape[1]<2: return pred.sum()*0
    mp=mask[:,1:]*mask[:,:-1]
    return masked_mse(pred[:,1:]-pred[:,:-1], target[:,1:]-target[:,:-1], mp)

def acceleration_loss(pred,target,mask):
    if pred.shape[1]<3: return pred.sum()*0
    mp=mask[:,2:]*mask[:,1:-1]*mask[:,:-2]
    ap=pred[:,2:]-2*pred[:,1:-1]+pred[:,:-2]
    at=target[:,2:]-2*target[:,1:-1]+target[:,:-2]
    return masked_mse(ap,at,mp)


def train_one(movement:str,args:argparse.Namespace,device:torch.device):
    mov_dir=Path(args.dataset_dir)/movement; out_dir=Path(args.out_dir)/movement; ensure_dir(out_dir)
    X=np.load(mov_dir/'X_train_healthy.npy').astype(np.float32); M=np.load(mov_dir/'M_train_healthy.npy').astype(np.uint8)
    if X.shape[0]<2:
        print(f"[SKIP] {movement}: need at least 2 samples"); return
    mean,std=compute_norm(X,M); Z=normalize(X,M,mean,std)
    ds=TrajectoryDataset(Z,M); n_val=max(1,int(round(len(ds)*args.val_fraction))); n_train=max(1,len(ds)-n_val)
    tr_ds,va_ds=random_split(ds,[n_train,n_val],generator=torch.Generator().manual_seed(args.seed))
    tr_loader=DataLoader(tr_ds,batch_size=args.batch_size,shuffle=True); va_loader=DataLoader(va_ds,batch_size=args.batch_size,shuffle=False)
    model=DynamicMotionProjector(n_markers=X.shape[2],n_frames=X.shape[1],hidden=args.hidden,latent=args.latent,dropout=args.dropout).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best=float('inf'); best_state=None; bad=0; hist=[]
    for epoch in range(1,args.epochs+1):
        model.train(); trs=[]
        for x,m in tr_loader:
            x=x.to(device); m=m.to(device); xin,minp=corrupt(x,m,args); pred=model(xin,minp)
            loss=masked_mse(pred,x,m)+args.lambda_velocity*velocity_loss(pred,x,m)+args.lambda_acceleration*acceleration_loss(pred,x,m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); trs.append(float(loss.detach().cpu()))
        model.eval(); vals=[]
        with torch.no_grad():
            for x,m in va_loader:
                x=x.to(device); m=m.to(device); pred=model(x,m)
                loss=masked_mse(pred,x,m)+args.lambda_velocity*velocity_loss(pred,x,m)+args.lambda_acceleration*acceleration_loss(pred,x,m)
                vals.append(float(loss.detach().cpu()))
        row={"epoch":epoch,"train_loss":float(np.mean(trs)),"val_loss":float(np.mean(vals))}; hist.append(row)
        print(f"{movement} epoch {epoch:04d} train={row['train_loss']:.6f} val={row['val_loss']:.6f}")
        if row['val_loss']<best-1e-6:
            best=row['val_loss']; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=args.patience:
            print(f"{movement}: early stopping at {epoch}"); break
    if best_state is not None: model.load_state_dict(best_state)
    pd.DataFrame(hist).to_csv(out_dir/'train_history.csv',index=False)
    write_json(out_dir/'normalization.json',{"mean":mean.tolist(),"std":std.tolist()})
    write_json(out_dir/'config.json',vars(args))
    torch.save({"model_state_dict":model.state_dict(),"n_markers":int(X.shape[2]),"n_frames":int(X.shape[1]),"hidden":args.hidden,"latent":args.latent,"dropout":args.dropout,"mean":mean,"std":std,"config":vars(args)},out_dir/'model.pt')


def parse_args():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset_dir',required=True); ap.add_argument('--out_dir',required=True)
    ap.add_argument('--movements',nargs='+',default=['M1','M2','M3','M4','M5']); ap.add_argument('--epochs',type=int,default=300); ap.add_argument('--batch_size',type=int,default=8); ap.add_argument('--patience',type=int,default=40)
    ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=1e-4); ap.add_argument('--hidden',type=int,default=160); ap.add_argument('--latent',type=int,default=40); ap.add_argument('--dropout',type=float,default=0.20)
    ap.add_argument('--point_mask_prob',type=float,default=0.10); ap.add_argument('--marker_mask_prob',type=float,default=0.35); ap.add_argument('--time_block_prob',type=float,default=0.50); ap.add_argument('--n_time_blocks',type=int,default=2); ap.add_argument('--time_block_min',type=int,default=8); ap.add_argument('--time_block_max',type=int,default=25); ap.add_argument('--noise_std',type=float,default=0.02)
    ap.add_argument('--lambda_velocity',type=float,default=0.20); ap.add_argument('--lambda_acceleration',type=float,default=0.05); ap.add_argument('--val_fraction',type=float,default=0.20); ap.add_argument('--seed',type=int,default=17); ap.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    return ap.parse_args()

def main():
    args=parse_args(); set_seed(args.seed); ensure_dir(Path(args.out_dir)); write_json(Path(args.out_dir)/'train_config.json',vars(args))
    device=torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) else ('cpu' if args.device=='auto' else args.device)); print(f"Using device: {device}")
    for movement in [m.upper() for m in args.movements]: train_one(movement,args,device)
    print(f"[OK] Wrote dynamic motion projectors: {args.out_dir}")

if __name__=='__main__': main()
