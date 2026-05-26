#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_healthy_twin_v3.py

Predict subject-specific healthy twins using AE-v3:
  static branch:  observed neutral face -> predicted healthy neutral face
  dynamic branch: observed displacement -> predicted healthy displacement

Final healthy twin:
  X_twin_face(t) = N_pred_healthy + D_pred_healthy(t)

Also writes observed face:
  X_obs_face(t) = N_obs + D_obs(t)

Outputs
-------
healthy_twin_v3_eval/
  twin_sample_scores.csv
  twin_marker_scores.csv
  twin_region_scores.csv
  twin_region_side_scores.csv
  M1/twin_eval_healthy.npz
  M1/twin_pathological.npz
  ...
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:
    raise RuntimeError("This script requires PyTorch.") from exc


def ensure_dir(p: Path)->None: p.mkdir(parents=True,exist_ok=True)
def write_json(p: Path,obj:object)->None: p.write_text(json.dumps(obj,indent=2,ensure_ascii=False))

def normalize_static(N,M,mean,std):
    Z=(N.astype(np.float32)-mean.reshape(1,1,3))/std.reshape(1,1,3); Z=np.where(M[...,None].astype(bool),Z,0.0); return np.where(np.isfinite(Z),Z,0.0).astype(np.float32)
def denorm_static(Z,mean,std): return Z.astype(np.float32)*std.reshape(1,1,3)+mean.reshape(1,1,3)
def normalize_dyn(X,M,mean,std):
    Z=(X.astype(np.float32)-mean.reshape(1,1,1,3))/std.reshape(1,1,1,3); Z=np.where(M[...,None].astype(bool),Z,0.0); return np.where(np.isfinite(Z),Z,0.0).astype(np.float32)
def denorm_dyn(Z,mean,std): return Z.astype(np.float32)*std.reshape(1,1,1,3)+mean.reshape(1,1,1,3)

class StaticNeutralAE(nn.Module):
    def __init__(self,n_markers=105,hidden=256,latent=48,dropout=0.20):
        super().__init__(); self.n_markers=n_markers
        self.net=nn.Sequential(nn.Linear(n_markers*4,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,latent),nn.GELU(),nn.Linear(latent,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,n_markers*3))
    def forward(self,n,m):
        B,N,_=n.shape; x=torch.cat([n,m.unsqueeze(-1)],dim=-1).reshape(B,N*4); return self.net(x).reshape(B,N,3)

class DynamicMotionProjector(nn.Module):
    def __init__(self,n_markers=105,n_frames=100,hidden=160,latent=40,dropout=0.20):
        super().__init__(); self.n_markers=n_markers; self.n_frames=n_frames; self.in_dim=n_markers*4; self.out_dim=n_markers*3
        self.enc1=nn.Sequential(nn.Conv1d(self.in_dim,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU(),nn.Dropout(dropout))
        self.enc2=nn.Sequential(nn.Conv1d(hidden,hidden,5,stride=2,padding=2),nn.BatchNorm1d(hidden),nn.GELU(),nn.Dropout(dropout))
        self.enc3=nn.Sequential(nn.Conv1d(hidden,latent,5,stride=2,padding=2),nn.BatchNorm1d(latent),nn.GELU(),nn.Dropout(dropout))
        self.dec1=nn.Sequential(nn.Conv1d(latent,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU())
        self.dec2=nn.Sequential(nn.Conv1d(hidden,hidden,5,padding=2),nn.BatchNorm1d(hidden),nn.GELU())
        self.out=nn.Conv1d(hidden,self.out_dim,3,padding=1)
    def forward(self,x,m):
        B,T,N,_=x.shape; y=torch.cat([x,m.unsqueeze(-1)],dim=-1).reshape(B,T,N*4).permute(0,2,1)
        y=self.enc1(y); y=self.enc2(y); y=self.enc3(y); y=F.interpolate(y,size=max(1,math.ceil(T/2)),mode='linear',align_corners=False); y=self.dec1(y); y=F.interpolate(y,size=T,mode='linear',align_corners=False); y=self.dec2(y); y=self.out(y)
        return y.permute(0,2,1).reshape(B,T,N,3)

def torch_load(path,device):
    try: return torch.load(path,map_location=device,weights_only=False)
    except TypeError: return torch.load(path,map_location=device)

def load_static(path:Path,device):
    ck=torch_load(path,device); model=StaticNeutralAE(int(ck.get('n_markers',105)),int(ck.get('hidden',256)),int(ck.get('latent',48)),float(ck.get('dropout',0.20))).to(device); model.load_state_dict(ck['model_state_dict']); model.eval(); return model,ck

def load_dynamic(path:Path,device):
    ck=torch_load(path,device); model=DynamicMotionProjector(int(ck.get('n_markers',105)),int(ck.get('n_frames',100)),int(ck.get('hidden',160)),int(ck.get('latent',40)),float(ck.get('dropout',0.20))).to(device); model.load_state_dict(ck['model_state_dict']); model.eval(); return model,ck

def load_labels(path:Path)->pd.DataFrame:
    df=pd.read_csv(path); df['marker_id']=pd.to_numeric(df['marker_id'],errors='coerce').astype('Int64'); df=df.dropna(subset=['marker_id']).copy(); df['marker_id']=df['marker_id'].astype(int); return df

def marker_ref_arrays(env:pd.DataFrame,N:int)->Dict[str,np.ndarray]:
    out={k:np.full(N,np.nan,dtype=np.float32) for k in ['amp_low','amp_high','amp_scale','rmse_high','rmse_scale']}
    for _,r in env.iterrows():
        j=int(r['facial_marker_id']) if 'facial_marker_id' in r else int(r['marker_id'])
        if 0<=j<N:
            for k in out: out[k][j]=float(r.get(k,np.nan))
    out['amp_scale']=np.where(np.isfinite(out['amp_scale'])&(out['amp_scale']>1e-9),out['amp_scale'],1.0)
    out['rmse_scale']=np.where(np.isfinite(out['rmse_scale'])&(out['rmse_scale']>1e-9),out['rmse_scale'],1.0)
    return out

def abnormal_mask(X,M,Xref,ref,args):
    valid=M.astype(bool); amp=np.nanmax(np.where(valid,np.linalg.norm(X,axis=-1),np.nan),axis=1); amp_ref=np.nanmax(np.linalg.norm(Xref,axis=-1),axis=0)
    rmse=np.sqrt(np.nanmean(np.where(valid,np.linalg.norm(X-Xref[None],axis=-1)**2,np.nan),axis=1))
    hypo=np.maximum(0,(ref['amp_low'][None]-amp)/ref['amp_scale'][None]); hyper=np.maximum(0,(amp-ref['amp_high'][None])/ref['amp_scale'][None]); absdev=np.maximum(0,(rmse-ref['rmse_high'][None])/ref['rmse_scale'][None])
    sev=np.nanmax(np.stack([hypo,hyper,absdev],axis=0),axis=0); sev=np.where(np.isfinite(sev),sev,0.0)
    mask=sev>args.abnormal_threshold
    # Also mask top fraction of markers per sample to force projection of worst markers.
    if args.extra_mask_fraction>0:
        n=max(1,int(round(X.shape[2]*args.extra_mask_fraction)))
        for s in range(X.shape[0]):
            idx=np.argsort(sev[s])[-n:]; mask[s,idx]=True
    return mask,sev

def region_summaries(marker_df,labels,metric_col,group_cols):
    df=marker_df.merge(labels[['marker_id','region','side','label']],on='marker_id',how='left'); df['region']=df['region'].fillna('unlabeled'); df['side']=df['side'].fillna('unknown')
    rows=[]
    for keys,g in df.groupby(group_cols):
        if not isinstance(keys,tuple): keys=(keys,)
        rec={c:v for c,v in zip(group_cols,keys)}; vals=pd.to_numeric(g[metric_col],errors='coerce').dropna()
        rec.update({'n_markers':int(len(g)),'score_mean':float(vals.mean()) if len(vals) else np.nan,'score_median':float(vals.median()) if len(vals) else np.nan,'score_max':float(vals.max()) if len(vals) else np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)

def project_group(movement,group,args,static_model,static_ck,dyn_model,dyn_ck,device,labels):
    mov_dir=Path(args.dataset_dir)/movement; out_mov=Path(args.out_dir)/movement; ensure_dir(out_mov)
    X=np.load(mov_dir/f'X_{group}.npy').astype(np.float32); M=np.load(mov_dir/f'M_{group}.npy').astype(np.uint8); N=np.load(mov_dir/f'N_{group}.npy').astype(np.float32); meta=pd.read_csv(mov_dir/f'metadata_{group}.csv')
    if X.shape[0]==0: return [],[],[],[]
    Xref=np.load(mov_dir/'reference_displacement.npy').astype(np.float32)[:, :X.shape[2], :]
    env=pd.read_csv(mov_dir/'reference_envelope_per_marker.csv'); ref=marker_ref_arrays(env,X.shape[2])
    Mneutral=np.isfinite(N).all(axis=-1).astype(np.uint8); Nzero=np.where(np.isfinite(N),N,0.0).astype(np.float32)
    Nmean=np.asarray(static_ck['mean'],dtype=np.float32); Nstd=np.asarray(static_ck['std'],dtype=np.float32); Dmean=np.asarray(dyn_ck['mean'],dtype=np.float32); Dstd=np.asarray(dyn_ck['std'],dtype=np.float32)
    ZN=normalize_static(Nzero,Mneutral,Nmean,Nstd)
    with torch.no_grad():
        predNn=static_model(torch.from_numpy(ZN).to(device),torch.from_numpy(Mneutral.astype(np.float32)).to(device)).cpu().numpy()
    Npred=denorm_static(predNn,Nmean,Nstd)
    # Blend: 0 = full projection, 1 = original neutral. Default preserves some identity but corrects asymmetry.
    Nproj=(1-args.neutral_identity_blend)*Npred + args.neutral_identity_blend*Nzero
    abn,sev=abnormal_mask(X,M,Xref,ref,args)
    Min=M.copy(); Xin=X.copy(); Min[abn[:,None,:].repeat(X.shape[1],axis=1)]=0; Xin[Min==0]=0.0
    Z=normalize_dyn(Xin,Min,Dmean,Dstd)
    with torch.no_grad():
        predDn=dyn_model(torch.from_numpy(Z).to(device),torch.from_numpy(Min.astype(np.float32)).to(device)).cpu().numpy()
    Dpred=denorm_dyn(predDn,Dmean,Dstd)
    Dproj=(1-args.motion_anchor_strength)*Dpred + args.motion_anchor_strength*Xref[None]
    obs_face=Nzero[:,None,:,:]+X
    twin_face=Nproj[:,None,:,:]+Dproj
    # Distances.
    marker_full=np.sqrt(np.nanmean((obs_face-twin_face)**2,axis=(1,3))) # S,N
    marker_motion=np.sqrt(np.nanmean((X-Dproj)**2,axis=(1,3)))
    marker_neutral=np.linalg.norm(Nzero-Nproj,axis=-1)
    sample_rows=[]; marker_rows=[]
    for s in range(X.shape[0]):
        row=meta.iloc[s].to_dict() if s < len(meta) else {}
        sample_id=str(row.get('sample_id',f'{movement}_{group}_{s:04d}'))
        valid=M[s].astype(bool)
        full_d=np.linalg.norm(obs_face[s]-twin_face[s],axis=-1)
        motion_d=np.linalg.norm(X[s]-Dproj[s],axis=-1)
        sample_rows.append({**row,'group':group,'movement':movement,'sample_index':s,'sample_id':sample_id,
            'twin_full_rmse':float(np.sqrt(np.nanmean(full_d[valid]**2))) if valid.any() else np.nan,
            'twin_full_mae':float(np.nanmean(full_d[valid])) if valid.any() else np.nan,
            'twin_motion_rmse':float(np.sqrt(np.nanmean(motion_d[valid]**2))) if valid.any() else np.nan,
            'twin_static_neutral_rmse':float(np.sqrt(np.nanmean(marker_neutral[s]**2))),
            'twin_marker_rmse_median':float(np.nanmedian(marker_full[s])),
            'twin_marker_rmse_p95':float(np.nanpercentile(marker_full[s],95)),
            'abnormal_marker_fraction_masked':float(abn[s].mean()),
            'abnormal_marker_severity_mean':float(np.nanmean(sev[s])),
        })
        for j in range(X.shape[2]):
            marker_rows.append({'group':group,'movement':movement,'sample_id':sample_id,'sample_index':s,'marker_id':j,
                'full_marker_rmse':float(marker_full[s,j]),'motion_marker_rmse':float(marker_motion[s,j]),'neutral_marker_shift':float(marker_neutral[s,j]),'abnormal_masked':bool(abn[s,j]),'abnormal_severity':float(sev[s,j])})
    np.savez_compressed(out_mov/f'twin_{group}.npz', observed_face=obs_face.astype(np.float32), twin_face=twin_face.astype(np.float32), observed_displacement=X.astype(np.float32), projected_displacement=Dproj.astype(np.float32), observed_neutral=Nzero.astype(np.float32), projected_neutral=Nproj.astype(np.float32), mask=M.astype(np.uint8), abnormal_mask=abn.astype(np.uint8), abnormal_severity=sev.astype(np.float32), sample_ids=np.asarray([r['sample_id'] for r in sample_rows],dtype=object))
    mdf=pd.DataFrame(marker_rows)
    rdf=region_summaries(mdf,labels,'full_marker_rmse',['group','movement','sample_id','region'])
    rsdf=region_summaries(mdf,labels,'full_marker_rmse',['group','movement','sample_id','region','side'])
    return sample_rows, marker_rows, rdf.to_dict('records'), rsdf.to_dict('records')

def parse_args():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset_dir',required=True); ap.add_argument('--static_model_dir',required=True); ap.add_argument('--dynamic_model_dir',required=True); ap.add_argument('--out_dir',required=True); ap.add_argument('--semantic_labels',required=True)
    ap.add_argument('--movements',nargs='+',default=['M1','M2','M3','M4','M5']); ap.add_argument('--groups',nargs='+',default=['eval_healthy','pathological'])
    ap.add_argument('--neutral_identity_blend',type=float,default=0.35); ap.add_argument('--motion_anchor_strength',type=float,default=0.30); ap.add_argument('--abnormal_threshold',type=float,default=0.25); ap.add_argument('--extra_mask_fraction',type=float,default=0.15); ap.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    return ap.parse_args()

def main():
    args=parse_args(); out=Path(args.out_dir); ensure_dir(out); write_json(out/'predict_config.json',vars(args)); labels=load_labels(Path(args.semantic_labels))
    device=torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) else ('cpu' if args.device=='auto' else args.device)); print(f'Using device: {device}')
    static_model,static_ck=load_static(Path(args.static_model_dir)/'model.pt',device)
    all_s=[]; all_m=[]; all_r=[]; all_rs=[]
    for movement in [m.upper() for m in args.movements]:
        dyn_model,dyn_ck=load_dynamic(Path(args.dynamic_model_dir)/movement/'model.pt',device)
        for group in args.groups:
            print(f'Projecting {movement} {group}')
            s,m,r,rs=project_group(movement,group,args,static_model,static_ck,dyn_model,dyn_ck,device,labels)
            all_s+=s; all_m+=m; all_r+=r; all_rs+=rs
    pd.DataFrame(all_s).to_csv(out/'twin_sample_scores.csv',index=False); pd.DataFrame(all_m).to_csv(out/'twin_marker_scores.csv',index=False); pd.DataFrame(all_r).to_csv(out/'twin_region_scores.csv',index=False); pd.DataFrame(all_rs).to_csv(out/'twin_region_side_scores.csv',index=False)
    print(f'[OK] Wrote healthy twin predictions: {out}')
if __name__=='__main__': main()
