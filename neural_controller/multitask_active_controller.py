from __future__ import annotations
import joblib, numpy as np, pandas as pd
from feature_extractor import WINDOW, extract_window_features

def management_action(fault_type, pmu, confidence):
    if confidence < 0.70:
        return "HOLD / REQUEST MORE DATA", [1.0,1.0,1.0]
    if fault_type == "NORMAL": return "ACCEPT ALL PMUs", [1.0,1.0,1.0]
    w=[1.0,1.0,1.0]; idx=pmu-1
    if fault_type == "BAD_DATA": w[idx]=0.10; action=f"DOWN-WEIGHT PMU{pmu}"
    elif fault_type == "SYNC": w[idx]=0.20; action=f"DOWN-WEIGHT PHASE DATA OF PMU{pmu}"
    elif fault_type == "CLOCK_DRIFT": w[idx]=0.20; action=f"DOWN-WEIGHT PMU{pmu} AND APPLY TIMING CHECK"
    else: w[idx]=0.0; action=f"ISOLATE PMU{pmu}"
    return action,w

def load(path): return joblib.load(path)

def predict_window(window,bundle):
    f=extract_window_features(window)
    X=pd.DataFrame([[f[n] for n in bundle['feature_names']]],columns=bundle['feature_names'])
    tm=bundle['type_model']; probs=tm.predict_proba(X)[0]; ti=int(np.argmax(probs)); typ=str(tm.named_steps['mlp'].classes_[ti]); tc=float(probs[ti])
    if typ == 'NORMAL': pmu='NONE'; pc=1.0
    else:
        pm=bundle['pmu_model']; pp=pm.predict_proba(X)[0]; pi=int(np.argmax(pp)); pmu=str(pm.named_steps['mlp'].classes_[pi]); pc=float(pp[pi])
    conf=min(tc,pc)
    action,weights=management_action(typ,int(pmu[-1]) if pmu!='NONE' else 1,conf)
    return {'fault_type':typ,'faulty_pmu':pmu,'confidence':conf,'action':action,'weights':weights}

def scan_csv(csv_path,model_path,out_path='Neural_Active_PDC_Results.csv'):
    b=load(model_path); df=pd.read_csv(csv_path); req=[]
    for p in (1,2,3): req += [f'PMU{p} Voltage Magnitude',f'PMU{p} Voltage Phase',f'PMU{p} Current Magnitude',f'PMU{p} Current Phase']
    rows=[]
    for end in range(WINDOW-1,len(df),WINDOW):
        w=df.iloc[end-WINDOW+1:end+1]
        if len(w)!=WINDOW or w[req].isna().any().any(): continue
        r=predict_window(w,b); r['time_s']=float(w['Time (s)'].iloc[-1]); rows.append(r)
    out=pd.DataFrame(rows); out.to_csv(out_path,index=False); return out
