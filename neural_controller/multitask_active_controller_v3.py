"""Inference for the v3 hierarchical Neural Active Fault Management Controller."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from feature_extractor_v2 import WINDOW, TIMING_LONG, extract_window_features

REQUIRED = []
for p in (1,2,3):
    REQUIRED += [f"PMU{p} Voltage Magnitude", f"PMU{p} Voltage Phase",
                 f"PMU{p} Current Magnitude", f"PMU{p} Current Phase"]

def _truth(v):
    if pd.isna(v): return False
    return str(v).strip().lower() in {"true","1","1.0","yes","y"}

def raw_fault_info(window):
    active=[]
    for p in (1,2,3):
        if any(_truth(v) for v in window.get(f"PMU{p} Bad Data", [])): active.append((f"PMU{p}","BAD_DATA"))
        if any(_truth(v) for v in window.get(f"PMU{p} Sync Fault Active", [])): active.append((f"PMU{p}","SYNC"))
        if any(_truth(v) for v in window.get(f"PMU{p} Clock Drift Fault", [])): active.append((f"PMU{p}","CLOCK_DRIFT"))
    if not active: return "NORMAL","NONE"
    if len(active)==1: return active[0][1],active[0][0]
    return "MIXED",",".join(x[0] for x in active)

def _management(ft, pmu, conf):
    if conf < .70: return "HOLD / REQUEST MORE DATA",[1,1,1],[1]*12
    if ft=="NORMAL" or pmu=="NONE": return "ACCEPT ALL PMUs",[1,1,1],[1]*12
    p=int(pmu[-1]); w=[1.,1.,1.]; w[p-1]=.10
    m=[1.]*12; b=(p-1)*4
    if ft=="BAD_DATA":
        m[b:b+4]=[.10]*4
        return f"DOWN-WEIGHT PMU{p}",w,m
    m[b+1]=.10; m[b+3]=.10
    if ft=="SYNC": return f"DOWN-WEIGHT PMU{p} AND APPLY PHASE CHECK",w,m
    if ft=="CLOCK_DRIFT": return f"DOWN-WEIGHT PMU{p} AND APPLY TIMING CHECK",w,m
    return "HOLD / REQUEST MORE DATA",[1,1,1],[1]*12

def predict_window(window, history, bundle):
    f=extract_window_features(window, history=history)
    X=pd.DataFrame([[f[n] for n in bundle["feature_names"]]],columns=bundle["feature_names"])
    tm=bundle["type_model"]; probs=tm.predict_proba(X)[0]
    tc=list(tm.named_steps["mlp"].classes_); ti=int(np.argmax(probs))
    primary=str(tc[ti]); primary_conf=float(probs[ti])

    # Timing specialist arbitrates whenever the primary model believes the
    # sample is timing-related. This is still entirely measurement-derived.
    timing_conf=0.0
    if primary in {"SYNC","CLOCK_DRIFT"}:
        tf=bundle["timing_feature_names"]
        Xt=pd.DataFrame([[f[n] for n in tf]],columns=tf)
        sm=bundle["timing_model"]; sp=sm.predict_proba(Xt)[0]
        sc=list(sm.named_steps["mlp"].classes_); si=int(np.argmax(sp))
        fault_type=str(sc[si]); timing_conf=float(sp[si])
    else:
        fault_type=primary

    # If primary is non-timing but its timing specialist is strongly decisive,
    # allow timing evidence to recover a timing fault.
    if primary not in {"SYNC","CLOCK_DRIFT"}:
        tf=bundle["timing_feature_names"]
        Xt=pd.DataFrame([[f[n] for n in tf]],columns=tf)
        sm=bundle["timing_model"]; sp=sm.predict_proba(Xt)[0]
        sc=list(sm.named_steps["mlp"].classes_); si=int(np.argmax(sp))
        if float(sp[si]) >= .90:
            fault_type=str(sc[si]); timing_conf=float(sp[si])

    pm=bundle["pmu_model"]
    if fault_type=="NORMAL":
        pmu="NONE"; pmu_conf=1.0
    else:
        pp=pm.predict_proba(X)[0]; pc=list(pm.named_steps["mlp"].classes_); pi=int(np.argmax(pp))
        pmu=str(pc[pi]); pmu_conf=float(pp[pi])
    conf=min(primary_conf, pmu_conf, timing_conf if fault_type in {"SYNC","CLOCK_DRIFT"} else 1.0)
    action,weights,mweights=_management(fault_type,pmu,conf)
    return dict(fault_type=fault_type,faulty_pmu=pmu,type_confidence=primary_conf,
                timing_confidence=timing_conf,pmu_confidence=pmu_conf,
                neural_confidence=conf,
                management_state="NORMAL" if fault_type=="NORMAL" else ("FAULT" if conf>=.70 else "UNCERTAIN"),
                management_action=action,pmu_weights=weights,measurement_weights=mweights)

def scan_csv(csv_path,model_path,out_path=None):
    b=joblib.load(model_path); df=pd.read_csv(csv_path).reset_index(drop=True)
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"Missing PMU measurement columns: {missing}")
    rows=[]; first_end=max(WINDOW-1,TIMING_LONG-1)
    for end in range(first_end,len(df),WINDOW):
        w=df.iloc[end-WINDOW+1:end+1]
        if len(w)!=WINDOW or w[REQUIRED].isna().any().any(): continue
        hs=max(0,end-TIMING_LONG+1); hist=df.iloc[hs:end+1]
        raw_type,raw_pmu=raw_fault_info(w); p=predict_window(w,hist,b)
        rows.append({"time_s":float(w["Time (s)"].iloc[-1]) if "Time (s)" in w else end/1000.,
                     "raw_fault_type":raw_type,"raw_faulty_pmu":raw_pmu,
                     "type_confidence":p["type_confidence"],"timing_confidence":p["timing_confidence"],
                     "pmu_confidence":p["pmu_confidence"],"neural_confidence":p["neural_confidence"],
                     "management_state":p["management_state"],"active_fault_type":p["fault_type"],
                     "active_faulty_pmu":p["faulty_pmu"],"management_action":p["management_action"],
                     "pmu_weights":str(p["pmu_weights"]),"measurement_weights":str(p["measurement_weights"])})
    out=pd.DataFrame(rows)
    if out_path: out.to_csv(out_path,index=False)
    return out
