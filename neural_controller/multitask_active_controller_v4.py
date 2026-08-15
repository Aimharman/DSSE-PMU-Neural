"""v4 inference for the gated physics-informed neural controller."""
from __future__ import annotations
import joblib, numpy as np, pandas as pd
from feature_extractor_v4 import WINDOW,TIMING_LONG,BASELINE_SAMPLES,_baseline_from_history,extract_window_features_v4

REQUIRED=[]
for p in (1,2,3):
    REQUIRED += [f"PMU{p} Voltage Magnitude",f"PMU{p} Voltage Phase",
                 f"PMU{p} Current Magnitude",f"PMU{p} Current Phase"]


def truth(v):
    if pd.isna(v): return False
    return str(v).strip().lower() in {"true","1","1.0","yes","y"}


def raw_fault_info(w):
    active=[]
    for p in (1,2,3):
        if any(truth(v) for v in w.get(f"PMU{p} Bad Data",[])): active.append((f"PMU{p}","BAD_DATA"))
        if any(truth(v) for v in w.get(f"PMU{p} Sync Fault Active",[])): active.append((f"PMU{p}","SYNC"))
        if any(truth(v) for v in w.get(f"PMU{p} Clock Drift Fault",[])): active.append((f"PMU{p}","CLOCK_DRIFT"))
    if not active:return "NORMAL","NONE"
    if len(active)==1:return active[0][1],active[0][0]
    return "MIXED",",".join(x[0] for x in active)


def _argmax(model,X):
    p=model.predict_proba(X)[0]
    classes=list(model.named_steps["mlp"].classes_)
    i=int(np.argmax(p))
    return str(classes[i]),float(p[i])


def management(ft,pmu,conf):
    if conf<0.70:return "HOLD / REQUEST MORE DATA",[1.,1.,1.],[1.]*12
    if ft=="NORMAL" or pmu=="NONE":return "ACCEPT ALL PMUs",[1.,1.,1.],[1.]*12
    p=int(pmu[-1]); w=[1.,1.,1.];w[p-1]=.10
    m=[1.]*12;b=(p-1)*4
    if ft=="BAD_DATA":
        m[b:b+4]=[.10]*4
        return f"DOWN-WEIGHT PMU{p}",w,m
    m[b+1]=.10;m[b+3]=.10
    if ft=="SYNC":return f"DOWN-WEIGHT PMU{p} AND APPLY PHASE CHECK",w,m
    if ft=="CLOCK_DRIFT":return f"DOWN-WEIGHT PMU{p} AND APPLY TIMING CHECK",w,m
    return "HOLD / REQUEST MORE DATA",[1.,1.,1.],[1.]*12


def predict_window(window,history,bundle,baseline):
    f=extract_window_features_v4(window,history,baseline)
    X=pd.DataFrame([[f[n] for n in bundle["feature_names"]]],columns=bundle["feature_names"])

    # Stage 1: normal/fault gate. No timing specialist is allowed to override
    # a confident NORMAL decision.
    state,state_conf=_argmax(bundle["state_model"],X)
    if state=="NORMAL" and state_conf>=0.70:
        return dict(fault_type="NORMAL",faulty_pmu="NONE",
                    type_confidence=state_conf,timing_confidence=0.,
                    pmu_confidence=1.,neural_confidence=state_conf,
                    management_state="NORMAL",management_action="ACCEPT ALL PMUs",
                    pmu_weights=[1.,1.,1.],measurement_weights=[1.]*12)

    # Stage 2: bad data vs timing.
    bt,bt_conf=_argmax(bundle["bad_timing_model"],X)
    if bt=="BAD_DATA":
        fault_type="BAD_DATA"; timing_conf=0.
    else:
        # Stage 3: timing specialist.
        Xt=X[bundle["timing_feature_names"]]
        fault_type,timing_conf=_argmax(bundle["timing_model"],Xt)

    # Faulty PMU is always inferred from measurements for fault states.
    pmu,pmu_conf=_argmax(bundle["pmu_model"],X)
    conf=min(state_conf,bt_conf,pmu_conf,
             timing_conf if fault_type in {"SYNC","CLOCK_DRIFT"} else 1.)
    action,weights,mweights=management(fault_type,pmu,conf)
    return dict(fault_type=fault_type,faulty_pmu=pmu,
                type_confidence=state_conf,timing_confidence=timing_conf,
                pmu_confidence=pmu_conf,neural_confidence=conf,
                management_state="FAULT" if conf>=.70 else "UNCERTAIN",
                management_action=action,pmu_weights=weights,
                measurement_weights=mweights)


def scan_csv(csv_path,model_path,out_path=None):
    b=joblib.load(model_path)
    df=pd.read_csv(csv_path).reset_index(drop=True)
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing:raise ValueError(f"Missing PMU measurement columns: {missing}")

    baseline=_baseline_from_history(df.iloc[:min(BASELINE_SAMPLES,len(df))])
    rows=[]
    first_end=max(WINDOW-1,TIMING_LONG-1)
    for end in range(first_end,len(df),WINDOW):
        w=df.iloc[end-WINDOW+1:end+1]
        if len(w)!=WINDOW or w[REQUIRED].isna().any().any():continue
        hs=max(0,end-TIMING_LONG+1);hist=df.iloc[hs:end+1]
        raw_type,raw_pmu=raw_fault_info(w)
        p=predict_window(w,hist,b,baseline)
        rows.append({
          "time_s":float(w["Time (s)"].iloc[-1]) if "Time (s)" in w else end/1000.,
          "raw_fault_type":raw_type,"raw_faulty_pmu":raw_pmu,
          "type_confidence":p["type_confidence"],
          "timing_confidence":p["timing_confidence"],
          "pmu_confidence":p["pmu_confidence"],
          "neural_confidence":p["neural_confidence"],
          "management_state":p["management_state"],
          "active_fault_type":p["fault_type"],
          "active_faulty_pmu":p["faulty_pmu"],
          "management_action":p["management_action"],
          "pmu_weights":str(p["pmu_weights"]),
          "measurement_weights":str(p["measurement_weights"])
        })
    out=pd.DataFrame(rows)
    if out_path:out.to_csv(out_path,index=False)
    return out
