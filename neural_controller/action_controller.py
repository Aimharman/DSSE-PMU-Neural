from __future__ import annotations
import joblib, numpy as np, pandas as pd
from feature_extractor import WINDOW, extract_window_features

def management_policy(action, confidence):
    if confidence < 0.70: return 'HOLD / REQUEST MORE DATA',[1.0,1.0,1.0]
    if action=='ACCEPT': return 'ACCEPT ALL PMUs',[1.0,1.0,1.0]
    pmu=int(action[-1]); w=[1.0,1.0,1.0]; w[pmu-1]=0.10
    return f'DOWN-WEIGHT PMU{pmu}',w

def scan_csv(csv_path,model_path,out_path='Neural_Active_Controller_Results.csv'):
    b=joblib.load(model_path); df=pd.read_csv(csv_path)
    req=[]
    for p in (1,2,3): req += [f'PMU{p} Voltage Magnitude',f'PMU{p} Voltage Phase',f'PMU{p} Current Magnitude',f'PMU{p} Current Phase']
    rows=[]
    for end in range(WINDOW-1,len(df),WINDOW):
        w=df.iloc[end-WINDOW+1:end+1]
        if len(w)!=WINDOW or w[req].isna().any().any(): continue
        f=extract_window_features(w); X=pd.DataFrame([[f[n] for n in b['feature_names']]],columns=b['feature_names'])
        m=b['model']; p=m.predict_proba(X)[0]; i=int(np.argmax(p)); action=str(m.named_steps['mlp'].classes_[i]); conf=float(p[i]); act,weights=management_policy(action,conf)
        rows.append({'time_s':float(w['Time (s)'].iloc[-1]),'neural_action':action,'confidence':conf,'management_action':act,'pmu_weights':str(weights)})
    out=pd.DataFrame(rows); out.to_csv(out_path,index=False); return out
