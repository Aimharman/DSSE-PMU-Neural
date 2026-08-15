"""Train the neural active management decision directly.

Targets are controller actions, not simulator fault labels:
    ACCEPT, DOWNWEIGHT_PMU1, DOWNWEIGHT_PMU2, DOWNWEIGHT_PMU3

Fault type remains available as post-run diagnostic information, but is not the
primary neural objective. This matches the active fault-management task.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_extractor import build_dataset

RANDOM_STATE=42

def action_label(label):
    if label == 'NORMAL': return 'ACCEPT'
    return 'DOWNWEIGHT_' + label.split('_')[0]

def make_model():
    return Pipeline([('scale',StandardScaler()),('mlp',MLPClassifier(
        hidden_layer_sizes=(32,16),activation='relu',solver='adam',alpha=1e-4,
        learning_rate_init=1e-3,max_iter=500,early_stopping=False,random_state=RANDOM_STATE))])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('csv',nargs='+'); ap.add_argument('--model-out',default='neural_active_fault_controller.joblib'); args=ap.parse_args()
    X,y,meta=build_dataset(args.csv)
    ya=y.map(action_label); groups=meta.source.to_numpy(); classes=set(ya.unique())
    splitter=GroupShuffleSplit(n_splits=200,test_size=0.40,random_state=RANDOM_STATE); chosen=None
    for tr,te in splitter.split(X,ya,groups):
        if classes.issubset(set(ya.iloc[tr])) and classes.issubset(set(ya.iloc[te])): chosen=(tr,te); break
    if chosen is None: raise SystemExit('Could not construct group-held-out split with every action in both sets.')
    tr,te=chosen
    print('Held-out scenario files:')
    for f in sorted(set(meta.iloc[te].source)): print(' ',f)
    model=make_model(); model.fit(X.iloc[tr],ya.iloc[tr]); pred=model.predict(X.iloc[te])
    print('\nNEURAL ACTIVE ACTION CONTROLLER')
    print(classification_report(ya.iloc[te],pred,digits=4,zero_division=0))
    labels=list(model.named_steps['mlp'].classes_); cm=confusion_matrix(ya.iloc[te],pred,labels=labels)
    print(pd.DataFrame(cm,index=labels,columns=labels).to_string())
    bundle={'model':model,'feature_names':list(X.columns),'classes':labels,'window_samples':20,'pdc_rate_hz':50.0}
    joblib.dump(bundle,args.model_out)
    Path(args.model_out).with_suffix('.json').write_text(json.dumps({'windows':len(X),'classes':labels,'test_windows':len(te),'model':'MLP(32,16)'},indent=2))
    print('\nSaved',args.model_out)

if __name__=='__main__': main()
