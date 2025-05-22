import pickle
import numpy as np
import wandb
import math
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp
from joblib import Parallel, delayed

# === 1. Custom LVQ with per‐epoch F1 logging ===
class CustomLVQ(BaseEstimator, ClassifierMixin):
    def __init__(self, n_prototypes_per_class=5, learning_rate=0.01,
                 n_epochs=1000, batch_size=32, random_state=42):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.history_ = []

    def _initialize_prototypes(self, X, y):
        np.random.seed(self.random_state)
        protos, labels = [], []
        for cls in np.unique(y):
            Xc = X[y == cls]
            idx = np.random.choice(len(Xc), self.n_prototypes_per_class, replace=True)
            protos.append(Xc[idx])
            labels += [cls] * self.n_prototypes_per_class
        self.prototypes_ = np.vstack(protos)
        self.prototype_labels_ = np.array(labels)

    def fit(self, X, y, X_val=None, y_val=None, log_wandb=False):
        self._initialize_prototypes(X, y)
        self.history_ = []
        for epoch in range(1, self.n_epochs + 1):
            Xb, yb = shuffle(X, y, random_state=self.random_state + epoch)
            for i in range(0, len(Xb), self.batch_size):
                xb, ybi = Xb[i:i+self.batch_size], yb[i:i+self.batch_size]
                for xi, yi in zip(xb, ybi):
                    d = np.linalg.norm(self.prototypes_ - xi, axis=1)
                    w = np.argmin(d)
                    sign = 1 if self.prototype_labels_[w] == yi else -1
                    self.prototypes_[w] += sign * self.learning_rate * (xi - self.prototypes_[w])
            # compute F1
            train_f1 = f1_score(y, self.predict(X), zero_division=0)
            rec = {"epoch": epoch, "train_f1": train_f1}
            if X_val is not None:
                val_f1 = f1_score(y_val, self.predict(X_val), zero_division=0)
                rec["val_f1"] = val_f1
            self.history_.append(rec)
            if log_wandb:
                wandb.log(rec)
        return self

    def predict(self, X):
        d = np.linalg.norm(self.prototypes_[None,:,:] - X[:,None,:], axis=2)
        return self.prototype_labels_[np.argmin(d, axis=1)]


# === 2. Hyperparameter grids ===
param_grids = {
    "Decision Tree":    {"max_depth":[None,10,20],    "min_samples_split":[2,5]},
    "Random Forest":    {"n_estimators":[50,100],     "max_depth":[None,10], "min_samples_split":[2,5]},
    "SVM":              {"C":[0.1,1],                "kernel":["linear","rbf"]},
    "MLP":              {"hidden_layer_sizes":[(10,),(50,)], "alpha":[1e-4,1e-3], "max_iter":[500,1000]},
    "KNN":              {"n_neighbors":[3,5,7],      "weights":["uniform","distance"]},
    "LVQ":              {"n_prototypes_per_class":[2,5,10], "learning_rate":[1e-3,1e-2,1e-1], "n_epochs":[200,500]}
}


# === 3. Statistical comparison helpers ===
def compute_cd(k, N):
    q = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850}
    return q[k] * math.sqrt((k*(k+1))/(6.0*N))

def evaluate_fold(ti, vi, models, X, y, metric, mname):
    scores=[]
    for name,clf in models.items():
        clf.fit(X[ti], y[ti])
        yp = clf.predict(X[vi])
        proba = clf.predict_proba(X[vi])[:,1] if hasattr(clf,"predict_proba") else None
        if mname=="roc_auc" and proba is not None:
            s=metric(y[vi], proba)
        elif mname in ("precision","recall","f1"):
            s=metric(y[vi], yp, zero_division=0)
        else:
            s=metric(y[vi], yp)
        scores.append(s)
    return scores

def run_stat_comparison(X, y, models, project="ml-comparisons", n_splits=10):
    mets = {
        "accuracy": accuracy_score, "precision": precision_score,
        "recall": recall_score, "f1": f1_score, "roc_auc": roc_auc_score
    }
    names = list(models.keys())
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for mname, metric in mets.items():
        wandb.init(project=project, name=f"CD-{mname}", reinit=True)
        folds = Parallel(n_jobs=-1)(
            delayed(evaluate_fold)(ti,vi,models,X,y,metric,mname)
            for ti,vi in kf.split(X,y)
        )
        F = np.array(folds)
        ranks = np.array([ rankdata(-row, method="average") for row in F ])
        avg_ranks = ranks.mean(axis=0)
        stat,p = friedmanchisquare(*[F[:,i] for i in range(F.shape[1])])
        wandb.log({f"{mname}/friedman_stat":stat, f"{mname}/p_value":p})
        summary=[]
        for i,n in enumerate(names):
            mu,sd = F[:,i].mean(),F[:,i].std()
            summary.append([n,mu,sd])
            wandb.log({f"{n}/{mname}_mean":mu, f"{n}/{mname}_std":sd})
        wandb.log({f"{mname}/summary": wandb.Table(data=summary,columns=["Model","Mean","Std"])})
        if p<0.05:
            post=sp.posthoc_nemenyi_friedman(F)
            post.columns=post.index=names
            wandb.log({f"{mname}/nemenyi":wandb.Table(data=post.values.tolist(),columns=names)})
            cd=compute_cd(len(names),n_splits)
            wandb.log({f"{mname}/CD":cd})
            ranktbl=[[names[i],float(avg_ranks[i])] for i in range(len(names))]
            wandb.log({f"{mname}/avg_ranks": wandb.Table(data=ranktbl,columns=["Model","AvgRank"])})
        wandb.finish()


# === 4. Main ===
if __name__=="__main__":
    with open(r"machine_learning_mini_project\adult_income_train_val.pkl","rb") as f:
        X_tr, X_val, y_tr, y_val = pickle.load(f)
    with open(r"machine_learning_mini_project\adult_income_test.pkl","rb") as f:
        X_test, y_test = pickle.load(f)

    X_full, y_full = np.vstack((X_tr,X_val)), np.concatenate((y_tr,y_val))

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "MLP": MLPClassifier(),
        "KNN": KNeighborsClassifier()
    }
    best_models, best_scores = {}, {}
    for name,mdl in base.items():
        print(f"Tuning {name}…")
        gs = GridSearchCV(mdl, param_grids[name], scoring="f1", cv=cv5, n_jobs=-1)
        gs.fit(X_full,y_full)
        best_models[name], best_scores[name] = gs.best_estimator_, gs.best_score_
        print(f" → {name} cv-f1 = {best_scores[name]:.4f}")

    print("Tuning LVQ…")
    gs_lvq = GridSearchCV(CustomLVQ(), param_grids["LVQ"], scoring="f1", cv=cv5, n_jobs=-1)
    gs_lvq.fit(X_full,y_full)
    best_lvq_params = gs_lvq.best_params_
    best_scores["LVQ"] = gs_lvq.best_score_
    print(f" → LVQ cv-f1 = {best_scores['LVQ']:.4f}, params = {best_lvq_params}")

    wandb.init(project="ml-comparisons", name="LVQ_curve", reinit=True)
    lvq = CustomLVQ(**best_lvq_params)
    lvq.fit(X_tr,y_tr, X_val=X_val,y_val=y_val, log_wandb=True)
    wandb.log({"LVQ/train_curve": wandb.Table(dataframe=pd.DataFrame(lvq.history_))})
    wandb.finish()

    best_models["LVQ"] = CustomLVQ(**best_lvq_params).fit(X_full,y_full)
    run_stat_comparison(X_full, y_full, best_models)

    test_metrics = {}
    for name,clf in best_models.items():
        yp = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:,1] if hasattr(clf,"predict_proba") else None
        m = {
            "accuracy": accuracy_score(y_test,yp),
            "precision": precision_score(y_test,yp,zero_division=0),
            "recall": recall_score(y_test,yp,zero_division=0),
            "f1": f1_score(y_test,yp,zero_division=0)
        }
        if proba is not None:
            m["roc_auc"] = roc_auc_score(y_test,proba)
        for k,v in m.items():
            test_metrics[f"{name}/{k}"] = v

    wandb.init(project="ml-comparisons", name="test_all_models", reinit=True)
    wandb.log(test_metrics)
    best_test = max(best_models.keys(), key=lambda nm: test_metrics[f"{nm}/f1"])
    wandb.log({"selection_consistent": best_test == max(best_scores, key=best_scores.get)})
    wandb.finish()

    print("\nCV-best model:", max(best_scores, key=best_scores.get))
    print("Test-best model:", best_test)
    print("Test metrics per model:")
    for k,v in test_metrics.items():
        print(f" {k}: {v:.4f}")
