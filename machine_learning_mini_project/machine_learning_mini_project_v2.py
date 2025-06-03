import pickle
import numpy as np
import wandb
import math

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

# === 1. Custom LVQ ===
class CustomLVQ(BaseEstimator, ClassifierMixin):
    def __init__(self, n_prototypes_per_class=5, learning_rate=0.01,
                 n_epochs=1000, batch_size=32, random_state=42):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _initialize_prototypes(self, X, y):
        np.random.seed(self.random_state)
        protos, labels = [], []
        for cls in np.unique(y):
            Xc = X[y==cls]
            idx = np.random.choice(len(Xc), self.n_prototypes_per_class, replace=True)
            protos.append(Xc[idx])
            labels += [cls]*self.n_prototypes_per_class
        self.prototypes_ = np.vstack(protos)
        self.prototype_labels_ = np.array(labels)

    def fit(self, X, y):
        self._initialize_prototypes(X, y)
        for epoch in range(self.n_epochs):
            Xb, yb = shuffle(X, y, random_state=self.random_state+epoch)
            for i in range(0, len(Xb), self.batch_size):
                xb, yb_i = Xb[i:i+self.batch_size], yb[i:i+self.batch_size]
                for x_i, y_i in zip(xb, yb_i):
                    dists = np.linalg.norm(self.prototypes_ - x_i, axis=1)
                    w = np.argmin(dists)
                    sign = 1 if self.prototype_labels_[w] == y_i else -1
                    self.prototypes_[w] += sign * self.learning_rate * (x_i - self.prototypes_[w])
        return self

    def predict(self, X):
        d = np.linalg.norm(self.prototypes_[None,:,:] - X[:,None,:], axis=2)
        winners = np.argmin(d, axis=1)
        return self.prototype_labels_[winners]

# === 2. Grades de hiperparâmetros ===
param_grids = {
    "Decision Tree":    {"max_depth":[None,10,20], "min_samples_split":[2,5]},
    "Random Forest":    {"n_estimators":[50,100],  "max_depth":[None,10], "min_samples_split":[2,5]},
    "SVM":              {"C":[0.1,1],           "kernel":["linear","rbf"]},
    "MLP":              {"hidden_layer_sizes":[(10,),(50,)], "alpha":[1e-4,1e-3], "max_iter":[500,1000]},
    "KNN":              {"n_neighbors":[3,5,7], "weights":["uniform","distance"]},
    "LVQ":              {"n_prototypes_per_class":[2,5,10], "learning_rate":[1e-3,1e-2,1e-1], "n_epochs":[500,1000]}
}

# === 3. Métricas e estatística ===
metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score
}

def compute_cd(k, N):
    q = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850}
    return q[k] * math.sqrt((k*(k+1))/(6.0*N))

def evaluate_fold(ti, vi, models, X, y, metric, name):
    scores = []
    for m, clf in models.items():
        clf.fit(X[ti], y[ti])
        yp = clf.predict(X[vi])
        proba = None
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X[vi])[:,1]
        if name=="roc_auc" and proba is not None:
            s = metric(y[vi], proba)
        elif name in ("precision","recall","f1"):
            s = metric(y[vi], yp, zero_division=0)
        else:
            s = metric(y[vi], yp)
        scores.append(s)
    return scores

def run_statistical_comparison(X, y, models, project="ml-comparisons", n_splits=10):
    names = list(models.keys())
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for name, metric in metrics.items():
        wandb.init(project=project, name=f"CD3-{name}", reinit=True)
        folds = Parallel(n_jobs=-1)(
            delayed(evaluate_fold)(ti,vi,models,X,y,metric,name)
            for ti,vi in kf.split(X,y)
        )
        F = np.array(folds)
        ranks = np.array([ rankdata(-f, method="average") for f in F ])
        avg_r = ranks.mean(axis=0)

        stat, p = friedmanchisquare(*[F[:,i] for i in range(F.shape[1])])
        wandb.log({f"{name}/friedman_stat":stat, f"{name}/p_value":p})

        summary = []
        for i,n in enumerate(names):
            mu, sd = F[:,i].mean(), F[:,i].std()
            summary.append([n, mu, sd])
            wandb.log({f"{n}/{name}_mean":mu, f"{n}/{name}_std":sd})
        wandb.log({f"{name}/summary":wandb.Table(data=summary, columns=["Model","Mean","Std"])})

        if p<0.05:
            post = sp.posthoc_nemenyi_friedman(F)
            post.columns = post.index = names
            wandb.log({f"{name}/nemenyi": wandb.Table(data=post.values.tolist(), columns=names)})
            cd = compute_cd(len(names), n_splits)
            wandb.log({f"{name}/CD": cd})
            wandb.log({
                f"{name}/avg_ranks": wandb.Table(
                    data=[[names[i], float(avg_r[i])] for i in range(len(names))],
                    columns=["Model", "AvgRank"]
                )
            })
        wandb.finish()


# === 4. MAIN ===
if __name__ == "__main__":
    with open(r"adult_income.pkl", "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)

    print("Train+Val:", X_train.shape, y_train.shape)
    print("Test:     ", X_test.shape, y_test.shape)

    base_models = {
        "Decision Tree":    DecisionTreeClassifier(),
        "Random Forest":    RandomForestClassifier(),
        "SVM":              SVC(probability=True),
        "MLP":              MLPClassifier(),
        "KNN":              KNeighborsClassifier(),
        "LVQ":              CustomLVQ()
    }

    best_models, best_scores = {}, {}
    cv5 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for name, model in base_models.items():
        print(f"Tuning {name}…")
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring="f1",
            cv=cv5,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        best_scores[name] = grid.best_score_
        print(f"→ {name} best_params={grid.best_params_} f1={grid.best_score_:.4f}")

    run_statistical_comparison(X_train, y_train, best_models,
                               project="ml-comparisons", n_splits=10)

    best_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_name]
    print(f"\nMelhor modelo (F1 CV): {best_name}")

    y_pred = best_model.predict(X_test)
    y_proba = (best_model.predict_proba(X_test)[:, 1]
               if hasattr(best_model, "predict_proba") else None)

    print("\n=== Avaliação no Conjunto de Teste ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
    print("F1       :", f1_score(y_test, y_pred, zero_division=0))
    if y_proba is not None:
        print("ROC AUC  :", roc_auc_score(y_test, y_proba))
