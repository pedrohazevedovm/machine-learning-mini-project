import pickle
import numpy as np
import wandb
import math

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
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

# === 1. IMPLEMENTAÇÃO CUSTOM LVQ ===
class CustomLVQ(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_prototypes_per_class=5,
        learning_rate=0.01,
        n_epochs=1000,
        batch_size=32,
        random_state=42
    ):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _initialize_prototypes(self, X, y):
        np.random.seed(self.random_state)
        classes = np.unique(y)
        protos, labels = [], []
        for cls in classes:
            Xc = X[y == cls]
            idx = np.random.choice(len(Xc), self.n_prototypes_per_class, replace=True)
            protos.append(Xc[idx])
            labels += [cls] * self.n_prototypes_per_class
        self.prototypes_ = np.vstack(protos)
        self.prototype_labels_ = np.array(labels)

    def fit(self, X, y):
        self._initialize_prototypes(X, y)
        for epoch in range(self.n_epochs):
            Xb, yb = shuffle(X, y, random_state=self.random_state + epoch)
            for i in range(0, len(Xb), self.batch_size):
                xb, ybi = Xb[i:i+self.batch_size], yb[i:i+self.batch_size]
                self._update_batch(xb, ybi)
        return self

    def _update_batch(self, X, y):
        for x_i, y_i in zip(X, y):
            dists = np.linalg.norm(self.prototypes_ - x_i, axis=1)
            w = np.argmin(dists)
            direction = 1 if self.prototype_labels_[w] == y_i else -1
            self.prototypes_[w] += direction * self.learning_rate * (x_i - self.prototypes_[w])

    def predict(self, X):
        d = np.linalg.norm(
            self.prototypes_[np.newaxis,:,:] - X[:,np.newaxis,:],
            axis=2
        )
        winners = np.argmin(d, axis=1)
        return self.prototype_labels_[winners]


# === 2. PARÂMETROS PARA GRID-SEARCH ===
param_grids = {
    "Decision Tree": {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "MLP": {
        "hidden_layer_sizes": [(10,), (50,)],
        "alpha": [0.0001, 0.001],
        "max_iter": [500, 1000]
    },
    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },
    "LVQ": {
        "n_prototypes_per_class": [2, 5, 10],
        "learning_rate": [0.001, 0.01, 0.1],
        "n_epochs": [500, 1000]
    }
}

# === 3. MÉTRICAS E FUNÇÕES AUXILIARES ===
metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score
}

def compute_cd(k, N, alpha=0.05):
    q = {2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,9:3.102,10:3.164}
    if k not in q:
        raise ValueError(f"q_alpha não definido para {k}")
    return q[k] * math.sqrt((k*(k+1)) / (6.0 * N))

def evaluate_fold(train_idx, test_idx, models, X, y, metric, name):
    scores = []
    for m in models:
        clf = models[m]
        clf.fit(X[train_idx], y[train_idx])
        yp = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx])[:,1] if hasattr(clf, "predict_proba") else None

        if name == "roc_auc" and proba is not None:
            s = metric(y[test_idx], proba)
        elif name in ["precision","recall","f1"]:
            s = metric(y[test_idx], yp, zero_division=0)
        else:
            s = metric(y[test_idx], yp)
        scores.append(s)
    return scores

def run_pipeline(X, y, models, project_name="ml-comparisons", n_splits=10, n_jobs=-1):
    model_names = list(models.keys())
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, metric in metrics.items():
        wandb.init(project=project_name, name=f"CD-{name}", reinit=True)

        folds = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_fold)(ti, vi, models, X, y, metric, name)
            for ti, vi in kf.split(X, y)
        )
        F = np.array(folds)
        ranks = np.array([rankdata(-f, method="average") for f in F])
        avg_ranks = np.mean(ranks, axis=0)

        stat, p = friedmanchisquare(*[F[:,i] for i in range(F.shape[1])])
        wandb.log({f"{name}/friedman_stat": stat, f"{name}/p_value": p})

        summary = []
        for i, m in enumerate(model_names):
            mu, sd = F[:,i].mean(), F[:,i].std()
            wandb.log({f"{m}/{name}_mean": mu, f"{m}/{name}_std": sd})
            summary.append([m, mu, sd])
        wandb.log({f"{name}/summary": wandb.Table(data=summary, columns=["Model","Mean","Std"])})

        if p < 0.05:
            post = sp.posthoc_nemenyi_friedman(F)
            post.columns = post.index = model_names
            wandb.log({f"{name}/nemenyi": wandb.Table(data=post.values.tolist(), columns=model_names)})
            cd = compute_cd(len(models), n_splits)
            wandb.log({f"{name}/CD": cd})
            wandb.log({
                f"{name}/avg_ranks": wandb.Table(
                    data=[[model_names[i], float(avg_ranks[i])] for i in range(len(models))],
                    columns=["Model","AvgRank"]
                )
            })

        wandb.finish()


# === 4. MAIN ===
if __name__ == "__main__":
    with open(r"machine_learning_mini_project\adult_income.pkl", "rb") as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    base_models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "MLP": MLPClassifier(),
        "KNN": KNeighborsClassifier(),
        "LVQ": CustomLVQ()
    }

    best_models = {}
    for name, model in base_models.items():
        print(f"Tuning {name}...")
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring="f1",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1
        )
        grid.fit(X, y)
        print(f"→ {name} best_params: {grid.best_params_} (f1={grid.best_score_:.4f})")
        best_models[name] = grid.best_estimator_

    run_pipeline(X, y, best_models, project_name="ml-comparisons", n_splits=10, n_jobs=-1)
