import numpy as np
import wandb
import tempfile
import os
import math

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn_lvq import GlvqModel

from scipy.stats import friedmanchisquare, rankdata
import scikit_posthocs as sp
from joblib import Parallel, delayed

# === MODELOS ===
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "MLP": MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "LVQ": GlvqModel()
}
model_names = list(models.keys())

# === MÉTRICAS ===
metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score
}


# === FUNÇÃO DE DIFERENÇA CRÍTICA ===
def compute_cd(k, N, alpha=0.05):
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    if k not in q_alpha_table:
        raise ValueError(f"q_alpha não definido para {k} algoritmos.")
    q_alpha = q_alpha_table[k]
    return q_alpha * math.sqrt((k * (k + 1)) / (6.0 * N))


# === FUNÇÃO DE AVALIAÇÃO DE UMA FOLD ===
def evaluate_fold(train_idx, test_idx, models, X, y, metric_func, metric_name):
    fold_scores = []
    for name in models:
        model = models[name]
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_proba = model.predict_proba(X[test_idx])[:, 1] if hasattr(model, "predict_proba") else None

        if metric_name == "roc_auc" and y_proba is not None:
            score = metric_func(y[test_idx], y_proba)
        else:
            if metric_name in ["precision", "recall", "f1"]:
                score = metric_func(y[test_idx], y_pred, zero_division=0)
            else:
                score = metric_func(y[test_idx], y_pred)

        fold_scores.append(score)
    return fold_scores


# === PIPELINE PRINCIPAL POR MÉTRICA ===
def run_pipeline(X, y, project_name="ml-comparisons", n_splits=10, n_jobs=-1):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for metric_name, metric_func in metrics.items():
        wandb.init(project=project_name, name=f"CD-{metric_name}", reinit=True)

        # Avaliação paralela das folds
        scores_per_fold = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_fold)(train_idx, test_idx, models, X, y, metric_func, metric_name)
            for train_idx, test_idx in kf.split(X, y)
        )
        scores_per_fold = np.array(scores_per_fold)  # shape: (folds, models)
        rankings = np.array([rankdata(-row, method='average') for row in scores_per_fold])
        avg_ranks = np.mean(rankings, axis=0)

        # Friedman
        stat, p = friedmanchisquare(*[scores_per_fold[:, i] for i in range(scores_per_fold.shape[1])])
        wandb.log({f"{metric_name}/friedman_statistic": stat, f"{metric_name}/p_value": p})

        # Média e desvio
        summary_data = []
        for i, name in enumerate(model_names):
            scores = scores_per_fold[:, i]
            mean = np.mean(scores)
            std = np.std(scores)
            wandb.log({
                f"{name}/{metric_name}_mean": mean,
                f"{name}/{metric_name}_std": std
            })
            summary_data.append([name, mean, std])

        wandb.log({
            f"{metric_name}/summary_table": wandb.Table(data=summary_data, columns=["Model", "Mean", "Std"])
        })

        # Se Friedman significativo → Nemenyi + CD Diagram
        if p < 0.05:
            nemenyi = sp.posthoc_nemenyi_friedman(scores_per_fold)
            nemenyi.columns = model_names
            nemenyi.index = model_names
            wandb.log({
                f"{metric_name}/nemenyi_pvalues": wandb.Table(data=nemenyi.values.tolist(), columns=model_names)
            })

            cd = compute_cd(len(models), n_splits)
            wandb.log({f"{metric_name}/CD": cd})

            # Logar rankings médios em uma tabela
            rank_table = [[model_names[i], float(avg_ranks[i])] for i in range(len(model_names))]
            wandb.log({
                f"{metric_name}/avg_ranks": wandb.Table(data=rank_table, columns=["Model", "Average Rank"])
            })

        wandb.finish()
