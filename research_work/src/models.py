"""
================================================================================
Model Hierarchy & Factory
================================================================================
Defines the standardized 9-model hierarchy for disease prediction benchmarking:
1. Dummy Classifier (stratified prior floor)
2. Logistic Regression (multinomial regularized linear)
3. Naive Bayes (Bernoulli NB for binary symptom indicators)
4. Decision Tree (CART tree benchmark)
5. Random Forest (bagging ensemble)
6. Extra Trees (extremely randomized trees)
7. SVM (Support Vector Machine with probability scaling)
8. XGBoost (Extreme Gradient Boosted trees)
9. MLP (Multi-Layer Perceptron neural network: 128 -> 64 -> output)
================================================================================
"""

from typing import Dict, Any, Callable
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from experiment_config import DEFAULT_CONFIG


def get_model_registry(seed: int = DEFAULT_CONFIG.random_seed) -> Dict[str, Any]:
    """
    Returns an instantiated dictionary of all 9 benchmark models configured
    with fixed random seeds for strict scientific reproducibility.
    """
    return {
        "Dummy Classifier": DummyClassifier(strategy="stratified", random_state=seed),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
            n_jobs=-1,
        ),
        "Naive Bayes": BernoulliNB(alpha=1.0),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=seed,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        ),
    }
