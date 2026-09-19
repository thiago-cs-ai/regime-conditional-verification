"""Probe implementations and held-out hyperparameter selection."""

from collections import namedtuple
from itertools import product

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LinearProbe:
    scores_are_continuous = True
    needs_validation = False
    selection = None

    def __init__(self, random_state=None, C=1.0, max_iter=1000,
                 l1_ratio=0.0, dual=False, tol=1e-4, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, solver="lbfgs", with_mean=True, with_std=True):
        self._pipeline = Pipeline([
            ("scale_invariance", StandardScaler(with_mean=with_mean, with_std=with_std)),
            ("logistic", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state,
                                            l1_ratio=l1_ratio, dual=dual, tol=tol,
                                            fit_intercept=fit_intercept,
                                            intercept_scaling=intercept_scaling,
                                            class_weight=class_weight, solver=solver)),
        ])

    def fit(self, representation, agreement):
        self._pipeline.fit(np.asarray(representation, dtype=np.float64), agreement)
        return self

    def logit_scores(self, representation):
        return self._pipeline.decision_function(np.asarray(representation, dtype=np.float64))


class MLPProbe:
    scores_are_continuous = True
    needs_validation = False
    selection = None

    def __init__(self, random_state=None, hidden_layer_sizes=(16,), max_iter=500,
                 activation="relu", solver="adam", alpha=1e-4, batch_size="auto",
                 learning_rate="constant", learning_rate_init=1e-3, power_t=0.5, shuffle=True,
                 tol=1e-4, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 n_iter_no_change=10, max_fun=15000, with_mean=True, with_std=True):
        if activation != "relu":
            raise ValueError(f"MLP logit extraction supports activation='relu' only; got "
                             f"{activation!r}.")
        self._pipeline = Pipeline([
            ("scale_invariance", StandardScaler(with_mean=with_mean, with_std=with_std)),
            ("mlp", MLPClassifier(hidden_layer_sizes=tuple(hidden_layer_sizes),
                                  activation=activation, max_iter=max_iter,
                                  random_state=random_state, solver=solver, alpha=alpha,
                                  batch_size=batch_size, learning_rate=learning_rate,
                                  learning_rate_init=learning_rate_init, power_t=power_t,
                                  shuffle=shuffle, tol=tol, momentum=momentum,
                                  nesterovs_momentum=nesterovs_momentum,
                                  early_stopping=early_stopping,
                                  validation_fraction=validation_fraction, beta_1=beta_1,
                                  beta_2=beta_2, epsilon=epsilon,
                                  n_iter_no_change=n_iter_no_change, max_fun=max_fun)),
        ])

    def fit(self, representation, agreement):
        self._pipeline.fit(np.asarray(representation, dtype=np.float64), agreement)
        return self

    def logit_scores(self, representation):
        """Return float64 pre-sigmoid scores from the fitted ReLU network."""
        scaled = self._pipeline.named_steps["scale_invariance"].transform(
            np.asarray(representation, dtype=np.float64))
        mlp = self._pipeline.named_steps["mlp"]
        activation = scaled
        for layer in range(mlp.n_layers_ - 2):
            activation = np.maximum(activation @ mlp.coefs_[layer] + mlp.intercepts_[layer],
                                    0.0)
        return (activation @ mlp.coefs_[-1] + mlp.intercepts_[-1]).ravel().astype(np.float64)


class GradientBoostedProbe:
    scores_are_continuous = False
    needs_validation = False
    selection = None

    def __init__(self, random_state=None, max_iter=100, loss="log_loss", learning_rate=0.1,
                 max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0,
                 max_features=1.0, max_bins=255, categorical_features="from_dtype",
                 monotonic_cst=None, interaction_cst=None, early_stopping=False,
                 scoring="loss", validation_fraction=0.1, n_iter_no_change=10, tol=1e-7,
                 class_weight=None):
        self._model = HistGradientBoostingClassifier(
            max_iter=max_iter, random_state=random_state, loss=loss,
            learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
            max_features=max_features, max_bins=max_bins,
            categorical_features=categorical_features, monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst, early_stopping=early_stopping, scoring=scoring,
            validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change,
            tol=tol, class_weight=class_weight)

    def fit(self, representation, agreement):
        self._model.fit(np.asarray(representation, dtype=np.float64), agreement)
        return self

    def logit_scores(self, representation):
        return np.asarray(self._model.decision_function(
            np.asarray(representation, dtype=np.float64)), dtype=np.float64)


PROBE_FAMILIES = {"linear": LinearProbe, "mlp": MLPProbe,
                  "gradient_boosted": GradientBoostedProbe}

SELECTION_KEY = "select"

SelectableParameter = namedtuple("SelectableParameter",
                                 ("weakest_regularisation", "as_candidate", "as_recorded"))


def unchanged(value):
    return value


SELECTABLE_PARAMETERS = {
    "linear": {"C": SelectableParameter(max, unchanged, unchanged)},
    "mlp": {"alpha": SelectableParameter(min, unchanged, unchanged),
            "hidden_layer_sizes": SelectableParameter(max, tuple, list)},
}


def _what_is_authorised(family):
    parameters = SELECTABLE_PARAMETERS.get(family, {})
    if not parameters:
        return f"No selectable parameters are registered for {family!r}."
    return (f"Selectable parameters for {family!r}: "
            f"{', '.join(repr(name) for name in parameters)}.")


def assert_probe_spec(spec):
    """Validate an optional hyperparameter-selection grid."""
    if SELECTION_KEY not in spec:
        return spec
    grid_block = spec[SELECTION_KEY]
    if not isinstance(grid_block, dict):
        raise ValueError(f"Probe {SELECTION_KEY!r} block must be a mapping; got "
                         f"{type(grid_block).__name__}.")
    family = spec.get("family")
    parameters = SELECTABLE_PARAMETERS.get(family, {})
    for parameter in sorted(grid_block):
        if parameter not in parameters:
            raise ValueError(f"Probe {SELECTION_KEY!r} block for {family!r} names unsupported "
                             f"parameter {parameter!r}. {_what_is_authorised(family)}")
    if set(grid_block) != set(parameters):
        raise ValueError(f"Probe {SELECTION_KEY!r} block for {family!r} must name exactly "
                         f"{list(parameters)}; got {sorted(grid_block)}.")
    for parameter, selectable in parameters.items():
        if parameter in spec:
            raise ValueError(f"Probe specification names {parameter!r} directly and in its "
                             f"{SELECTION_KEY!r} block.")
        candidates = grid_block[parameter]
        if not isinstance(candidates, list):
            raise ValueError(f"Probe {SELECTION_KEY!r} grid for {parameter!r} must be a list; "
                             f"got {type(candidates).__name__}.")
        if not candidates:
            raise ValueError(f"Probe {SELECTION_KEY!r} grid for {parameter!r} is empty.")
        canonical = _canonical_candidates(family, parameter, selectable, candidates)
        repeated = sorted({value for value in canonical if canonical.count(value) > 1})
        if repeated:
            raise ValueError(f"Probe {SELECTION_KEY!r} grid for {parameter!r} repeats "
                             f"{repeated}.")
    return spec


def _canonical_candidates(family, parameter, selectable, candidates):
    try:
        return [selectable.as_candidate(value) for value in candidates]
    except TypeError as malformed:
        raise ValueError(f"Probe {SELECTION_KEY!r} grid for {family!r}.{parameter} has an "
                         f"invalid candidate: {malformed}") from malformed


class SelectingProbe:
    """Select a probe configuration by held-out AUROC."""

    needs_validation = True

    def __init__(self, family, grid_block, base_spec, random_state=None):
        self._family = family
        self._table = SELECTABLE_PARAMETERS[family]
        self._parameters = tuple(self._table)
        self._grid = tuple(
            tuple(self._table[parameter].as_candidate(value)
                  for value in grid_block[parameter])
            for parameter in self._parameters)
        self._base_spec = dict(base_spec)
        self._random_state = random_state
        self.scores_are_continuous = PROBE_FAMILIES[family].scores_are_continuous

    def fit(self, representation, agreement, held_out_representation, held_out_agreement):
        scored = {}
        fitted = {}
        for candidate in product(*self._grid):
            probe = PROBE_FAMILIES[self._family](
                random_state=self._random_state,
                **{**self._base_spec, **dict(zip(self._parameters, candidate))})
            probe.fit(representation, agreement)
            scored[candidate] = float(roc_auc_score(
                held_out_agreement, probe.logit_scores(held_out_representation)))
            fitted[candidate] = probe
        best = max(scored.values())
        chosen = self._weakest_of([candidate for candidate, auroc in scored.items()
                                   if auroc == best])
        self.held_out_auroc = scored
        self.selection = [{"parameter": parameter,
                           "value": self._table[parameter].as_recorded(value)}
                          for parameter, value in zip(self._parameters, chosen)]
        self._probe = fitted[chosen]
        return self

    def _weakest_of(self, tied):
        """Break AUROC ties toward the weakest registered regularization."""
        for position, parameter in enumerate(self._parameters):
            weakest = self._table[parameter].weakest_regularisation(
                candidate[position] for candidate in tied)
            tied = [candidate for candidate in tied if candidate[position] == weakest]
        return tied[0]

    def logit_scores(self, representation):
        return self._probe.logit_scores(representation)


def build_probe(spec, random_state=None):
    """Build a fixed or held-out-selected probe from a specification."""
    spec = dict(assert_probe_spec(spec))
    grid_block = spec.pop(SELECTION_KEY, None)
    family = spec.pop("family")
    if family not in PROBE_FAMILIES:
        raise ValueError(f"Unknown probe family {family!r}; available families: "
                         f"{sorted(PROBE_FAMILIES)}.")
    if grid_block is None:
        return PROBE_FAMILIES[family](random_state=random_state, **spec)
    PROBE_FAMILIES[family](**spec)
    return SelectingProbe(family, grid_block, spec, random_state=random_state)
