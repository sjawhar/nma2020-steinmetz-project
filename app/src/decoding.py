import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from .data import get_spikes, reshape_by_bins
from .plotting import plot_confusion


def cross_validate(clf, X, y):
    trials = np.arange(X.shape[1])
    num_bins = X.shape[2]

    skf = StratifiedKFold(n_splits=5)
    y_pred = np.zeros((len(y) * num_bins))
    scores = []
    for train_index, val_index in skf.split(trials, y):
        X_train = reshape_by_bins(X[:, train_index, -num_bins:])
        X_val = reshape_by_bins(X[:, val_index, -num_bins:])
        y_train = np.repeat(y[train_index], num_bins)
        y_val = np.repeat(y[val_index], num_bins)

        split_clf = type(clf)(**clf.get_params())
        split_clf.fit(X_train, y_train)
        split_pred = split_clf.predict(X_val)
        scores.append(np.mean(split_pred == y_val))
        y_pred[
            (val_index[:, np.newaxis] * num_bins + np.arange(num_bins)).flatten()
        ] = split_pred

    return y_pred, scores


def cv_and_fit(
    clf,
    session,
    selector,
    neurons,
    trials,
    bins,
    labels,
    class_names,
    align=50,
    baseline_bins=None,
    smoothing=(17, 2.5),
    cv=True,
    fit=True,
):
    num_classes = len(class_names)

    spikes = get_spikes(
        session,
        neurons,
        trials,
        bins,
        align=align,
        baseline_bins=baseline_bins,
        smoothing=smoothing,
    )
    labels = labels[trials]

    confidence_threshold = None
    if cv:
        y_pred, scores = cross_validate(clf, spikes, labels)
        print(scores)
        print(np.mean(scores))
        confusion = confusion_matrix(np.repeat(labels, spikes.shape[2]), y_pred)
        plot_confusion(confusion, class_names, title="Cross-val perf by bin")
        confidence_threshold = (
            np.diag(confusion).astype(np.float) / confusion.sum(axis=0)
        )[1:]

    if fit:
        clf.fit(reshape_by_bins(spikes), np.repeat(labels, spikes.shape[2]))

    return confidence_threshold


def decode(clf, spikes, threshold):
    _, num_trials, num_bins = spikes.shape
    num_classes = len(np.atleast_1d(threshold)) + 1

    X = reshape_by_bins(spikes)
    decisions = np.zeros((num_trials, 2), np.int)
    posteriors = np.zeros((num_trials, num_bins, num_classes), np.float)
    eps = np.finfo(np.float).eps

    for trial_num in range(num_trials):
        population_activity = X[
            (trial_num * num_bins) : ((trial_num + 1) * num_bins), :
        ]
        likelihoods = clf.predict_log_proba(population_activity)

        posteriors[trial_num, 0] = likelihoods[0]
        for i in range(1, num_bins):
            prob = np.exp(posteriors[trial_num, i - 1] + likelihoods[i]) + eps
            posteriors[trial_num, i] = np.log(prob / prob.sum())
        posteriors[trial_num] = np.exp(posteriors[trial_num])

        decision = 0
        decided = np.argmax(posteriors[trial_num, :, 1:] > threshold, axis=0)
        if np.any(decided):
            decision = num_classes - np.argmax(decided[::-1] > 0) - 1
            decisions[trial_num] = [decision, decided[decision - 1]]

    return decisions, posteriors
