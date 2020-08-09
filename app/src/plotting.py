import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, ttest_ind
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_decisions(decisions, posteriors, y, class_names):
    num_trials, num_bins, num_classes = posteriors.shape

    xticks = np.arange(num_bins // 10) * 10
    xticklabels = [f"{i*10:4d}" for i in xticks]
    fig, axes = plt.subplots(num_trials, 1, sharey=True)
    fig.set_size_inches(16, 12)
    fig.suptitle(f"Posteriors Over Time (true class in green)")

    for trial_num in range(num_trials):
        ax = axes[trial_num]
        correct_class = y[trial_num]
        decision, decision_time = decisions[trial_num]
        im = ax.imshow(posteriors[trial_num].T, vmin=0, vmax=1, aspect="auto")

        ax.axhspan(
            ymin=correct_class - 0.5,
            ymax=correct_class + 0.5,
            xmin=0,
            xmax=1 / num_bins,
            color=(0, 1, 0),
        )
        if decision > 0:
            ax.axhspan(
                ymin=decision - 0.5,
                ymax=decision + 0.5,
                xmin=decision_time / num_bins,
                xmax=1,
                edgecolor="r" if decision != correct_class else (0, 1, 0),
                facecolor=(0, 0, 0, 0),
                lw=2,
            )
        if decision != correct_class:
            ax.axhspan(
                ymin=decision - 0.5,
                ymax=decision + 0.5,
                xmin=0,
                xmax=1 / num_bins,
                color="r",
            )

        ax.set_yticks(np.arange(num_classes))
        ax.set_yticklabels(class_names)

        if trial_num != num_trials - 1:
            ax.set_xticks([])
            continue

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Time since stimulus onset (ms)")

    fig.show()


def plot_confusion(perf, class_names, title="Performance", norm_range=(0, 1)):
    fig = plt.figure(figsize=(12, 5))
    (ax1, ax2) = fig.subplots(1, 2, sharex=True, sharey=True)
    fig.set_tight_layout(True)

    ax1.set_title(title)
    ConfusionMatrixDisplay(perf, display_labels=class_names).plot(ax=ax1)

    perf = perf / perf.sum(axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(perf, display_labels=class_names).plot(ax=ax2)
    if norm_range is not None:
        vmin, vmax = norm_range
        disp.im_.colorbar.remove()
        disp.im_ = ax2.imshow(perf, vmin=vmin, vmax=vmax)
        fig.colorbar(disp.im_, ax=ax2)
        ax2.set_title(f"{title} (normalized by true label)")
        ax2.set_ylabel("")
        ax2.set_yticklabels([])

    fig.show()


def plot_confusion_by_correct_choice(decisions, labels, mouse_correct, class_names):
    correct_perf = confusion_matrix(labels[mouse_correct], decisions[mouse_correct, 0])
    incorrect_perf = confusion_matrix(
        labels[~mouse_correct], decisions[~mouse_correct, 0]
    )

    data = [
        (correct_perf + incorrect_perf, "Overall"),
        (correct_perf, "Correct Choice"),
        (incorrect_perf, "Incorrect Choice"),
    ]
    for i, (perf, title) in enumerate(data):
        plot_confusion(perf, class_names, title=title)


def _plot_hist(ax, data, class_name):
    if len(data) == 0:
        return
    times = data * 10
    ax.hist(times)
    mean = times.mean()
    ax.axvline(x=mean, color="r", linestyle="--")
    median = np.median(times)
    ax.axvline(x=median, color="g", linestyle="--")
    ax.set_title(f"{class_name} - Mean {int(mean)}ms - Median {int(median)}ms")


def plot_decision_metrics(
    decisions,
    posteriors,
    selector,
    trials,
    labels,
    class_names,
    num_timelines=20,
    hist_xlabel="after stimulus presentation",
    regress=None,
):
    num_classes = len(class_names)
    labels = labels[trials]

    if num_timelines > 0:
        plot_decisions(
            decisions[:num_timelines],
            posteriors[:num_timelines],
            labels[:num_timelines],
            class_names,
        )

    mouse_correct = selector["CHOICE_CORRECT"][trials]
    plot_confusion_by_correct_choice(decisions, labels, mouse_correct, class_names)

    fig = plt.figure(figsize=(16, (3 * num_classes) - 1))
    axes = fig.subplots(3, num_classes - 1, sharey=True, sharex=True)
    fig.suptitle("Decision Times (mean in red, median in green)")
    if num_classes == 2:
        axes = axes[:, np.newaxis]

    decision_times = decisions[:, 1] - (regress if regress is not None else 0)
    for i, class_name in enumerate(class_names):
        if i == 0:
            continue
        ax = axes[:, i - 1]
        decoder_classes = decisions[:, 0] == i
        true_classes = labels == i
        _plot_hist(ax[0], decision_times[decoder_classes], f"{class_name} \nOverall")
        _plot_hist(ax[1], decision_times[decoder_classes & true_classes], f"Correct")
        _plot_hist(ax[2], decision_times[decoder_classes & ~true_classes], f"Incorrect")

        ax[0].set_ylabel("Number of trials")
        ax[-1].set_xlabel(f"Time {hist_xlabel} (ms)")

    fig.set_tight_layout(True)
    fig.show()

    if regress is None:
        return

    is_decoding_correct = (regress > 0) & (decisions[:, 0] > 0)
    correct_decisions = decisions[is_decoding_correct, 1]
    regress = regress[is_decoding_correct]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots()
    ax.scatter(regress, correct_decisions)
    ax.set_ylabel("Decoder decision time (ms)")
    ax.set_xlabel("Animal reaction time (ms)")

    slope, intercept, r, pvalue, stderr = linregress(correct_decisions, regress)
    linreg_x = np.arange(0, correct_decisions.max())
    ax.plot(linreg_x, slope * linreg_x + intercept)
    ax.set_title(
        f"Decoder decision time vs animal reaction time ($R^2$={r**2:0.2f}, p={pvalue:0.2e})"
    )
    fig.show()


def plot_boxplot(data, labels, title):
    fig = plt.figure(figsize=(16, 5))
    ax = fig.subplots()
    ax.boxplot(np.array(data, dtype=object), labels=labels)
    ax.set_title(f"Deliberation time {title}")
    ax.set_ylabel("Deliberation time (ms)")
    fig.show()

    if len(data) == 2:
        print(ttest_ind(*data, equal_var=False))
