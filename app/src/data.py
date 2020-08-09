import numpy as np
import os
import pickle
import requests
import tempfile
from pathlib import Path
from scipy import signal


SESSION_FILE_NAME = "steinmetz_part"


def download_sessions(data_dir):
    if type(data_dir) is str:
        data_dir = Path(data_dir)

    urls = [
        "https://osf.io/agvxh/download",
        "https://osf.io/uv3mw/download",
        "https://osf.io/ehmw2/download",
    ]
    for i, url in enumerate(urls):
        file_path = data_dir / f"{SESSION_FILE_NAME}{i}.npz"
        if os.path.isfile(file_path):
            continue

        print(f"Downloading {file_path}...")
        r = requests.get(url)
        if r.status_code != requests.codes.ok:
            raise requests.ConnectionError()
        with open(file_path, "wb") as f:
            f.write(r.content)


def load_sessions(data_dir=None, cleanup=True):
    tmp_dir = None
    if data_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()
        data_dir = Path(tmp_dir.name)

    if type(data_dir) is str:
        data_dir = Path(data_dir)

    file_glob = f"{SESSION_FILE_NAME}*.npz"
    files = sorted(data_dir.glob(file_glob))
    if len(files) == 0:
        download_sessions(data_dir)
        files = sorted(data_dir.glob(file_glob))

    sessions = [None] * len(files)
    for i, f in enumerate(files):
        print(f"Loading {f}...")
        sessions[i] = np.load(data_dir / f, allow_pickle=True)["dat"]

    if tmp_dir is not None and cleanup is True:
        tmp_dir.cleanup()

    return np.hstack(sessions)


def get_spikes(
    session, neurons, trials, bins, align=50, baseline_bins=None, smoothing=None
):
    all_spikes = session["spks"][neurons][:, trials]

    if smoothing is not None:
        pad = smoothing[0] // 2
        bins = np.array(bins) + [-pad, pad]

    bins = np.atleast_1d(align)[:, np.newaxis] + np.arange(*bins)
    bins += np.minimum(all_spikes.shape[2] - bins[:, -1] - 1, 0)[:, np.newaxis]
    bins -= np.minimum(bins[:, 0], 0)[:, np.newaxis]
    bins = bins[np.newaxis, :, :]

    spikes = np.take_along_axis(all_spikes, bins, 2)

    if baseline_bins is not None:
        start, end = baseline_bins
        baseline = all_spikes[:, :, start:end]
        baseline = baseline.mean(axis=2, keepdims=True)
        spikes = (spikes - baseline) / (baseline + 0.5)

    if smoothing is not None:
        size, std = smoothing
        half_gaussian = signal.gaussian(size, std)
        half_gaussian[size // 2 + 1 :] = 0
        spikes = signal.convolve(
            spikes,
            half_gaussian[np.newaxis, np.newaxis, :],
            mode="valid",
            method="direct",
        )

    return spikes


def reshape_by_bins(spikes):
    return spikes.transpose((1, 2, 0)).reshape(
        spikes.shape[1] * spikes.shape[2], spikes.shape[0]
    )


def save_decoder_results(
    data_dir, session_number, trials_selector, area, decisions, decoder,
):
    results = {}
    results_file = data_dir / "decoder_results.pickle"
    if results_file.exists():
        with open(results_file, "rb") as f:
            results = pickle.load(f)

    if session_number not in results:
        results[session_number] = {}

    results[session_number][area] = {
        "trial_numbers": np.where(trials_selector)[0],
        "decision_times": decisions[:, 1] * 10,
        "decisions": decisions[:, 0],
        "decoder": decoder,
    }

    with open(results_file, "wb") as f:
        pickle.dump(results, f)
