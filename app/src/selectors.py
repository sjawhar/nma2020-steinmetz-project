import numpy as np
from .brain_areas import AREAS_ACTION, AREAS_MOTOR, AREAS_VISUAL


def get_action_times(
    wheel,
    response_time,
    bins_before=20,
    bins_after=5,
    move_min=5,
    move_window=5,
    stim_time=50,
    stim_buffer=20,
):
    """
    DEPRECTATED: Use session['reaction_time'] instead

    Finds the action-aligned indices for each trial.

    Movement time is determined by taking the peak-to-peak wheel position in
    a sliding leading window to find the time bin initiating a large, sustained
    movement.

    Returns an array of size num_trials x (bin_before + bins_after).
    The returned indices are guaranteed to fit within the available sessiona, (i.e.
    movement time will not be before stimulus time or extend past the response
    time) but might not be perfectly aligned with actual movement if it starts
    too soon or too late.

    Use np.take_along_axis with spike sessiona and returned indices

    Arguments:
    wheel -- array of wheel positions for each trial

    Keyword Arguments:
    bins_before -- number of bins before first movement point to include
    bins_after -- number of bins after first movement point to include
    move_window -- sliding window size in which movement will be searched
    move_min -- minimum amount of peak-to-peak wheel movement required in window
    stim_time -- bin number of stimulus presentation
    stim_buffer -- number of mansessionory "no movement" bins before stim_time
    """
    num_bins = wheel.shape[2]
    windows = np.arange(stim_time, num_bins - move_window)[:, np.newaxis] + np.arange(
        move_window
    )
    is_moving = np.ptp(wheel[0][:, windows], axis=2) > move_min
    action_time = is_moving.argmax(axis=1) + stim_time

    # Fit within available window
    min_time = stim_time - stim_buffer
    action_time = np.maximum(action_time, min_time + bins_before)

    max_time = (response_time / 0.01).astype(np.int).flatten() + stim_time
    action_time = np.minimum(action_time, max_time - bins_after)

    return action_time[np.newaxis, :, np.newaxis] + np.arange(-bins_before, bins_after)


def get_selectors(session):
    no_movement = np.ptp(session["wheel"][0], 1) < 3
    selector = {
        "NEURON_VISUAL": np.isin(session["brain_area"], AREAS_VISUAL),
        "NEURON_MOTOR": np.isin(session["brain_area"], AREAS_MOTOR),
        "NEURON_ACTION": np.isin(session["brain_area"], AREAS_ACTION),
        "CHOICE_RIGHT": (session["response"] == -1) & (~no_movement),
        "CHOICE_LEFT": (session["response"] == 1) & (~no_movement),
        "CHOICE_NONE": (session["response"] == 0) | (no_movement),
        "STIM_RIGHT": session["contrast_right"] > session["contrast_left"],
        "STIM_LEFT": session["contrast_right"] < session["contrast_left"],
        "STIM_EQUAL": (session["contrast_right"] == session["contrast_left"])
        & (session["contrast_right"] > 0),
        "STIM_NONE": (session["contrast_right"] == session["contrast_left"])
        & (session["contrast_right"] == 0),
        "STIM_RIGHT_HIGH": session["contrast_right"] == 1,
        "STIM_RIGHT_MEDIUM": session["contrast_right"] == 0.5,
        "STIM_RIGHT_LOW": session["contrast_right"] == 0.25,
        "STIM_RIGHT_NONE": session["contrast_right"] == 0,
        "TIMES_ACTION": get_action_times(
            session["wheel"],
            session["response_time"],
            bins_before=20,
            bins_after=5,
            move_window=4,
            move_min=2,
            stim_time=50,
            stim_buffer=20,
        ),
    }
    selector.update(
        {
            "CHOICE_CORRECT": (
                (selector["STIM_RIGHT"] & selector["CHOICE_RIGHT"])
                | (selector["STIM_LEFT"] & selector["CHOICE_LEFT"])
                | (selector["STIM_NONE"] & selector["CHOICE_NONE"])
            ),
            "CHOICE_MISS": ~selector["STIM_NONE"] & selector["CHOICE_NONE"],
        }
    )
    return selector
