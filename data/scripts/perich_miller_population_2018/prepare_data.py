import argparse
import datetime
import logging

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.model_selection import train_test_split

from mojo_kirby.data import Data, IrregularTimeSeries, Interval, DatasetBuilder
from mojo_kirby.data.dandi_utils import extract_spikes_from_nwbfile, extract_subject_from_nwb
from mojo_kirby.utils import find_files_by_extension
from mojo_kirby.taxonomy import RecordingTech, Task

logging.basicConfig(level=logging.INFO)


def extract_behavior_and_trials(nwbfile, task):
    """Extract behavior from the NWB file."""
    # cursor, hand and eye share the same timestamps (verified)
    timestamps = nwbfile.processing["behavior"]["Position"]["cursor_pos"].timestamps[:]
    cursor_pos = nwbfile.processing["behavior"]["Position"]["cursor_pos"].data[:]  # 2d
    cursor_vel = nwbfile.processing["behavior"]["Velocity"]["cursor_vel"].data[:]
    cursor_acc = nwbfile.processing["behavior"]["Acceleration"]["cursor_acc"].data[:]

    # normalization
    cursor_vel = cursor_vel / 20.0

    hand_acc_norm = np.linalg.norm(cursor_acc, axis=1)
    mask_acceleration = hand_acc_norm > 1500.0
    mask_acceleration = binary_dilation(
        mask_acceleration, structure=np.ones(2, dtype=bool)
    )

    # we also want to identify out of bound segments
    mask_position = np.logical_or(cursor_pos[:, 0] < -10, cursor_pos[:, 0] > 10)
    mask_position = np.logical_or(mask_position, cursor_pos[:, 1] < -10)
    mask_position = np.logical_or(mask_position, cursor_pos[:, 1] > 10)
    # dilate than erode
    mask_position = binary_dilation(mask_position, np.ones(400, dtype=bool))
    mask_position = binary_erosion(mask_position, np.ones(100, dtype=bool))

    outlier_mask = np.logical_or(mask_acceleration, mask_position)

    cursor_domain_end = timestamps[-1]
    assert np.max(timestamps) == cursor_domain_end

    if task == "center_out_reaching":
        trials = extract_center_out_reaching_trials(nwbfile, cursor_domain_end)
    elif task == "random_target_reaching":
        trials = extract_random_target_reaching_trials(nwbfile)

    # create a behavior type segmentation mask
    subtask_index = np.ones_like(timestamps, dtype=np.int64) * int(Task.REACHING.RANDOM)

    if task == "center_out_reaching":
        for i in range(len(trials)):
            # first we check whether the trials are valid or not
            if trials.is_valid[i]:
                subtask_index[
                    (timestamps >= trials.target_on_time[i])
                    & (timestamps < trials.go_cue_time[i])
                ] = int(Task.REACHING.HOLD)
                subtask_index[
                    (timestamps >= trials.go_cue_time[i])
                    & (timestamps < trials.stop_time[i])
                ] = int(Task.REACHING.REACH)
                subtask_index[
                    (timestamps >= trials.stop_time[i]) & (timestamps < trials.end[i])
                ] = int(Task.REACHING.RETURN)
            else:
                subtask_index[
                    (timestamps >= trials.start[i]) & (timestamps < trials.end[i])
                ] = int(Task.REACHING.INVALID)

    elif task == "random_target_reaching":
        for i in range(len(trials)):
            if not np.isnan(trials.go_cue_time_array[i][0]):
                subtask_index[
                    (timestamps >= trials.start[i])
                    & (timestamps < trials.go_cue_time_array[i][0])
                ] = int(Task.REACHING.HOLD)

    # make sure outlier is set correctly
    subtask_index[outlier_mask] = int(Task.REACHING.OUTLIER)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel,
        acc=cursor_acc,
        subtask_index=subtask_index,
        domain="auto",
    )

    return cursor, trials


def extract_center_out_reaching_trials(nwbfile, cursor_domain_end):
    r"""Extract trial information from the NWB file."""
    trial_table = nwbfile.trials.to_dataframe()

    # infer the trial structure from the trial table
    trial_grid = np.append(
        trial_table.target_on_time.iloc[:],
        min(trial_table.stop_time.iloc[-1] + 1.0, cursor_domain_end),
    )
    default_value = np.append(
        trial_table.start_time.iloc[0], trial_table.stop_time.values + 1.0
    )
    default_value[1:-1] = np.minimum(
        default_value[1:-1], trial_table.stop_time.values[1:]
    )
    nan_mask = np.isnan(trial_grid)
    trial_grid[nan_mask] = default_value[nan_mask]
    trial_grid = trial_grid.astype(np.float64)
    trial_table["end"] = trial_grid[1:]
    trial_table["start"] = trial_grid[:-1]

    trials = Interval.from_dataframe(trial_table)
    assert trials.is_disjoint()

    # find valid trials
    success_mask = trials.result == "R"
    valid_target_mask = ~np.isnan(trials.target_id)
    max_duration_mask = (trials.end - trials.start) < 6.0
    min_duration_mask = (trials.end - trials.start) > 0.5

    trials.is_valid = (
        success_mask & valid_target_mask & max_duration_mask & min_duration_mask
    )

    return trials


def extract_random_target_reaching_trials(nwbfile):
    r"""Extract trial information from the NWB file."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )

    trials = Interval.from_dataframe(trial_table)

    # find valid trials
    success_mask = trials.result == "R"
    valid_num_attempts = trials.num_attempted == 4
    max_duration_mask = (trials.end - trials.start) < 10.0
    min_duration_mask = (trials.end - trials.start) > 2.0

    trials.is_valid = (
        success_mask & valid_num_attempts & max_duration_mask & min_duration_mask
    )

    return trials


def split_trials(trials, test_size=0.2, valid_size=0.1, random_state=42):
    num_trials = len(trials)
    train_size = 1.0 - test_size - valid_size

    train_valid_ids, test_ids = train_test_split(
        np.arange(num_trials),
        test_size=test_size,
        random_state=random_state,
        shuffle=False,
    )
    train_ids, valid_ids = train_test_split(
        train_valid_ids,
        test_size=valid_size / (train_size + valid_size),
        random_state=random_state,
        shuffle=False,
    )

    train_trials = trials.select_by_mask(np.isin(np.arange(num_trials), train_ids))
    valid_trials = trials.select_by_mask(np.isin(np.arange(num_trials), valid_ids))
    test_trials = trials.select_by_mask(np.isin(np.arange(num_trials), test_ids))

    return train_trials, valid_trials, test_trials


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    # intiantiate a DatasetBuilder which provides utilities for processing data
    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name="perich_miller_population_2018",
        origin_version="dandi/000688/draft",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000688",
        description="This dataset contains electrophysiology and behavioral data from "
        "three macaques performing either a center-out task or a continuous random "
        "target acquisition task. Neural activity was recorded from "
        "chronically-implanted electrode arrays in the primary motor cortex (M1) or "
        "dorsal premotor cortex (PMd) of four rhesus macaque monkeys. A subset of "
        "sessions includes recordings from both regions simultaneously. The data "
        "contains spiking activity—manually spike sorted in three subjects, and "
        "threshold crossings in the fourth subject—obtained from up to 192 electrodes "
        "per session, cursor position and velocity, and other task related metadata.",
    )

    # iterate over the .nwb files and extract the data from each
    for file_path in find_files_by_extension(db.raw_folder_path, ".nwb"):
        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:
            # open file
            io = NWBHDF5IO(file_path, "r")
            nwbfile = io.read()

            # extract subject metadata
            subject = extract_subject_from_nwb(nwbfile)
            session.register_subject(subject)

            # extract experiment metadata
            recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
            sortset_id = f"{subject.id}_{recording_date}"
            task = (
                "center_out_reaching" if "CO" in file_path else "random_target_reaching"
            )
            session_id = f"{sortset_id}_{task}"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.REACHING,
            )

            # extract spiking activity
            spikes, units = extract_spikes_from_nwbfile(
                nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            # extract behavior
            cursor, trials = extract_behavior_and_trials(nwbfile, task)

            # close file
            io.close()

            # register session
            data = Data(
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                cursor=cursor,
                # domain
                domain=cursor.domain,
            )

            session.register_data(data)

            # split trials into train, validation and test
            print(np.sum(~trials.is_valid), "is invalid out of", len(trials))
            successful_trials = trials.select_by_mask(trials.is_valid)
            _, valid_trials, test_trials = split_trials(
                successful_trials, test_size=0.2, valid_size=0.1, random_state=42
            )

            train_sampling_intervals = data.domain.difference(
                (valid_trials | test_trials).dilate(1.0)
            )

            session.register_split("train", train_sampling_intervals)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            assert len(train_sampling_intervals & valid_trials) == 0
            assert len(train_sampling_intervals & test_trials) == 0
            assert len(valid_trials & test_trials) == 0

            # trials object is added with train/valid/test_mask at this step
            print(
                np.sum(
                    trials.is_valid[
                        (
                            trials.train_mask.astype(int)
                            + trials.valid_mask.astype(int)
                            + trials.test_mask.astype(int)
                        )
                        != 1
                    ]
                ),
                "valid trials skipped",
            )
            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
