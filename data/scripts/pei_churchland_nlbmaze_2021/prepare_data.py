"""Load data, processes it, save it."""

import argparse
import datetime
import logging

import numpy as np
from pynwb import NWBHDF5IO

from mojo_kirby.data import Data, IrregularTimeSeries, Interval, DatasetBuilder
from mojo_kirby.data.dandi_utils import (
    extract_subject_from_nwb,
)
from mojo_kirby.taxonomy.task import REACHING
from mojo_kirby.utils import find_files_by_extension
from mojo_kirby.taxonomy import (
    RecordingTech,
    Task,
)
import pandas as pd
from mojo_kirby.data import ArrayDict

logging.basicConfig(level=logging.INFO)


def extract_spikes_from_nwbfile_mcmaze(nwbfile, recording_tech):
    # spikes
    timestamps = []
    unit_index = []

    # units
    unit_meta = []

    units = nwbfile.units.spike_times_index[:]
    electrodes = nwbfile.units.electrodes.table

    # all these units are obtained using threshold crossings
    for i in range(len(units)):
        if recording_tech == RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS:
            # label unit
            group_name = electrodes["group_name"][i]
            unit_id = f"group_{group_name}/elec{i}/multiunit_{0}"
        elif recording_tech == RecordingTech.UTAH_ARRAY_SPIKES:
            # label unit
            electrode_id = nwbfile.units[i].electrodes.item().item()
            unit_identifier = str(nwbfile.units[i].index.item())
            if unit_identifier.startswith("1"):
                # unit in pmd
                pass
            elif unit_identifier.startswith("2"):
                # known dataset conversion issue with M1 electrode
                # https://neurallatents.github.io/datasets.html
                electrode_id = electrode_id + 96
            group_name = electrodes["group_name"][electrode_id]
            unit_id = f"group_{group_name}/elec{electrode_id}/unit_{i}"
        else:
            raise ValueError(f"Recording tech {recording_tech} not supported")

        # extract spikes
        spiketimes = units[i]
        timestamps.append(spiketimes)

        if len(spiketimes) > 0:
            unit_index.append([i] * len(spiketimes))

        # extract unit metadata
        unit_meta.append(
            {
                "id": unit_id,
                "unit_number": i,
                "count": len(spiketimes),
                "type": int(recording_tech),
            }
        )

    # convert unit metadata to a Data object
    unit_meta_df = pd.DataFrame(unit_meta)  # list of dicts to dataframe
    units = ArrayDict.from_dataframe(
        unit_meta_df,
        unsigned_to_long=True,
    )

    # concatenate spikes
    timestamps = np.concatenate(timestamps)
    unit_index = np.concatenate(unit_index)

    # create spikes object
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        domain="auto",
    )

    # make sure to sort ethe spikes
    spikes.sort()

    return spikes, units


def extract_trials(nwbfile):
    r"""Extract trial information from the NWB file."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
            "split": "split_indicator",
        }
    )
    trials = Interval.from_dataframe(trial_table)

    # the dataset has pre-defined train/valid splits, we will use the valid split
    train_mask_nwb = trial_table.split_indicator.to_numpy() == "train"
    test_mask_nwb = trial_table.split_indicator.to_numpy() == "val"

    trials.train_mask_nwb = train_mask_nwb
    trials.test_mask_nwb = test_mask_nwb

    return trials


def extract_behavior(nwbfile, trials):
    """Extract behavior from the NWB file."""
    # cursor, hand and eye share the same timestamps (verified)
    timestamps = nwbfile.processing["behavior"]["hand_vel"].timestamps[:]
    hand_pos = nwbfile.processing["behavior"]["hand_pos"].data[:]
    hand_vel = nwbfile.processing["behavior"]["hand_vel"].data[:]
    eye_pos = nwbfile.processing["behavior"]["eye_pos"].data[:]

    # normalization
    hand_vel = hand_vel / 100.0

    # create a behavior type segmentation mask
    behavior_type = np.ones_like(timestamps, dtype=np.int64) * int(REACHING.RANDOM)

    for i in range(len(trials)):
        behavior_type[
            (timestamps >= (trials.move_onset_time[i] - 0.15))
            & (timestamps < (trials.move_onset_time[i] + 0.55))
        ] = int(REACHING.EVAL)

    behavior = IrregularTimeSeries(
        timestamps=timestamps,
        pos=hand_pos,
        vel=hand_vel,
        eye_pos=eye_pos,
        subtask_index=behavior_type,
        domain="auto",
    )

    return behavior


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
        experiment_name="mc_maze",
        origin_version="dandi/000128/0.220113.0400",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000128",
        description="This dataset contains sorted unit spiking times and behavioral"
        " data from a macaque performing a delayed reaching task. The experimental task"
        " was a center-out reaching task with obstructing barriers forming a maze,"
        " resulting in a variety of straight and curved reaches.",
    )

    # iterate over the .nwb files and extract the data from each
    for file_path in find_files_by_extension(db.raw_folder_path, ".nwb"):
        if "test" in file_path:
            # test file does not have behavior, skipping
            continue

        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:
            # open file
            io = NWBHDF5IO(file_path, "r")
            nwbfile = io.read()

            subject = extract_subject_from_nwb(nwbfile)
            session.register_subject(subject)

            # extract experiment metadata
            recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
            subject_id = subject.id
            sortset_id = f"{subject_id}_{recording_date}"
            session_id = f"{sortset_id}_maze"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.REACHING,
            )

            # extract spiking activity
            spikes, units = extract_spikes_from_nwbfile_mcmaze(
                nwbfile,
                recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            # extract data about trial structure
            trials = extract_trials(nwbfile)

            # extract behavior
            behavior = extract_behavior(nwbfile, trials)

            # close file
            io.close()

            data = Data(
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                cursor=behavior,
                # domain
                domain=behavior.domain,
            )

            session.register_data(data)

            [
                train_trials,
                valid_trials,
                test_trials,
            ] = trials.split([0.8, 0.1, 0.1], shuffle=True, random_seed=42)

            session.register_split("train", train_trials)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
