import argparse
import datetime
import logging
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import pandas as pd

from mojo_kirby.data import (
    Data,
    ArrayDict,
    IrregularTimeSeries,
    Interval,
    DatasetBuilder,
)
from mojo_kirby.taxonomy import (
    Macaque,
    RecordingTech,
    Species,
    SubjectDescription,
    Task,
    Channel,
    Probe,
)
from mojo_kirby.utils import find_files_by_extension

logging.basicConfig(level=logging.INFO)


SAMPLE_FREQUENCY = 24414.0625

channel_map = np.array(
    [
        [0, 0, -1000],
        [0, 2000, -1000],
        [1200, 800, -1000],
        [800, 1600, -1000],
        [400, 400, -1000],
        [-400, 2000, -1000],
        [800, 1200, -1000],
        [400, 2000, -1000],
        [0, 400, -1000],
        [2400, -1200, -1000],
        [1200, 1200, -1000],
        [2400, -800, -1000],
        [0, 800, -1000],
        [2000, -1200, -1000],
        [400, 800, -1000],
        [2000, -800, -1000],
        [0, 1200, -1000],
        [1600, -1200, -1000],
        [400, 1200, -1000],
        [1600, -800, -1000],
        [0, 1600, -1000],
        [1200, -1200, -1000],
        [400, 1600, -1000],
        [1200, -800, -1000],
        [-800, 0, -1000],
        [800, -1200, -1000],
        [-400, 0, -1000],
        [800, -800, -1000],
        [-400, 400, -1000],
        [400, -1200, -1000],
        [-800, 400, -1000],
        [400, -800, -1000],
        [-400, 800, -1000],
        [0, -1200, -1000],
        [-800, 800, -1000],
        [0, -800, -1000],
        [-400, 1200, -1000],
        [-400, -1200, -1000],
        [-800, 1200, -1000],
        [-400, -800, -1000],
        [-800, 1600, -1000],
        [-800, -800, -1000],
        [-400, 1600, -1000],
        [0, -400, -1000],
        [-400, 2400, -1000],
        [-800, -400, -1000],
        [-800, 2000, -1000],
        [-400, -400, -1000],
        [2800, -400, -1000],
        [2000, 1200, -1000],
        [2800, -800, -1000],
        [2800, 2000, -1000],
        [2800, 0, -1000],
        [1600, 800, -1000],
        [2800, 400, -1000],
        [2400, 2000, -1000],
        [2000, 400, -1000],
        [2000, 800, -1000],
        [2800, 800, -1000],
        [2400, 2400, -1000],
        [2400, 400, -1000],
        [1600, 1200, -1000],
        [2800, 1200, -1000],
        [2000, 2000, -1000],
        [2400, 800, -1000],
        [1200, 1600, -1000],
        [2800, 1600, -1000],
        [2000, 2400, -1000],
        [2400, 1200, -1000],
        [2000, 1600, -1000],
        [2400, 1600, -1000],
        [1600, 2400, -1000],
        [1600, -400, -1000],
        [1600, 1600, -1000],
        [1600, 0, -1000],
        [1200, 2400, -1000],
        [1200, -400, -1000],
        [1600, 2000, -1000],
        [1200, 0, -1000],
        [800, 2400, -1000],
        [800, -400, -1000],
        [1200, 2000, -1000],
        [1200, 400, -1000],
        [400, 2400, -1000],
        [800, 0, -1000],
        [800, 2000, -1000],
        [800, 400, -1000],
        [0, 2400, -1000],
        [400, -400, -1000],
        [2400, -400, -1000],
        [800, 800, -1000],
        [2400, 0, -1000],
        [400, 0, -1000],
        [2000, -400, -1000],
        [1600, 400, -1000],
        [2000, 0, -1000],
    ]
)


def extract_relevant_probes(probes, chan_names, animal, sortset_id):
    """
    Assemble probe information."""
    relevant_probe_names = set(
        [f"odoherty_sabes_{animal}_{x[:2]}".lower() for x in chan_names]
    )
    relevant_probes = []
    for probe in probes:
        if probe.id in relevant_probe_names:
            relevant_probes.append(probe)
    if len(relevant_probes) == 0:
        raise ValueError(f"No probes found for {sortset_id}")


def extract_probe_description() -> list[Probe]:
    infos = [
        ("indy_m1", Macaque.primary_motor_cortex, "M1"),
        ("indy_s1", Macaque.primary_somatosensory_cortex, "S1"),
        ("loco_m1", Macaque.primary_motor_cortex, "M1"),
        ("loco_s1", Macaque.primary_somatosensory_cortex, "S1"),
    ]

    descriptions = []
    for suffix, area, name in infos:
        channels = [
            Channel(
                id=f"{name} {i + 1:03}",
                local_index=i,
                relative_x_um=channel_map[i, 0] if "suffix" == "indy_m1" else 0,
                relative_y_um=channel_map[i, 1] if "suffix" == "indy_m1" else 0,
                relative_z_um=channel_map[i, 2] if "suffix" == "indy_m1" else 0,
                area=area,
            )
            for i in range(96)
        ]

        description = Probe(
            id=f"odoherty_sabes_{suffix}",
            type=RecordingTech.UTAH_ARRAY,
            wideband_sampling_rate=SAMPLE_FREQUENCY,
            waveform_sampling_rate=SAMPLE_FREQUENCY,
            lfp_sampling_rate=500,
            waveform_samples=48,
            channels=channels,
        )
        descriptions.append(description)

    return descriptions


def extract_behavior(h5file):
    """Extract the behavior from the h5 file."""

    cursor_pos = h5file["cursor_pos"][:].T
    finger_pos = h5file["finger_pos"][:].T
    target_pos = h5file["target_pos"][:].T
    timestamps = h5file["t"][:][0]

    expected_period = 0.004
    assert np.all(np.abs(np.diff(timestamps) - expected_period) < 1e-4)

    # calculate the velocity of the cursor
    cursor_vel = np.gradient(cursor_pos, timestamps, edge_order=1, axis=0)
    cursor_acc = np.gradient(cursor_vel, timestamps, edge_order=1, axis=0)
    finger_vel = np.gradient(finger_pos, timestamps, edge_order=1, axis=0)

    subtask_index = np.ones(len(timestamps), dtype=np.int64) * int(Task.REACHING.RANDOM)

    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_pos,
        vel=cursor_vel / 200.0,  # TODO: this is used to match the other datasets
        acc=cursor_acc,
        subtask_index=subtask_index,
        domain="auto",
    )

    finger = IrregularTimeSeries(
        timestamps=timestamps,
        pos=finger_pos[:, :3],
        vel=finger_vel[:, :3],
        subtask_index=subtask_index,
        domain="auto",
    )
    if finger_pos.shape[1] == 6:
        finger.orientation = finger_pos[:, 3:]
        finger.angular_vel = finger_vel[:, 3:]

    assert cursor.is_sorted()
    assert finger.is_sorted()

    nomove_mask = np.linalg.norm(cursor_vel, axis=1) < 10.0

    nomove_start = timestamps[np.where(np.diff(nomove_mask.astype(int)) == 1)[0]]
    if nomove_mask[0]:
        nomove_start = np.insert(nomove_start, 0, timestamps[0])

    nomove_end = timestamps[np.where(np.diff(nomove_mask.astype(int)) == -1)[0]]
    if nomove_mask[-1]:
        nomove_end = np.append(nomove_end, timestamps[-1])

    no_movement_segments = Interval(start=nomove_start, end=nomove_end)
    assert no_movement_segments.is_disjoint()

    no_movement_segments = no_movement_segments.select_by_mask(
        (no_movement_segments.end - no_movement_segments.start) > 10.0
    )

    return cursor, finger, no_movement_segments


def extract_spikes(h5file: h5py.File):
    r"""This dataset has a mixture of sorted and unsorted (threshold crossings)
    units.
    """

    # helpers specific to spike extraction
    def _to_ascii(vector):
        return ["".join(chr(char) for char in row) for row in vector]

    def _load_references_2d(h5file, ref_name):
        return [[h5file[ref] for ref in ref_row] for ref_row in h5file[ref_name][:]]

    spikesvec = _load_references_2d(h5file, "spikes")
    waveformsvec = _load_references_2d(h5file, "wf")

    # this is slightly silly but we can convert channel names back to an ascii token
    # this way.
    chan_names = _to_ascii(
        np.array(_load_references_2d(h5file, "chan_names")).squeeze()
    )

    spikes = []
    unit_index = []
    types = []
    waveforms = []
    unit_meta = []

    # The 0'th spikesvec corresponds to unsorted thresholded units, the rest are sorted.
    suffixes = ["unsorted"] + [f"sorted_{i:02}" for i in range(1, 11)]
    type_map = [int(RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS)] + [
        int(RecordingTech.UTAH_ARRAY_SPIKES)
    ] * 10

    # Map from common names to brodmann areas
    bas = {"s1": 3, "m1": 4}

    encountered = set()

    unit_index_delta = 0
    for j in range(len(spikesvec)):
        crossings = spikesvec[j]
        for i in range(len(crossings)):
            spiketimes = crossings[i][:][0]
            if spiketimes.ndim == 0:
                continue

            spikes.append(spiketimes)
            area, channel_number = chan_names[i].split(" ")

            unit_name = f"{chan_names[i]}/{suffixes[j]}"

            unit_index.append([unit_index_delta] * len(spiketimes))
            types.append(np.ones_like(spiketimes, dtype=np.int64) * type_map[j])

            if unit_name in encountered:
                raise ValueError(f"Duplicate unit name: {unit_name}")
            encountered.add(unit_name)

            wf = np.array(waveformsvec[j][i][:])
            unit_meta.append(
                {
                    "count": len(spiketimes),
                    "channel_name": chan_names[i],
                    "id": unit_name,
                    "area_name": area,
                    "channel_number": channel_number,
                    "unit_number": j,
                    "ba": bas[area.lower()],
                    "type": type_map[j],
                    "average_waveform": wf.mean(axis=1)[:48],
                    # Based on https://zenodo.org/record/1488441
                    "waveform_sampling_rate": SAMPLE_FREQUENCY,
                }
            )
            waveforms.append(wf.T)
            unit_index_delta += 1

    spikes = np.concatenate(spikes)
    waveforms = np.concatenate(waveforms)
    unit_index = np.concatenate(unit_index)

    spikes = IrregularTimeSeries(
        timestamps=spikes,
        unit_index=unit_index,
        waveforms=waveforms,
        domain="auto",
    )
    spikes.sort()

    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta))
    return spikes, units, chan_names


def split_intervals(data, no_movement_segments):
    task_domain = data.domain

    if len(no_movement_segments) > 0:
        task_domain = task_domain.difference(no_movement_segments)
        task_domain = task_domain.select_by_mask(
            task_domain.end - task_domain.start > 10.0
        )

    intervals = Interval.linspace(task_domain.start[0], task_domain.end[-1], 10)
    if len(no_movement_segments) > 0:
        task_ratio = []
        for start, end in zip(intervals.start, intervals.end):
            intersection = task_domain & Interval(start, end)
            duration = np.sum(intersection.end - intersection.start)
            task_ratio.append(duration / (end - start))

        task_ratio = np.array(task_ratio)
        task_ratio_with_index = np.column_stack((np.arange(len(intervals)), task_ratio))
        rng = np.random.default_rng(42)
        rng.shuffle(task_ratio_with_index)
        sorted_index = task_ratio_with_index[task_ratio_with_index[:, 1].argsort()][
            :, 0
        ].astype(int)

        train_sampling_intervals = Interval(
            start=intervals.start[sorted_index[:8]], end=intervals.end[sorted_index[:8]]
        )
        valid_sampling_intervals = Interval(
            start=intervals.start[sorted_index[8:9]],
            end=intervals.end[sorted_index[8:9]],
        )
        test_sampling_intervals = Interval(
            start=intervals.start[sorted_index[9:]], end=intervals.end[sorted_index[9:]]
        )

        train_sampling_intervals.sort()
        valid_sampling_intervals.sort()
        test_sampling_intervals.sort()

        train_sampling_intervals = train_sampling_intervals.difference(
            no_movement_segments
        )
        valid_sampling_intervals = valid_sampling_intervals.difference(
            no_movement_segments
        )
        test_sampling_intervals = test_sampling_intervals.difference(
            no_movement_segments
        )

        train_sampling_intervals = train_sampling_intervals.select_by_mask(
            train_sampling_intervals.end - train_sampling_intervals.start > 10.0
        )
        valid_sampling_intervals = valid_sampling_intervals.select_by_mask(
            valid_sampling_intervals.end - valid_sampling_intervals.start > 10.0
        )
        test_sampling_intervals = test_sampling_intervals.select_by_mask(
            test_sampling_intervals.end - test_sampling_intervals.start > 10.0
        )
    else:
        [
            train_sampling_intervals,
            valid_sampling_intervals,
            test_sampling_intervals,
        ] = intervals.split([8, 1, 1], shuffle=False, random_seed=42)
    return train_sampling_intervals, valid_sampling_intervals, test_sampling_intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name="odoherty_sabes_nonhuman_2017",
        origin_version="583331",
        derived_version="1.0.0",
        source="https://zenodo.org/record/583331",
        description="The behavioral task was to make self-paced reaches to targets "
        "arranged in a grid (e.g. 8x8) without gaps or pre-movement delay intervals. "
        "One monkey reached with the right arm (recordings made in the left hemisphere)"
        "The other reached with the left arm (right hemisphere). In some sessions "
        "recordings were made from both M1 and S1 arrays (192 channels); "
        "in most sessions M1 recordings were made alone (96 channels).",
    )

    # common to all sessions
    probes = extract_probe_description()

    # find all files with extension .mat in folder_path
    for file_path in sorted(find_files_by_extension(db.raw_folder_path, ".mat")):
        logging.info(f"Processing file: {file_path}")

        with db.new_session() as session:
            # load file
            h5file = h5py.File(file_path, "r")

            # extract experiment metadata
            session_id = Path(file_path).stem

            sortset_id = session_id[:-3]
            assert sortset_id.count("_") == 1, f"Unexpected file name: {sortset_id}"

            # get subject and recording date from sortset_id
            animal, recording_date = sortset_id.split("_")
            subject = SubjectDescription(
                id=animal,
                species=Species.MACACA_MULATTA,
            )
            session.register_subject(subject)

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.REACHING,
            )

            # extract spiking activity, unit metadata and channel names info
            spikes, units, chan_names = extract_spikes(h5file)

            # register sortset
            session.register_sortset(
                id=session_id,
                units=units,
            )

            # extract behavior
            cursor, finger, no_movement_segments = extract_behavior(h5file)

            relevant_probes = extract_relevant_probes(
                probes, chan_names, animal, sortset_id
            )

            data = Data(
                probes=relevant_probes,
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                cursor=cursor,
                finger=finger,
                domain=cursor.domain,
            )

            session.register_data(data)

            (
                train_sampling_intervals,
                valid_sampling_intervals,
                test_sampling_intervals,
            ) = split_intervals(data, no_movement_segments)

            # save samples
            session.register_split("train", train_sampling_intervals)
            session.register_split("valid", valid_sampling_intervals)
            session.register_split("test", test_sampling_intervals)

            # save data to disk
            session.save_to_disk()

            # close the file
            h5file.close()

    db.finish()


if __name__ == "__main__":
    main()
