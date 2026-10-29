import numpy as np
import pandas as pd

from mojo_kirby.data import ArrayDict, IrregularTimeSeries

from mojo_kirby.taxonomy import (
    RecordingTech,
    Sex,
    Species,
    SubjectDescription,
)


def extract_subject_from_nwb(nwbfile):

    species = nwbfile.subject.species

    if "NCBITaxon" in species:
        species = "NCBITaxon_" + species.split("_")[-1]

    return SubjectDescription(
        id=nwbfile.subject.subject_id.lower(),
        species=Species.from_string(species),
        sex=Sex.from_string(nwbfile.subject.sex),
    )


def extract_spikes_from_nwbfile(nwbfile, recording_tech):
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
