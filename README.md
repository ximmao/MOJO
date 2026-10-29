# MOJO

Code for [Leveraging unlabelled data for generalizable neural population decoding](https://arxiv.org/abs/2607.14086).

**MOJO** (Masked autOencoder-based JOint training) combines self-supervised masked autoencoding with supervised behavioural decoding to learn from labelled and unlabelled neural recordings. This repository provides data-preparation pipelines for monkey reaching datasets and multi-session training with GRU and transformer backbones, plus optional Mamba and xLSTM support.

This repository builds on an earlier version of Kirby, now known as `torch_brain`. We plan to integrate **MOJO** into `torch_brain`. Stay tuned for updates!

## Installation

### Environment setup with `uv`

```bash
uv venv --python 3.10
source .venv/bin/activate
module load cuda/12.6.0 # install cuda 12.6
uv pip install pydantic~=1.10.0 numpy~=1.23.5 torch==2.5.1 setuptools
uv pip install -e .
```

### Optional backbones

After setting up the environment, install the dependency for your selected backbone.

For Mamba (`ssm_mamba` or `ssm_mamba2`):

```bash
uv pip install "mamba_ssm[causal_conv1d]" --no-build-isolation
```

For xLSTM (`ssm_xlstm` or `ssm_xlstm-mixed`):

```bash
uv pip install xlstm
```

### Data preparation

Data preparation is managed via Snakemake. Each dataset has its own `Snakefile` and `prepare_data.py` under `data/scripts/`. The pipeline downloads raw data, processes it into per-session HDF5 files, and produces a `description.mpk` (MessagePack) manifest.

#### Monkey motor cortex

```bash
snakemake --rerun-triggers mtime --config tmp_dir=$SLURM_TMPDIR -c1 perich_miller_population_2018_unfreeze
snakemake --rerun-triggers mtime --config tmp_dir=$SLURM_TMPDIR -c1 odoherty_sabes_nonhuman_2017_unfreeze
snakemake --rerun-triggers mtime --config tmp_dir=$SLURM_TMPDIR -c1 churchland_shenoy_neural_2012_unfreeze
snakemake --rerun-triggers mtime --config tmp_dir=$SLURM_TMPDIR -c1 pei_churchland_nlbmaze_2021_unfreeze
```

#### Output format

Each dataset produces:
- Per-session HDF5 files containing spike times (as irregular event streams), behavioral time series, stimulus/trial intervals, unit metadata, and train/valid/test split masks
- A `description.mpk` manifest listing all sessions, subjects, and available splits

## Training

```bash
python train_join_ssl_sl.py data_root="path_of_uncompressed_dataset[data/uncompressed/]"
```

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{mao2026mojo,
      title={Leveraging unlabelled data for generalizable neural population decoding}, 
      author={Ximeng Mao and Nanda H. Krishna and Avery Hee-Woon Ryoo and Matthew G. Perich and Guillaume Lajoie},
      year={2026},
      eprint={2607.14086},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.14086}, 
}
```
