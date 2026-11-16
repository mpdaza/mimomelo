# MeloTTS: Staged Multi-Objective Scheduling for TTS Training

Experimental code for studying speaker adaptation in MeloTTS through progressive
activation and weighting of training losses. This repository accompanies
**“Staged Multi-Objective Scheduling for text-to-speech training”**, presented
at IberSPEECH 2026 by M. P. Daza-Llin, N. Solis-Verdesoto,
J. L. Blanco-Murillo, and L. Hernández-Gómez.

**Paper:** [IberSPEECH 2026, DOI: 10.21437/IberSPEECH.2026-2](https://doi.org/10.21437/IberSPEECH.2026-2).
The results and figures quoted below come from the paper. The example
configuration in this repository does not reproduce all its experiments.

## How the paper maps to the code

The paper compares a MeloTTS reference (**Base**) with four scheduling
strategies. The controller observes individual losses and selects their weights
and masks. It also coordinates the learning rates of the model components.

| Paper configuration | Main idea | Status in this repository |
| --- | --- | --- |
| Base | Fixed loss weights and learning-rate decay | `melo/train_wandb.py` uses exponential decay, but its loss sum does not fully match Base. The exact profile is missing. |
| C1 | Progressive activation of Mel, duration, and KL losses | No ready-to-run C1 profile is included. |
| C2 | Spectral losses in the early stages | No ready-to-run C2 profile is included. |
| NF | Loss-trend-based transitions and an ADSR learning-rate schedule | `melo/train.py` implements transitions based on Mel, duration, and KL; ADSR can be enabled in the configuration. |
| NFL | NF with relaxed constraints and later refinement | No separate, validated NFL profile is included. |

In `melo/train.py`, `losses_g` is ordered as follows: generator adversarial,
feature matching, Mel, duration, KL, and spectral. The controller applies masks
and weights in that order. The scheduler uses the same learning-rate function
for the generator and discriminators; independent controls described in the
paper are not exposed as configuration profiles. Also, `train.py` limits the
number of batches during some early stages. **This checkout therefore does not
directly reproduce the paper's five rows of results.**

The paper evaluates Mark F. Smith and Karen Savage using LibriVox recordings
at 44.1 kHz, with training, validation, and test splits of 10%, 10%, and 80%.
It reports up to approximately 14 times less adaptation time than the
reference, with comparable quality and intelligibility under the evaluated
conditions. These are published results, not a performance guarantee for other
datasets or configurations.

## Requirements and installation

- Python 3.10, a GPU-compatible PyTorch build, and CUDA for training and
  `preprocess_text.py`.
- FFmpeg for audio preparation.
- A network connection to download pretrained checkpoints on first use.

From the repository root:

```bash
conda env create -f environment.yml
conda activate melotts-win
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install wandb huggingface_hub click nltk
python -m unidic download
```

The environment name comes from `environment.yml` and can also be used on
Linux. `melo/train.py` initializes W&B, so an available W&B session is needed
to run it. Install PyTorch and CUDA versions that match the training machine.

## Prepare data

Place **44,100 Hz WAV** files at a path accessible from `melo/`. Each metadata
line has four `|`-separated fields:

```text
data/my_voice/wavs/001.wav|my_voice|EN|Example sentence to synthesize.
```

From `melo/`, preprocessing creates `train.list`, `val.list`, text features,
and a `config.json` next to the metadata file:

```bash
cd melo
python preprocess_text.py --metadata data/my_voice/metadata.list --config_path configs/config.json
```

Review the paths and parameters in the generated JSON before training.
`melo/data/example/config.json` and its example lists illustrate the format;
some entries refer to audio files that are not distributed here.

**Data splits:** the current preprocessor initially reserves 20% for
validation and caps that list at eight examples by default. It does not create
the paper's test split. To follow the paper's 10/10/80 protocol, prepare the
three splits separately and use the same test recordings for every
configuration.

`resample.py` and `transcript.py` contain experiment-specific paths that must
be adapted before using them with a new corpus.

## Train

Run this from `melo/` with a `config.json` pointing to existing lists and WAV
files:

```bash
python train.py -c data/example/config.json -m example
```

On Windows, `train.bat` runs that example once. On Linux, run `bash train.sh`
from `melo/`. The name passed to `-m` sets the output directory to
`melo/logs/<name>/`. If needed, the script downloads initial generator and
discriminator checkpoints. It records metrics in TensorBoard and W&B.

To select the ADSR learning-rate schedule described for NF/NFL, add this key
to the `train` object in the JSON configuration:

```json
"lr_schedule": "adsr"
```

Without this key, `train.py` uses its warmup and decay schedule. To view local
logs from `melo/`:

```bash
tensorboard --logdir logs
```

`melo/train_TFM.py` is a diagnostic variant. Results also depend on weights,
thresholds, the number of examples, the random seed, and checkpoint selection.
These values are not collected into separate manifests for Base, C1, C2, NF,
and NFL.

## Inference with a trained checkpoint

`infer.py` expects `config.json` beside the `G_*.pth` checkpoint. The required
`--type` argument selects the output subdirectory. From `melo/`:

```bash
python infer.py --text "Example sentence." --language EN --ckpt_path logs/example/G_1000.pth --type test --output_dir outputs
```

Valid `--type` values are `val`, `test`, and `balanced`. `G_1000.pth` is an
example; use a checkpoint that exists in your training directory.

## Included data and analysis

The CSV files in `melo/infer/loss_study/` are inputs to
`melo/infer/loss_study/min_loss_study.ipynb` and are retained for that analysis.
LibriVox audio, trained weights, and listening-test responses are not bundled
with this repository. The main training script no longer writes per-batch loss
CSVs; inspect its metrics in TensorBoard and W&B.

## Citation

> M. P. Daza-Llin, N. Solis-Verdesoto, J. L. Blanco-Murillo, and
> L. Hernández-Gómez, “Staged Multi-Objective Scheduling for text-to-speech
> training,” *IberSPEECH 2026*, Madrid, Spain, 2026.
> DOI: [10.21437/IberSPEECH.2026-2](https://doi.org/10.21437/IberSPEECH.2026-2).
