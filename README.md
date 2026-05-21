# Investigating the Entanglement in Subliminal Prompting

This repository contains the code, data and write-up for a study of **token entanglement under subliminal prompting**.
We inject an emotion $e_n$ towards a three-digit number $n$ into a model's system prompt, then measure how this shifts the model's emotion $e_a$ towards an animal $a$.
By sweeping over emotions, numbers and animals we map the structure of the entanglement that connects number tokens and animal tokens.

The full write-up is in [report/out/report.pdf](report/out/report.pdf).
Headline findings, copied from the abstract:

1. Emotion transfer is **sentiment-specific rather than emotion-specific**.
   Numbers entangled with positive emotions form a cluster independent of numbers entangled with negative emotions, so a single number can simultaneously make an animal more likely to be mentioned as the most loved and as the most hated.
2. Entanglement extends across **semantically related animals beyond synonyms** (e.g. koala/kangaroo, hog/cattle).
3. Cosine similarity of unembedding vectors **does not** predict shared entanglement strength beyond near-synonyms, which argues against the softmax bottleneck in the unembedding layer being the primary driver of subliminal prompting.

This project was forked from [loftusa/owls](https://github.com/loftusa/owls) (the implementation behind the [It's Owl in the Numbers](https://owls.baulab.info) blog post and Cloud et al. 2025).
The original Llama-3.2 1B owl demo is preserved in [experiments/Subliminal Learning.ipynb](experiments/Subliminal%20Learning.ipynb) and [experiments/Subliminal Learning.py](experiments/Subliminal%20Learning.py) for reference.
Everything else, including the experiment drivers, the SLURM pipeline, the analysis notebooks and the report, is new.

## Repository layout

```
.
├── subliminal_prompting.py     Main experiment driver. Reads HF model, writes per-config logprob CSVs.
├── unembeddings.py             Dumps lm_head unembedding vectors for animals/numbers/relations.
├── animals.py                  Upstream multi-model driver. Kept for the legacy results/<model>/*.csv layout.
├── compare_tokenization.py     Small standalone diff utility (upstream leftover).
├── utils/
│   ├── animals_utils.py        Prompt templates, RELATION_MAP, animal lists, synonym groups, get_numbers().
│   ├── data_loading.py         Loads the CSVs produced by subliminal_prompting.py back into nested dicts.
│   └── plotting.py             All figures used in the report.
├── slurm/
│   ├── gradual_emotions/       Sweep for the EvilOwls/CompareTokenization style experiments.
│   ├── animal_synonyms/        Sweep used for the report figures (PostiveAndNegativeTokens, IncreaseNumberFocus).
│   └── unembeddings/           Single job that runs unembeddings.py.
├── experiments/                Jupyter notebooks. See "Notebooks" below.
├── results/                    Output CSVs. Files at the top of results/<model>/ come from animals.py.
│                               Subdirectories (base_prompting/, allow_hate_prompting/,
│                               subliminal_prompting/) and unembeddings.csv come from subliminal_prompting.py
│                               and unembeddings.py.
├── data/animal_preference_numbers/   Teacher-generated number sequences from upstream. Only used by EvilOwls.ipynb.
├── report/                     LaTeX sources and PDF.
└── scripts/                    Upstream wrappers around animals.py. Superseded by slurm/.
```

## How the data is generated

There are two drivers and three sweeps.
Outputs always land under `results/<model_short_name>/`.

> Only `Qwen/Qwen2.5-7B-Instruct` is supported end to end.
> `unembeddings.py` has the model name hardcoded, the single-token tables in [utils/animals_utils.py](utils/animals_utils.py) and the analysis paths in the notebooks and `utils/plotting.py` all assume Qwen.
> Passing `--model` to `subliminal_prompting.py` will probably run, but downstream analysis will need manual fixes.

### `subliminal_prompting.py`

The core experiment.
For each configuration it writes one CSV per prompt and a sibling `.txt` containing the literal prompt.

Relevant CLI flags:

| Flag                  | Values                                                                                       | Notes                                                                  |
| --------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `--model`             | `Qwen/Qwen2.5-7B-Instruct` (default). Other HF ids will run but trigger the caveats in the note above. | default `Qwen/Qwen2.5-7B-Instruct`.                                    |
| `--template-types`    | `full`, `withoutthinking`, `withoutthinkingallowhate`, `respondwithnumber`, `onlythinking`, `empty`, `brood`, `ponder`, `full2`, or `all`, comma separated | the system-prompt template, defined in `SUBLIMINAL_PROMPT_TEMPLATES`.  |
| `--number-relations`  | any key of `RELATION_MAP` (`love`, `hate`, ...) or `all`                                     | emotion $e_n$ injected towards the number.                             |
| `--animal-relations`  | same as above                                                                                | emotion $e_a$ asked about in the user question.                        |
| `--response-start`    | `spaceinprompt`, `spaceinanimal`                                                             | controls whether the leading space sits in the prompt or in the animal token. The report uses `spaceinanimal`. |
| `--animal-set`        | `default`, `synonyms`                                                                        | `synonyms` adds the [SYNONYM_GROUPS](utils/animals_utils.py) animals on top of the per-model default list. |
| `--baseline-only`     | flag                                                                                         | skip the main sweep and only run the two baselines.                    |

Output paths (for `model_short = Qwen2.5-7B-Instruct`):

```
results/Qwen2.5-7B-Instruct/
├── base_prompting/<response_start>_<animal_relation>.csv
├── allow_hate_prompting/<response_start>_<animal_relation>.csv
└── subliminal_prompting/<response_start>_<template_type>_<number_relation>_<animal_relation>.csv
```

Each CSV is indexed by number (000 ... 999), columns are animals + a trailing `number` column with the log-probability of generating the number itself.

### `unembeddings.py`

Hardcoded to `Qwen/Qwen2.5-7B-Instruct`.
Downloads `lm_head.weight` from HF and writes one row per text (numbers, animals in singular/plural, with and without leading space, relation verbs and attributes).

Output: `results/Qwen2.5-7B-Instruct/unembeddings.csv` (large, around 500 MB).

### `animals.py`

The upstream-style driver.
Single positional `--model` flag, no template/relation/response-start configuration.
It writes flat CSVs at the top of `results/<model>/` (`base_prompting.csv`, `subliminal_prompting_*.csv`, `logit.csv`, `unembedding.csv`).
None of the new analysis notebooks read these files.
The Llama/Gemma/OLMo CSVs currently committed in `results/*/` were produced by this driver and only feed into legacy upstream notebooks like `experiments/Subliminal Learning.ipynb`.

### SLURM sweeps

Each `slurm/<name>/run_jobs.sh` submits one `job.slurm` per cell of a parameter grid.
Pass `--local` to run sequentially on the current machine instead of submitting.
Defaults below are for the committed scripts and can be edited at the top of each `run_jobs.sh`.

| Sweep                                                         | Driver                  | Parameter grid                                                                                                                                                                                          | Produces                                                                                                                                                                                    |
| ------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [slurm/gradual_emotions/](slurm/gradual_emotions/run_jobs.sh) | `subliminal_prompting.py` | templates ∈ {`empty`, `withoutthinking`, `full`}, number-relations ∈ {`love`, `hate`}, animal-relations = `all`, response-starts ∈ {`spaceinprompt`, `spaceinanimal`}, animal-set = `default`. 12 jobs. | The matched cells of `subliminal_prompting/<response_start>_<template>_<number_rel>_<animal_rel>.csv` plus the baselines under `base_prompting/` and `allow_hate_prompting/`. |
| [slurm/animal_synonyms/](slurm/animal_synonyms/run_jobs.sh)   | `subliminal_prompting.py` | templates ∈ {`withoutthinking`, `withoutthinkingallowhate`, `respondwithnumber`, `empty`, `onlythinking`}, number-relations ∈ {`love`, `hate`}, animal-relations = `all`, response-starts = `spaceinanimal`, animal-set = `synonyms`. 10 jobs. | Same layout, but the animal columns include the synonym groups. This is what the report-figure notebooks consume. |
| [slurm/unembeddings/](slurm/unembeddings/run_jobs.sh)         | `unembeddings.py`         | none                                                                                                                                                                                                    | `results/Qwen2.5-7B-Instruct/unembeddings.csv`.                                                                                                                                              |

The `slurm/job.slurm` files target the FAU `tinygpu` partition (`sbatch.tinygpu`); switch to plain `sbatch` if you are on a different cluster, or use `--local`.

## Notebooks

All notebooks live under [experiments/](experiments/) and assume they run from that directory (paths are `Path.cwd().parent / ...`).
They all import from `utils/` via a `sys.path.append`.

| Notebook                                                                            | What it does                                                                                                                                                                | Required data                                                                                                                                          |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [PostiveAndNegativeTokens.ipynb](experiments/PostiveAndNegativeTokens.ipynb)        | Generates most figures in the report body and appendix. Uses `withoutthinking` and `withoutthinkingallowhate` templates with $e_n \in \{\text{love}, \text{hate}\}$ and $e_a$ across many emotions. | `slurm/animal_synonyms` + `slurm/unembeddings`. The `hate_hate` baseline uses `allow_hate_prompting/`. |
| [IncreaseNumberFocus.ipynb](experiments/IncreaseNumberFocus.ipynb)                  | Same analysis but for the `respondwithnumber` template. Generates the `*_respondwithnumber.png` appendix figures.                                                           | `slurm/animal_synonyms` + `slurm/unembeddings`.                                                                                                        |
| [EvilOwls.ipynb](experiments/EvilOwls.ipynb)                                        | Wild collection of exploratory plots. The seed for most of what later became the report.                                                                                    | `slurm/gradual_emotions` covers most cells. Some combinations (`brood`, `ponder`, `onlythinking` with `oldtokenization` response-start) need manual runs. Also reads `data/animal_preference_numbers/Qwen2.5-7B-Instruct/` (the teacher-generated number sequences from upstream) for `load_dataset_frequency_ratios`. |
| [CompareTokenization.ipynb](experiments/CompareTokenization.ipynb)                  | Diagnostic. Compares the legacy "subtract empty-prompt baseline" tokenization in `animals.py` against the cleaner per-token gather in `subliminal_prompting.py`, and explains a batched-bfloat16 non-determinism bug. Generates `results/Qwen2.5-7B-Instruct/tokenization_differences_combined.csv`. | `slurm/gradual_emotions` (needs both `spaceinprompt` and `spaceinanimal` outputs).                                                                     |
| [ClosestTokens.ipynb](experiments/ClosestTokens.ipynb)                              | Quick lookup of nearest neighbours under unembedding cosine for each single-token animal. Standalone.                                                                       | HF download of `Qwen2.5-7B-Instruct` weights only. No experiment outputs needed.                                                                       |
| [CheckTopTokens.ipynb](experiments/CheckTopTokens.ipynb)                            | Interactive scratchpad. Loads the model once and prints the top-k most likely next tokens for a baseline or subliminal prompt. Useful for eyeballing what the model wants to say before computing full logprob sweeps. | HF download of `Qwen2.5-7B-Instruct` weights only.                                                                                                     |
| [Subliminal Learning.ipynb](experiments/Subliminal%20Learning.ipynb) / [.py](experiments/Subliminal%20Learning.py) | Upstream demo (Llama-3.2 1B, owls and threshold sampling). Kept for reference, not used by this project.                                                                    | None.                                                                                                                                                  |

### Notebook → SLURM cheat sheet

```
PostiveAndNegativeTokens.ipynb  ─►  slurm/animal_synonyms + slurm/unembeddings
IncreaseNumberFocus.ipynb       ─►  slurm/animal_synonyms + slurm/unembeddings
CompareTokenization.ipynb       ─►  slurm/gradual_emotions
EvilOwls.ipynb                  ─►  slurm/gradual_emotions + slurm/animal_synonyms +
                                    slurm/unembeddings + data/animal_preference_numbers/ +
                                    a few manual subliminal_prompting.py runs
ClosestTokens.ipynb             ─►  (none, just HF model)
```

To reproduce all report figures from scratch:

```bash
# from the repo root, on a machine with a GPU and HF login
bash slurm/animal_synonyms/run_jobs.sh --local
bash slurm/unembeddings/run_jobs.sh --local
# then open and run-all the two report notebooks
jupyter lab experiments/PostiveAndNegativeTokens.ipynb
jupyter lab experiments/IncreaseNumberFocus.ipynb
```

Figures are written into [report/figures/](report/figures/), and the report itself is built with `make` (or `pdflatex report.tex`) in [report/](report/).

## Environment

`uv` for Python deps, `nix-shell` (via [shell.nix](shell.nix) and [.envrc](.envrc)) for the system toolchain.

```bash
uv sync
huggingface-cli login   # needed for Llama and Qwen weights
```

GPU recommended.
The Qwen sweeps each run roughly a few hours on a single A100; CPU is not realistic.

## Citation

If the report is useful, cite the project as

```
@misc{rossbach2026entanglement,
  title  = {Investigating the Entanglement in Subliminal Prompting},
  author = {Ro{\ss}bach, Christopher and Riess, Christian},
  year   = {2026},
  note   = {FAU-Erlangen-Nuernberg, technical report}
}
```

The upstream demo and the wider blog post are at [owls.baulab.info](https://owls.baulab.info); the underlying subliminal-learning finding is [Cloud et al. (2025)](https://arxiv.org/abs/2507.14805).
