"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E flood-segmentation workflow: the pinned Prithvi flood checkpoint (a Lightning
pickle) is digest-verified, statically audited and converted once into safetensors, 44 hand-labelled Sen1Floods11
chips are fetched from the public bucket, validated and assigned the official roles, the frozen model is scored
against the no-water baseline, a bounded decoder fine-tuning runs in the kernel, the held-out chips are scored
again, a pre-registered fresh test is scored under fail-closed score-once rules, the upstream example chips are
segmented, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "prithvi-flood-segmentation-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/prithvi_flood_segmentation_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ibm--nasa--geospatial%2FPrithvi--EO--2.0--300M--TL--Sen1Floods11-ffcc4d?style=flat",
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-NASA--IMPACT%2FPrithvi--EO--2.0-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/NASA-IMPACT/Prithvi-EO-2.0",
    ),
    ("Paper", "https://img.shields.io/badge/arXiv-2412.02732-b31b1b.svg", "https://arxiv.org/abs/2412.02732"),
]

TEMPLATE = {
    "package": "prithvi_flood_segmentation_pipeline",
    "repo_name": REPO,
    "stem": "prithvi_flood_segmentation",
    "notebook_name": "prithvi_flood_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime builds an isolated environment from the hash-locked pins (torch, torchvision, "
        "terratorch and its stack, tifffile, numpy, safetensors, huggingface-hub); nothing is installed into the notebook's own Python, so "
        "no restart is needed and Run all completes in one pass. It then stages and digest-verifies the pinned Prithvi flood checkpoint "
        "(1.28 GB) from the Hub, statically audits the Lightning pickle against an allow-list, converts it once into safetensors "
        "with a pinned digest, rebuilds the architecture from the installed `terratorch` package and loads it strictly, fetches 44 "
        "hand-labelled Sen1Floods11 chips (104 MB of digest-verified GeoTIFFs, no credential), validates them and assigns the "
        "official roles (24 training, 8 validation, 12 exposed development test), scores the frozen model against the no-water "
        "baseline, runs the pre-registered validation grid of three bounded fine-tunings on a GPU (deterministic kernels; on CPU "
        "the grid is off and the run is exploratory), compares the selected system with the frozen model on the exposed "
        "development test, scores the pre-registered 23-chip fresh test (53 MB more from the same bucket) only when every "
        "pre-registered setting holds on a T4, segments the three example chips shipped with the upstream repository, exports the adapter as safetensors with a "
        "manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC "
        "2.0 §5). On a T4 the whole path takes a few minutes of model time after the downloads; building the isolated "
        "environment is the slowest step. Set the environment variable `DIMER_MASK_FRESH_TEST=1` to skip fetching and scoring "
        "the fresh test (the run is then exploratory)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled chips — as `BYOD_PATH` (a path in the runtime, which works on Colab, Kaggle and Jupyter) or, when it is empty, "
        "through the Colab upload dialog — as a zip (or folder) holding `pairs.csv` (columns `id`, `image`, `label`) beside six-band 512 × 512 GeoTIFF chips "
        "(blue, green, red, narrow NIR, SWIR 1, SWIR 2 — Sentinel-2 **L1C top-of-atmosphere** reflectance in [0, 1] or × 10 000, "
        "the model's training input; not L2A surface reflectance, not HLS) and single-band label rasters "
        "(0 = no water, 1 = water, −1 = no data); at least **seven** labelled chips, with some water among them (the seeded "
        "split keeps four for training, one for validation and two for test). An optional `group` column (the scene or event "
        "a chip was cut from) keeps each group in one role, and an optional `split` column (`train`, `validation`, `test`) sets "
        "the roles yourself; without either, chips are split one by one with a printed warning. Your chips then flow through "
        "the same contract — validation, frozen baseline, adaptation, "
        "held-out evaluation, inference, artifact export and reload parity. The expected schema, the ceilings and the privacy "
        "guidance are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. A BYOD run is "
        "exploratory: it never scores the pre-registered fresh test. BYOD is optional and never part of the default path."
    ),
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python, has used Colab or Jupyter, and wants to see how a geospatial '
            'foundation model fine-tuned to map flood water in Sentinel-2 imagery is evaluated and adapted honestly: how its water masks are scored against a '
            'baseline that predicts *land* everywhere, what a bounded fine-tuning of the decoder does to a model that already trained on this dataset, and how the '
            'change is exported and reloaded. No prior experience with Prithvi, TerraTorch or remote sensing models is assumed; terms are explained where they '
            'first matter and again in the **Glossary** at the end. A GPU runtime (T4 or better) is expected.\n\n**Input → Model → Output.**\n\n| | Segmentation | '
            'Bounded fine-tuning |\n|---|---|---|\n| Input | six-band 512 × 512 Sentinel-2 L1C top-of-atmosphere reflectance chips (blue, green, red, narrow NIR, SWIR 1, SWIR 2) | '
            'hand-labelled Sen1Floods11 chips with 0 / 1 / −1 masks (24 training, 8 validation, 12 exposed development test, official splits; 23 fresh test chips) |\n| Model | the Prithvi-EO-2.0 ViT-L '
            'encoder, pyramid neck, UPerNet decoder and two-class head, loaded from audited, converted safetensors | the same model; only the neck, decoder and '
            'head (15.1 M parameters) are trained, the encoder and BatchNorm statistics stay frozen |\n| Output | a per-pixel water mask, softmax scores, water '
            "fraction; pixel IoU / F1 beside the no-water baseline | the adapted model's numbers beside the frozen model's on the same held-out chips, and a 60 MB "
            'safetensors adapter that reloads with parity |\n\n**How to use this notebook.** Choose a GPU runtime (**Runtime → Change runtime type → T4 GPU**), then '
            "**Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 "
            'are **infrastructure** — the isolated environment, the carried package and the audited model snapshot — and their cells are collapsed; you may run '
            'them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults '
            'reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible '
            '**Check your reasoning** with a worked answer from the recorded run (the evaluation of record: the hosted Colab CLI T4 run of 19 October 2026 of commit `9fd2b76`, '
            '`docs/execution-evidence/2026-10-19-9fd2b76/`). Every adaptation starts from the pinned base, so re-running Section 6 with other settings is a fresh, exploratory experiment. '
            '**Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 '
            'infrastructure → 4 the labelled chips, validation and refusals *(evaluation practice)* → 5 the frozen model against the no-water baseline *(core '
            'concept)* → 6 the pre-registered validation grid of bounded fine-tunings *(core concept)* → 7 the exposed development test *(evaluation practice)* → 8 the '
            'pre-registered fresh test *(evaluation practice)* → 9 upstream example chips, export and fresh reload *(engineering)* → interpretation, troubleshooting, glossary and your conclusion.'
        ],
    },
    "pipeline_class": "PrithviFloodPipeline",
    "model_load": "PrithviFloodPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), report=print)",
    "weights_key": "prithvi-eo-2.0-300m-tl-sen1floods11",
    "modules": ["pipeline.py", "samples.py", "metrics.py", "scoring.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "timm", "lightning", "tifffile"],
    "title": "Prithvi-EO-2.0 flood mapping — DIMER E2E segmentation fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "flood-extent segmentation of six-band Sentinel-2 chips with a Prithvi-EO-2.0 ViT-L encoder and UPerNet decoder, held-out IoU/F1 against a no-water baseline, and bounded fine-tuning of the decoder to labelled chips",
    "intro": (
        "Prithvi-EO-2.0 [1] is NASA and IBM's foundation model for Harmonized Landsat Sentinel-2 imagery: a "
        "ViT-L masked autoencoder pretrained on 4.2 M global multispectral samples, here in its `TL` variant with temporal and "
        "location embeddings. The checkpoint packaged here is the upstream authors' fine-tune for flood mapping — the 300 M-parameter "
        "encoder, a learned pyramid neck over four encoder depths, a UPerNet decoder and a two-class head — fine-tuned with "
        "TerraTorch on the 252-chip official training split of Sen1Floods11's hand-labelled Sentinel-2 L1C chips, with the 89-chip "
        "validation split used for early stopping (the 90 test chips and the 15 Bolivia chips were not trained on; inferred from the "
        "checkpoint's 630 steps = 42 × ⌊252/16⌋). Prithvi's pretraining used HLS imagery, which is derived from Sentinel-2 L1C, so "
        "the same acquisitions may appear, unlabelled, in pretraining.\n\n"
        "Two things about this row are handled in the open. **The upstream asset is a pickle** — a PyTorch Lightning checkpoint. "
        "Section 3 downloads and digest-verifies it, statically lists every global the pickle would import (a state dict of tensors "
        "and nothing else), refuses anything outside that allow-list, unpickles it exactly once through torch's weights-only loader, "
        "and writes a safetensors file whose digest is pinned in the carried module; the model you run is rebuilt from the installed "
        "`terratorch` package and loads that file strictly. **The model is already fine-tuned on this dataset**, so the bounded "
        "adaptation in Section 6 is a demonstration of the contract on chips it has seen the distribution of, selected by validation "
        "loss with the frozen model as epoch 0; the honest number is the paired held-out comparison in Section 7, and the point of "
        "the contract is the same recipe applied to *your* labelled chips from another sensor, season or region. The grid, the "
        "fresh test and the reading rule are fixed in advance in `docs/prithvi-flood-preregistration.md`."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify a pickled "
        "checkpoint, read its static audit and see it converted into safetensors; fetch and validate real labelled multispectral "
        "chips with an ignore class; read pixel IoU, F1, precision and recall against a no-water baseline; run a bounded decoder "
        "fine-tuning with explicit hyperparameters and frozen BatchNorm statistics; select a configuration on validation by a "
        "pre-registered rule; tell an exposed development test from a pre-registered fresh test and read a bootstrap interval "
        "against a practical tolerance; segment unlabelled upstream example chips; and export a safetensors adapter that reloads "
        "against the pinned base with verified parity."
    ),
    "exclusions": (
        "the Sentinel-1 (SAR) route of Sen1Floods11, the temporal and location embeddings (the packaged fine-tune ran without "
        "metadata, and so does this pipeline), tiling of scenes larger than 512 × 512, atmospheric correction, cloud masking, the "
        "published benchmark scores, and any claim that a 44-chip sample stands in for an operational evaluation. The repository "
        "exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12): the ViT-L encoder runs in float16 autocast and the default adaptation needs about 2.5 GB of GPU memory; on CPU one 512 × 512 chip takes tens of seconds and the adaptation would take an hour. About 3 GB of disk is needed for the checkpoint and its conversion; building the isolated environment (terratorch pulls torchgeo, lightning and their dependencies) takes several minutes the first time and is reused on a re-run.",
        "- **Knowledge:** what a multispectral reflectance chip is (bands, scaling, no-data), what a pixel-wise segmentation mask and an ignore class are, and how IoU, precision and recall are read against a majority baseline.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint is a pickle. It is digest-verified, statically audited against an allow-list (audit digest pinned) and unpickled **once** through torch's weights-only loader to produce the safetensors the model is actually loaded from. No Hub-hosted Python module is imported; `terratorch` is installed from PyPI at a pinned version.",
        "- **Data contract:** a record is `{id, image, label}` — a (6, 512, 512) Sentinel-2 **L1C top-of-atmosphere** reflectance array (or a GeoTIFF; 13-band L1C files are reduced to the six Prithvi bands B2, B3, B4, B8A, B11, B12) with values in [0, 1] or × 10 000, on the 10 m grid (20 m bands resampled), no-data 0 or −9999, and a (512, 512) mask with 0 / 1 / −1. This is what the packaged model was trained on. An atmospherically corrected L2A product or an HLS product is a different input distribution. L1C products from processing baseline 04.00 onwards (from 25 January 2022) carry a +1000 DN offset that must be subtracted before scaling; left in, it adds about 0.1 to every band and still passes the range check. Validation is structural: nothing checks the band order, the product level, the offset, the pixel size or CRS, or that the label belongs to the chip.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — commercial imagery under licence or unreleased disaster assessments are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 88 pinned objects (44 chips and their labels, about 104 MB) and, on a pre-registered T4 run, 46 more (the 23 fresh-test chips and labels, 53 MB) from the public Sen1Floods11 bucket `storage.googleapis.com/sen1floods11` over HTTPS, digest-verified before decoding; Sen1Floods11 is CC BY 4.0 (Cloud to Street).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample chips, validation and roles\n\n"
                "The default dataset is 44 hand-labelled Sentinel-2 chips of Sen1Floods11 — 24 from the official training split, "
                "8 from the validation split and 12 from the test split (the **exposed development test**: scored in every earlier "
                "run and read during the 2026-09-19 build sweep), drawn across the ten flood-event regions from the chips "
                "whose label is at least 60 % valid and 3 % water — fetched by object path from the public bucket and refused on "
                "any byte-size or SHA-256 mismatch (`fetch_corpus`). Each 13-band `S2Hand` GeoTIFF is reduced to the six Prithvi "
                "bands, scaled from reflectance × 10 000 to reflectance, and its no-data replaced by 0, exactly as the upstream "
                "datamodule does; each `LabelHand` mask keeps −1 for cloud / no-data, which every metric ignores. `dataset_manifest` "
                "validates every split, checks that no chip appears twice and records a digest.\n\n"
                "With `USE_BYOD = True` the cell reads your `pairs.csv`, checks that every file it names is present and is a "
                "GeoTIFF (a mistake names the row, the file and the fix), and splits your chips: by your `split` column if there is "
                "one, else by your `group` column so a scene or event never sits in two roles, else chip by chip with a warning. "
                "The per-chip split needs at least seven chips (`byod_minimum_chips`). Every run of this cell first removes this "
                "notebook's earlier exports from `outputs/`, so the files there always describe the current run.\n\n"
                "Look for: 24 / 8 / 12 chips from the official splits (`split_rule`) with water fractions around 0.2..0.4, a written "
                "sample pair (`outputs/{stem}_sample_chip.tif` + `_sample_label.tif`, the BYOD shape), and three refusal probes — a "
                "five-band chip, a label with an unknown class, a chip with reflectance far outside range — each padded with other "
                "chips to the dataset minimum and rejected for its own rule before the model runs.\n\n"
                "*Evaluation practice.* **Predict before running:** cloud and no-data pixels are labelled −1. What would happen to the "
                "baseline's accuracy if they were counted as land?"
            ),
            "code": (
                'import csv\n'
                'import io\n'
                'import json\n'
                'import os\n'
                'import random\n'
                'import shutil\n'
                'import zipfile\n'
                'from pathlib import Path\n'
                '\n'
                'import numpy as np\n'
                'import tifffile\n'
                '\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                '# A .zip or a folder already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n'
                'BYOD_PATH = \'\'  # @param {{type:"string"}}\n'
                '\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                "# A re-run (for example with USE_BYOD = True) first removes this notebook's earlier exports, so outputs/ describes this run only.\n"
                "for stale in sorted(Path('outputs').glob('{stem}_*')):\n"
                '    shutil.rmtree(stale) if stale.is_dir() else stale.unlink()\n'
                '\n'
                '\n'
                'def byod_minimum_records(val_fraction=0.2, test_fraction=0.25):\n'
                '    """The smallest dataset the seeded per-chip split accepts: it keeps MIN_RECORDS chips for training."""\n'
                '    n = MIN_RECORDS\n'
                '    while n - max(1, round(n * test_fraction)) - round(n * val_fraction) < MIN_RECORDS:\n'
                '        n += 1\n'
                '    return n\n'
                '\n'
                '\n'
                'BYOD_MINIMUM = byod_minimum_records()\n'
                '\n'
                '\n'
                'def read_byod_table(path):\n'
                '    """Read pairs.csv and check every file it names before anything is decoded, so a mistake names the row, the file and the fix."""\n'
                '    source = Path(path)\n'
                '    if source.is_dir():\n'
                '        present = lambda name: bool(name) and (source / name).is_file()  # noqa: E731\n'
                '        read = lambda name: (source / name).read_bytes()  # noqa: E731\n'
                "    elif source.is_file() and source.suffix.lower() == '.zip':\n"
                '        archive = zipfile.ZipFile(source)\n'
                "        members = {{Path(n).name: n for n in archive.namelist() if not n.endswith('/')}}\n"
                '        present = lambda name: name in members  # noqa: E731\n'
                '        read = lambda name: archive.read(members[name])  # noqa: E731\n'
                '    else:\n'
                "        raise ValueError(f'{{source.name}}: give a .zip or a folder holding pairs.csv and the GeoTIFF files.')\n"
                "    if not present('pairs.csv'):\n"
                "        raise ValueError(f'{{source.name}} has no pairs.csv: add one with the columns id, image, label (and optionally group or split).')\n"
                "    rows = list(csv.DictReader(io.StringIO(read('pairs.csv').decode('utf-8'))))\n"
                '    for row in rows:\n'
                "        for column, kind in (('image', 'a six-band 512 x 512 GeoTIFF'), ('label', 'a single-band GeoTIFF of 0 / 1 / -1')):\n"
                "            name = (row.get(column) or '').strip()\n"
                '            if not present(name):\n'
                '                raise ValueError(f"pairs.csv row {{row.get(\'id\')!r}}: {{column}} file {{name!r}} is listed but not in {{source.name}}; add the file or correct the row.")\n'
                '            try:\n'
                '                tifffile.TiffFile(io.BytesIO(read(name))).close()\n'
                '            except Exception:\n'
                '                raise ValueError(f"pairs.csv row {{row.get(\'id\')!r}}: {{column}} file {{name!r}} is not a readable GeoTIFF; save it as {{kind}}.") from None\n'
                "    if 'split' not in (rows[0] if rows else {{}}) and len(rows) < BYOD_MINIMUM:\n"
                "        raise ValueError(f'pairs.csv lists {{len(rows)}} chips; bring at least {{BYOD_MINIMUM}} labelled chips (the seeded split keeps {{MIN_RECORDS}} for training).')\n"
                '    return rows\n'
                '\n'
                '\n'
                'def split_byod(records, rows, seed=0, val_fraction=0.2, test_fraction=0.25):\n'
                '    """A `split` column fixes the roles; a `group` column (scene or event) keeps each group in one role; without\n'
                '    either, the carried seeded per-chip split runs and a warning is printed."""\n'
                '    columns = set(rows[0]) if rows else set()\n'
                "    by_id = {{row['id']: row for row in rows}}\n"
                "    if not columns & {{'split', 'group'}}:\n"
                "        print({{'warning': 'pairs.csv has no group or split column, so chips are split one by one: chips cut from one scene or event can land in training and test. Add a group column to keep them together.'}})\n"
                "        return split_dataset(records, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction), 'per chip (seeded; no group column)'\n"
                "    if 'group' in columns:\n"
                "        missing = [r['id'] for r in records if not (by_id[r['id']].get('group') or '').strip()]\n"
                '        if missing:\n'
                "            raise ValueError(f'pairs.csv: {{len(missing)}} chips have no group (first: {{missing[0]!r}}); give every chip a group or remove the column.')\n"
                "        records = [{{**r, 'region': by_id[r['id']]['group'].strip()}} for r in records]\n"
                "    checked = validate_dataset(records)['records']\n"
                "    if 'split' in columns:\n"
                '        role_of = {{}}\n'
                '        for r in checked:\n'
                "            role = (by_id[r['id']].get('split') or '').strip().lower()\n"
                '            if role not in ROLES:\n'
                '                raise ValueError(f"pairs.csv row {{r[\'id\']!r}}: split {{role!r}} must be one of {{\', \'.join(ROLES)}}.")\n'
                "            role_of[r['id']] = role\n"
                "        rule = 'by the split column'\n"
                '    else:\n'
                "        groups = sorted({{r['region'] for r in checked}})\n"
                '        if len(groups) < 3:\n'
                "            raise ValueError(f'{{len(groups)}} groups; a group split needs at least 3 (one each for training, validation and test).')\n"
                '        random.Random(seed).shuffle(groups)\n'
                '        n_test = max(1, round(len(groups) * test_fraction))\n'
                '        n_val = max(1, round(len(groups) * val_fraction))\n'
                "        group_role = {{g: 'test' if i < n_test else 'validation' if i < n_test + n_val else 'train' for i, g in enumerate(groups)}}\n"
                "        role_of = {{r['id']: group_role[r['region']] for r in checked}}\n"
                "        rule = f'by group ({{len(groups)}} groups)'\n"
                "    splits = {{role: [r for r in checked if role_of[r['id']] == role] for role in ROLES}}\n"
                '    empty = [role for role, part in splits.items() if not part]\n'
                '    if empty:\n'
                "        raise ValueError(f'the split {{rule}} leaves no chip for {{empty}}; every role needs at least one.')\n"
                "    if len(splits['train']) < MIN_RECORDS:\n"
                '        raise ValueError(f"the split {{rule}} leaves {{len(splits[\'train\'])}} training chips; at least {{MIN_RECORDS}} are required.")\n'
                '    check_split_disjoint(splits)\n'
                '    return splits, rule\n'
                '\n'
                '\n'
                'if USE_BYOD:\n'
                '    if BYOD_PATH.strip():\n'
                '        byod_path = Path(BYOD_PATH.strip()).expanduser()\n'
                '        if not byod_path.exists():\n'
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding pairs.csv and the GeoTIFF files.')\n"
                '        file_name = byod_path.name\n'
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                '        uploaded = files.upload() or {{}}\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                '        file_name, payload = next(iter(uploaded.items()))\n'
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding pairs.csv and the GeoTIFF files.')\n"
                "        byod_path = Path('work') / file_name\n"
                '        byod_path.parent.mkdir(parents=True, exist_ok=True)\n'
                '        byod_path.write_bytes(payload)\n'
                '    byod_rows = read_byod_table(byod_path)\n'
                '    splits, split_rule = split_byod(load_byod_dataset(byod_path), byod_rows, seed=0)\n'
                "    data_source = 'BYOD (' + file_name + ')'\n"
                'else:\n'
                "    splits = fetch_sample_dataset(cache_dir='weights/sen1floods11')\n"
                "    split_rule = 'the official Sen1Floods11 splits'\n"
                '    data_source = SAMPLE_LABEL_SOURCE\n'
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                '\n'
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'split_rule': split_rule, 'byod_minimum_chips': BYOD_MINIMUM, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'disjoint': dataset_report['disjoint'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "for name, part in dataset_report['splits'].items():\n"
                "    print({{name: {{'water_fraction': part['class_pixel_fraction']['water'], 'ignored_pixels': part['ignored_pixels'], 'regions': part['regions']}}}})\n"
                "print({{'first_test_chip': validate_inputs(test_records[0])}})\n"
                "sample_pair = write_sample_pair(test_records[0], 'outputs/{stem}_sample_chip.tif', 'outputs/{stem}_sample_label.tif')\n"
                "print({{'sample_pair': sample_pair, 'pairs_csv': str(write_dataset_csv(test_records, 'outputs/{stem}_sample_pairs.csv'))}})\n"
                '\n'
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                '# Each probe is padded with other chips up to the dataset minimum, so it is refused for its own rule, never for its size.\n'
                'probe_fill = (test_records[1:] + train_records + val_records)[:MIN_RECORDS - 1]\n'
                'probes = {{\n'
                "    'five-band chip': [{{**test_records[0], 'image': test_records[0]['image'][:5]}}, *probe_fill],\n"
                "    'unknown label class': [{{**test_records[0], 'label': np.where(test_records[0]['label'] == 1, 7, test_records[0]['label'])}}, *probe_fill],\n"
                "    'reflectance out of range': [{{**test_records[0], 'image': test_records[0]['image'] * 50000.0}}, *probe_fill],\n"
                '}}\n'
                'for name, records in probes.items():\n'
                '    try:\n'
                '        validate_dataset(records)\n'
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                '    except (TypeError, ValueError) as exc:\n'
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '**What to notice:** 24 / 8 / 12 chips, the water fractions per split, the ignored-pixel counts, and the three refusals.\n\n<details><summary>Check your '
                'reasoning</summary>It would rise for free: every cloud pixel would count as a correct *land* prediction. Pixels labelled −1 are excluded from every '
                "metric instead, so the baseline's accuracy is exactly the labelled land fraction (0.8049 on the test chips in the recorded run). The refusals (five "
                'bands, an unknown class, reflectance far out of range) stop before any model call and name the rule.</details>'
            ),
        },
        {
            "md": (
                "## 5. The frozen model against the no-water baseline\n\n"
                "`pipe.predict` standardises each chip with the upstream datamodule's band statistics, runs the encoder, neck, "
                "decoder and head in float16 autocast, and returns the argmax mask, the softmax scores (the model's outputs, not "
                "calibrated probabilities) and the water fraction per chip. `pipe.evaluate` pools the labelled pixels of every "
                "chip into one confusion matrix (−1 pixels excluded) and reports the per-class IoU, mean IoU, accuracy, "
                "and the water class's precision, recall and F1; the **no-water baseline** — every pixel predicted as land — is "
                "scored on the same pixels, so its accuracy is exactly the land fraction and its water IoU is 0.\n\n"
                "**Decision rule and who owns the threshold.** Each pixel takes the class with the higher score (argmax, the same "
                "as a 0.5 water score); no threshold is tuned here. Choosing a water threshold — trading missed water against "
                "false alarms — and calibrating the scores belong to whoever deploys the model, on validation chips from their own "
                "area and sensor; this notebook does neither.\n\n"
                "This section scores the validation chips and the 12 chips of the **exposed development test**. They are held out "
                "only at chip level: all ten of their events are in the upstream training split and six of them touch a training "
                "or validation chip, so their numbers are within-event, and they were already read when the defaults were chosen. "
                "The fresh test is scored only in Section 8.\n\n"
                "Look for: a water IoU well above 0 on the exposed development test (0.730, F1 0.844, in the Colab CLI T4 run of "
                "2026-10-08), a validation water IoU around 0.86, precision against recall, and per-chip water fractions that track "
                "the labels. These are sample-sanity numbers on 12 and 8 chips, not the benchmark. If you re-run this cell after "
                "Section 6, it first puts the adapted tensors back to the pinned base, so *frozen* always means the packaged model.\n\n"
                "**Predict before running:** the packaged model was fine-tuned on Sen1Floods11. Will the exposed development chips "
                "(ten regions) or the validation chips be easier for it?"
            ),
            "code": (
                "import time\n\n"
                "t0 = time.perf_counter()\n"
                "# SWP-F: the pinned base of every tensor adaptation may change is kept once, before any training; restore_pinned_base()\n"
                "# puts it back. Section 5 scores the pinned base on a re-run (then re-run Sections 6-9 in order), and Section 6 starts\n"
                "# every adaptation from it.\n"
                "if '_PINNED_BASE' not in globals():\n"
                "    _adaptable = set(pipe._trainable('decoder+last_block'))\n"
                "    _PINNED_BASE = {{k: v.detach().clone() for k, v in pipe.model.state_dict().items() if k in _adaptable}}\n\n"
                "def restore_pinned_base(pipeline):\n"
                "    current = pipeline.model.state_dict()\n"
                "    changed = sorted(k for k, v in _PINNED_BASE.items() if not bool((current[k] == v.to(current[k].device)).all()))\n"
                "    if changed:\n"
                "        pipeline.model.load_state_dict({{**current, **{{k: _PINNED_BASE[k].to(current[k].device) for k in changed}}}}, strict=True)\n"
                "        pipeline.model.eval()\n"
                "    pipeline.adapter = None\n"
                "    return changed\n\n"
                "restored_tensors = restore_pinned_base(pipe)\n"
                "if restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'adapted tensors put back to the pinned base; re-run Sections 6-9 in order'}})\n"
                "# The exposed development test (the 12 former test chips) and validation; the fresh test is scored only in Section 8.\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "frozen_val = pipe.evaluate(val_records)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'split': 'exposed development test', 'metric': frozen_test['metric']}})\n"
                "print({{'baseline_no_water_test': {{k: frozen_test['baseline_no_water'][k] for k in ('iou', 'accuracy', 'f1')}}}})\n"
                "print({{'frozen_test': {{k: frozen_test['model'][k] for k in ('iou', 'mean_iou', 'accuracy', 'precision', 'recall', 'f1')}}}})\n"
                "print({{'frozen_validation': {{k: frozen_val['model'][k] for k in ('iou', 'f1')}}}})\n"
                "frozen_predictions = pipe.predict(test_records)\n"
                "for record, pred in list(zip(test_records, frozen_predictions['predictions']))[:6]:\n"
                "    labelled = record['label'] >= 0\n"
                "    print({{'chip': record.get('source_id', record['id']), 'water_label': round(float((record['label'] == 1).sum() / labelled.sum()), 3), 'water_predicted': pred['class_fraction']['water'], 'ignored': int((~labelled).sum())}})\n"
                "print({{'decision_rule': frozen_predictions['decision_rule'], 'scores_shape': frozen_predictions['predictions'][0]['scores'].shape}})"
            ),
        },
        {
            "md": (
                '**What to notice:** `baseline_no_water_test` beside `frozen_test`, precision against recall, the validation IoU, and the per-chip water fractions.\n\n<details><summary>Check '
                "your reasoning</summary>Validation. In the recorded run the frozen exposed-development water IoU was 0.7301 (F1 0.8440, accuracy 0.9440 against the baseline's "
                '0.8049) while validation reached 0.8614. Twelve chips from ten regions include harder scenes; differences between splits this small say more '
                'about which chips were drawn than about the model. Note also the direction of the errors: precision 0.9251 is above recall 0.776, so '
                'the frozen model misses some water rather than inventing it.</details>'
            ),
        },
        {
            "md": (
                "## 6. The pre-registered validation grid of bounded fine-tunings\n\n"
                "`pipe.adapt` trains the 46 tensors of the pyramid neck, the UPerNet decoder and the head (15.1 M parameters — "
                "4.7 % of the model) and nothing else: the ViT-L encoder is frozen, and every BatchNorm layer keeps its running "
                "statistics, because batches of two chips would corrupt them. Each step takes two chips with a seeded flip, computes "
                "the cross-entropy over the labelled pixels (−1 ignored) and takes an AdamW step with gradient-norm clipping and "
                "float16 loss scaling. Epoch 0 records the frozen model's validation loss; the epoch with the lowest validation loss "
                "is kept — which can be epoch 0, since the packaged model already trained on this dataset.\n\n"
                "**Where the defaults came from, and the grid.** The default learning rate 10⁻⁵ came from a 2026-09-19 build sweep "
                "(`MODEL_CARD.md`) that read the 12 test chips' IoU for each arm — so those chips are now the *exposed development "
                "test* — and that sweep's numbers also carried a reflectance-scaling bug fixed on 2026-09-25. The choice is now made "
                "on validation only, by the rule fixed in `docs/prithvi-flood-preregistration.md`: three configurations — **C0** "
                "(10⁻⁵, `decoder`, the default), **C1** (3 × 10⁻⁵, `decoder`) and **C2** (10⁻⁵, `decoder+last_block`, 27.7 M "
                "parameters) — each trained from the pinned base with everything else fixed; a challenger replaces C0 only if its "
                "kept epoch's validation loss is lower by at least 0.005. The 8 validation chips also steered the upstream early "
                "stopping, so this selection is not independent of the packaged model.\n\n"
                "**On a GPU the grid runs** (about a minute on a T4). **On CPU it is off**: the cell trains the form's settings "
                "(the recorded selection, C0, until a hosted record exists) and the run is exploratory. Training is deterministic: "
                "`set_determinism` switches on `torch.use_deterministic_algorithms(True)` without `warn_only` and the cuDNN flags, "
                "and because torch has no deterministic CUDA kernel for the bilinear resize, the adaptive pooling in the UperNet "
                "decoder or the masked cross-entropy, the carried pipeline replaces those three with equivalent matmul and "
                "elementwise versions (`DETERMINISTIC_REPLACEMENTS`, patched into the rebuilt TerraTorch model; the library is not "
                "edited). The printed `grid_hash` lets two runs be compared digit for digit.\n\n"
                "**Predict before running:** will either challenger beat C0 by 0.005 in validation loss?"
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "BATCH_SIZE = 2  # @param {{type:\"integer\"}}\n"
                "RUN_GRID = True  # @param {{type:\"boolean\"}}\n"
                "# Used only when the grid is off (CPU, or RUN_GRID = False): any value other than the recorded selection is exploratory.\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "TRAINABLE = 'decoder'  # @param [\"decoder\", \"decoder+last_block\"]\n\n"
                "determinism = set_determinism(0)\n"
                "print({{'determinism': determinism, 'replaced_ops': list(DETERMINISTIC_REPLACEMENTS), 'patched_sites': len(getattr(pipe.model, 'deterministic_patches', []))}})\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'val' in entry:\n"
                "        row['val_water_iou'] = entry['val']['iou']['water']\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "def run_config(lr, trainable):\n"
                "    restore_pinned_base(pipe)  # SWP-F: every adaptation starts from the pinned base\n"
                "    result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=lr, batch_size=BATCH_SIZE, trainable=trainable, seed=0, progress=report)\n"
                "    names = set(result['trainable_names'])\n"
                "    return result, {{k: v.detach().cpu().clone() for k, v in pipe.model.state_dict().items() if k in names}}\n\n"
                "grid_on = RUN_GRID and pipe.device.startswith('cuda')\n"
                "t0 = time.perf_counter()\n"
                "grid_runs = {{}}\n"
                "if grid_on:\n"
                "    for cfg in GRID:\n"
                "        print({{'config': cfg['name'], 'lr': cfg['lr'], 'trainable': cfg['trainable']}})\n"
                "        grid_runs[cfg['name']] = run_config(cfg['lr'], cfg['trainable'])\n"
                "    grid_rows = [grid_row(name, run[0]) for name, run in grid_runs.items()]\n"
                "    selection = select_config(grid_rows)\n"
                "else:\n"
                "    name = config_for(LEARNING_RATE, TRAINABLE) or 'off-grid'\n"
                "    grid_runs[name] = run_config(LEARNING_RATE, TRAINABLE)\n"
                "    grid_rows = [grid_row(name, grid_runs[name][0])]\n"
                "    selection = {{'selected': name, 'rule': 'grid off (' + ('RUN_GRID = False' if RUN_GRID is False else 'no GPU') + '): trained the form settings; exploratory, validation only'}}\n"
                "selected_name = selection['selected']\n"
                "GRID_HASH = grid_hash(grid_rows, selected_name)\n"
                "# The selected system: reset to the pinned base, then overlay the selected config's kept state.\n"
                "adapt_result, selected_state = grid_runs[selected_name]\n"
                "restore_pinned_base(pipe)\n"
                "_current = pipe.model.state_dict()\n"
                "pipe.model.load_state_dict({{**_current, **{{k: v.to(_current[k].device) for k, v in selected_state.items()}}}}, strict=True)\n"
                "pipe.model.eval()\n"
                "pipe.adapter = dict(adapt_result)\n"
                "SELECTED_FINGERPRINT = state_fingerprint(pipe.model, adapt_result['trainable_names'])\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "for row in grid_rows:\n"
                "    print({{'grid_row': row}})\n"
                "print({{'grid': 'on' if grid_on else 'off (exploratory)', 'selection': selection, 'grid_hash': GRID_HASH}})\n"
                "print({{'selected': selected_name, 'trainable_parameters': adapt_result['n_trainable'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['precision'], 'batchnorm': adapt_result['batchnorm'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** each config\'s epoch 0 (`note: frozen model`) and validation loss per epoch, the `selection` line, and the `grid_hash`.\n\n<details><summary>Check '
                'your reasoning</summary>In the evaluation of record (hosted T4, 19 October 2026) C0\'s validation loss went from 0.1306 (frozen) to its minimum 0.1178 at '
                'epoch 2 and rose again by epoch 4, so epoch 2 was kept (validation water IoU 0.8614 → 0.8747). A challenger had to reach 0.1128 or lower to replace C0: '
                'C1 kept 0.1262 and C2 0.1177, only 0.00005 below C0, so C0 was kept and `grid_hash` began `68b0a351`. Two masked T4 runs printed the same hash, '
                'so under the deterministic settings this table repeats exactly on a T4 (`docs/release-verification.md`). Validation selection with the frozen model as a candidate stops drift.</details>'
            ),
        },
        {
            "md": (
                "## 7. The exposed development test: a paired comparison for continuity\n\n"
                "The selected system is scored on the same 12 chips as the frozen model in Section 5, side by side with the "
                "baseline. These chips are the **exposed development test**: they were scored in every earlier run and their IoU "
                "was read during the 2026-09-19 build sweep that fixed the default learning rate, so they are reported for "
                "continuity and are *not* the endpoint — that is the fresh test in Section 8. The cell stops only on what the "
                "procedure guarantees — the kept epoch's validation loss is no higher than the frozen model's, and re-scoring the "
                "validation chips reproduces the kept epoch's positive-class IoU within 0.01 — and writes "
                "`outputs/{stem}_evaluation_report.json` atomically (a temporary file, then a rename). In the recorded run the water "
                "IoU on these chips moved from 0.730 to 0.746: 12 chips, one seed, no interval, not a finding. A figure then shows "
                "two chips in false colour beside their label, the frozen and adapted masks, and where the adapted mask finds water "
                "(blue), misses it (red) or invents it (amber).\n\n"
                "*Evaluation practice.* **Predict before running:** if the water IoU here rises by about 0.015, is that evidence the "
                "fine-tuning helped?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{'split': 'exposed development test (12 chips; scored in every earlier run; descriptive only)'}}\n"
                "for key in ('mean_iou', 'accuracy', 'precision', 'recall', 'f1'):\n"
                "    comparison[key] = {{'baseline_no_water': frozen_test['baseline_no_water'][key], 'frozen': frozen_test['model'][key], 'adapted': adapted_test['model'][key]}}\n"
                "comparison['water_iou'] = {{'baseline_no_water': frozen_test['baseline_no_water']['iou']['water'], 'frozen': frozen_test['model']['iou']['water'], 'adapted': adapted_test['model']['iou']['water']}}\n"
                "comparison['change_water_iou'] = round(adapted_test['model']['iou'][CLASS_NAMES[1]] - frozen_test['model']['iou'][CLASS_NAMES[1]], 4)\n"
                "comparison['reading'] = 'descriptive: the exposed development test decides nothing; the endpoint is the fresh test (Section 8)'\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "print({{'validation_water_iou': {{'frozen': frozen_val['model']['iou']['water'], 'adapted': adapted_val['model']['iou']['water']}}, 'validation_loss': {{'frozen': adapt_result['history'][0]['val_loss'], 'kept_epoch': adapt_result['history'][adapt_result['best_epoch']]['val_loss']}}}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'frozen': {{'exposed_development_test': frozen_test, 'validation': frozen_val}},\n"
                "    'adapted': {{'exposed_development_test': adapted_test, 'validation': adapted_val}},\n"
                "    'comparison': comparison,\n"
                "    'grid': {{'on': grid_on, 'rows': grid_rows, 'selection': selection, 'grid_hash': GRID_HASH}},\n"
                "    'determinism': determinism,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "atomic_write_json('outputs/{stem}_evaluation_report.json', evaluation_report)\n"
                "# Contract integrity (not model quality): epoch 0 is a selection candidate, and re-scoring reproduces the kept epoch.\n"
                "if adapt_result['history'][adapt_result['best_epoch']]['val_loss'] > adapt_result['history'][0]['val_loss']:\n"
                "    raise RuntimeError('contract: the kept epoch has a higher validation loss than the frozen model, which epoch selection cannot produce')\n"
                "if abs(adapted_val['model']['iou'][CLASS_NAMES[1]] - adapt_result['history'][adapt_result['best_epoch']]['val']['iou'][CLASS_NAMES[1]]) >= 1e-2:\n"
                "    raise RuntimeError('contract: re-scoring the validation chips does not reproduce the kept epoch')\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})\n"
                '\n'
                '# Two exposed development chips: false colour, label, frozen and adapted masks, and where the adapted model is right or wrong.\n'
                'import matplotlib.pyplot as plt\n'
                '\n'
                'shown = test_records[:2]\n'
                "adapted_shown = pipe.predict(shown)['predictions']\n"
                'fig, axes = plt.subplots(len(shown), 5, figsize=(17, 3.6 * len(shown)), squeeze=False)\n'
                "for row, (record, frozen_pred, adapted_pred) in enumerate(zip(shown, frozen_predictions['predictions'], adapted_shown)):\n"
                "    label, mask = record['label'], adapted_pred['mask']\n"
                "    composite = np.clip(np.stack([record['image'][5], record['image'][3], record['image'][2]], axis=-1) / 0.4, 0.0, 1.0)\n"
                '    errors = np.full(label.shape + (3,), 0.95)\n'
                '    errors[(label == 1) & (mask == 1)] = (0.2, 0.45, 0.9)\n'
                '    errors[(label == 1) & (mask == 0)] = (0.85, 0.15, 0.15)\n'
                '    errors[(label == 0) & (mask == 1)] = (0.95, 0.7, 0.1)\n'
                '    errors[label < 0] = (0.55, 0.55, 0.55)\n'
                '    panels = (\n'
                "        (composite, record.get('source_id', record['id']) + ': false colour (SWIR 2, NIR, red)'),\n"
                "        (np.ma.masked_less(label, 0), 'label (water dark; blank = no data)'),\n"
                "        (frozen_pred['mask'], 'frozen mask'),\n"
                "        (mask, 'adapted mask'),\n"
                "        (errors, 'adapted: blue found, red missed, amber false'),\n"
                '    )\n'
                '    for ax, (image, title) in zip(axes[row], panels):\n'
                "        ax.imshow(image, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')\n"
                '        ax.set_title(title, fontsize=9)\n'
                "        ax.axis('off')\n"
                'fig.tight_layout()\n'
                'plt.show()'
            ),
        },
        {
            "md": (
                '**What to notice:** the `water_iou` row (baseline, frozen, adapted), `change_water_iou`, and the validation numbers beside it.\n\n<details><summary>Check '
                'your reasoning</summary>No. In the evaluation of record the water IoU on these chips moved from 0.7301 to 0.7459 (+0.0158; F1 0.8440 → 0.8544, '
                'accuracy 0.9440 → 0.9468): recall rose from 0.776 to 0.8004 while precision fell from 0.9251 to 0.9164. These 12 chips were already read when the default '
                'learning rate was chosen, they come from events in the training split, and there is one seed and no interval; the change shows the contract '
                'ran and did no harm. The pre-registered fresh test in Section 8 is where a change is read, with an interval and a tolerance; in the '
                'evaluation of record this gain did not show up there.</details>'
            ),
        },
        {
            "md": (
                "## 8. The pre-registered fresh test\n\n"
                "The **fresh test** is 23 official Sen1Floods11 test chips that this notebook never scored: the test chips outside "
                "the sample that pass the same label filter (at least 60 % valid, 3 % water), minus the two upstream example chips "
                "shown in Section 9, pinned by size and SHA-256 (`FRESH_TEST_RECORDS`, digest `FRESH_POOL_DIGEST`). **Plainly:** "
                "they come from the same 8 flood events as the training data and 20 of the 23 lie within one chip width of a "
                "training or validation chip, so this measures *within-event generalisation to unseen chips*, not new events (the "
                "official Bolivia event, held out from everything, would be an out-of-event check; it is not used here).\n\n"
                "**Reading rule (pre-registered).** One primary endpoint: the pooled water IoU of the selected system minus the "
                "frozen model on the fresh test, from unrounded per-chip confusion counts, with a paired bootstrap 95 % interval "
                "over chips (10 000 resamples, seed 0). The verdict is *improved* or *worse* only if the interval excludes 0 and "
                "the change is at least 0.02; otherwise *no detectable change*. With 23 correlated chips from 8 events the power "
                "is low: the expected verdict for a small fine-tuning effect is *no detectable change*, and the interval "
                "understates event-level uncertainty.\n\n"
                "**Score-once.** The cell scores the fresh test only if every pre-registered setting holds — the pinned sample, "
                "epochs 4, batch 2, seed 0, the full grid and its selection, float16 on a **Tesla T4**, the determinism flags, "
                "the bootstrap — and re-checks, just before scoring, that the grid is complete and matches, that the model state "
                "is the selected system, and the fresh-pool digest. It resets to the pinned base to score the frozen model, then "
                "reloads the selected state and checks its fingerprint before scoring it. Any other run (CPU, another GPU, BYOD, a "
                "form edit, the optional activity, `DIMER_MASK_FRESH_TEST=1`) is **exploratory**: the fresh test is not fetched "
                "or scored, and a record of validation and the exposed development test goes to "
                "`outputs/score_once/{stem}_fresh_test_exploratory_<n>.json` with the settings that differ. The first completed "
                "scoring in a runtime is labelled *pre-registered evaluation (this runtime)*; a repeat is a *reproduction* "
                "(`_reproduction_<n>`). A runtime cannot know about other runtimes: the evaluation of record is the first hosted "
                "T4 run listed in `docs/release-verification.md`. It is recorded (19 October 2026), so every scoring here is a "
                "*reproduction*, and the cell prints `matches_record` with the fields that differ from it. A failed "
                "fetch, decode or check is not a look; once scoring completes it is a look, even if a later write fails. A look "
                "marker is kept beside the fresh-test cache (`weights/sen1floods11_fresh/score_once_look_marker.json`), so "
                "deleting `outputs/` or `outputs/score_once/` does **not** reset the look; the report and the ledger are "
                "written atomically. The cell prints `masked`, `preregistered` and `will_fetch_fresh_test` before anything is "
                "fetched, and `DIMER_MASK_FRESH_TEST` is re-read where the fresh test would be fetched and scored, so setting it "
                "in any cell before Section 8 masks the run.\n\n"
                "**Predict before running:** will the interval exclude 0?"
            ),
            "code": (
                "MASKED = os.environ.get(MASK_ENV) == '1'  # re-read here; score_fresh_test and fetch_fresh_test re-read it again\n"
                "device_name = torch.cuda.get_device_name(0) if pipe.device.startswith('cuda') else 'cpu'\n"
                "run_settings = {{\n"
                "    'data_source': data_source,\n"
                "    'splits': {{'train': len(train_records), 'validation': len(val_records), 'exposed_development_test': len(test_records), 'fresh_test': len(FRESH_TEST_RECORDS)}},\n"
                "    'dataset_digest': dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})['digest'],  # recomputed from the live records\n"
                "    'fresh_pool_digest': fresh_pool_digest(),\n"
                "    'epochs': adapt_result['epochs'],\n"
                "    'batch_size': adapt_result['batch_size'],\n"
                "    'seed': adapt_result['seed'],\n"
                "    'grid': [{{'name': n, 'lr': r[0]['lr'], 'trainable': r[0]['trainable']}} for n, r in grid_runs.items()],\n"
                "    'selection_margin': SELECTION_MARGIN,\n"
                "    'precision': adapt_result['precision'],\n"
                "    'device': device_name,\n"
                "    'determinism': determinism_state(),\n"
                "    'n_boot': N_BOOT,\n"
                "    'boot_seed': BOOT_SEED,\n"
                "    'masked': MASKED,\n"
                "}}\n"
                "checks = {{\n"
                "    'grid_complete_and_matching': grid_on and [r['name'] for r in grid_rows] == [c['name'] for c in GRID] and select_config(grid_rows)['selected'] == selected_name and grid_hash(grid_rows, selected_name) == GRID_HASH,\n"
                "    'model_is_selected_system': pipe.adapter is not None and state_fingerprint(pipe.model, adapt_result['trainable_names']) == SELECTED_FINGERPRINT,\n"
                "    'fresh_pool_digest': fresh_pool_digest() == FRESH_POOL_DIGEST,\n"
                "}}\n\n"
                "def score_fresh():\n"
                "    fresh_records = fetch_fresh_test(cache_dir='weights/sen1floods11_fresh')  # a failure here is not a look\n"
                "    try:\n"
                "        restore_pinned_base(pipe)  # reset: the frozen model is recomputed, never taken from memory\n"
                "        frozen = chip_confusions(pipe, fresh_records)\n"
                "    finally:\n"
                "        current = pipe.model.state_dict()\n"
                "        pipe.model.load_state_dict({{**current, **{{k: v.to(current[k].device) for k, v in selected_state.items()}}}}, strict=True)\n"
                "        pipe.model.eval()\n"
                "        pipe.adapter = dict(adapt_result)\n"
                "    if state_fingerprint(pipe.model, adapt_result['trainable_names']) != SELECTED_FINGERPRINT:\n"
                "        raise RuntimeError('the reloaded state is not the selected system')\n"
                "    selected = chip_confusions(pipe, fresh_records)\n"
                "    return {{'chips': [r['source_id'] for r in fresh_records], 'frozen': frozen, 'selected': selected}}\n\n"
                "exploratory_payload = {{'validation_water_iou': {{'frozen': frozen_val['model']['iou']['water'], 'adapted': adapted_val['model']['iou']['water']}}, 'exposed_development_test': comparison, 'grid': grid_rows, 'selection': selection, 'grid_hash': GRID_HASH, 'determinism': run_settings['determinism'], 'device': device_name}}\n"
                "pre_check = preregistration_check(run_settings)\n"
                "print({{'masked': MASKED, 'preregistered': pre_check['preregistered'], 'will_fetch_fresh_test': pre_check['preregistered'] and all(checks.values()), 'settings_differing': pre_check['settings_differing']}})  # before any fetch\n"
                "try:\n"
                "    fresh_result = score_fresh_test(settings=run_settings, checks=checks, score=score_fresh, score_dir='outputs/score_once', stem='{stem}', exploratory_payload=exploratory_payload, marker=LOOK_MARKER)\n"
                "except Exception as exc:  # a failed fetch, decode or check BEFORE scoring completed: not a look; the rest of the notebook still runs\n"
                "    fresh_result = {{'test_look': 'not scored (failed before scoring completed; not a look)', 'error': str(exc)[:200]}}\n"
                "if fresh_result['test_look'] == LABEL_EXPLORATORY:\n"
                "    print({{'fresh_test': 'not scored (exploratory)', 'masked': MASKED, 'settings_differing': fresh_result['settings_differing'], 'record': fresh_result['file']}})\n"
                "elif 'bootstrap' in fresh_result:\n"
                "    boot = fresh_result['bootstrap']\n"
                "    print({{'test_look': fresh_result['test_look'], 'chips': boot['n_chips'], 'frozen_water_iou': round(boot['frozen_water_iou'], 4), 'selected_water_iou': round(boot['selected_water_iou'], 4), 'change': round(boot['delta'], 4), 'ci95': [round(v, 4) for v in boot['ci95']], 'verdict': fresh_result['verdict'], 'report': fresh_result['file'], 'write_errors': fresh_result.get('write_errors', [])}})\n"
                "    comparison_to_record = compare_with_record(fresh_result, GRID_HASH, selected_name)  # None without a record\n"
                "    if comparison_to_record is not None:\n"
                "        print({{'reproduction_of': comparison_to_record['record'], 'matches_record': comparison_to_record['matches_record'], 'differs': comparison_to_record['differs']}})\n"
                "else:\n"
                "    print(fresh_result)"
            ),
        },
        {
            "md": (
                '**What to notice:** `test_look` (or why the fresh test was not scored and which settings differ), the `change` with its `ci95`, and the `verdict`.\n\n<details><summary>Check '
                'your reasoning</summary>Probably not. On 23 within-event chips a fine-tuning effect of the size seen on the exposed development test (about 0.015) is '
                'below the 0.02 tolerance, so the pre-registered reading is *no detectable change* unless the interval is narrow and the change large. The '
                'evaluation of record (hosted T4, 19 October 2026, commit `9fd2b76`) read exactly that: frozen 0.8815, selected C0 0.8816, change 0.0, '
                '95 % interval [−0.0024, 0.0021], *no detectable change*. The +0.0158 on the exposed development test did not show up on the 23 fresh '
                'chips: the frozen model is already strong on these in-distribution events (0.88), and 12 already-read chips overstated the gain. '
                'Your T4 run with the defaults is a *reproduction*; `matches_record: True` means the deterministic settings reproduced it, and `differs` '
                'names any field that did not. `docs/release-verification.md` lists the record and its reproductions. On '
                'CPU the cell reports *not scored (exploratory)* and names the settings that differ (`grid`, `precision`, `device`; and '
                '`masked` when `DIMER_MASK_FRESH_TEST=1` is set).</details>'
            ),
        },
        {
            "md": (
                "## 9. Upstream example chips, artifact export and fresh reload\n\n"
                "The selected system segments the three example chips that ship with the upstream repository (`India_900498`, "
                "`Spain_7370579`, `USA_430764` — 13-band Sentinel-2 L1C files reduced to the six bands). All three come from the "
                "upstream hand-labelled **test** split, so the packaged model never trained on them, but they are not new here: "
                "`USA_430764` is one of the 12 exposed development chips (the cell prints `also_a_sample_chip: 'test'` for it), and "
                "`India_900498` and `Spain_7370579` are excluded from the fresh test because they are shown here. Their upstream "
                "labels are not fetched, so the predicted water fraction and the written mask are a sanity check, not an "
                "evaluation.\n\n"
                "`pipe.save_artifact` writes the trained tensors as `adapter.safetensors` into a temporary directory with a "
                "`manifest.json` (artifact format, base model id and revision, the digest of the converted base file, the adaptation "
                "scope, the tensor names, file size and SHA-256, training configuration and epoch history; OUT8) and renames it into "
                "place only when complete. `PrithviFloodPipeline.from_artifact` re-verifies the base file, checks the artifact "
                "manifest, scope and digest **before** deserialising, rebuilds the model and overlays the tensors — a fresh object "
                "from files, not the in-memory model (VER2). The cell asserts the same exposed-development positive-class IoU within 0.001 and score maps within 0.01 (VER4: float16 tolerances; on one device they are usually identical)."
            ),
            "code": (
                "import importlib.metadata\n"
                "import io\n"
                "import platform\n\n"
                "import tifffile\n\n"
                "example_dir = WEIGHTS_DIR / 'examples'\n"
                "new_records = [{{'id': path.stem, 'image': read_chip(path, band_indices=S2_L1C_BAND_INDICES), 'source': str(path.name)}} for path in sorted(example_dir.glob('*.tif'))]\n"
                "new_predictions = pipe.predict(new_records)\n"
                "sample_role = {{name: role for name, role, *_ in SAMPLE_RECORDS}}  # the 44 sample chips by scene\n"
                "for record, pred in zip(new_records, new_predictions['predictions']):\n"
                "    buffer = io.BytesIO()\n"
                "    tifffile.imwrite(buffer, pred['mask'])\n"
                "    atomic_write_bytes(f'outputs/{stem}_mask_' + record['id'] + '.tif', buffer.getvalue())\n"
                "    print({{'chip': record['id'], 'water_fraction': pred['class_fraction']['water'], 'also_a_sample_chip': sample_role.get(record['id'].removesuffix('_S2Hand')), 'note': 'upstream test-split chip; no label fetched here'}})\n"
                "atomic_write_json('outputs/{stem}_predictions.json', {{'model': new_predictions['model'], 'classes': new_predictions['classes'], 'decision_rule': new_predictions['decision_rule'], 'predictions': [{{'id': p['id'], 'class_fraction': p['class_fraction']}} for p in new_predictions['predictions']]}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source, 'selected_config': selected_name, 'grid_hash': GRID_HASH}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'trainable': artifact_manifest['adapter']['trainable'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = PrithviFloodPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)  # the exposed development test, never the fresh test\n"
                "before = pipe.predict(test_records[:2])['predictions']\n"
                "after = reloaded.predict(test_records[:2])['predictions']\n"
                "parity = {{'positive_iou_diff': round(abs(reloaded_test['model']['iou'][CLASS_NAMES[1]] - adapted_test['model']['iou'][CLASS_NAMES[1]]), 6), 'metrics_identical': reloaded_test['model'] == adapted_test['model'], 'max_abs_score_diff': max(float(np.abs(a['scores'] - b['scores']).max()) for a, b in zip(before, after))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['positive_iou_diff'] < 1e-3 and parity['max_abs_score_diff'] < 1e-2\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'device_name': device_name, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_asset': [e for e in MANIFEST['files'] if e['path'] == SOURCE_CKPT_NAME],\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickle_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "        'deterministic_patches': list(getattr(pipe.model, 'deterministic_patches', [])),\n"
                "        'data_objects': 2 * len(SAMPLE_RECORDS),\n"
                "        'fresh_test_objects': 2 * len(FRESH_TEST_RECORDS),\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'lightning': lightning.__version__, 'tifffile': tifffile.__version__, 'terratorch': importlib.metadata.version('terratorch')}},\n"
                "    'data_source': data_source,\n"
                "    'determinism': run_settings['determinism'],\n"
                "    'grid': {{'on': grid_on, 'rows': grid_rows, 'selection': selection, 'grid_hash': GRID_HASH}},\n"
                "    'exposed_development_test': comparison,\n"
                "    'fresh_test': {{k: fresh_result.get(k) for k in ('test_look', 'settings_differing', 'bootstrap', 'verdict', 'file', 'error')}},\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "atomic_write_json('outputs/{stem}_result.json', result_payload)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
        {
            "md": (
                "**What to notice:** the water fraction of the three example chips (and which one is an exposed development chip), the artifact's size and tensor count, and `reload_parity`.\n\n<details><summary>Check "
                'your reasoning</summary>No labels are fetched for the example chips, so their water fractions are a sanity check only; `USA_430764` repeats an '
                'exposed development chip. In the recorded run the 46-tensor, about 60 MB adapter reloaded into a fresh pipeline with identical metrics '
                '(`positive_iou_diff` 0.0, `metrics_identical` True, `max_abs_score_diff` 0.0); identity was observed there, and the check itself is the float16 '
                'tolerance.</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**On the exposed development test**, 12 chips from the official test split, the packaged flood model finds water "
        "with an IoU near 0.73 (F1 0.84 in the recorded run) against a no-water baseline that scores 0, and its precision "
        "(0.93) is well above its recall (0.78): it misses some water rather than inventing it. The selected fine-tuning "
        "moved that IoU from 0.7301 to 0.7459 in the evaluation of record: recall rose from 0.776 to 0.8004 while precision "
        "fell from 0.9251 to 0.9164. Those chips are held out only "
        "at chip level and were read when the defaults were chosen, so they describe, they do not decide. **The fresh test** "
        "(Section 8) is the one pre-registered reading: 23 unseen chips from the same 8 events, a paired bootstrap interval "
        "and a 0.02 tolerance, scored only on a T4 under the pre-registered settings; it measures within-event "
        "generalisation, not new events, and its power is low. In the evaluation of record (hosted T4, 19 October 2026) it read "
        "*no detectable change* (frozen 0.8815, selected 0.8816, 95 % interval [−0.0024, 0.0021]): the exposed-test gain did not "
        "show up on fresh chips. **If you ran your own chips (BYOD),** the run is exploratory, "
        "the numbers describe your chips under the split printed as `split_rule` in Section 4, and the frozen numbers are "
        "the packaged model's. That is the claim: the adaptation contract runs end to end on real labelled multispectral "
        "chips with deterministic training, the pickle is audited and converted rather than served, and the 60 MB artifact "
        "reloads with outputs equal within the float16 tolerances (identical in the recorded run). It is not a claim that "
        "this sample improves the model — the model already trained on this dataset — nor that these chips measure its "
        "skill.\n\n"
        "The numbers are sample-sanity evidence: pixel-pooled metrics that let large chips dominate, hand labels that "
        "carry their own uncertainty at water edges and under thin cloud, and validation chips that also steered the "
        "upstream early stopping. Nothing here measures the model on Sentinel-1, on scenes larger than a chip, or on "
        "regions, events and seasons outside Sen1Floods11.\n\n"
        "Three things to carry to real data. **The input is Sentinel-2 L1C top-of-atmosphere reflectance in six bands:** "
        "blue, green, red, narrow NIR, SWIR 1, SWIR 2 in that order, reflectance in [0, 1], with the post-2022 +1000 DN "
        "offset removed and the 20 m bands on the 10 m grid. An L2A (surface reflectance) or HLS product, a different band "
        "order or a left-in offset is segmented without complaint and silently shifted from what the model learned. "
        "**Split by scene or event, not by chip:** neighbouring chips are near-duplicates, and a random split makes "
        "memorisation look like skill — which is why the exposed development and fresh tests here are labelled "
        "within-event. **Read the baseline first:** on a chip with 5 % water the no-water baseline is 95 % accurate; only "
        "the water IoU, precision and recall say whether the model did anything.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify a pickled upstream checkpoint, audit and convert it into safetensors without "
        "executing anything outside the audited allow-list, rebuild the model from the installed package, fetch and validate "
        "digest-pinned real labelled chips, run a pre-registered validation grid with deterministic kernels, evaluate against "
        "a baseline and the frozen model, apply fail-closed score-once rules to a fresh test, and emit the shown "
        "machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, production fitness, or flood-mapping skill beyond the checks shown.\n\n"
        "## Optional activity: Predict → Change → Run → Observe → Explain\n\n"
        "This does not affect the default path. **Any change here makes the run exploratory:** Section 8 then does not fetch "
        "or score the fresh test and records validation and the exposed development test only, with the settings that "
        "differ.\n\n"
        "1. **Predict:** with `TRAINABLE = 'decoder+last_block'` the last encoder block (27.7 M more parameters) is trained "
        "too. Will the kept epoch's validation loss be lower than with `'decoder'`? Write your guess down.\n"
        "2. **Change** `RUN_GRID = False` and `TRAINABLE` in Section 6; keep `EPOCHS`, `LEARNING_RATE` and `BATCH_SIZE`.\n"
        "3. **Run** Sections 6, 7, 8 and 9 in that order. Section 6 first restores the pinned base, so this is a fresh "
        "experiment rather than continued training, and epoch 0 is still the frozen model.\n"
        "4. **Observe** `best_epoch`, the validation loss per epoch, the exposed-development `water_iou`, `precision` and "
        "`recall` rows, the Section 8 line *not scored (exploratory)*, and the figure.\n"
        "5. **Explain** whether a larger trainable share helped on 24 chips whose distribution the model has already seen, "
        "and why an exploratory run must not look at the fresh test.\n\n"
        "**Scope of a re-run:** every variation here is Sections 6 → 9; to return to the pre-registered settings, restore the "
        "defaults and run the same sections (on a T4 a second scoring in the same runtime is labelled a *reproduction*). "
        "Other variations to predict before running: raise `EPOCHS` (does the validation loss keep falling?) or set "
        "`LEARNING_RATE = 1e-4` (which epoch does validation selection keep, when the frozen epoch 0 is a candidate?). For "
        "your own chips set `USE_BYOD = True` and run from Section 4 onwards.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Environment variables the notebook reads** — `DIMER_ISOLATED_ENV` (Section 1: where the '
        'isolated environment is built or reused; default `dimer_isolated_env_<lock digest>` in the working directory), `DIMER_NOTEBOOK_CI_PREINSTALLED=1` '
        '(Section 1: skip building it, for CI images that already have the pins) and `DIMER_MASK_FRESH_TEST=1` (Section 8: do not fetch or score the fresh '
        'test; the run is exploratory).\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 3 stops during the pickle audit or '
        'conversion** — the audit refuses any global outside the allow-list and names it; the downloaded checkpoint is not the pinned one. Delete the snapshot '
        'folder and run Section 3 again.\n- **Section 4 reports a size or SHA-256 mismatch for a chip** — the message names the object; a download from '
        '`storage.googleapis.com/sen1floods11` was cut short or altered. Run Section 4 again.\n- **CUDA out of memory in Section 6** — another notebook holds '
        "the GPU. Restart the session, keep `BATCH_SIZE = 2`, and choose **Run all**; a smaller batch makes the run exploratory.\n- "
        '**Section 6 raises "… does not have a deterministic implementation"** — an operation without a deterministic CUDA kernel was reached; the run stops '
        'rather than training non-deterministically. Report it with the printed operation name.\n- '
        '**Section 8 prints "not scored (exploratory)"** — at least one pre-registered setting differs (it lists them: on CPU `device` and `grid`). This is '
        'expected off a T4.\n- **Section 8 prints "failed before completion; not a look"** — the fresh chips could not be fetched or verified, or a '
        'pre-scoring check failed; nothing was recorded. Run Section 8 again.\n- '
        '**Section 5 prints `restored_pinned_base`** — you re-ran it after Section 6; the adapted tensors were put back to the base. Re-run Sections 6–9 in order.\n- '
        '**BYOD: a band, shape or label refusal** — the message names the rule; chips must be six-band 512 × 512 GeoTIFFs in the documented band order (13-band '
        'Sentinel-2 L1C files are reduced automatically) with masks of 0 / 1 / −1, and the dataset needs some water.\n- **BYOD: "bring at least 7 labelled '
        'chips"** — the per-chip split keeps four chips for training, one for validation and two for test; add chips, or give a `split` column.\n- **BYOD: '
        '"pairs.csv row … is listed but not in …" or "… is not a readable GeoTIFF"** — the message names the row and the file: add the file, correct the '
        'row, or save the file again as a GeoTIFF.\n- **BYOD: a group or split refusal** — a group split needs at least three groups and four training '
        'chips; a `split` column needs `train`, `validation` and `test` each at least once.\n- **BYOD: "BYOD_PATH … does not '
        'exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on '
        'Kaggle or Jupyter, put the zip (or folder) in the runtime and set `BYOD_PATH` to its path.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was '
        "cancelled or several files were chosen; run the cell again.\n\n## Glossary\n\n- **Sentinel-2 L1C / Sen1Floods11:** ESA's optical satellites; L1C is "
        'top-of-atmosphere reflectance (not atmospherically corrected); Sen1Floods11 is a dataset of flood-event chips with hand-drawn water labels over eleven events.\n- **Reflectance and the six bands:** the fraction of sunlight reflected '
        'in blue, green, red, narrow NIR, SWIR 1 and SWIR 2; water is dark in NIR and SWIR.\n- **Segmentation mask / ignore class:** one class per pixel; −1 '
        'marks cloud or no-data pixels that are excluded from training and scoring.\n- **No-water baseline:** predicting *land* for every pixel; its accuracy '
        'equals the land fraction and its water IoU is 0.\n- **IoU, precision, recall, F1:** overlap of predicted and true water; the share of predicted water '
        'that is real; the share of real water found; their harmonic mean.\n- **Encoder, neck, decoder, head:** the ViT-L backbone; the pyramid that rescales '
        'its features; the UPerNet that upsamples them to pixels; the final two-class layer.\n- **Frozen / adapted / pinned base:** the packaged model; the '
        'model after Section 6; the verified packaged weights every adaptation starts from (`restore_pinned_base` in Sections 5 and 6).\n- **BatchNorm statistics:** running means and '
        'variances in the decoder; kept fixed because two-chip batches would corrupt them.\n- **Validation-loss selection:** keeping the epoch with the lowest '
        'validation loss, with the frozen model as epoch 0.\n- **Exposed development test / fresh test:** chips that were already looked at while choosing '
        'settings, reported for continuity; chips never scored before, read once under settings fixed in advance.\n- **Pre-registration / exploratory / '
        'reproduction:** fixing the settings, endpoint and reading rule before looking; any run that differs from them; a later run with identical settings.\n- '
        '**Deterministic training:** the same inputs and seed give bit-identical results; here by flags plus matmul versions of three operations.\n- **Pickle audit / safetensors:** the upstream checkpoint is a pickle that could run code when '
        'loaded; it is statically checked against an allow-list and converted once to safetensors, a format that stores only tensors.\n- **Isolated '
        'environment:** the separate Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## Conclusion (your notes)\n\nComplete '
        "these in your own words; the recorded run's values are in the **Check your reasoning** answers above.\n\n- The frozen model's water IoU on the exposed development test was ___ "
        "against the no-water baseline's ___.\n- The grid selected ___; on the fresh test the change was ___ with interval ___, which the reading rule calls ___.\n- The number I would not trust on its own "
        'is ___, because ___.\n- Before adapting on my own chips I would check the product level, band order and offset, split by ___, and compare against ___.\n\n'
        "## References\n\n"
        "[1] D. Szwarcman, S. Roy, P. Fraccaro, et al., \"Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications,\" arXiv:2412.02732, 2024. https://arxiv.org/abs/2412.02732\n\n"
        "[2] D. Bonafilia, B. Tellman, T. Anderson, and E. Issenberg, \"Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1,\" in *Proc. IEEE/CVF CVPR Workshops*, 2020. https://github.com/cloudtostreet/Sen1Floods11\n\n"
        "[3] IBM, \"TerraTorch,\" GitHub repository. https://github.com/IBM/terratorch\n\n"
        "[4] IBM and NASA, \"Prithvi-EO-2.0-300M-TL-Sen1Floods11,\" Hugging Face model repository, revision `{MODEL_REVISION}`. https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11\n\n"
        "[5] European Space Agency, \"Sentinel-2 processing baseline 04.00 (radiometric offset),\" Sentinel Online. https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/processing-baseline\n\n"
        "[6] K. Valcorza, \"prithvi-flood-segmentation-pipeline: README, model card, weights notes and pre-registration,\" GitHub. https://github.com/kurtvalcorza/prithvi-flood-segmentation-pipeline/blob/main/MODEL_CARD.md\n\n"
        "[7] DIMER Notebook Specification 2.5 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
