"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E flood-segmentation workflow: the pinned Prithvi flood checkpoint (a Lightning
pickle) is digest-verified, statically audited and converted once into safetensors, 44 hand-labelled Sen1Floods11
chips are fetched from the public bucket, validated and assigned the official roles, the frozen model is scored
against the no-water baseline, a bounded decoder fine-tuning runs in the kernel, the held-out chips are scored
again, new chips are segmented, and the adapter is exported and reloaded.
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
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime installs the pinned dependencies (torch, torchvision, terratorch and its "
        "stack, tifffile, numpy, safetensors, huggingface-hub), stages and digest-verifies the pinned Prithvi flood checkpoint "
        "(1.28 GB) from the Hub, statically audits the Lightning pickle against an allow-list, converts it once into safetensors "
        "with a pinned digest, rebuilds the architecture from the installed `terratorch` package and loads it strictly, fetches 44 "
        "hand-labelled Sen1Floods11 chips (104 MB of digest-verified GeoTIFFs, no credential), validates them and assigns the "
        "official roles (24 training, 8 validation, 12 test), segments the held-out chips with the frozen model and scores them "
        "against the no-water baseline, runs a bounded fine-tuning of the neck, decoder and head, scores the same chips again, "
        "segments the three example chips shipped with the upstream repository, exports the adapter as safetensors with a "
        "manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC "
        "2.0 §5). On a T4 the whole path takes a few minutes of model time after the downloads; the terratorch install is the "
        "slowest step."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled chips as a zip holding `pairs.csv` (columns `id`, `image`, `label`) beside six-band 512 × 512 GeoTIFF chips "
        "(blue, green, red, narrow NIR, SWIR 1, SWIR 2 — reflectance in [0, 1] or × 10 000) and single-band label rasters "
        "(0 = no water, 1 = water, −1 = no data); at least four chips with some water. Your chips are split by seed into "
        "training, validation and test sets and flow through the same contract — validation, frozen baseline, adaptation, "
        "held-out evaluation, inference, artifact export and reload parity. The expected schema, the ceilings and the privacy "
        "guidance are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional "
        "and never part of the default path."
    ),
    "pipeline_class": "PrithviFloodPipeline",
    "model_load": "PrithviFloodPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), report=print)",
    "weights_key": "prithvi-eo-2.0-300m-tl-sen1floods11",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "timm", "lightning", "tifffile"],
    "title": "Prithvi-EO-2.0 flood mapping — DIMER E2E segmentation fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "flood-extent segmentation of six-band Sentinel-2 chips with a Prithvi-EO-2.0 ViT-L encoder and UPerNet decoder, held-out IoU/F1 against a no-water baseline, and bounded fine-tuning of the decoder to labelled chips",
    "intro": (
        "Prithvi-EO-2.0 (Szwarcman et al., 2024) is NASA and IBM's foundation model for Harmonized Landsat Sentinel-2 imagery: a "
        "ViT-L masked autoencoder pretrained on 4.2 M global multispectral samples, here in its `TL` variant with temporal and "
        "location embeddings. The checkpoint packaged here is the upstream authors' fine-tune for flood mapping — the 300 M-parameter "
        "encoder, a learned pyramid neck over four encoder depths, a UPerNet decoder and a two-class head — trained on the 446 "
        "hand-labelled Sentinel-2 chips of Sen1Floods11 with TerraTorch.\n\n"
        "Two things about this row are handled in the open. **The upstream asset is a pickle** — a PyTorch Lightning checkpoint. "
        "Section 3 downloads and digest-verifies it, statically lists every global the pickle would import (a state dict of tensors "
        "and nothing else), refuses anything outside that allow-list, unpickles it exactly once through torch's weights-only loader, "
        "and writes a safetensors file whose digest is pinned in the carried module; the model you run is rebuilt from the installed "
        "`terratorch` package and loads that file strictly. **The model is already fine-tuned on this dataset**, so the bounded "
        "adaptation in Section 6 is a demonstration of the contract on chips it has seen the distribution of, selected by validation "
        "loss with the frozen model as epoch 0; the honest number is the paired held-out comparison in Section 7, and the point of "
        "the contract is the same recipe applied to *your* labelled chips from another sensor, season or region."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and metrics modules; stage and digest-verify a pickled "
        "checkpoint, read its static audit and see it converted into safetensors; fetch and validate real labelled multispectral "
        "chips with an ignore class; read pixel IoU, F1, precision and recall against a no-water baseline; run a bounded decoder "
        "fine-tuning with explicit hyperparameters and frozen BatchNorm statistics; compare the adapted and frozen models on the "
        "same held-out chips; segment new chips; and export a safetensors adapter that reloads against the pinned base with "
        "verified parity."
    ),
    "exclusions": (
        "the Sentinel-1 (SAR) route of Sen1Floods11, the temporal and location embeddings (the packaged fine-tune ran without "
        "metadata, and so does this pipeline), tiling of scenes larger than 512 × 512, atmospheric correction, cloud masking, the "
        "published benchmark scores, and any claim that a 44-chip sample stands in for an operational evaluation. The repository "
        "exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU and Python 3.12): the ViT-L encoder runs in float16 autocast and the default adaptation needs about 2.5 GB of GPU memory; on CPU one 512 × 512 chip takes tens of seconds and the adaptation would take an hour. About 3 GB of disk is needed for the checkpoint and its conversion; the `terratorch` install pulls torchgeo, lightning and their dependencies and takes several minutes.",
        "- **Knowledge:** what a multispectral reflectance chip is (bands, scaling, no-data), what a pixel-wise segmentation mask and an ignore class are, and how IoU, precision and recall are read against a majority baseline.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint is a pickle. It is digest-verified, statically audited against an allow-list (audit digest pinned) and unpickled **once** through torch's weights-only loader to produce the safetensors the model is actually loaded from. No Hub-hosted Python module is imported; `terratorch` is installed from PyPI at a pinned version.",
        "- **Data contract:** a record is `{{id, image, label}}` — a (6, 512, 512) reflectance array (or a GeoTIFF; 13-band Sentinel-2 L1C files are reduced to the six Prithvi bands) with values in [0, 1] or × 10 000, no-data 0 or −9999, and a (512, 512) mask with 0 / 1 / −1. Validation is structural: nothing checks that the bands are the right six in the right order, that the reflectance is corrected, or that the label belongs to the chip.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — commercial imagery under licence or unreleased disaster assessments are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 88 pinned objects (44 chips and their labels, about 104 MB) from the public Sen1Floods11 bucket `storage.googleapis.com/sen1floods11` over HTTPS, digest-verified before decoding; Sen1Floods11 is CC BY 4.0 (Cloud to Street).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample chips, validation and roles\n\n"
                "The default dataset is 44 hand-labelled Sentinel-2 chips of Sen1Floods11 — 24 from the official training split, "
                "8 from the validation split and 12 from the test split, drawn across the ten flood-event regions from the chips "
                "whose label is at least 60 % valid and 3 % water — fetched by object path from the public bucket and refused on "
                "any byte-size or SHA-256 mismatch (`fetch_corpus`). Each 13-band `S2Hand` GeoTIFF is reduced to the six Prithvi "
                "bands, scaled from reflectance × 10 000 to reflectance, and its no-data replaced by 0, exactly as the upstream "
                "datamodule does; each `LabelHand` mask keeps −1 for cloud / no-data, which every metric ignores. `dataset_manifest` "
                "validates every split, checks that no chip appears twice and records a digest.\n\n"
                "Look for: 24 / 8 / 12 chips with water fractions around 0.2..0.4, a written sample pair "
                "(`outputs/{stem}_sample_chip.tif` + `_sample_label.tif`, the BYOD shape), and three refusal probes — a five-band "
                "chip, a label with an unknown class, a chip with reflectance far outside range — each rejected before the model runs."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/sen1floods11')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'disjoint': dataset_report['disjoint'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "for name, part in dataset_report['splits'].items():\n"
                "    print({{name: {{'water_fraction': part['class_pixel_fraction']['water'], 'ignored_pixels': part['ignored_pixels'], 'regions': part['regions']}}}})\n"
                "print({{'first_test_chip': validate_inputs(test_records[0])}})\n"
                "sample_pair = write_sample_pair(test_records[0], 'outputs/{stem}_sample_chip.tif', 'outputs/{stem}_sample_label.tif')\n"
                "print({{'sample_pair': sample_pair, 'pairs_csv': str(write_dataset_csv(test_records, 'outputs/{stem}_sample_pairs.csv'))}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'five-band chip': [{{**test_records[0], 'image': test_records[0]['image'][:5]}}, *test_records[1:4]],\n"
                "    'unknown label class': [{{**test_records[0], 'label': np.where(test_records[0]['label'] == 1, 7, test_records[0]['label'])}}, *test_records[1:4]],\n"
                "    'reflectance out of range': [{{**test_records[0], 'image': test_records[0]['image'] * 50000.0}}, *test_records[1:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. The frozen model against the no-water baseline\n\n"
                "`pipe.predict` standardises each chip with the upstream datamodule's band statistics, runs the encoder, neck, "
                "decoder and head in float16 autocast, and returns the argmax mask, the softmax scores (the model's outputs, not "
                "calibrated probabilities) and the water fraction per chip. `pipe.evaluate` pools the labelled pixels of every "
                "held-out chip into one confusion matrix (−1 pixels excluded) and reports the per-class IoU, mean IoU, accuracy, "
                "and the water class's precision, recall and F1; the **no-water baseline** — every pixel predicted as land — is "
                "scored on the same pixels, so its accuracy is exactly the land fraction and its water IoU is 0.\n\n"
                "Look for: a water IoU well above 0 on the test chips (in the build record about 0.59, with recall above "
                "precision — the model over-predicts water on some regions), a validation water IoU around 0.86, and per-chip "
                "water fractions that track the labels. These are sample-sanity numbers on 12 and 8 chips, not the benchmark."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "frozen_val = pipe.evaluate(val_records)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'metric': frozen_test['metric']}})\n"
                "print({{'baseline_no_water_test': {{k: frozen_test['baseline_no_water'][k] for k in ('iou', 'accuracy', 'f1')}}}})\n"
                "print({{'frozen_test': {{k: frozen_test['model'][k] for k in ('iou', 'mean_iou', 'accuracy', 'precision', 'recall', 'f1')}}}})\n"
                "print({{'frozen_validation': {{k: frozen_val['model'][k] for k in ('iou', 'f1')}}}})\n"
                "frozen_predictions = pipe.predict(test_records)\n"
                "for record, pred in list(zip(test_records, frozen_predictions['predictions']))[:6]:\n"
                "    labelled = record['label'] >= 0\n"
                "    print({{'chip': record['source_id'], 'water_label': round(float((record['label'] == 1).sum() / labelled.sum()), 3), 'water_predicted': pred['class_fraction']['water'], 'ignored': int((~labelled).sum())}})\n"
                "print({{'decision_rule': frozen_predictions['decision_rule'], 'scores_shape': frozen_predictions['predictions'][0]['scores'].shape}})"
            ),
        },
        {
            "md": (
                "## 6. Bounded fine-tuning of the neck, decoder and head\n\n"
                "`pipe.adapt` trains the 46 tensors of the pyramid neck, the UPerNet decoder and the head (15.1 M parameters — "
                "4.7 % of the model) and nothing else: the ViT-L encoder is frozen (no gradient is stored for it), and every "
                "BatchNorm layer keeps its running statistics, because batches of two chips would corrupt them. Each step takes "
                "two chips with a seeded horizontal or vertical flip, computes the cross-entropy over the labelled pixels "
                "(−1 ignored) and takes an AdamW step at a small fixed learning rate with gradient-norm clipping and float16 loss "
                "scaling. Epoch 0 records the frozen model's validation loss and metrics; the epoch with the lowest validation "
                "loss is kept — which can be epoch 0, since the packaged model already trained on this dataset.\n\n"
                "Watch the validation loss: in the build record it dipped at epoch 2 and rose again by epoch 4 — the sign that a "
                "small learning rate and validation selection are doing their job on a model that has little left to learn from "
                "24 chips it has already seen the distribution of. Four epochs (48 steps) take under a minute on a T4. "
                "`TRAINABLE = 'decoder+last_block'` also unfreezes the last encoder block (27.7 M parameters)."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 2  # @param {{type:\"integer\"}}\n"
                "TRAINABLE = 'decoder'  # @param [\"decoder\", \"decoder+last_block\"]\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'val' in entry:\n"
                "        row['val_water_iou'] = entry['val']['iou']['water']\n"
                "        row['val_f1'] = entry['val']['f1']\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable=TRAINABLE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['precision'], 'batchnorm': adapt_result['batchnorm'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 7. Held-out evaluation: the paired comparison\n\n"
                "The test chips were never used for training or epoch selection (they come from the official test split). The "
                "adapted model is scored exactly as the frozen model was in Section 5, and the table puts the baseline, the "
                "frozen and the adapted numbers side by side. The cell asserts what the procedure guarantees — the kept epoch's "
                "validation loss is no higher than the frozen model's — and prints the test numbers without asserting a "
                "direction: on this sample the water IoU moved from about 0.59 to 0.60 in the build record, a sample-sanity "
                "observation on 12 chips with no dispersion estimate, not a quality claim. Twelve chips from seven regions "
                "cannot separate a real gain from noise; with your own chips from a new sensor or region, the gap between "
                "frozen and adapted is the number to watch."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{}}\n"
                "for key in ('mean_iou', 'accuracy', 'precision', 'recall', 'f1'):\n"
                "    comparison[key] = {{'baseline_no_water': frozen_test['baseline_no_water'][key], 'frozen': frozen_test['model'][key], 'adapted': adapted_test['model'][key]}}\n"
                "comparison['water_iou'] = {{'baseline_no_water': frozen_test['baseline_no_water']['iou']['water'], 'frozen': frozen_test['model']['iou']['water'], 'adapted': adapted_test['model']['iou']['water']}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "print({{'validation_water_iou': {{'frozen': frozen_val['model']['iou']['water'], 'adapted': adapted_val['model']['iou']['water']}}, 'validation_loss': {{'frozen': adapt_result['history'][0]['val_loss'], 'kept_epoch': adapt_result['history'][adapt_result['best_epoch']]['val_loss']}}}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'frozen': {{'test': frozen_test, 'validation': frozen_val}},\n"
                "    'adapted': {{'test': adapted_test, 'validation': adapted_val}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "assert adapt_result['history'][adapt_result['best_epoch']]['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert adapted_val['model'] == adapt_result['history'][adapt_result['best_epoch']]['val']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 8. New chips, artifact export and fresh reload\n\n"
                "The adapted model segments the three example chips that ship with the upstream repository (India, Spain, USA — "
                "13-band Sentinel-2 L1C files reduced to the six bands), which carry no labels here: the predicted water fraction "
                "per chip and a written mask are a sanity check, not an evaluation.\n\n"
                "`pipe.save_artifact` writes the trained tensors (about 60 MB) as `adapter.safetensors`, with a `manifest.json` "
                "recording the artifact format, the base model id and revision, the digest of the converted base file, the "
                "adaptation scope, the tensor names, the file size and SHA-256, the training configuration and the epoch history "
                "(OUT8). `PrithviFloodPipeline.from_artifact` re-verifies the base file, checks the artifact manifest, scope and "
                "digest **before** deserialising, rebuilds the model and overlays the tensors — a fresh object from files, not "
                "the in-memory model (VER2). The cell asserts identical held-out metrics and identical score maps (VER4)."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "import tifffile\n\n"
                "example_dir = WEIGHTS_DIR / 'examples'\n"
                "new_records = [{{'id': path.stem, 'image': read_chip(path, band_indices=S2_L1C_BAND_INDICES), 'source': str(path.name)}} for path in sorted(example_dir.glob('*.tif'))]\n"
                "new_predictions = pipe.predict(new_records)\n"
                "for record, pred in zip(new_records, new_predictions['predictions']):\n"
                "    tifffile.imwrite(f'outputs/{stem}_mask_' + record['id'] + '.tif', pred['mask'])\n"
                "    print({{'chip': record['id'], 'water_fraction': pred['class_fraction']['water'], 'note': 'sanity check, no label'}})\n"
                "with open('outputs/{stem}_predictions.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'model': new_predictions['model'], 'classes': new_predictions['classes'], 'decision_rule': new_predictions['decision_rule'], 'predictions': [{{'id': p['id'], 'class_fraction': p['class_fraction']}} for p in new_predictions['predictions']]}}, f, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'trainable': artifact_manifest['adapter']['trainable'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = PrithviFloodPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "before = pipe.predict(test_records[:2])['predictions']\n"
                "after = reloaded.predict(test_records[:2])['predictions']\n"
                "parity = {{'metrics_identical': reloaded_test['model'] == adapted_test['model'], 'max_abs_score_diff': max(float(np.abs(a['scores'] - b['scores']).max()) for a, b in zip(before, after))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['metrics_identical'] and parity['max_abs_score_diff'] < 1e-4\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_asset': [e for e in MANIFEST['files'] if e['path'] == SOURCE_CKPT_NAME],\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickle_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "        'data_objects': 2 * len(SAMPLE_RECORDS),\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'lightning': lightning.__version__, 'tifffile': tifffile.__version__, 'terratorch': importlib.metadata.version('terratorch')}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 12 held-out chips from the official test split the packaged flood model finds water with an IoU near 0.6 and a "
        "recall above its precision, against a no-water baseline that scores 0; a bounded fine-tuning of its neck, decoder "
        "and head on 24 chips, selected by validation loss with the frozen model as a candidate, leaves those numbers about "
        "where they were. That is the claim: the adaptation contract runs end to end on real labelled multispectral chips, "
        "the pickle is audited and converted rather than served, and the artifact that carries the change is 60 MB and "
        "reloads with identical outputs. It is not a claim that this sample improves the model — the model already "
        "trained on this dataset — nor that 12 chips measure its skill.\n\n"
        "The numbers are sample-sanity evidence: one seeded run, 12 test chips from seven regions, no dispersion estimate, "
        "pixel-pooled metrics that let large chips dominate, and hand labels that carry their own uncertainty at water edges "
        "and under thin cloud. Nothing here measures the model on Sentinel-1, on scenes larger than a chip, or on regions and "
        "seasons outside Sen1Floods11.\n\n"
        "Three things to carry to real data. **The six bands and their scaling are the contract:** blue, green, red, narrow "
        "NIR, SWIR 1, SWIR 2 in that order, reflectance in [0, 1]; a different band order or an uncorrected product is "
        "segmented without complaint and silently wrong. **Split by scene or event, not by chip:** neighbouring chips are "
        "near-duplicates, and a random split makes memorisation look like skill. **Read the baseline first:** on a chip with "
        "5 % water the no-water baseline is 95 % accurate; only the water IoU, precision and recall say whether the model "
        "did anything.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify a pickled upstream checkpoint, audit and convert it into safetensors without "
        "executing anything outside the audited allow-list, rebuild the model from the installed package, fetch and validate "
        "digest-pinned real labelled chips, execute bounded fine-tuning, evaluate against a baseline and the frozen model on "
        "held-out chips, and emit the shown machine-readable artifacts — without the repository being reachable. It does "
        "**not** establish benchmark superiority, production fitness, or flood-mapping skill beyond the checks shown.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE = 'decoder+last_block'`; raise "
        "`EPOCHS` and watch the validation loss turn; try `LEARNING_RATE = 1e-4` to see the frozen model win every epoch; or "
        "bring your own labelled chips through BYOD and read the baseline before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/prithvi-flood-segmentation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/prithvi-flood-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/prithvi-flood-segmentation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 (revision `{MODEL_REVISION}`)\n"
        "- Szwarcman, D., Roy, S., Fraccaro, P., et al. (2024). Prithvi-EO-2.0: A versatile multi-temporal foundation model for Earth observation applications. arXiv:2412.02732: https://arxiv.org/abs/2412.02732\n"
        "- Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. CVPR Workshops: https://github.com/cloudtostreet/Sen1Floods11\n"
        "- TerraTorch: https://github.com/IBM/terratorch\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
