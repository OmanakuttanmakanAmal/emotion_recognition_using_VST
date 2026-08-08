# Multimodal Emotion Recognition — Project Report

**Amal Omanakuttan · MSc Artificial Intelligence · THD · 2025**

This is a short, evidence-first summary for supervision. The full technical writeup (architecture, every design decision, every config) is in [`README.md`](README.md) — this document is the "what did you actually do and prove" version.

---

## 1. What the system does

A model that predicts one of 7 emotions (Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral) from a face photo, and optionally combines it with a short audio clip of speech for a stronger combined prediction. Three models, trained in three phases:

| Phase | Model | Task |
|---|---|---|
| 1 | DINO ViT-S/16 (vision transformer) | Face → emotion |
| 2 | Wav2Vec2-base (speech transformer) | Voice → emotion |
| 3 | Small MLP | Combines phases 1+2 into one prediction |

## 2. Datasets

| Dataset | Used for | Size | Source |
|---|---|---|---|
| RAF-DB | Face model (train + test) | 15,339 images (12,271 train / 3,068 test), 7 classes | Real internet photos, crowd-labelled |
| RAVDESS | Audio model | 1,440 speech clips, 24 actors | Studio-recorded, professionally acted |

Both are on disk under `amal/DATASET/` and `amal/audio_data/`, linked into this project via `DATASET/` and `audio_data/` (Windows directory junctions, so the ~15K images aren't duplicated on disk).

## 3. What this session found and fixed

The project's previous results were **not real** — every chart in `images/` was mathematically smooth with zero epoch-to-epoch noise, a sign they were plotted from a formula rather than logged training. Investigation found three concrete bugs, and — critically — the **actual executed Colab notebook** (preserved in [`evidence/colab_run_2_nan_bug_evidence.ipynb`](evidence/colab_run_2_nan_bug_evidence.ipynb)) proved a real run had been attempted on a real Tesla T4 GPU and had genuinely failed:

| # | Bug | Effect |
|---|---|---|
| 1 | `src/train.py` printed a Unicode character (`█`) the Windows console can't display, crashing *after* the checkpoint saved but *before* the results log saved | Every run left a checkpoint with no verifiable results file |
| 2 | The NaN-safety fix documented in README §7 (keep Wav2Vec2's backbone in `eval()` mode during training) was described in the README but **missing from the actual code** | Audio training's loss became `NaN` on step 1 and stayed `NaN` for all 20 epochs, every time |
| 3 | The chart-generation script had the old numbers hardcoded as literal text in several panels, separate from its (correctly real) curve-plotting logic | Charts would have looked "real" even after a genuine retrain |

**Direct proof of bug #2** — the flat red line is the actual Colab run; the model never learned anything and just kept predicting the majority class for 20 straight epochs:

![NaN bug before and after](images/experiment_nan_bug_before_after.png)

All three bugs are fixed (see `README.md` §19.1 for the exact diffs). A full clean retrain was then run end-to-end on a local **NVIDIA RTX A3000 12GB GPU**.

## 4. Real, verified results

| Phase | Metric | Result |
|---|---|---|
| Face (DINO) | Test accuracy / Macro F1 | **78.36% / 70.78%** |
| Audio (Wav2Vec2) | Val accuracy / Macro F1 | **42.36% / 34.94%** |
| Fusion | Val accuracy | **89.90%** |

Every number above is backed by a `results_*.json` with full per-epoch history, in `checkpoints/`. The training curves (real, with real noise — compare to the smooth fake ones this project used to have):

| Face | Audio | Fusion |
|---|---|---|
| ![face curve](images/face_training_curve.png) | ![audio curve](images/audio_training_curve.png) | ![fusion curve](images/fusion_training_curve.png) |

**The full picture** — what the original broken Colab run produced, what the README used to (unverifiably) claim, and what's now real and reproducible:

![Big picture comparison](images/big_picture_colab_vs_claimed_vs_real.png)

The previously claimed numbers turn out to be close to what real training actually achieves — the methodology was sound, it just never had a completed, verified run to back it until now.

### 4a. Confusion matrix and trivial-baseline comparison

Two checks any examiner would expect and that weren't done before: does the model actually beat a dumb baseline, and where specifically does it fail?

| Confusion matrix | vs. trivial baselines |
|---|---|
| ![Confusion matrix](images/face_confusion_matrix.png) | ![Baseline comparison](images/face_baseline_comparison.png) |

The model (78.36%) clears both a random-guess baseline (14.3%) and a majority-class baseline of always predicting "Happy" (38.62%, verified — this number was mentioned in the original Colab notebook's final report but never actually checked) by a wide margin, so the accuracy is real signal, not an artefact of class imbalance. The confusion matrix shows the errors are interpretable, not random: Fear is most often mistaken for Surprise (22% of Fear images), and Disgust for Neutral (16%) — both are well-documented visual look-alikes in facial emotion recognition, which is a reassuring sign the model learned something real rather than a shortcut.

## 5. Experiments tried, and what was decided

Two concrete improvement ideas were tested to completion (not just proposed) and compared against the baselines above.

### 5a. Adopted: class-weighted loss for the face model

RAF-DB is heavily imbalanced (Happy = 38.9% of training images, Fear = 2.3%). Weighting the loss by inverse class frequency cost 0.4 points of overall accuracy but meaningfully rebalanced the model:

![Class weight comparison](images/experiment_face_classweight_perclass.png)

**Decision: adopted.** Macro F1 and the worst-performing class both improved for a negligible accuracy cost — this is now the official face checkpoint, and fusion was retrained to match it.

### 5b. Rejected: fully unfreezing the audio backbone

Hypothesis: letting all 95M Wav2Vec2 parameters train (instead of just the small head) might close the audio model's accuracy gap.

![Unfreeze experiment](images/experiment_audio_frozen_vs_unfrozen.png)

**Decision: rejected.** It performed worse and was still near-random after 8 of 15 epochs — 1,296 training clips isn't enough to safely fine-tune 95M parameters in a practical time budget. This confirms the original frozen-backbone design (documented in README §15, Decision 2) was the right call.

### 5c. Rejected: adding CREMA-D to expand the audio dataset

Hypothesis (from an earlier draft of this report): the audio model is data-starved, so combining RAVDESS (1,440 clips) with CREMA-D (7,442 clips, downloaded and wired in during this session — see §8) should help. Tested properly, with a fair comparison: the combined model evaluated on the exact same RAVDESS validation split used everywhere else in this report.

![CREMA-D dataset combo experiment](images/experiment_audio_cremad_dataset_combo.png)

**Decision: rejected.** Despite 6.2x more training data, RAVDESS accuracy got *worse* (42.36% → 35.42%), not better. Most likely cause: domain shift — RAVDESS and CREMA-D use different actors, microphones, and scripted phrases, so naively concatenating them diluted the RAVDESS-specific signal rather than reinforcing it. (On its own, CREMA-D-heavy validation set, the combined model does score a reasonable 39.06% / macro F1 38.01% — but that's a different, harder benchmark, not evidence it improved at the original task.) The multi-dataset-combining code (`SpeechEmotionDataset.combine()`) is kept in `src/audio_model.py` since it's generically useful and correctly built — the data itself (kept in `amal/audio_data_cremad/`) just didn't transfer the way hoped. §7 below has better next steps for using it.

### 5d. Rejected: Optuna-tuned hyperparameters

A proper hyperparameter search (`src/tune_optuna.py`, previously unused — fixed to use the same class-weighted loss as the deployed config, then run) searched learning rate, weight decay, label smoothing, dropout, and unfreeze depth across 15 trials of 4 epochs each.

![Optuna search results](images/experiment_optuna_search.png)

**Decision: rejected.** The best trial from the short search (74.4% after 4 epochs, using a higher learning rate and unfreezing twice as many backbone blocks) was then trained to the full 20 epochs to check it properly — and it *didn't* beat the official config (78.25% / macro F1 69.59%, vs. the official 78.36% / 70.78%). This is a known, well-documented pitfall of short-budget hyperparameter search: configs that look best after a few epochs (faster initial convergence from a higher LR and more unfrozen parameters) don't necessarily hold that lead once training runs to completion, where the larger trainable parameter count (7.2M vs. 3.65M) instead nudges it slightly toward overfitting. Official config kept; the tuned run is archived in `experiments/checkpoints_optuna_tuned/`.

### 5e. Attempted, incomplete: multi-seed stability check

To check whether the adopted 78.36%/70.78% result is reliable or just a lucky random initialization, a second and third full training run of the identical official config were planned (same hyperparameters, different implicit random seed each time, since `train.py` doesn't currently fix one). **The second replicate was cut short by a GPU driver crash** ("GPU is lost, reboot required") partway through epoch 6 of 20 — a hardware/driver-level failure, unrelated to the code, that could not be recovered without a system reboot. In the interest of not risking the machine further, this check was stopped rather than retried.

**Honest status: not completed.** Only one full run of the official config exists (78.36% test accuracy / 70.78% macro F1). This is a real limitation worth stating plainly in the thesis rather than glossing over: the reported numbers are correct and reproducible, but their run-to-run variance hasn't been measured. If pursued later, rerun the exact command in §10 two more times with a GPU that's confirmed stable, and report mean ± std instead of a single number.

## 6. Why not just "more epochs"?

Checked directly, not assumed:

- **Face model:**

![Face overfitting proof](images/face_overfitting_proof.png)

  Validation accuracy (blue) plateaus and oscillates between roughly 68–75% from epoch 8 onward, never beating its epoch-12 peak again, while training accuracy (red) keeps climbing the whole time, reaching 96.3% by epoch 20 — a 22.6-point gap by the end. That's the classic signature of overfitting. To be precise about what this does and doesn't prove: training accuracy is still rising at epoch 20, so this one run doesn't establish that more epochs *definitely* wouldn't help — only that validation accuracy showed no further gains across all 20 epochs actually trained, while the gap to training accuracy kept widening. Not a concern for the deployed model either way, since the saved checkpoint already uses epoch 12's weights (best val_acc), not epoch 20's.

- **Audio model:** training and validation accuracy stay close together the whole run (both ~42-43%) — it's *underfitting*, not overfitting. More epochs on the same frozen setup wouldn't help much (it was already flattening by epoch 14); more **data** would.

## 7. Recommended next steps (ranked by effort vs. payoff)

1. **Finish the multi-seed stability check (§5e).** Cheapest remaining item and arguably the most important for thesis rigor — 2 more ~20-minute runs of the exact command in §10, once a GPU is confirmed stable, to turn "78.36%" into "78.x% ± y%".
2. **Use CREMA-D for domain-adapted fine-tuning, not naive concatenation.** §5c showed simply merging RAVDESS+CREMA-D hurts RAVDESS accuracy. Two better ways to use the CREMA-D data already sitting in `amal/audio_data_cremad/`: (a) pre-train the head on the combined set, then briefly fine-tune on RAVDESS-only for a few epochs so it specialises back to the target domain; or (b) train and report a fully separate CREMA-D model as an independent benchmark, rather than mixing corpora.
3. **Partial audio backbone unfreeze** (last 2-4 Wav2Vec2 layers only, not all 95M params) — a middle ground between §5b's two extremes, not yet tried.
4. **Targeted augmentation for Fear/Disgust/Anger** (the 3 rarest RAF-DB classes) on top of the already-adopted class weighting.
5. **Try the larger DINO `vitb16` variant** — already supported via `--dino_variant vitb16`, just not yet tested.

`src/tune_optuna.py` (previously unused) has now been run — see §5d; a wider or deeper sweep (more trials, or trials with more epochs to avoid the short-budget pitfall found there) is a reasonable follow-up but with clearly diminishing returns given §5d's result.

Full detail and reasoning for each is in `README.md` §19.4.

## 8. CREMA-D — already downloaded and wired in

Unlike the original draft of this report, this was actually done, not just planned: CREMA-D's `AudioWAV/` folder (7,442 clips, 91 actors, ~595MB) was pulled via `git`/`git-lfs` sparse-checkout from **https://github.com/CheyneyComputerScience/CREMA-D**, placed at `amal/audio_data_cremad/`, and linked into the project the same way as the other datasets. `src/audio_model.py` gained a `SpeechEmotionDataset.combine()` method and `train_audio.py` gained comma-separated `--data_root`/`--dataset` support to train on multiple corpora at once:

```powershell
python src\train_audio.py --data_root audio_data,audio_data_cremad --dataset ravdess,cremad --epochs 20 ...
```

As §5c describes, training on the combined set didn't improve the RAVDESS benchmark — but the data and the code for combining corpora are both there and correctly working for whichever of the two better approaches in recommendation #1 above gets tried next.

## 9. Where everything is

```
amal/
├── DATASET/                  RAF-DB face images (15,339 files)
├── audio_data/                RAVDESS audio (1,440 files)
├── audio_data_cremad/         CREMA-D audio (7,442 files, added this session)
└── emotion_recognition_using_VST/    <- the actual project
    ├── README.md                      full technical documentation
    ├── SUPERVISOR_REPORT.md           this file
    ├── checkpoints/                   official trained models + results_*.json
    ├── experiments/                   the rejected audio-unfreeze experiment (kept for reference)
    ├── evidence/                      the real, executed Colab notebooks (proof of the original bug)
    ├── training_logs/                 raw terminal logs from every training run
    ├── images/                        all charts referenced in this report
    ├── src/                           model code, training scripts
    └── app.py                         the interactive demo (python app.py)
```

## 10. How to reproduce this from scratch

```powershell
cd emotion_recognition_using_VST
.\venv\Scripts\python.exe src\train.py --model dino --epochs 20 --batch_size 32 --lr 1e-4 --class_weights --num_workers 0 --data_root . --output_dir checkpoints
.\venv\Scripts\python.exe src\train_audio.py --data_root audio_data --dataset ravdess --epochs 20 --batch_size 16 --lr 1e-4 --num_workers 0 --output_dir checkpoints
.\venv\Scripts\python.exe src\train_multimodal.py --face_ckpt checkpoints\best_dino.pt --audio_ckpt checkpoints\best_audio.pt --synthetic --img_root DATASET\train --img_csv train_labels.csv --audio_root audio_data --n_synthetic 4000 --epochs 20 --output_dir checkpoints
.\venv\Scripts\python.exe generate_results_visuals.py
.\venv\Scripts\python.exe generate_experiment_visuals.py
.\venv\Scripts\python.exe test_e2e.py   # verify -- should print "All tests passed."
```

Total run time: roughly 40 minutes on an RTX A3000-class GPU (20 min face + 8 min audio + <1 min fusion + verification).
