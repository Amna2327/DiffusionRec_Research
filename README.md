# DiffusionRec — Repository Reconstruction Ledger

This document tracks what the shared code in this repository actually does, verified by direct inspection, versus what the camera-ready paper ("DiffusionRec: Recognition-Guided Diffusion for Content-Aware Urdu Handwriting Generation," Kausar et al.) claims. It is **not** an official audit — it's a working ledger built by tracing individual scripts, config files, and data artifacts, kept so that anyone (human or LLM) picking this repo up can get oriented quickly without re-deriving everything from scratch.

**Epistemic labeling used throughout:** findings are marked **Confirmed** (directly verified by reading the relevant code or data), **Likely** (strong circumstantial evidence, not fully verified), or **Open / Hypothesis** (plausible but unverified). At least one finding below was already revised once during this investigation after new evidence surfaced — status labels are not permanent and should be re-checked against the actual code before being treated as settled.

---

## 1. Background: Why This Repo Is in the State It's In

The repository's own README states: *"This repository is not up to date. We will update it after the conference."*

Based on patterns across multiple files (see Sections 3 and 6), the working explanation is: the original, clean training pipeline (run on a lab server with multi-GPU support, e.g. hardcoded `device_ids=[3,4]`) was lost due to a disk failure. What has been reconstructed since lives across at least two different local Windows machines (evidenced by backslash paths, `CUDA_VISIBLE_DEVICES` hardcoding, `num_workers=0` "Windows multiprocessing crash" workarounds, and a `try/except` handler explicitly commented as a *"known Windows multiprocessing issue"*). This explains most of the repo's structural oddities: multiple near-duplicate training scripts representing successive rebuild attempts, inconsistent path conventions, disabled validation calls, and manually patched resume commands (see Section 5).

---

## 2. Datasets

| Corpus | Type | Size / Split | Used For |
|---|---|---|---|
| **IIIT-INDIC-HW-WORDS-Urdu** | Real handwritten word images | 100,630 total — 71,207 train / 13,906 val / 15,517 test; 11,936 unique words | Word-level generation (Table 3) |
| **TUKL UPTI 2.0** | Printed, synthetic Urdu sentences | 100,000 total; 20,000 (Alvi Nastaleeq style) train / 20,000 test | Sentence-level pretraining, Stage I |
| **UNHD (Urdu-Nastaleeq Handwritten Dataset)** | Real handwritten text lines | 7,610 total from 500 writers; 5,842 train / 717 test | Sentence-level fine-tuning, Stage II |

Locally, the repo references these under different internal names/paths: `Urdu_Word_Dataset/` (word-level), `images_upti2_2/` + `groundtruth_upti2/` (UPTI), and `unhd` (sentence-level, via `UNHDDataset`). A fourth, unexpected data source was also found — see Section 6.3.

---

## 3. File-by-File Ledger

| File | Level | Recognizer integrated? | Status / notable finding |
|---|---|---|---|
| `training_without_style_upti_with_recognizer.py` | Sentence (UPTI pretrain) | ✅ full curriculum | Default args point at UPTI. `validate()` call is disabled. Has `--load_check` resume support. Leading hypothesis: reused for **both** Stage I (UPTI) and Stage II (UNHD fine-tune) via CLI args, rather than a separate fine-tune file existing — unconfirmed, see Q5. |
| `inference_without_style_upti_with_recognizer.py` | Sentence (UPTI eval) | ✅ (eval only) | Points at UPTI test set, **not UNHD**. Computes WER/BLEU, neither of which is reported anywhere in the paper. |
| `train_new_without_style.py` | Sentence (UNHD, pre-recognizer) | ❌ none | Precursor file; classifier-free guidance (condition-dropout) added over the base DiffusionPen fork it descends from. Likely candidate for Table 4's weaker "Direct Training on UNHD" row (FID 51.3). |
| `train_word_level.py` | Word | ✅ full curriculum | No resume/checkpoint-restore logic; `validate()` is active (unlike the UPTI script). |
| `train_word_level_with_resume.py` | Word | ✅ | Adds full checkpoint/resume logic (model, optimizer, EMA step, RNG states). Windows-style paths first appear here. |
| `train_word_level_with_resume_v2.py` | Word | ✅ | Adds `try/except` handling for Windows DataLoader-worker crashes; `num_workers` defaults to 0. Confirmed to be the target of `scripts/train.ps1` (Section 5) — the strongest current candidate for the script that produced the paper's headline word-level numbers. |
| `unet.py` | Shared (word + sentence) | n/a | Near-verbatim fork of Stable Diffusion's `openaimodel.py`. **Confirmed:** `label_emb` (writer/style embedding) is live on the default forward path — `emb = emb + self.label_emb(y)` — feeding every ResBlock. This contradicts the paper's stated "content-only" design (see Section 6.1). CANINE-C text conditioning is genuinely wired via cross-attention as described in the paper. `label_emb` is declared twice redundantly; `Style_Text_Encoder` and an unused `vocab_size` parameter appear to be inert DiffusionPen remnants. |
| `testing.py` | Word (single image) | n/a (third-party OCR) | Tests `nomypython/urdu-ocr-deepseek` via Unsloth — **not Qwen**. No batching, no metrics; a one-off exploratory script, not a real evaluation harness. |
| `writers_dict.json` | Master writer dict | — | 421 entries. Mixes zero-padded numeric IDs, unpadded numeric IDs, and filename-style keys (`n1.png`...`n20.png`, 20 entries that are not writer IDs at all). Contains at least one confirmed duplicate-writer-under-two-keys bug (`"025"` → 24 vs. `"25"` → 220). |
| `writers_dict_train.json` / `writers_dict_test.json` | Writer split | — | 339 + 161 = exactly 500 writers, zero overlap — a clean, writer-disjoint split. Which dataset these 500 writers actually belong to is **reopened** (see Section 6.2) after IAM English-language artifacts were found elsewhere in the repo. |
| `split_words/test.txt` (and presumed siblings) | Word (English) | n/a | **Confirmed** to be IAM English handwriting data — records follow IAM's exact form-naming convention (e.g. `c04-110`), matching IAM's published structure (657 writers, 1,539 forms). Whether any script actually reads this file is unresolved. |
| `vocab/ved/` (`vocab.json`, `merges.txt`) | Shared (small recognizer) | n/a | Special-token layout (`<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>` at IDs 0–4) matches stock `roberta-base` exactly. Used only by `load_urdu_recognizer()` for the Conv-Transformer recognizer's GPT-2 decoder — unrelated to CANINE-C (needs no vocab file) and unrelated to Qwen3-VL (ships its own tokenizer). |

---

## 4. Dataset Class / Path Used Per Script

| Script | Dataset class / path |
|---|---|
| `training_without_style_upti_with_recognizer.py` | `UPTIDataset` (`utils.upti_dataset_subset`), pointed at `images_upti2_2/` |
| `inference_without_style_upti_with_recognizer.py` | Same `UPTIDataset`, UPTI test split |
| `train_new_without_style.py` | `UNHDDataset` (`utils.unhd_dataset`) |
| `train_word_level.py` | `WordGenerationDataset` (`utils.word_generation_dataset`), pointed at `Urdu_Word_Dataset/train/processed_images` etc. |
| `train_word_level_with_resume.py` | Same `WordGenerationDataset`, same `Urdu_Word_Dataset/...` paths |
| `train_word_level_with_resume_v2.py` | Same again |

**Note:** no script inspected so far references `split_words/` or an IAM-style path — see Section 6.3.

---

## 5. `scripts/` Folder Findings

Four run-command files exist: `sampling.ps1`, `train.ps1`, `train_legacy.ps1` (all PowerShell/Windows), and `train_legacy.sh` (Linux) — consistent with the repo's "Linux support" commit added after the fact.

- **`train.ps1`** invokes `train_word_level_with_resume_v2.py` with `--rec_start_epoch 50 --rec_weight_start 0.001 --rec_weight_max 0.05 --rec_curriculum_epochs 150` — an **exact match** to the paper's Eq. 3 parameters (e_start=50, λ_start=0.001, λ_max=0.05, E_curr=150). This is the strongest evidence yet that `train_word_level_with_resume_v2.py`, run via this file, is the script behind the paper's headline word-level numbers.
- **`train_legacy.ps1`** invokes the older `train_word_level.py` with patched values (`--rec_start_epoch 1 --rec_weight_start 0.03170666666666667 --rec_curriculum_epochs 57`), with an explicit comment in the file: *"currently we started after 93 epochs."* The decimal matches the original curriculum formula evaluated at epoch ≈94, confirming this is a **manually recomputed resume** — done because `train_word_level.py` predates proper RNG/optimizer-state checkpointing.
- **`train_legacy.sh`** shows a second, different manual resume point (`--rec_weight_start 0.032669473684210526`, `--rec_curriculum_epochs 54`), mathematically consistent with resuming at epoch ≈147 under the same formula. This suggests **at least two separate manual resume incidents** occurred under the legacy script before `_v2.py`'s proper checkpointing was built.
- **`sampling.ps1`** also invokes `train_word_level.py`, in `sampling` mode, using the *original* (non-patched) paper hyperparameters.
- **None of the four files touch `training_without_style_upti_with_recognizer.py` or any sentence-level script.** The sentence-level (UPTI → UNHD, Table 4) reproduction path has no visible run command anywhere in this folder.

---

## 6. Cross-Cutting Findings

### 6.1 Writer/style conditioning is live in the code, but its real-world effect differs by level

The paper states, as a deliberate design choice, that DiffusionRec *"conditions solely on textual content,"* unlike style-conditioned frameworks such as DiffusionPen. Direct inspection of `unet.py`'s `forward()` shows the default execution path (`interpolation=False`, `style_extractor=None` — true for every training/sampling call seen so far) unconditionally executes `emb = emb + self.label_emb(y)`, injecting a writer/style embedding into the same conditioning pathway as the timestep embedding, feeding every ResBlock. **This contradicts the paper's stated design at the mechanism level.**

However, whether this produces *genuine* per-writer conditioning depends on what value the label actually takes:

- **Word-level (IIIT-INDIC-HW-WORDS-Urdu): Confirmed dummy/constant.** The word-level dataset class hardcodes `self.wclasses = 1`, `self.writer_ids = [0]`, and every sample's `style_id = 0`. With `num_classes = 1`, `label_emb` is a single-row embedding table — `emb = emb + self.label_emb(y)` reduces to adding one fixed learned vector to every sample, always. This is an undisclosed constant additive bias, not personalization. A related dead-weight artifact: the dataset also loads 5 "style reference" images per sample (`data[7]`) onto GPU, which are then never used (`style_features = None` downstream) — wasted I/O/VRAM from unwired DiffusionPen scaffolding.
- **Sentence-level (UNHD): Open, but plausibly real.** Given the writer-disjoint 500-writer split in `writers_dict_train/test.json` (see 6.2), this pathway is much more likely to carry genuine per-writer signal at sentence level — not yet directly confirmed by inspecting `UNHDDataset`'s label output.

The 10%-probability condition-dropout (classifier-free-guidance-style training) already present in the training loop is wired to this writer label specifically, **not** to the CANINE text condition.

### 6.2 The 500-writer split: reopened, not settled

`writers_dict_train.json` and `writers_dict_test.json` contain 339 and 161 writer IDs respectively — summing to exactly 500, with zero overlap (a clean, writer-disjoint split). This initially read as a strong match to the paper's stated UNHD writer count. That reading is now less certain: `split_words/test.txt` was confirmed to contain genuine IAM English handwriting data (Section 6.3), demonstrating that substantial unmodified English-dataset artifacts exist elsewhere in this repository. Given that, the 500-writer split could equally be a subset of IAM's writer pool (IAM has 657 writers total) carried over from the DiffusionPen fork, rather than UNHD-specific. Resolving this requires checking the actual file paths referenced inside these two JSON files (Urdu filenames vs. IAM-style `c04-110` paths) — not yet done.

### 6.3 IAM English handwriting data is present in the repo

`split_words/test.txt` (and presumably sibling files) contain IAM-format word-level records — image path, writer ID, transcribed word — following IAM's exact form-naming convention, confirmed by cross-referencing IAM's published structure. What remains unresolved is whether any training script actually reads this data: no dataset class in Section 4's table references `split_words/` or an IAM-style path, which argues against it being used by the word-level or sentence-level generation scripts inspected so far. It remains open whether it feeds a not-yet-located recognizer-pretraining script, or is simply an unused leftover from the DiffusionPen fork.

### 6.4 No trace of the Qwen3-VL-4B-Instruct evaluation pipeline

No file inspected so far fine-tunes, loads, or calls Qwen3-VL-4B-Instruct — the OCR system responsible for every headline Rec. Acc. number in the paper (45.2% word-level, and every Qwen column in Tables 1, 3, and 4). The only third-party OCR test located (`testing.py`) uses a different model entirely (`nomypython/urdu-ocr-deepseek` via Unsloth) in a non-batched, non-metric, exploratory script. Between this and the EasyOCR/Google Vision API inference scripts also present in the repo, at least **four different OCR systems** were experimented with at some point — but the actual Qwen fine-tuning/scoring code that produced the paper's numbers has not been located.

### 6.5 Location of UNHD Stage II fine-tuning is unconfirmed

The paper describes a two-stage sentence-level recipe: pretrain on UPTI 2.0, then fine-tune on UNHD. No dedicated "fine-tune" script has been located, and the `scripts/` folder's four run-command files all target word-level scripts only (Section 5) — none touch `training_without_style_upti_with_recognizer.py` or any other sentence-level script. The leading hypothesis remains that `training_without_style_upti_with_recognizer.py` is reused for both stages via CLI args (UNHD paths in place of UPTI paths, plus `--load_check True`), but this is now confirmed **not** to be recoverable from anything in `scripts/`.

### 6.6 Whether the training-time recognition loss actually reaches the UNet's weights

In the recognition-loss block of the word-level training loop, the approximate clean latent is explicitly detached (`x_approx = x_approx.detach()`, commented *"Detach to stop backprop through loop"*) before being decoded and shown to the frozen recognizer. Depending on exactly where this detach sits relative to the loss computation, this could mean the recognition loss value is computed and logged but does **not** actually backpropagate into the UNet's weights — a meaningful gap between the code's actual training dynamics and the paper's Eq. 3 (`L_total = L_MSE + λ_e · L_rec`). This is more consequential than initially assessed, given Section 5's finding that the exact script containing this detach call (`train_word_level_with_resume_v2.py`) is the strongest candidate for the paper's actual reported-results run. Not yet definitively resolved — requires tracing the full computation graph from `x_approx` through to `loss.backward()`.

---

## 7. Open Questions — Prioritized

| # | Question | Status | Suggested next step |
|---|---|---|---|
| 1 | Is word-level generation genuinely writer-conditioned, or fed a dummy value? | **Resolved** | Confirmed dummy/constant (`style_id=0` always, `num_classes=1`) via `WordGenerationDataset` inspection. |
| 2 | Do the 500 writers in `writers_dict_train/test.json` belong to UNHD or IAM's writer pool? | Reopened | Check actual file paths inside these two JSON files for Urdu vs. IAM-style (`c04-110`) filenames. |
| 3 | Is `split_words/` (IAM data) read by any script at all? | Open | Search all `.py` files for `split_words` references or matching CSV-parsing logic; check for a not-yet-located recognizer-pretraining script. |
| 4 | Where is the Qwen3-VL-4B-Instruct fine-tuning / word-level scoring code? | Not found in shared repo | Confirm whether this lives in a separate repo/notebook. |
| 5 | Where does UNHD Stage II fine-tuning actually happen? | Open — `scripts/` folder ruled out | No run command found anywhere in `scripts/`; check for a location outside this folder, or confirm it was never committed. |
| 6 | Does the recognition loss actually backprop to the UNet given the `.detach()` call? | Open, higher priority than previously assessed | Trace the computation graph from `x_approx` to `loss.backward()`; check if `outputs.loss` requires grad at that point. |
| 7 | Is `vocab/ved` literally the unmodified stock `roberta-base` tokenizer? | Likely, unconfirmed | Count total keys in `vocab.json`; compare to 50,265. |
| 8 | Was Qwen selected after a real comparison against other OCR systems? | Open | At least 4 systems (EasyOCR, Google Vision API, `nomypython/urdu-ocr-deepseek`, Qwen) show evidence of being tried; no comparison results located yet. |
| 9 | How many manual resume incidents occurred during word-level training, and did they affect final results? | Open | At least two distinct manual resume points identified mathematically (≈epoch 94, ≈epoch 147) under the legacy (pre-checkpointing) script. |

---

## 8. Summary Read

The strongest, most load-bearing finding in this ledger is **Section 6.1**: the writer/style embedding is genuinely live in the default forward pass, directly contradicting the paper's explicit "content-only, deliberate" framing — though its practical effect is confirmed dummy at word level and still open at sentence level. Everything else uncovered so far — the IAM artifacts, the near-stock English tokenizer, the unused `Style_Text_Encoder` class, the CFG machinery pointed at the writer label rather than the text condition — is consistent with a single unifying explanation: **this codebase is a fork of an English-language repository (DiffusionPen, built and evaluated on IAM) that was adapted for Urdu incompletely, with meaningful original-language scaffolding never fully removed or verified as inert.**

Combined with the reconstruction-after-data-loss context (Section 1) and the `scripts/` folder findings (Section 5), a second load-bearing gap emerges: the actual Qwen-based evaluation pipeline and the actual UNHD Stage II fine-tuning run are both currently unlocated in what's been shared, meaning the paper's headline Rec. Acc. numbers (word- and sentence-level alike) cannot yet be fully traced back to a specific, inspectable script.
