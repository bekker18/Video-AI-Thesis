# Video AI Thesis

Turns a video into an **identikit**: one JSON record of how the video looks (plastic level), what it
shows (figurative level) and how it is filmed (enunciative level), over a shared timeline. A
FRESCO-style score then compares videos through their identikits, including how each one changes
from keyframe to keyframe.

| Stage                      | What it does                                                             |
| -------------------------- | ------------------------------------------------------------------------ |
| `01-sampling`            | optional: writes frames to disk                                          |
| `02-segmentation`        | shot cuts (OmniShotCut) and keyframes (MobileCLIP): one per 3 s, 3 to 10 |
| `03-profiler`            | what each keyframe holds (YOLO11)                                        |
| `04-router`              | which experts run on which shot                                          |
| `05-global-experts`      | scene, tags, depth, segmentation and colour per keyframe                 |
| `06-detection-tracking`  | people and faces, tracked within each shot                               |
| `07-conditional-experts` | pose, gaze, emotion, age, gender and text on the tracks                  |
| `08-identity`            | joins tracks into people across shots                                    |
| `09-aggregation`         | statistics per person, per shot and per keyframe; camera movement        |
| `10-relations`           | who looks at whom, and whose emotions move together                      |
| `11-caption`             | one sentence for the video (Qwen3-VL)                                    |
| `12-identikit`           | assembles `identikit.json`                                             |

To replicate everything, follow the sections in order; [section 9](#9-replication-order-and-limits)
is the checklist. Commands run from the repository root, and work the same in PowerShell and in a Linux shell. Replace `YYYY-MM-DD` with the date, so that earlier reports are kept.

## 1. Setup

You only need Docker with NVIDIA GPU support. The times below were measured on an RTX 4070 Laptop GPU
(8 GB).

```powershell
docker compose build
```

## 2. Model weights

Weights (9 GB) go to `models/`. They download automatically before each stage; to fetch them all first:

```powershell
docker compose run --rm --entrypoint python3 pipeline -m src.tools.download_models
```

## 3. Datasets

| Dataset     | Used for                        | How to get it                                                                                       | Size   | Folder                            |
| ----------- | ------------------------------- | --------------------------------------------------------------------------------------------------- | ------ | --------------------------------- |
| AVE         | pipeline, validation            | command below                                                                                       | 2.1 GB | `data/raw/ave/`                 |
| Dior reels  | pipeline, validation, score     | **not public**: supply your own `.mp4` files                                                | 7.6 GB | `data/raw/diorreels/diorreels/` |
| Test clips  | sanity checks                   | **not public**: `anime`, `cat1`, `cat2`, `messi`, `neymar`, `patrick`, `thanos` | small  | `data/raw/test/`                |
| CameraBench | camera movement (09)            | [section 8](#camera-movement-on-camerabench)                                                         | 4.1 GB | `data/raw/camerabench/`         |
| ShotBench   | shot size (12)                  | [section 8](#shot-size-on-shotbench)                                                                 | 341 MB | `data/raw/shotbench/`           |
| UCO-LAEO    | who looks at whom (10)          | [section 8](#looks-at-on-uco-laeo)                                                                   | 4.1 GB | `data/raw/ucolaeo/`             |
| VSGaze      | the MTGS comparison on UCO-LAEO | [section 8](#looks-at-on-uco-laeo)                                                                   | 103 MB | `data/raw/vsgaze/`              |
| VPCD        | identity (08)                   | [section 8](#identity-on-vpcd)                                                                       | 5.7 GB | `data/raw/vpcd/`                |

AVE's labels and a seeded sample of its scenes come from [AVE](https://github.com/dawitmureja/AVE).
The videos come from YouTube; removed ones are skipped and listed in `unavailable.txt`.

```powershell
docker compose run --rm --entrypoint python3 pipeline -m src.tools.download_ave --count 100
```

## 4. Run the pipeline

### One video

```powershell
# whole pipeline
docker compose run --rm pipeline data/raw/test/messi.mp4

# one stage on its own, once its inputs exist
docker compose run --rm pipeline data/raw/test/messi.mp4 --stages 02-segmentation

# every option and its default
docker compose run --rm pipeline --help
```

Output lands in `data/processed/<video_name>/`:

```
identikit.json   the final record (12)
json/            every other stage's metadata
embeddings/      keyframe, face and body vectors (.npy)
previews/        rendered videos and pages, from section 5
frames/          decoded frames        (01)
shots/           first frame per shot  (02)
keyframes/       representative frames (02)
global/          dense maps per keyframe (05)
tracks/          best crops per track  (06)
```

The options you are most likely to change:

| Flag                            | Default         | Meaning                                                |
| ------------------------------- | --------------- | ------------------------------------------------------ |
| `--stages <name> ...`         | all but 01      | run a subset, always in pipeline order                 |
| `--device cpu\|cuda\|auto`      | `auto`        | where models run                                       |
| `--keyframe-every 3`          | `3`           | seconds of a shot per keyframe, so long shots get more |
| `--keyframes-min 3`           | `3`           | fewest keyframes in a shot                             |
| `--keyframes-max 10`          | `10`          | most keyframes in a shot                               |
| `--detector yolo11m`          | `yolo11s`     | yolo11n/s/m or rtdetr-l                                |
| `--det-stride 1`              | `2`           | detect every Nth frame; the rest are interpolated      |
| `--series-stride 2`           | `2`           | observed rows between samples in 07                    |
| `--caption-model qwen3-vl-4b` | `qwen3-vl-2b` | captioner used by 11                                   |
| `--caption-frames 8`          | `8`           | keyframes shown to the captioner                       |

### A whole collection

Each video's output is moved into a collection folder. Finished videos are skipped, so you can stop
and restart at any time. A third argument processes only that many videos: a fixed random sample, so
the same clips come up each time and a larger number adds to them.

```powershell
docker compose run --rm -T --entrypoint bash pipeline src/tools/run_collection.sh data/raw/test tests
docker compose run --rm -T --entrypoint bash pipeline src/tools/run_collection.sh data/raw/ave ave_processed
docker compose run --rm -T --entrypoint bash pipeline src/tools/run_collection.sh data/raw/diorreels/diorreels diorreels
docker compose run --rm -T --entrypoint bash pipeline src/tools/run_collection.sh data/raw/diorreels/diorreels diorreels 100
```

It prints only failures; each video's log is in `data/logs/<name>/`.

`evaluate_ave` and the camera refit in section 7 also need the first 50 AVE scenes, which no threshold
was fitted on:

```powershell
docker compose run --rm -T --entrypoint bash pipeline src/tools/copy_ave_subset.sh
```

Expect about 100 MB of output per minute of video, so roughly 100 GB for the Dior reels.

## 5. Previews

Draws what the stages produced over the source video, and the identikit as a page. The name is the
folder under `data/processed`.

| View            | Shows                                                                                                      |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `tracks`      | 06: person boxes, dashed where interpolated, faces, and the frames chosen as crops                         |
| `conditional` | 07: pose, head pose, gaze, emotion with valence and arousal, recognised text                               |
| `identity`    | 08: global person ids over the same boxes                                                                  |
| `relations`   | 10: where each gaze lands, who looks at whom, synchrony                                                    |
| `identikit`   | 12: the record as a page, then a second page, *Order over keyframes*, with every arc the score compares  |
| `similarity`  | the score's ranking, after FRESCO's Figure 11, with the order-only ranking. Only when named; see section 6 |

```powershell
# one clip, all views (similarity excluded)
docker compose run --rm --entrypoint python3 pipeline -m src.tools.preview tests/messi

# one clip, one view
docker compose run --rm --entrypoint python3 pipeline -m src.tools.preview tests/messi --views relations

# a whole collection, identikit pages only
docker compose run --rm -T --entrypoint bash pipeline src/tools/preview_collection.sh diorreels
```

Files appear in `data/processed/<video>/previews/`: `<view>_preview.mp4`, `identikit.png` and
`similarity.png`. For a collection, name more views after it to render those too, as in
`preview_collection.sh diorreels tracks identikit`; video views are slow and take a lot of disk.

## 6. Similarity between videos

A distance between every pair of processed videos, from their `identikit.json` files only: one distance
per field, calibrated over the collection, averaged per plastic, figurative and enunciative level.
Each level also has an **arc group**, which compares how a video changes from keyframe to keyframe.
It reports each video's nearest neighbours and k-medoids clusters. Only videos in the same folder are
compared, and identikits written before the keyframe lists existed are refused until their video is
rerun.

```powershell
# score a collection
docker compose run --rm -T --entrypoint python3 pipeline -m src.score.similarity --processed data/processed/diorreels --out data/score/diorreels_YYYY-MM-DD.json

docker compose run --rm -T --entrypoint python3 pipeline -m src.score.similarity --processed data/processed/ave_processed --out data/score/ave_YYYY-MM-DD.json

# also print one pair, field by field, and choose the number of clusters
docker compose run --rm -T --entrypoint python3 pipeline -m src.score.similarity --processed data/processed/tests --out data/score/tests_YYYY-MM-DD.json --clusters 3 --explain messi anime

# one video's nearest, median and farthest, by level and by order alone
docker compose run --rm -T --entrypoint python3 pipeline -m src.tools.preview diorreels/0-3657668001364317985-0 --views similarity --score data/score/diorreels_YYYY-MM-DD.json
```

## 7. Validation on AVE and Dior reels

Each report lands where `--out` says, with its figures beside it.

**Against AVE's per-shot labels** (AVE only, since Dior has no labels). The camera, shot-size and
indoor/outdoor rules were fitted on AVE scenes 51-100, so for those three fields quote the 50-scene
report; for everything else, such as cuts and people, use all 100. In `contrasts`, only the steady
against handheld check uses a fitted threshold.

```powershell
# every scene against AVE's labels
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_ave.py --processed data/processed/ave --out data/validation/ave_50_YYYY-MM-DD.json
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_ave.py --processed data/processed/ave_processed --out data/validation/ave_full_YYYY-MM-DD.json

# 11 groups the labels tell apart, such as exterior against interior
docker compose run --rm -T --entrypoint python3 pipeline src/validation/contrasts.py --out data/validation/contrasts/contrasts_YYYY-MM-DD.json
```

**Distributions and consistency** compare AVE with the Dior reels. Both need at least 20 videos in each
collection. Expectations are checked against intervals from resampled videos.

```powershell
# every identikit field's distribution, after FRESCO's Figures 9 and 10; test clips appear as marks
docker compose run --rm -T --entrypoint python3 pipeline src/validation/distributions.py --out data/validation/distributions/distributions_YYYY-MM-DD.json

# people and topics across models per keyframe, after FRESCO's Tables 3-5
docker compose run --rm -T --entrypoint python3 pipeline src/validation/consistency.py --out data/validation/consistency/consistency_YYYY-MM-DD.json
```

`consistency` caches YuNet's faces and a Mask2Former rerun beside its report, and fetches CLIP (1.6 GB)
into `models/validation/` on the first run. Add `--encoders` to also compare CLIP with the pipeline's
MobileCLIP.

**Refitting 09's camera thresholds** (optional). Each rule is fitted on AVE scenes 51-100 and scored on
1-50, so `data/processed/ave` must exist:

```powershell
docker compose run --rm -T --entrypoint python3 pipeline src/validation/tune_camera.py pairs --collections ave
docker compose run --rm -T --entrypoint python3 pipeline src/validation/tune_camera.py pairs --collections ave --stride 2
docker compose run --rm -T --entrypoint python3 pipeline src/validation/tune_camera.py evaluate --out data/validation/camera/camera_YYYY-MM-DD.json
```

## 8. Comparisons with the state of the art

These run on the benchmarks' own data, so they do not need your processed collections, apart from
CameraBench's `gate` step.

### Camera movement on CameraBench

09's rule on [CameraBench](https://github.com/sy77777en/CameraBench)'s 1,071 test clips, with the
benchmark's own scorer. `gate` checks the code against 06's and 09's output on the test clips; skip it if
you do not have them.

```powershell
docker compose run --rm -T --entrypoint python3 pipeline -m src.tools.download_camerabench
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_camera.py gate
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_camera.py shots
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_camera.py boxes
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_camera.py pairs
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_camera.py score --out data/validation/camera/camerabench_YYYY-MM-DD.json
```

### Shot size on ShotBench

12's shot-size rule on ShotBench's shot-size questions.

```powershell
docker compose run --rm -T --entrypoint python3 pipeline -m src.tools.download_shotbench
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_shotbench.py detect
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_shotbench.py features
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_shotbench.py score --out data/validation/shot/shotbench_YYYY-MM-DD.json
```

### Looks-at on UCO-LAEO

10's `looks_at` rule on [UCO-LAEO](https://uco.es/investiga/grupos/ava/?p=941). `--protocol mtgs` scores
it on MTGS's test pairs from VSGaze, the like-for-like comparison. Gaze-LLE's output is cached beside the
report (about 25 minutes the first time), so a rerun only rescores.

```powershell
docker compose run --rm -T --entrypoint bash pipeline src/tools/download_ucolaeo.sh
docker compose run --rm -T --entrypoint uv pipeline run --no-project --python 3.10 --with pandas --with tables python src/tools/download_vsgaze.py

docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_laeo.py --protocol mtgs --out data/validation/laeo_mtgs_YYYY-MM-DD.json
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_laeo.py --out data/validation/laeo_YYYY-MM-DD.json
```

### Identity on VPCD

08's rule on [VPCD](https://www.robots.ox.ac.uk/~vgg/data/vpcd/)'s person clustering, from its own features.

```powershell
docker compose run --rm -T --entrypoint python3 pipeline -m src.tools.download_vpcd
docker compose run --rm -T --entrypoint python3 pipeline src/validation/evaluate_identity.py --out data/validation/identity/vpcd_YYYY-MM-DD.json
```

## 9. Replication order and limits

1. Set up, and fetch the weights (sections 1 and 2).
2. Download AVE and put the Dior reels and test clips in place (section 3).
3. Run the test clips, AVE and the Dior reels (section 4).
4. Draw previews (section 5).
5. Score the collections and draw similarity pages (section 6).
6. Run the validators (section 7).
7. Run the four benchmark comparisons (section 8).

**What cannot be replicated exactly:**

- The Dior reels and test clips are not public, so your numbers on them need your own copies.
- YouTube removals can change which AVE scenes download.
