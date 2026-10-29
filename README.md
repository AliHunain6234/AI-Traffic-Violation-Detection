# AI Traffic Violation Detection System

Detects vehicles that cross a stop line while the signal is **red** in recorded video (or an RTSP stream), and produces evidence for every violation: annotated frames, vehicle and plate crops, OCR text, a CSV log and a SHA-256 manifest. A web dashboard (FastAPI + React) lets you browse events, watch a live annotated stream and review benchmark and evaluation results.

Built with Pakistani roads in mind (per-camera configuration, polygon violation zones for faded or missing stop lines, a class map ready for rickshaws and local vehicles), but it works on any intersection.

> **Status.** Every module is implemented and unit/integration tested on synthetic data and a few real clips. **It has not been evaluated on real Pakistani footage**, and the plate-detector stage ships as a placeholder (no plate model). The results table below is intentionally empty until you run `evaluate.py` and `benchmark.py` on your own labelled footage.
>
> Output is for **human review**, not an automatic legal determination.

---

## Architecture

```mermaid
flowchart LR
    V[Video file / RTSP] --> M[main.py Pipeline]
    C[config.yaml] --> M
    M --> D[detector + tracker<br/>YOLO26 / YOLOv8 +<br/>BoT-SORT / ByteTrack]
    D --> L[line_crossing<br/>bottom-centre, sign change,<br/>direction, once per ID]
    V --> S[signal_classifier<br/>HSV + majority vote + lag]
    L --> E{event classifier}
    S --> E
    E -->|red / amber| EV[evidence<br/>before / crossing / after<br/>vehicle crop]
    EV --> P[plate_reader<br/>detect, enhance, EasyOCR]
    E --> LG[logger<br/>events.csv + stats.json]
    P --> LG
    M --> O[annotated video]
    LG --> API[FastAPI<br/>REST, WebSocket, MJPEG]
    M -.live frames.-> API
    API --> W[React dashboard]
    LG --> EVAL[evaluate.py / benchmark.py]
    EVAL --> API
```

Event rules: **red + crossing → `red_light_violation`**, yellow + crossing → `amber_crossing`, green + crossing → `legal_crossing` (logged, no evidence). If the signal state is unknown (ROI occluded, light off) the crossing is **suppressed**, never guessed.

Speed (optional, per camera): every tracked vehicle's speed is measured with a **two-line** or **homography** calibration and compared with the limit for its class (car, motorcycle, rickshaw, bus, truck). Above limit + tolerance → `overspeed` event with evidence. See [Speed measurement](#speed-measurement).

## Project layout

```
config.yaml            every threshold, path and model name
main.py                CLI + Pipeline class
evaluate.py            predictions vs ground truth (precision / recall / F1)
benchmark.py           model x tracker comparison
src/                   detector, tracker, line_crossing, signal_classifier,
                       plate_reader, evidence, logger, utils
tools/                 define_line.py, define_signal_roi.py, define_speed.py (camera setup)
                       extract_frames.py, prelabel.py, relabel.py, train_detector.py (fine-tuning)
tests/                 pytest unit tests (line crossing, speed estimation)
ground_truth/          ground_truth_template.csv
dashboard/api          FastAPI backend
dashboard/web          React + Vite + TypeScript + Tailwind + Framer Motion + Recharts
notebooks/             colab_demo.ipynb
```

---

## Setup

Python 3.10+ is required.

### Local (CPU works; a GPU is faster)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Model weights (`yolo26n.pt`, `yolov8n.pt`, ...) are downloaded by Ultralytics on first use. EasyOCR downloads its English models the first time a plate is read (~100 MB).

### Google Colab (GPU)

Open `notebooks/colab_demo.ipynb`, choose **Runtime → Change runtime type → GPU**, and run the cells top to bottom. They install the requirements, process a video and show the results. Colab has no display, so the line and signal ROI are defined with the headless flags below.

---

## 1. Define the stop line and signal ROI

**Easiest: the dashboard's *Setup* page.** Open the dashboard → *Setup*, pick a video (or camera URL / `0`), press *Load frame*, then draw on the real frame: the stop line (click two points; an arrow shows the counted direction), the signal (drag a box), and the speed calibration (two lines + real distance, or a road rectangle with a live 1 m grid to check it). *Save* writes the same keys as the command-line tools below into `config.yaml` (comments preserved) and the next run uses them. It works for several approaches per camera, shows what is already configured, and can remove an approach. Editing the config from the browser is controlled by `dashboard.allow_config_edit`; set an API token before exposing the dashboard.

Or use the command-line tools. Coordinates are stored per camera in `config.yaml` (`camera.sites.<camera_id>`) together with the frame size they were defined on, so resizing the video later is safe.

**With a display (local):**

```bash
python tools/define_line.py --video traffic.mp4 --camera junction_1
python tools/define_signal_roi.py --video traffic.mp4 --camera junction_1 --frame 300
```

- *Line:* click two points, drag endpoints to adjust. The green **arrow shows the direction vehicles must travel** to count as crossing; press `d` to flip it. `Enter`/`s` saves, `r` resets, `q` quits.
- *Signal:* drag a tight box around the signal head. The window shows what the classifier reads inside it. Pick a `--frame` where the light is lit.

**Headless (Colab / SSH):**

```bash
python tools/define_line.py --video traffic.mp4 --export-frame frame.png     # look at frame.png, note two points
python tools/define_line.py --video traffic.mp4 --camera junction_1 \
       --points 120,540,880,560 --direction positive --preview line.png
python tools/define_signal_roi.py --video traffic.mp4 --camera junction_1 \
       --roi 812,96,38,110 --frame 300 --scan 40 --preview roi.png
```

`--scan N` classifies the ROI on N sampled frames and prints how often red / yellow / green appear, a quick sanity check of the HSV thresholds on your camera.

Select the camera for a run with `--camera junction_1` or by setting `camera.active`.

**No clear stop line?** Add a polygon in `camera.sites.<id>.violation_zones` (list of `[[x, y], ...]`); entering the zone counts as a crossing.

### Several stop lines, each with its own signal ("approaches")

One camera often sees more than one signal (e.g. a turn-arrow signal for the left lanes and a straight-ahead signal for the rest). Give every group of lanes its own **approach**: a stop line bound to the signal that controls it. A crossing is judged only by *its own* approach's signal, so a vehicle on a green lane is never flagged because another signal is red.

```bash
python tools/define_line.py       --video traffic.mp4 --camera junction_1 --approach left_lanes
python tools/define_signal_roi.py --video traffic.mp4 --camera junction_1 --approach left_lanes --frame 300
python tools/define_line.py       --video traffic.mp4 --camera junction_1 --approach main_road
python tools/define_signal_roi.py --video traffic.mp4 --camera junction_1 --approach main_road --frame 300
python tools/define_line.py       --video traffic.mp4 --camera junction_1 --remove-approach main_road   # delete one
```

(headless: add `--points` / `--roi` as above.) Approaches live in `camera.sites.<id>.approaches` and replace the single `stop_line` / `signal_roi` when present; older single-line configs keep working (they run as one approach named `main`). Two approaches that share a signal can point at the same ROI. `events.csv` gets an `approach` column, evidence frames highlight the event's own line, and the dashboard shows one signal light per approach. If one approach's signal cannot be read, only *its* crossings are suppressed.

## 2. Run

```bash
python main.py --video traffic.mp4 --config config.yaml --model yolo26n.pt \
               --tracker botsort --output outputs/annotated.mp4
```

| Flag | Meaning |
|---|---|
| `--video` | file path or `rtsp://` URL (default `video.source`) |
| `--model` | `yolo26n.pt`, `yolo26s.pt`, `yolov8n.pt`, or your fine-tuned weights |
| `--tracker` | `botsort` (default) or `bytetrack` |
| `--output` | annotated video path |
| `--show` | live window (needs a display) |
| `--camera`, `--device`, `--conf`, `--max-frames`, `--no-plates` | see `python main.py -h` |
| `--assume-signal red` | **testing only**: pretend the signal is always red (useful to exercise the line logic when no signal is visible) |

Outputs: `outputs/events.csv`, `outputs/stats.json`, `outputs/annotated.mp4`, `evidence/<event_id>/`.

**events.csv columns:** `event_id, track_id, vehicle_class, event_type, signal_state, frame_no, video_timestamp, wall_clock_time, confidence, plate_text, plate_conf, evidence_path` (+ `camera_id`). Set `video.start_time` (ISO-8601) in the config so `wall_clock_time` reflects when the footage was recorded; otherwise it is the processing time.

**Evidence folder:** `before.jpg`, `crossing.jpg`, `after.jpg` (annotated with box, ID, line, signal state and timestamps), `vehicle.jpg`, `plate.jpg` and `plate_enhanced.jpg` when a plate was found, `metadata.json`, `manifest.json` (SHA-256 of each file).

### Live camera (webcam, IP camera, phone)

`--video` accepts a **webcam index**, an **RTSP** URL or an **HTTP (MJPEG) URL** as well as a file:

```bash
python main.py --video 0 --camera junction_1                                    # laptop / USB webcam
python main.py --video "rtsp://user:pass@192.168.1.20:554/stream1" --camera junction_1   # CCTV / IP camera
python main.py --video "http://192.168.1.15:8080/video" --camera junction_1     # phone running "IP Webcam" / DroidCam
```

- **Set up the camera first, on a still frame:** `python tools/define_line.py --video <same source> --export-frame frame.png`, then use `--points` / `--roi` (the tools read one frame from the live source).
- **Real-time behaviour:** a background thread always keeps the newest frame, so if detection is slower than the camera it *skips* frames instead of falling behind. Timestamps (and therefore speed estimates) use real elapsed time. A dropped connection is re-opened automatically (`video.rtsp_reconnect_attempts`, `video.rtsp_reconnect_delay_s`); RTSP uses TCP.
- **Speed on CPU:** use a low-resolution sub-stream or `--resize-width 720`; the saved annotated file plays back at `output.live_fps` (10).
- **Video files play at real speed too.** Runs started from the dashboard pace a file by the clock like a camera (`dashboard.realtime_playback`): a detector that manages 3 frames per second skips the rest, so the clip is never slow motion, and speed estimates still use the video's own timestamps. From the command line add `--realtime`. The dashboard also turns off the annotated-file writer, downsizes to `dashboard.resize_width` and, without a GPU, uses `dashboard.cpu_imgsz` (640) for detection. On a plain CPU expect ~3-4 processed frames per second at 1080p; fast vehicles then move far between processed frames. A CUDA GPU removes the limit.
- **Dashboard:** *Live monitor* accepts the same sources (`0`, `rtsp://…`, `http://…`) when `dashboard.allow_live_sources` is true. Credentials in URLs are masked in logs and the UI. **Set `dashboard.api_token` before exposing the dashboard**, because a URL lets the server connect to any address it can reach.
- A webcam is only useful if it looks at a real road (stop line + signal in frame, fixed mount). Pointing a webcam at a screen playing a video is a handy functional test, not an accuracy test.
- Tested against a local fake MJPEG camera; not yet against a real RTSP CCTV camera.

### Crossing logic (why it is robust)

- Uses the **bottom-centre** of the box (where wheels meet the road), smoothed over a few frames.
- A crossing is a **sign change** relative to the line, but only after the point leaves a small dead-band, so jitter or a vehicle **stopped on the line** never flips the side by accident.
- Only the configured **direction** counts, and only if the point projects onto the **segment** (not the infinite line).
- **Once per track ID.** If a crossed vehicle gets a new ID right after the line (ID fragmentation) the new ID inherits the flag, so it is not counted twice.
- The signal state used is the state a few frames *before* the crossing (`signal.lag_frames`), compensating for the delay that majority-vote smoothing adds at transitions.

### Speed measurement

Speed needs a **calibration** that ties pixels to real metres. Pick one of two methods per camera (`tools/define_speed.py` saves it to `camera.sites.<id>.speed`):

| Method | You provide | Best for |
|---|---|---|
| `lines` | two lines across the road and the **real distance between them** (e.g. 20 m between two painted marks) | straight road, simplest and very accurate when the distance is measured well |
| `homography` | **4 road points forming a rectangle of known size** (e.g. one lane 3.5 m wide × 12 m long); the tool overlays a 1 m grid so you can check it | angled / perspective cameras, any direction of travel |

```bash
# two lines (interactive: click line A end 1, end 2, line B end 1, end 2)
python tools/define_speed.py --video traffic.mp4 --camera junction_1 --method lines --distance 20
# homography (interactive: click near-left, near-right, far-right, far-left corners)
python tools/define_speed.py --video traffic.mp4 --camera junction_1 --method homography --rect 3.5x12

# headless (Colab / SSH), with a preview image
python tools/define_speed.py --video traffic.mp4 --export-frame frame.png
python tools/define_speed.py --video traffic.mp4 --camera junction_1 --method lines \
       --lines "100,300,540,300;80,420,560,420" --distance 20 --preview speed.png
python tools/define_speed.py --video traffic.mp4 --camera junction_1 --method homography \
       --points "300,600,500,600,460,300,340,300" --rect 3.5x12 --preview speed.png
```

How it works: `lines` times the vehicle between the two lines (sub-frame interpolation, either direction); `homography` projects the bottom-centre of the box onto the road plane and measures displacement over time (median over the time inside the calibrated quadrilateral). Limits live in `config.yaml` under `speed.limits_kmh` (one value per detector class name, e.g. `car`, `motorcycle`, `bus`, `truck`, `rickshaw`, plus `default_limit_kmh` and `tolerance_kmh`). **The shipped limits are examples; set the ones that apply on your road.** Overspeed events are written to `events.csv` with `speed_kmh` and `limit_kmh`, get an evidence package with the speed burned into the frames, and appear on the dashboard. The annotated video shows each vehicle's speed.

Accuracy and caveats:
- It is only as good as the measured distance and the flatness of the road. **Verify** with a vehicle at a known steady speed (e.g. 40 km/h on its speedometer) before trusting numbers.
- Synthetic tests reproduce known speeds within a few percent; **real-world accuracy has not been measured.**
- Rickshaws are only reported as `rickshaw` once the detector is fine-tuned for them; until then they count as `car` and get the car limit.
- Vehicles must be tracked through the whole measured zone; heavy occlusion or ID switches inside it can drop or distort a measurement.
- These are estimates for human review, not calibrated enforcement-grade measurements.

### Plate reading

Two stages: plate detector → crop → upscale/denoise/CLAHE → EasyOCR. **No plate model is bundled.** Until you configure one, the reader returns an empty result with a note instead of guessing. To add one:

1. Label plates (one class, `plate`) from your own footage, covering Punjab, Sindh, KP, Balochistan and Islamabad formats, motorcycles and dirty/decorative plates.
2. Fine-tune: `yolo detect train data=plates.yaml model=yolo26n.pt imgsz=640 epochs=100`.
3. Set `plate.detector_model: runs/detect/train/weights/best.pt`.
4. Optionally add province regexes in `plate.format_patterns` to validate OCR text. The English EasyOCR model will misread Urdu/mixed-script plates; fine-tune or swap the OCR engine for those.

### Making it more reliable on your footage

| Problem | Switch | What it does |
|---|---|---|
| A **yellow taxi** (or any vehicle) passing the lamp reads as a yellow light | `signal.mask_vehicles` (on) | Vehicles overlapping the signal ROI are blacked out before classification; if they hide most of the ROI the state is `unknown` (violations suppressed, never guessed). |
| A **small / distant lamp** is never "seen" | `camera.sites.<id>.signal: {min_pixel_ratio: 0.005}` or `define_signal_roi.py --min-pixel-ratio 0.005` | Per-camera (or per-approach) sensitivity instead of lowering it for every camera. |
| **IDs break** under occlusion (one car counted twice, speed lost between the two lines) | `tracker.stitch.enabled: true` | A vehicle that vanishes mid-scene and reappears where it should be gets its old ID back. Conservative (never merges tracks that start/end at the frame border, different vehicle kinds or very different sizes) but a wrong merge is still possible in dense queues, so it is off until you have checked it on your footage. |
| **Handheld / wobbling camera** moves the scene under the fixed line and ROI | `stabilization.enabled: true` or `--stabilize` | Every frame is aligned to the first one (feature tracking + RANSAC). On the handheld Bangkok clip the road markings' shake dropped from ~22 px to ~1 px and the signal head's from ~9 px to <1 px. About 20 ms per 1080p frame. Not for a camera that is re-aimed or driven around (dashcam). |

### Teaching the detector new vehicle types (rickshaw, loader, ...) and plates

The stock COCO model has no rickshaw class (it says `car` / `truck`). Fine-tune it on frames from your own videos. The tools do everything except the actual checking of boxes, which only a person can do:

```bash
python tools/extract_frames.py --dataset rickshaw --video "videos/*.mp4" --every-sec 1.5     # 1. frames (skips near-duplicates)
python tools/prelabel.py --dataset rickshaw --classes car motorcycle bus truck rickshaw       # 2. first-draft boxes from the current model
python tools/relabel.py  --dataset rickshaw                                                   # 3. fix them: click a box, press its class key
python tools/train_detector.py all --dataset rickshaw --epochs 60                             # 4. split by video + train + install models/rickshaw.pt
```

* Step 3 is a keyboard/mouse editor (`n` next, `d` delete, `u` undo, drag to add a box). Pre-labels already contain the vehicles; you mostly press the rickshaw key on boxes that say `car` / `truck`. Aim for **300+ boxes per class** from several videos / locations. `train_detector.py` warns about classes with too few boxes and refuses unreviewed data unless you pass `--use-unreviewed`.
* The train/validation split is **by video** (a single video: the last 20 % of frames), never by frame, so the validation score is honest.
* On a CPU training is slow; zip `datasets/rickshaw` and run the same command on Colab's free GPU (use the notebook's GPU runtime), then copy `models/rickshaw.pt` back.
* The command prints the lines to put in `config.yaml`: `detector.model: models/rickshaw.pt` and `detector.classes: [car, motorcycle, bus, truck, rickshaw]` (class **names** work). Then add `rickshaw: 40` (or your limit) under `speed.limits_kmh`.
* **Plates** work the same way: `--classes plate`, label plates, train, and set `plate.detector_model: models/<name>.pt`.
* The fine-tuning pipeline was verified end to end on real frames (extract, pre-label, split, train one epoch, load by class names); the *quality* of a rickshaw model depends entirely on the boxes you label.

## 3. Evaluate

Fill `ground_truth/ground_truth.csv` by hand (copy the template):

```csv
frame_no,vehicle_description,event_type
412,white car left lane,red_light_violation
655,red motorcycle two riders,amber_crossing
790,blue bus right lane,legal_crossing
```

`frame_no` is the frame where the vehicle's wheels touch the line. Then:

```bash
python evaluate.py --pred outputs/events.csv --gt ground_truth/ground_truth.csv --tolerance 15
```

A prediction matches a ground-truth event if the type agrees and the frames differ by at most the tolerance. Matching is one-to-one (Hungarian assignment). The report lists precision, recall, F1, false positives and missed events, per type and overall, and writes `outputs/evaluation.json` (shown on the dashboard). Flags: `--ignore-type`, `--flagged-only`.

## 4. Benchmark

```bash
python benchmark.py --video traffic.mp4 --models yolov8n.pt yolo26n.pt \
                    --trackers botsort bytetrack --max-frames 300
```

Runs every model × tracker combination through the full pipeline and prints FPS, detect+track ms/frame, detections, unique track IDs, track-quality proxies and violations found; saves `outputs/benchmark.json`. **True ID switches need ground-truth tracks**, so the table reports proxies (`short_tracks`, `mean_track_len`): on the same video, fewer short tracks and longer tracks mean steadier IDs.

### Results (placeholder)

| Model + tracker | Precision | Recall | F1 | FPS (GPU) | FPS (CPU) |
|---|---|---|---|---|---|
| yolov8n + BoT-SORT | - | - | - | - | - |
| yolov8n + ByteTrack | - | - | - | - | - |
| yolo26n + BoT-SORT | - | - | - | - | - |
| yolo26n + ByteTrack | - | - | - | - | - |

*(Fill from `evaluate.py` and `benchmark.py` runs on your footage.)*

---

## Dashboard

FastAPI serves REST endpoints, a WebSocket for live updates, the MJPEG stream and evidence images. The React app has four pages: **Overview** (animated KPIs, charts, live feed), **Events** (search / filter / sort, evidence drawer with swipeable frames, CSV + PDF export), **Live monitor** (pick a video / camera setup, stream, pulsing signal light(s), live counters, start/stop a run), **Setup** (draw lines, signals and speed calibration on a frame) and **Analytics** (hourly heatmap, benchmark charts, evaluation gauges). Dark by default with a light toggle; honours `prefers-reduced-motion`; responsive down to phones.

### One command (recommended)

```bash
python run_dashboard.py                # uses .venv automatically, builds the UI when needed, opens the browser
python -m dashboard.api.main           # same thing without the .venv hand-off
```

The dashboard shows only what the pipeline produced from your own videos (`outputs/events.csv`, `outputs/stats.json`, `evidence/`); with no processed video it shows empty states. There is no demo or sample data.

Options: `--port 8080`, `--host 0.0.0.0`, `--no-browser`, `--rebuild`. Run it from any folder with the project's Python environment.

### Run (development)

```bash
uvicorn dashboard.api.main:app --reload
cd dashboard/web && npm install && npm run dev      # http://localhost:5173 (proxies /api to :8000)
```

### Run (single port, production-style)

```bash
cd dashboard/web && npm install && npm run build    # creates dashboard/web/dist
cd ../..
uvicorn dashboard.api.main:app --host 0.0.0.0 --port 8000   # serves API + built UI on :8000
```

Use another config with `TVD_CONFIG=path/to/config.yaml`.

### Security

The dashboard shows licence plates (personal data). **Set `dashboard.api_token`** (or `TVD_TOKEN`) before exposing it; the UI then asks for the token and sends it as a Bearer header (and as `?token=` for images, the stream and downloads, which the browser loads itself). Evidence files are served only from the evidence folder by validated IDs. The Live monitor can only start videos inside `dashboard.allowed_video_dirs`.

### Expose from Colab for a demo

Colab cannot open ports directly. Build the UI once so everything is on **one port**, then tunnel it:

```python
# 1) build the frontend (Node is needed once)
!curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs
!cd dashboard/web && npm ci && npm run build

# 2) set a token and start the API in the background
import os, subprocess
os.environ["TVD_TOKEN"] = "choose-a-long-random-token"
subprocess.Popen(["uvicorn", "dashboard.api.main:app", "--host", "0.0.0.0", "--port", "8000"])

# 3) open a public URL (Cloudflare quick tunnel, no account needed)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8000
```

Open the printed `https://*.trycloudflare.com` URL and enter the token. (`ngrok` via `pyngrok` works the same way.) The tunnel and the URL disappear when the Colab session ends; do not leave it running unattended.

---

## Configuration highlights

Everything lives in `config.yaml`: detector (model, conf, classes, `class_names` for custom models), tracker, crossing thresholds (dead-band, segment margin, dedup window), signal HSV ranges and smoothing, evidence frame offsets, plate OCR options, evaluation tolerance, benchmark model list and dashboard settings. Command-line flags override the matching keys.

## Tests

```bash
pytest -q          # line crossing (direction, jitter, stopped vehicles, ID fragmentation, zones) and speed estimation (both calibration methods, limits)
```

## Known limitations and future work

- **Not validated on real footage yet.** Thresholds (HSV ranges, dead-band, lag) are sensible defaults and need tuning per camera.
- **Night, rain, glare:** the HSV signal classifier degrades with LED flicker, glare and low light. Planned: a small CNN behind the same `SignalClassifier` interface, plus an `unknown` state that suppresses violations (already implemented).
- **Camera shake / moved cameras:** shake is compensated (`--stabilize`) by aligning frames to the first one; a camera that is re-aimed, zoomed or driven around (dashcam) is not supported, and near objects can still shift slightly because of parallax.
- **Pakistani traffic:** COCO classes lack rickshaws, loaders, Suzuki pickups and Hiace vans, and motorcycles with 2-3 riders are common. The fine-tuning tools above exist but **no rickshaw model has been trained yet** (it needs your labelled boxes). Heavy occlusion and weak lane discipline increase ID fragmentation; ID stitching helps but is unproven on Pakistani footage.
- **Pakistani plates:** per-province formats, Urdu/mixed script, non-standard plates and rear-only motorcycle plates. Needs a fine-tuned plate detector and OCR, plus format validation. Plates are often unreadable at CCTV resolution.
- **Stop line may be missing:** use polygon zones; a vehicle crossing diagonally through a zone edge is treated as one entry.
- **True ID-switch metrics** need annotated tracks; only proxies are reported.
- **Speed accuracy** depends on the calibration (measured distance, road flatness) and has only been verified on synthetic motion. A radar/known-speed validation run and per-class limits for Pakistani roads are still to be done; a tilted or moving camera invalidates the calibration.
- **Edge deployment:** export to ONNX / TensorRT, INT8 quantisation, and benchmark on Jetson-class hardware.
- **Dashboard data store:** events are read from CSV (fine for a single site or a few runs). A multi-camera deployment should move to a database.
- **Legal:** admissibility requirements differ by province and authority; this tool provides evidence for review, nothing more.
