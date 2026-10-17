# Stimulus and representation audit

## Media and alignment

- Representative training clips contain H.264 video plus AAC-LC stereo audio. Observed audio sample rates are 32 kHz and 48 kHz; the audio extractor decodes all clips to a consistent 16 kHz mono waveform.
- Training clips are 90 frames at 29.97 fps (about 3.003 s); the inspected test clip is 90 frames at 30 fps (3.000 s). All inspected containers start at 0 seconds.
- Organizer materials specify a 3-second stimulus and a matching 17-channel, 500 Hz, 1,500-sample EEG recording. Existing preprocessing treats sample zero as stimulus onset, downsamples to 100 Hz, and scores 50-600 ms. No offset correction is documented or applied.

## Cached representations

- Mean CLIP: `(1000, 512)`; OpenCLIP `ViT-B-32`, OpenAI checkpoint, QuickGELU, final projected `encode_image` embedding averaged over eight uniform video frames.
- Frame cache: `(1000, 8, 512)` and derived two-segment features `(1000, 1024)`.
- New intermediate CLIP caches: blocks 4, 8, 12 (zero-indexed blocks 3, 7, 11), each `(1000, 768)` CLS embeddings averaged across the same eight frames. These are normalized only inside each train fold by the encoder pipeline.
- New simple audio cache: `(1000, 76)`: 12 temporal RMS-envelope bins plus mean and standard deviation of 32 log-mel bands.
- Existing unrelated cached representations include AlexNet, DINOv2-S, VideoMAE, PANNs, and the final-test feature analogues; they were not re-evaluated here.
