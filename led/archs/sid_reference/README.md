# SID reference architecture snapshot

Copied from the sibling `raw_image_denoising` repository on 2026-10-26.
This is a local snapshot: training does not import or depend on that repository.

- `models/learning_dwt_repncb.py` and its wavelet/block dependencies preserve the original LearningDWT + RepNCB model.
- `models/mrlfn_arch.py` preserves the original mosaic-domain k=4 S2D/D2S path, including its shallow skip source.
- `models/splitternet.py` contains the SplitterNet classes extracted from `models/paper_denoisers.py`; the unrelated BRVE implementation is omitted.
- Imports are namespaced under `led.archs.sid_reference`. Model computations are unchanged.
- `../sid_reference_arch.py` registers the models and converts LED `[R,G1,B,G2]` to reference `[R,G1,G2,B]` at input and back at output.

The source repository license is retained in `LICENSE`; SplitterNet's upstream attribution/license is retained in `SplitterNet-CC-BY-NC-SA-4.0.md`.
