# Figure Sources

This folder contains TikZ sources referenced in `docs/architecture.md`.

## Files

- `docs/figures/ecg_resnet1d_pytorch2tikz.tex`
  - Generated with `pytorch2tikz` from `ECGResNet` in `scripts/train.py`.
  - Post-processed to remove `np.float64(...)` wrappers in shift values.
  - Note: `pytorch2tikz` is under an academic-use license (Fraunhofer HHI).
- `docs/figures/resblock_1d_nn_graphics.tex`
  - Residual block schematic styled after `nn_graphics` (MIT).
- `docs/figures/classifier_head_nntikz.tex`
  - Classifier head schematic styled after `nntikz` (MIT).

## Compile

Example (PDF output):

```bash
pdflatex docs/figures/ecg_resnet1d_pytorch2tikz.tex
pdflatex docs/figures/resblock_1d_nn_graphics.tex
pdflatex docs/figures/classifier_head_nntikz.tex
```

## References

- `https://github.com/fraunhoferhhi/pytorch2tikz`
- `https://github.com/mvoelk/nn_graphics`
- `https://github.com/fraserlove/nntikz`
