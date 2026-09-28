# Validation record

Repository preparation validation was completed on **2026-09-28**, using Python 3.10 on CPU. Package versions are recorded in `validation_environment.json`.

| Check | Outcome |
| --- | --- |
| Source inventory | 6,162 files; 3,294,239,638 bytes accounted for |
| Exact local archive | SHA-256 verified for every source file |
| Original parent folder | Every original file still matches its captured SHA-256 |
| Notebook reconstruction | All 1,060 notebooks restored semantically exactly, including outputs and metadata |
| Python export equivalence | All 1,060 exported code ASTs match the original notebook code; multiline string content is preserved |
| Core CPU tests | 12 tests passed, including original entropy-function parity and unpruned timm prediction parity |
| Editable installation | Installed successfully in a local virtual environment |
| Evaluation CLI | Completed single-image ViT-Tiny smoke test with random weights and explicit provenance |
| Visualization CLI | Generated 27 figure groups for 12-block pruned/unpruned comparison, with PDF/SVG/PNG and raw numerical arrays |
| Original figure restoration | Restored entropy visualization notebook and extracted 52 embedded figures without executing it |
| Experiment sweeps | Generated 27 combined-pruning configurations for Tiny, Small, and Base |
| Completed-run plotting | Generated labeled smoke-test comparison and provenance CSV |
| Published-data figures | Generated six figure groups in PDF/SVG/PNG |
| Synthetic walkthrough | Executed all code cells; retained 98 of 196 patches at 50% selection |
| Documentation links | All local links in README and documentation resolve |
| Ordinary Git file sizes | Large original notebooks, datasets, and generated binary outputs use Git LFS; the largest ordinary Git artifact is about 8 MB |

The random-weight smoke test validates software execution only. Its accuracy and timing are not research results. No pretrained checkpoints were downloaded, full ImageNet evaluation was not rerun, and FPGA execution was not tested. The validation distinguishes implementation checks, preservation checks, and the paper's published measurements.

The local validation environment inherits existing scientific packages through `--system-site-packages`; a fully isolated dependency installation was not tested here. The CI definition installs the declared package dependencies and CPU PyTorch on a fresh runner, but remote CI has not yet run because the repository has not been pushed.

## Complete experiment inclusion

The repository now tracks all 6,162 original source files through Git LFS, including exact original notebooks, the paper, all images, COCO data, annotations, and ZIP archives. The inventory was compared with the current parent source folder: no source files are missing from the manifest. All 176 generated run artifacts are also tracked. Only environment/build files and interpreter caches remain excluded.

`python tools/verify_archive.py --local --git-tracked --original-source ..` checks original bytes, notebook reconstruction, and all tracked original-file identities. Run `git lfs pull` before this check on a new clone; `git lfs fsck` verifies the committed LFS objects.
