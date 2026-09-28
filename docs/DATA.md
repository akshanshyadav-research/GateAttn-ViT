# Data and checkpoints

## Image classification data

The evaluator expects class-organized images:

```text
/path/to/validation/
  n01440764/
    image_001.jpg
  n01443537/
    image_002.jpg
```

Pass `--class-map` with a JSON mapping from these directory names to the selected checkpoint's classifier output indices. Both formats below are supported:

```json
{"n01440764": 0, "n01443537": 1}
```

```json
{"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"]}
```

`configs/class_map.example.json` is a two-class format example, not the full ImageNet mapping. The mapping used by the original notebooks lived outside the supplied folder. Do not create an alphabetical 0-to-N mapping for an ImageNet subset; this changes the target labels. Folder names missing from the mapping and out-of-range output indices cause explicit errors.

Historical notebooks point to dataset locations such as `/home/viraj/my_project/val/val2/`. The reusable runner replaces those hard-coded paths with command arguments. It does not automatically use COCO files for ImageNet classification.

## Assets found in the supplied folder

The local archive includes COCO validation images and annotation JSON files, ZIP archives, and images used for visual demonstrations. Each is listed in `archive/manifest.json`. Paths remain under `local_assets/<original-path>`. The archive also preserves notebooks mentioning ImageNet-A.

Raw COCO, ImageNet-A, CIFAR-100, ObjectNet, and ImageNet each have distinct label spaces and evaluation protocols. The new folder-based runner is not a detection evaluator or an automatic protocol adapter for those datasets. Dataset-specific exploratory code remains in the notebook archive. No external class mapping or evaluation protocol is fabricated.

## Model weights

A normal run loads a pretrained checkpoint through timm. Network access may be needed the first time. For offline reproduction, supply a compatible local checkpoint:

```bash
gateattn-evaluate --config configs/vit_base_combined.json \
  --checkpoint /path/to/exact_checkpoint.pth \
  --data /path/to/validation --class-map /path/to/class_map.json \
  --device cpu --output runs/local-checkpoint
```

Pinning the timm version is not sufficient to identify historical pretrained weights. The metadata records the resolved pretrained configuration and a SHA-256 over the actual model state. Share the original checkpoint identifier or file checksum with published runs.

Configurations include ViT Tiny/Small/Base/Large, DeiT Tiny/Small/Base, and an explicit DeiT3 Large variant present in the source. Historical `deit_large_patch16_224` calls are preserved in the archive; the new code does not silently reinterpret that name as DeiT3. Distilled DeiT notebooks are also preserved, but the reusable pruning wrapper requires a single CLS prefix token.
