# Publishing the complete experiment repository

The repository is prepared locally on the server. It has not been pushed to a hosting service. Every original experiment file is tracked: code, notebooks, original outputs, logs, image assets, COCO data and annotations, ZIP archives, and the supplied paper. Run records are retained; standalone generated run graphs are excluded. The curated graph collection contains the original Colab outputs corresponding to Figures 7–9.

## Git LFS

`archive/originals/` and `local_assets/` use Git LFS to retain exact source bytes, including files larger than ordinary Git hosting limits. Large generated run outputs also use LFS. Git commits store content-addressed pointers; the corresponding file objects are stored by Git LFS. Both must reach the remote to publish the complete repository.

After creating a remote repository, publish from this directory:

```bash
git lfs install --local
git remote add origin https://github.com/IITjodhpurbuddy/GateAttn-ViT.git
git lfs push --all origin main
git push -u origin main
```

The normal Git LFS pre-push hook also uploads required objects. The explicit LFS push above makes the complete-object upload clear. No remote has been configured or uploaded during preparation.

To obtain the complete experiment elsewhere:

```bash
git lfs install
git clone https://github.com/IITjodhpurbuddy/GateAttn-ViT.git
cd GateAttn-ViT
git lfs pull
python tools/verify_archive.py --local --git-tracked
git lfs fsck
```

A hosting-service ZIP or plain `git archive` may include LFS pointers instead of the original data. Use the LFS-enabled clone instructions above for the complete experiment.

Select the code license before public release. The supplied paper and datasets retain their respective rights and terms; their inclusion here does not change them. The reproduction and paper-to-code guides document which external checkpoints and implementation artifacts were not present in the original folder.
