# Publishing this local repository

The repository is prepared locally on the server. It has not been pushed to a hosting service.

Before choosing a public release, select the code license and replace `LICENSE.md`. The citation identifies the paper and its authors. Checkpoints and exact external dataset identities should accompany any claim of independently reproduced accuracy. The paper-to-code map explicitly records the experimental and hardware artifacts absent from the supplied folder.

The source Git payload includes code, documentation, all notebook sources and outputs, original logs, extracted observations, and paper-transcribed data. The full local copies under `archive/originals/` and `local_assets/` are intentionally ignored. A source clone therefore does not include the raw datasets or publisher PDF; the checksum inventory remains available.

Once a remote repository exists, connect and push from this directory:

```bash
git remote add origin <your-repository-url>
git push -u origin main
```

Create a versioned release after running the documented tests and archive verification. No remote URL, hosting-account identity, archival DOI, or open-source license is invented by the preparation process.
