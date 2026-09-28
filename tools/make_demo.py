#!/usr/bin/env python3
"""Generate a deterministic synthetic example of entropy gating; no model or dataset download."""
from pathlib import Path
import numpy as np
from PIL import Image
from gateattn.entropy import patch_entropy, select_mask
from gateattn.visualize import plot_entropy_panels


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'runs/demo'
    output.mkdir(parents=True,exist_ok=True)
    image=np.full((224,224,3),235,dtype=np.uint8)
    rng=np.random.default_rng(2026)
    image[48:176,64:160]=rng.integers(0,256,(128,96,3),dtype=np.uint8)
    yy,xx=np.indices((64,64))
    checker=((xx//4+yy//4)%2*180+30).astype(np.uint8)
    image[128:192,128:192]=checker[:,:,None]
    Image.fromarray(image).save(output/'synthetic_input.png')
    scores=patch_entropy(image)
    mask=select_mask(scores,.5,'stable')
    plot_entropy_panels(image,scores,mask,output,'Synthetic entropy-gating demonstration — not an evaluation image')
    np.savez_compressed(output/'patch_data.npz',entropy_bits=scores,keep_mask=mask,rgb=image)
    print(output)


if __name__=='__main__':
    main()
