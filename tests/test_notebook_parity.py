"""Compare the refactored EDPG against the actual archived function, without running its notebook."""
import ast
import json
from pathlib import Path
import unittest
import cv2
import numpy as np
from gateattn.entropy import patch_entropy, select_mask


class NotebookParityTests(unittest.TestCase):
    def test_entropy_mask_matches_original_function(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'archive/notebooks/version_40_combine_0nly_attention_and_similarity.ipynb'
        if not path.exists():
            self.skipTest('Archive not present in package-only installation')
        nb=json.loads(path.read_text())
        code='\n'.join(''.join(c.get('source',[])) for c in nb['cells'] if c['cell_type']=='code')
        tree=ast.parse(code)
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_image_and_create_entropy_binary_array')
        # Independent Shannon reference used instead of importing the whole notebook.
        def shannon_entropy(values):
            _,counts=np.unique(values,return_counts=True)
            p=counts/counts.sum()
            return -(p*np.log2(p)).sum()
        scope={'np':np,'cv2':cv2,'shannon_entropy':shannon_entropy,'percentage_entropy':[]}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
        image=np.random.default_rng(7).integers(0,256,(280,320,3),dtype=np.uint8)
        gray=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(224,224))
        for rate in [0,.1,.15,.2,.5]:
            original=scope[function.name](image,100-rate*100)
            current=select_mask(patch_entropy(gray),rate)
            np.testing.assert_array_equal(current,np.asarray(original[1:],dtype=bool))


if __name__=='__main__':
    unittest.main()
