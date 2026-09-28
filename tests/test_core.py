import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from timm.models.vision_transformer import VisionTransformer
from gateattn.entropy import patch_entropy, select_mask, baseline_mask
from gateattn.model import GateAttnViT, attention_keep
from gateattn.quantization import fixed_point
from gateattn.data import load_class_map, samples
from gateattn.evaluate import ExperimentConfig, run_evaluation
from gateattn.visualize import spatial_attention


def backbone():
    return VisionTransformer(img_size=32, patch_size=8, embed_dim=24, depth=3,
                             num_heads=3, mlp_ratio=2, num_classes=3).eval()


class EntropyTests(unittest.TestCase):
    def test_known_distributions(self):
        self.assertEqual(patch_entropy(np.zeros((16,16),dtype=np.uint8))[0], 0)
        self.assertAlmostEqual(patch_entropy(np.arange(256,dtype=np.uint8).reshape(16,16))[0],8)
        image=np.zeros((16,16),dtype=np.uint8); image[:,8:]=255
        self.assertAlmostEqual(patch_entropy(image)[0],1)

    def test_selection_ties_and_extremes(self):
        scores=np.ones(196)
        self.assertEqual(select_mask(scores,.5).sum(),98)
        self.assertEqual(select_mask(scores,.5,'threshold').sum(),196)
        self.assertTrue(select_mask(scores,.5,'stable')[:98].all())
        self.assertEqual(select_mask(scores,0).sum(),196)
        self.assertEqual(select_mask(scores,.999).sum(),1)
        for rate in [-1,1,float('nan')]:
            with self.assertRaises(ValueError):select_mask(scores,rate)

    def test_baselines(self):
        a=baseline_mask(14,.2,'random',np.random.default_rng(4))
        b=baseline_mask(14,.2,'random',np.random.default_rng(4))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.sum(),156)
        self.assertEqual(baseline_mask(14,.2,'center').sum(),156)
        self.assertEqual(baseline_mask(14,.2,'center-square').sum(),144)

    def test_quantization_saturates(self):
        result=fixed_point(torch.tensor([-1000.,-128.,1.001,127.999,1000.]))
        expected=torch.tensor([-128.,-128.,1.,127.99609375,127.99609375])
        torch.testing.assert_close(result,expected,rtol=0,atol=0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)

    def test_unpruned_matches_timm(self):
        reference=backbone()
        model=GateAttnViT(copy.deepcopy(reference),capture_attention=True).eval()
        image=torch.randn(1,3,32,32)
        with torch.inference_mode():
            expected=reference(image)
            actual=model(image,np.ones(16,dtype=bool))
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        self.assertEqual(model.token_counts,[17,17,17])

    def test_pruning_and_original_spatial_indices(self):
        model=GateAttnViT(backbone(),[0,1],.5,'topk',True).eval()
        mask=np.arange(16)%2==0
        with torch.inference_mode():out=model(torch.randn(1,3,32,32),mask)
        self.assertEqual(tuple(out.shape),(1,3))
        self.assertEqual(model.token_counts,[9,5,3])
        self.assertEqual(len(model.retained_indices),2)
        self.assertTrue(all(i%2==0 for i in model.retained_indices.tolist()))
        grid=spatial_attention(model.snapshots[1],16)
        self.assertEqual(np.isfinite(grid).sum(),4)
        self.assertEqual(model.snapshots[0].patch_indices.tolist(),list(range(0,16,2)))

    def test_cls_kept_and_threshold_ties(self):
        scores=torch.tensor([0.,1.,1.,1.])
        keep=attention_keep(scores,.5)
        self.assertTrue(keep.all())
        self.assertEqual(attention_keep(scores,.5,'topk').tolist(),[True,True,False,False])

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):GateAttnViT(backbone(),[3],.1)
        with self.assertRaises(ValueError):GateAttnViT(backbone(),[1,1],.1)
        model=GateAttnViT(backbone()).eval()
        with self.assertRaises(ValueError):model(torch.randn(2,3,32,32))
        with self.assertRaises(ValueError):model(torch.randn(1,3,32,32),np.zeros(16,dtype=bool))


class EvaluationTests(unittest.TestCase):
    def test_evaluation_and_output_provenance(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'data'/'class_a';data.mkdir(parents=True)
            Image.fromarray(np.zeros((32,32,3),dtype=np.uint8)).save(data/'sample.png')
            mapping=root/'mapping.json';mapping.write_text('{"class_a": 2}')
            config=ExperimentConfig(model='fixture',random_init=True,attention_blocks=(0,1),attention_drop=.25)
            with patch('timm.create_model',side_effect=lambda *a,**k:backbone()):
                result=run_evaluation(config,root/'data',mapping,root/'run')
            self.assertEqual(result['images'],1)
            self.assertEqual(result['status'],'complete')
            metadata=json.loads((root/'run/metadata.json').read_text())
            self.assertEqual(metadata['metric_kind'],'random_initialization_smoke_test')
            self.assertEqual(len(metadata['weights_sha256']),64)
            prediction=json.loads((root/'run/predictions.jsonl').read_text())
            self.assertEqual(prediction['target'],2)
            self.assertEqual(len(prediction['image_sha256']),64)
            with self.assertRaises(FileExistsError):run_evaluation(config,root/'data',mapping,root/'run')

    def test_class_map_preserves_model_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'map.json'
            path.write_text('{"5": ["n_class", "example"]}')
            self.assertEqual(load_class_map(path),{'n_class':5})
            path.write_text('{"a": 1, "b": 1}')
            with self.assertRaises(ValueError):load_class_map(path)

    def test_configuration_rejects_inconsistent_baseline(self):
        with self.assertRaises(ValueError):ExperimentConfig(method='none',drop_rate=.5).validate()


if __name__ == '__main__':
    unittest.main()
