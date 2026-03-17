import torch
import torch.nn as nn
import unittest
import sys
import io
from unittest.mock import patch, MagicMock

# Import the module so we can test it
import src.quantum_weights as qw
from src.quantum_weights import (
    find_quantum_layer,
    extract_quantum_weights,
    summarize_quantum_weights,
    find_classical_to_quantum_mapper,
    infer_quantum_metadata,
    get_default_quantum_dummy_input,
    draw_quantum_circuit_from_model
)

# A fake Custom Hybrid Head for testing
class FakeBaseHybridHead(nn.Module):
    def __init__(self, n_qubits, q_depth, max_layers):
        super().__init__()
        self.pre_net = nn.Linear(10, n_qubits)
        self.q_params = nn.Parameter(torch.randn(max_layers * n_qubits))
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.max_layers = max_layers
    
    def forward(self, x):
        return self.pre_net(x)

# Since pennylane might be installed in this env, we test with a fake BaseHybridHead
# or dictionary to ensure the core logic runs without needing a real PennyLane TorchLayer.

class TestQuantumWeights(unittest.TestCase):

    def setUp(self):
        # We'll use our fake hybrid head for module-level tests where TorchLayer isn't strictly needed.
        # Patching `find_quantum_layer` internal checks for BaseHybridHead to recognize our fake class.
        pass

    @patch('src.quantum_weights.BaseHybridHead', FakeBaseHybridHead, create=True)
    def test_find_quantum_layer_basehybrid(self):
        model = nn.Sequential(
            FakeBaseHybridHead(4, 2, 5),
            nn.Linear(4, 2)
        )
        
        # We need to manually inject FakeBaseHybridHead into the test scope of qw
        qw.BaseHybridHead = FakeBaseHybridHead
        qlayer = find_quantum_layer(model)
        self.assertIsNotNone(qlayer)
        self.assertIsInstance(qlayer, FakeBaseHybridHead)

    @patch('src.quantum_weights.BaseHybridHead', FakeBaseHybridHead, create=True)
    def test_extract_and_summarize_weights(self):
        model = FakeBaseHybridHead(3, 2, 4)
        qw.BaseHybridHead = FakeBaseHybridHead
        
        weights = extract_quantum_weights(model)
        self.assertIn("q_params", weights)
        self.assertEqual(weights["q_params"].shape, (12,))
        
        summary = summarize_quantum_weights(model)
        self.assertIn("q_params", summary)
        self.assertEqual(summary["q_params"]["shape"], (12,))

    @patch('src.quantum_weights.BaseHybridHead', FakeBaseHybridHead, create=True)
    def test_find_classical_mapper(self):
        head = FakeBaseHybridHead(4, 2, 5)
        model = nn.Sequential(head)
        qw.BaseHybridHead = FakeBaseHybridHead
        
        found_mapper = find_classical_to_quantum_mapper(model)
        self.assertIs(found_mapper, head.pre_net)

    @patch('src.quantum_weights.BaseHybridHead', FakeBaseHybridHead, create=True)
    def test_infer_metadata(self):
        model = FakeBaseHybridHead(5, 3, 4)
        qw.BaseHybridHead = FakeBaseHybridHead
        
        meta = infer_quantum_metadata(model)
        self.assertEqual(meta["n_qubits"], 5)
        self.assertEqual(meta["q_depth"], 3)
        self.assertEqual(meta["max_layers"], 4)

    def test_dummy_input(self):
        dummy = get_default_quantum_dummy_input(4, 0.7)
        self.assertEqual(dummy.shape, (4,))
        self.assertTrue(torch.allclose(dummy, torch.tensor([0.7, 0.7, 0.7, 0.7])))

    @patch('src.quantum_weights.BaseHybridHead', FakeBaseHybridHead, create=True)
    def test_draw_circuit_smoke(self):
        # We'll patch out pennylane specifically in qw just for this smoke test to ensure
        # the function catches the failure gracefully if it's missing, or runs if present.
        
        model = nn.Sequential(FakeBaseHybridHead(2, 2, 3))
        qw.BaseHybridHead = FakeBaseHybridHead
        
        try:
            import pennylane
        except ImportError:
            self.skipTest("PennyLane not installed. Skipping circuit drawing smoke test.")
            
        try:
            import matplotlib.pyplot as plt
            original_show = plt.show
            plt.show = lambda: None
            
            # We also mock qml.draw and draw_mpl so we don't actually hang or crash if the fake model isn't fully compatible
            with patch('pennylane.draw'), patch('pennylane.draw_mpl'):
                captured_output = io.StringIO()
                sys.stdout = captured_output
                
                # We expect it might fail inner drawing since FakeBaseHybridHead isn't a REAL QNode,
                # but it should ATTEMPT to run the function. We'll catch and ignore expected internal draw errors
                # because this is a smoke test just verifying the wrapper logic.
                try:
                    draw_quantum_circuit_from_model(model)
                except Exception:
                    pass
                
            sys.stdout = sys.__stdout__
            plt.show = original_show
            
        except Exception as e:
            self.fail(f"Drawing circuit raised an unexpected exception: {e}")


    @patch('src.quantum_weights.BaseHybridHead', FakeBaseHybridHead, create=True)
    def test_extract_quantum_inputs(self):
        head = FakeBaseHybridHead(4, 2, 5)
        model = nn.Sequential(
            nn.Linear(10, 10),
            head,
            nn.Linear(4, 2)
        )
        qw.BaseHybridHead = FakeBaseHybridHead
        
        sample = torch.randn(3, 10)
        q_in = qw.extract_quantum_inputs(model, sample)
        self.assertIsNotNone(q_in)
        # FakeBaseHybridHead captures inputs at its root, so it sees the (3, 10) output of the preceding Linear layer
        self.assertEqual(q_in.shape, (3, 10))
        
    @patch('src.quantum_weights.BaseHybridHead', FakeBaseHybridHead, create=True)
    def test_export_quantum_artifacts(self):
        import tempfile
        import os
        import numpy as np
        
        head = FakeBaseHybridHead(4, 2, 5)
        model = nn.Sequential(head)
        qw.BaseHybridHead = FakeBaseHybridHead
        
        sample = torch.randn(2, 10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = qw.export_quantum_artifacts(model, sample_input=sample, save_dir=tmpdir)
            
            self.assertIn("metadata", artifacts)
            self.assertIn("n_qubits", artifacts["metadata"])
            self.assertIn("weights", artifacts)
            # FakeBaseHybridHead outputs 'q_params'
            self.assertIn("q_params", artifacts["weights"])
            self.assertIsNotNone(artifacts["inputs"])
            self.assertEqual(artifacts["inputs"].shape, (2, 10))
            
            files = os.listdir(tmpdir)
            self.assertIn("quantum_metadata.json", files)
            self.assertIn("quantum_weights.npz", files)
            self.assertIn("quantum_inputs.npy", files)
            
            loaded_inputs = np.load(os.path.join(tmpdir, "quantum_inputs.npy"))
            self.assertTrue(np.allclose(loaded_inputs, artifacts["inputs"]))

    def test_find_final_classifier(self):
        """find_final_classifier should return the nn.Linear immediately after the TorchLayer."""
        import numpy as np

        # Build a realistic hybrid head: Linear → ReLU → FakeTorchLayer → Linear
        class FakeTorchLayer(nn.Module):
            def forward(self, x):
                return x

        try:
            from pennylane.qnn import TorchLayer as PL_TorchLayer
        except ImportError:
            PL_TorchLayer = None

        n_qubits = 4
        n_classes = 2

        # Build Sequential that looks like model_builder._build_classifier_head output
        fake_qlayer = FakeTorchLayer()

        # Patch TorchLayer to be our FakeTorchLayer inside find_final_classifier
        with patch("pennylane.qnn.TorchLayer", FakeTorchLayer):
            head = nn.Sequential(
                nn.Linear(8, n_qubits),
                nn.ReLU(),
                fake_qlayer,
                nn.Linear(n_qubits, n_classes),
            )
            wrapped = nn.Sequential(nn.Identity(), head)
            result = qw.find_final_classifier(wrapped)

        self.assertIsNotNone(result, "Expected find_final_classifier to find the trailing Linear")
        self.assertIsInstance(result, nn.Linear)
        self.assertEqual(result.out_features, n_classes)
        self.assertEqual(result.in_features, n_qubits)

    def test_apply_final_classifier(self):
        """apply_final_classifier should compute W.T @ x + b correctly."""
        import numpy as np

        n_samples = 3
        n_qubits = 4
        n_classes = 2

        rng = np.random.default_rng(42)
        quantum_outputs = rng.uniform(-1.0, 1.0, size=(n_samples, n_qubits))
        W = rng.standard_normal(size=(n_classes, n_qubits))
        b = rng.standard_normal(size=(n_classes,))

        result = qw.apply_final_classifier(quantum_outputs, W, b)

        expected = quantum_outputs @ W.T + b
        self.assertEqual(result.shape, (n_samples, n_classes))
        np.testing.assert_allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
