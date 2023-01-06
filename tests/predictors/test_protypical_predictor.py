import unittest
import torch
import json
import numpy as np
from pathlib import Path

from gaze_verification.data_objects import Sample
from gaze_verification.predictors import PrototypicalPredictor


class TestPrototypicalPredictor(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()

    def test_create(self):
        targets_config_path = str(
            self._current_dir.parent / "target_configurators" / "targets_config_test/targets_config.json")

        # Inference mode
        predictor = PrototypicalPredictor(
            targets_config_path=targets_config_path,
            hidden_size=128,
            embedding_size=56,
            n_support=5,
            confidence_level=0.5,
            p_dropout=0.6,
            do_compute_loss=False,
            is_predict=True
        )
        print(f"Predictor created!\n{predictor}")

    def test_compute_loss(self):
        targets_config_path = str(
            self._current_dir.parent / "target_configurators" / "targets_config_test/targets_config.json")

        # Train mode
        n_support = 5
        n_classes = 3
        predictor = PrototypicalPredictor(
            targets_config_path=targets_config_path,
            hidden_size=128,
            embedding_size=56,
            n_support=n_support,
            confidence_level=0.5,
            p_dropout=0.6,
            do_compute_loss=True,
            is_predict=False
        )
        print(f"Predictor created!\n{predictor}")

        # Create dummy inputs
        n_query = 5
        bs = n_classes * (n_support + n_query)
        input_size = (bs, 56)  # [bs, emedding_size]
        label_logits = torch.rand(*(input_size))
        labels = torch.cat([torch.full((n_query + n_support,), fill_value=i) for i in range(n_classes)])
        labels = labels[torch.randperm(len(labels))]  # shuffle
        loss = predictor.compute_loss(label_logits, labels)
        print(f"Loss: {loss}")

    def test_set_predicted_label(self):
        targets_config_path = str(
            self._current_dir.parent / "target_configurators" / "targets_config_test" / "targets_config.json")

        # Train mode
        n_support = 5
        predictor = PrototypicalPredictor(
            targets_config_path=targets_config_path,
            hidden_size=128,
            embedding_size=56,
            n_support=n_support,
            confidence_level=0.5,
            p_dropout=0.6,
            do_compute_loss=True,
            is_predict=False
        )
        print(f"Predictor created!\n{predictor}")

        predicted_class_id = 1
        predicted_class_proba = 0.09
        data = np.zeros((10, 4), dtype=np.float32)
        sample = Sample(guid=0, seq_id=0, session_id=0, data=data)
        sample = predictor.set_predicted_label(sample, predicted_class_id, predicted_class_proba)
        self.assertTrue(sample.predicted_label is not None)
        print(f"Predicted: {sample.predicted_label}")

    def test_predict(self):
        targets_config_path = str(
            self._current_dir.parent / "target_configurators" / "targets_config_test/targets_config.json")

        # Train mode
        n_support = 5
        n_classes = 3
        predictor = PrototypicalPredictor(
            targets_config_path=targets_config_path,
            hidden_size=128,
            embedding_size=56,
            n_support=n_support,
            confidence_level=0.5,
            p_dropout=0.6,
            do_compute_loss=False,
            is_predict=True
        )
        print(f"Predictor created!\n{predictor}")

        # Create dummy inputs
        n_query = 5
        emedding_size = 56
        bs = n_classes * (n_support + n_query)
        input_size = (bs, emedding_size)  # [bs, emedding_size]

        # support
        support_embeddings = torch.rand((n_classes * n_support, emedding_size))
        support_labels = torch.cat([torch.full((n_support,), fill_value=i) for i in range(n_classes)])
        support_labels = support_labels[torch.randperm(len(support_labels))]  # shuffle

        # query
        query_embeddings = torch.rand((n_classes * n_query, emedding_size))

        # Var. 1 with 'hot' prototypes calculation
        output = predictor.head.predict(query_embeddings=query_embeddings,
                                        support_embeddings=support_embeddings,
                                        support_labels=support_labels)
        print(f"Output:")
        for k, v in output.items():
            print(f"{k} with size {v.size()}\n\t{v}")


if __name__ == '__main__':
    unittest.main()
