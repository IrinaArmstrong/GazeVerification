import unittest
import torch
import json
import numpy as np
from pathlib import Path

from gaze_verification.data_objects import Sample, Samples
from gaze_verification.modelling import Model
from gaze_verification.embedders import Conv2DEmbedder, Conv2dEmbedderConfig
from gaze_verification.bodies import EmptyBody, EmptyBodyConfig
from gaze_verification.predictors import PrototypicalPredictor


class TestModel(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()

    def test_create(self):
        targets_config_path = str(
            self._current_dir / "target_configurators" / "targets_config_test/targets_config.json")

        embedder_config_path = str(self._current_dir / "embedders" / "test_embedder_4d_config_params.json")
        with open(embedder_config_path, encoding="utf-8") as f:
            embedder_config_json = json.load(f)

        body_config_path = str(self._current_dir / "bodies" / "test_empty_config_params.json")
        with open(body_config_path, encoding="utf-8") as f:
            body_config_json = json.load(f)

        # Create embedder
        conv_param = embedder_config_json.get("conv")
        fc_param = embedder_config_json.get("fc")
        embedder_config = Conv2dEmbedderConfig(
            input_size=(2, 4, 1000),
            embedding_size=128,
            num_conv_blocks=len(conv_param),
            num_fc_blocks=len(fc_param),
            conv_params=conv_param,
            fc_params=fc_param
        )
        embedder = Conv2DEmbedder(embedder_config)

        # Create body
        body_config = EmptyBodyConfig(
            input_size=body_config_json.get('input_size'),
            hidden_size=body_config_json.get('hidden_size')
        )
        body = EmptyBody(body_config)

        # Create predictor - in train mode
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

        # Model
        model = Model(embedder=embedder,
                      body=body,
                      predictor=predictor)
        print(f"Model created!\n{model}")

    def test_forward(self):
        # Hyperparameters
        n_support = 5
        n_classes = 3
        bs = 2
        input_size = (bs, 4, 1000)  # [bs, dims, seq_len]
        embedding_size = 128
        predictor_embedding_size = 56

        targets_config_path = str(
            self._current_dir / "target_configurators" / "targets_config_test/targets_config.json")

        embedder_config_path = str(self._current_dir / "embedders" / "test_embedder_4d_config_params.json")
        with open(embedder_config_path, encoding="utf-8") as f:
            embedder_config_json = json.load(f)

        body_config_path = str(self._current_dir / "bodies" / "test_empty_config_params.json")
        with open(body_config_path, encoding="utf-8") as f:
            body_config_json = json.load(f)

        # Create embedder
        conv_param = embedder_config_json.get("conv")
        fc_param = embedder_config_json.get("fc")
        embedder_config = Conv2dEmbedderConfig(
            input_size=input_size,
            embedding_size=embedding_size,
            num_conv_blocks=len(conv_param),
            num_fc_blocks=len(fc_param),
            conv_params=conv_param,
            fc_params=fc_param
        )
        embedder = Conv2DEmbedder(embedder_config)

        # Create body
        body_config = EmptyBodyConfig(
            input_size=body_config_json.get('input_size'),
            hidden_size=body_config_json.get('hidden_size')
        )
        body = EmptyBody(body_config)

        # Create predictor - in train mode
        predictor = PrototypicalPredictor(
            targets_config_path=targets_config_path,
            hidden_size=embedding_size,
            embedding_size=predictor_embedding_size,
            n_support=n_support,
            confidence_level=0.5,
            p_dropout=0.6,
            do_compute_loss=True,
            is_predict=False
        )

        # Model
        model = Model(embedder=embedder,
                      body=body,
                      predictor=predictor)
        print(f"Model created!\n{model}")

        # Create dummy inputs
        # [bs, dims, seq_len]
        batch = torch.rand(*(input_size))
        # support
        support_embeddings = torch.rand((n_classes * n_support, predictor_embedding_size))
        support_labels = torch.cat([torch.full((n_support,), fill_value=i) for i in range(n_classes)])
        support_labels = support_labels[torch.randperm(len(support_labels))]  # shuffle
        model.predictor.head.init_prototypes(support_embeddings, support_labels)

        # forward pass
        output = model(batch)
        predictions, scores, predictor_output, probabilities = output
        print(f"Output:")
        print(f"Predictions: {predictions.size()}")
        print(f"Scores: {scores.size()}")
        print(f"Predictor_output: {predictor_output.size()}")
        print(f"Probabilities: {probabilities.size()}")



if __name__ == '__main__':
    unittest.main()
