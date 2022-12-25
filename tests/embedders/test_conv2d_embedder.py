import unittest
import torch
import json
from pathlib import Path

from gaze_verification.embedders.conv2d_embedder import (Conv2dEmbedderConfig,
                                                         Conv2DEmbedder)

class TestSamplesDataset(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._config_path = self._current_dir / "test_embedder_2d_config_params.json"
        with open(self._config_path, encoding="utf-8") as f:
            self.json_config = json.load(f)

    def test_config_creation(self):
        conv_param = self.json_config.get("conv")
        fc_param = self.json_config.get("fc")
        config = Conv2dEmbedderConfig(
            input_size=(2, 2, 50),
            embedding_size=128,
            num_conv_blocks=len(conv_param),
            num_fc_blocks=len(fc_param),
            conv_params=conv_param,
            fc_params=fc_param
        )
        for k, v in config.__dict__.items():
            print(f"{k}: {v}")

    def test_create_embedder(self):
        conv_param = self.json_config.get("conv")
        fc_param = self.json_config.get("fc")
        config = Conv2dEmbedderConfig(
            input_size=(2, 2, 50),
            embedding_size=128,
            num_conv_blocks=len(conv_param),
            num_fc_blocks=len(fc_param),
            conv_params=conv_param,
            fc_params=fc_param
        )
        embedder = Conv2DEmbedder(config)
        print(f"Embedder created!")
        for name, params in embedder.named_children():
            if isinstance(params, torch.nn.Sequential):
                print(f"--- Block `{name}` ---")
                for inner_name, inner_params in params.named_children():
                    print(f"\t{inner_name}:   {inner_params}")
            else:
                print(f"{name}:   {params}")


if __name__ == '__main__':
    unittest.main()
