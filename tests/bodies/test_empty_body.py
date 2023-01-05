import unittest
import torch
import json
from pathlib import Path

from gaze_verification.bodies import EmptyBody, EmptyBodyConfig

class TestEmptyBody(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self._config_path = self._current_dir / "test_empty_config_params.json"
        with open(self._config_path, encoding="utf-8") as f:
            self.json_config = json.load(f)

    def test_creation(self):
        bs = 2
        input_size = self.json_config.get("input_size")
        hidden_size = self.json_config.get("hidden_size")
        # or:
        # input_size = (2, 128)
        config = EmptyBodyConfig(
            input_size=input_size,
            hidden_size=hidden_size
        )
        for k, v in config.__dict__.items():
            print(f"{k}: {v}")

        body = EmptyBody(config)
        print(f"Empty Body created!")
        for name, params in body.named_children():
            if isinstance(params, torch.nn.Sequential):
                print(f"--- Block `{name}` ---")
                for inner_name, inner_params in params.named_children():
                    print(f"\t{inner_name}:   {inner_params}")
            else:
                print(f"{name}:   {params}")


    def test_forward(self):
        bs = 2
        input_size = self.json_config.get("input_size")
        hidden_size = self.json_config.get("hidden_size")
        # or:
        # input_size = (2, 128)
        config = EmptyBodyConfig(
            input_size=input_size,
            hidden_size=hidden_size
        )
        for k, v in config.__dict__.items():
            print(f"{k}: {v}")

        body = EmptyBody(config)
        print(f"Empty Body created!")

        # Create dummy inputs
        # [bs, dims, seq_len]
        batch = torch.rand((bs, input_size))
        # forward pass
        output = body(batch)
        print(f"Output of size: {output.size()}")
        self.assertEqual(output.size(0), bs)
        self.assertEqual(output.size(-1), hidden_size)


if __name__ == '__main__':
    unittest.main()
