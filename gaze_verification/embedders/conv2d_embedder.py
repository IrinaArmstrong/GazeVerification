import torch

from typeguard import typechecked
from typing import Dict, Any, Type, List, Tuple

from gaze_verification.modelling.layers.cbam import CBAM
from gaze_verification.embedders import EmbedderAbstract, EmbedderConfigAbstract


class Conv2dEmbedderConfig(EmbedderConfigAbstract):
    """
    Config for Conv2DEmbedder class model.

    It allows to
        - set input size for the first layer;
        - set output embeddings size of the last layer;
        - select number of convolutional blocks and fully connected layers;
        - set custom values for model hyperparameters;
    """
    model_type = "Conv2DEmbedder"

    def __init__(
            self,
            input_size: Tuple[int, ...],
            embedding_size: int,
            num_conv_blocks: int,
            num_fc_blocks: int,
            conv_params: List[Dict[str, Any]],
            fc_params: List[Dict[str, Any]],
            **kwargs
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.num_conv_blocks = num_conv_blocks
        self.num_fc_blocks = num_fc_blocks
        self.conv_params = conv_params
        self.fc_params = fc_params

    @staticmethod
    def get_target_class() -> Type[EmbedderAbstract]:
        return Conv2DEmbedder


@typechecked
class Conv2DEmbedder(EmbedderAbstract):
    """
    Embedder with a convolutional backbone (including CBAM module, optionally)
    for making embeddings out of 2-dimensional input samples.
    For example, valid input data is:
        - a sequence of (x, y) coordinates,
        - or a sequence of (vel_x, vel_y) velocities of data points,
        - or a sequence of tuples containing (pitch, yaw) angles.
    """
    name = "Conv_2dim_input_with_CBAM"
    def __init__(self, config: Conv2dEmbedderConfig, *args, **kwargs):
        super().__init__(config)
        self.cnn = self.create_conv_blocks()
        self.fc = self.create_fc_blocks()
        self.model = torch.nn.ModuleDict({
            "cnn": self.cnn,
            "fc": self.fc
        })

    def get_embeddings_size(self) -> int:
        raise NotImplementedError

    def _get_conv_output_shape(self):
        """
        Calculates inner CNN module output tensor size.
        Requires single forward pass for output feature map size estimation.
        """
        output = self.cnn(torch.rand(*(self.config.input_size)))
        self._logger.info(f"Calculated CNN output size: {output.view(output.size()[0], -1).data.shape}")
        return output.view(output.size()[0], -1).data.shape[-1]

    def create_conv_blocks(self) -> torch.nn.Module:
        """
        Dynamically generated architecture of inner convolutional module (i.e. `cnn`)
        based on passed through config parameters.
        :param n_blocks: a general number of sequential layers stacks:
                    [
                        nn.Conv2d,
                        CBAM (Channel or Spatial or both),
                        pooling (Average/Max/...),
                        BatchNorm1d/LayerNorm (*optionally*),
                        activation (*optionally*)
                    ];
        :type n_blocks: int;
        :return: a created model;
        :rtype: torch.nn.Module.
        """
        layers = []
        for i in range(self.config.num_conv_blocks):
            layers.append(torch.nn.Conv1d(**self.config.conv_params[i].get("conv")))
            layers.append(CBAM(**self.config.conv_params[i].get("cbam")))

            # Pooling
            pooling_type = self.config.conv_params[i].get("pool_type")
            if pooling_type and hasattr(torch.nn, pooling_type):
                pooling = getattr(torch.nn, pooling_type)(**self.config.conv_params[i].get("pool"))
                layers.append(pooling)

            # Normalization
            if self.config.conv_params[i].get("norm_type", None):
                norm_type = self.config.conv_params[i].get("norm_type")
                if norm_type and hasattr(torch.nn, norm_type):
                    norm = getattr(torch.nn, norm_type)(**self.config.conv_params[i].get("norm"))
                    layers.append(norm)

            # Activations
            if self.config.conv_params[i].get("activation_type", None):
                activation_type = self.config.conv_params[i].get("activation_type")
                if activation_type and hasattr(torch.nn, activation_type):
                    activation = getattr(torch.nn, activation_type)(**self.config.conv_params[i].get("activation"))
                    layers.append(activation)

        seq = torch.nn.Sequential(*layers)
        return seq

    def create_fc_blocks(self) -> torch.nn.Module:
        """
        Dynamically generated architecture of top fully-connected block (i.e. `fc`)
        based on passed through config parameters and
        calculated convolutional module tensors output sizes.
        :return: a created model;
        :rtype: torch.nn.Module.
        """
        layers = []
        for i in range(self.config.num_fc_blocks):
            if i == 0:
                in_dim = self._get_conv_output_shape()
                layers.append(torch.nn.Linear(in_features=in_dim,
                                              **self.config.fc_params[i].get("linear")))
            else:
                layers.append(torch.nn.Linear(**self.config.fc_params[i].get("linear")))

            layers.append(torch.nn.Dropout(**self.config.fc_params[i].get("dropout")))
            # Activation
            if self.config.fc_params[i].get("activation_type", None):
                activation_type = self.config.fc_params[i].get("activation_type")
                if activation_type and hasattr(torch.nn, activation_type):
                    activation = getattr(torch.nn, activation_type)(**self.config.fc_params[i].get("activation"))
                    layers.append(activation)
        seq = torch.nn.Sequential(*layers)
        return seq

    def forward(self, x, **kwargs):
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, 1)
        elif len(x.size()) == 4:
            x = torch.squeeze(x, 1)
        output = self.model['cnn'](x.float())
        output = output.view(output.size()[0], -1)
        output = self.model['fc'](output)
        return output
