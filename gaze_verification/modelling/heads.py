import torch
from abc import ABC, abstractmethod
from torch import nn
from typing import Tuple, Union, Optional, Any, List, Dict
from gaze_verification.logging_handler import get_logger


class HeadAbstract(nn.Module, ABC):
    """
    Abstract class for all heads.
    """

    def __init__(self,
                 input_size: Optional[int],
                 is_predict: bool = False):
        super().__init__()
        self.logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.input_size = input_size
        self.is_predict = is_predict

    @abstractmethod
    def forward(
            self, body_output: torch.Tensor, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    @abstractmethod
    def score(
            self, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predict(
            self,
            label_logits: torch.Tensor,
            *args, **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Retrieves prediction labels from logits.
        :param label_logits: logits vector/tensor from model outputs;
        :type label_logits: torch.Tensor;
        :return: predictions;
        :rtype: torch.Tensor or any other type;
        """
        raise NotImplementedError


class LinearHead(HeadAbstract):
    """
    Simple linear single layer head for getting output embeddings data representation.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            p_dropout: float = 0.2,
            compute_loss: bool = False,
            is_predict: bool = False
    ):
        super().__init__(input_size, is_predict)
        self.output_size = output_size
        self.compute_loss = compute_loss

        self.dropout = nn.Dropout(p_dropout)
        self.linear = nn.Linear(self.input_size, self.output_size)

        self.loss = None
        if self.compute_loss:
            # a squared term if the absolute element-wise error (x - y)
            # falls below beta and an L1 term otherwise.
            loss_fn = torch.nn.SmoothL1Loss
            self.loss = loss_fn(reduction="mean")

    def forward(self, body_output: torch.Tensor, **kwargs) -> torch.Tensor:
        embeddings = self.linear(self.dropout(body_output))
        return embeddings

    def score(
            self,
            embeddings: torch.Tensor,
            gt_embeddings: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        """
        Calculates Smooth L1 loss:
        https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html.
        :param embeddings: data embeddings (batch_size, embedding_size == output_size);
        :param gt_embeddings: `ground truth` or reference embeddings (batch_size, embedding_size == output_size);
        :return: loss value;
        """
        loss = self.loss(embeddings, gt_embeddings)
        return loss

    def predict(
            self,
            label_logits: torch.Tensor,
            *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate predictions - i.e. most probable class based on probabilities.
        Here probabilities are softmaxed logits.
        :param label_logits: logits vector, of size: [bs, *] - ?;
        :type label_logits: torch.Tensor;
        :return:
        :rtype:
        """
        raise NotImplementedError


class LinearClassificationHead(HeadAbstract):
    """
    Simple linear single layer head for prediction fixed number of classes.
    """

    def __init__(
            self,
            input_size: int,
            num_classes: int,
            p_dropout: float = 0.2,
            confidence_level: float = 0.5,
            class_weights: Optional[torch.Tensor] = None,
            label_smoothing: Optional[float] = None,
            compute_loss: bool = False,
            is_predict: bool = False
    ):
        super().__init__(input_size, is_predict)
        self.num_classes = num_classes
        self.binary = self.num_classes == 2
        self.confidence_level = confidence_level
        self.compute_loss = compute_loss

        self.dropout = nn.Dropout(p_dropout)
        self.label_classifier = nn.Linear(self.input_size, self.num_classes)

        # Два лосса используются потому, что поведение может быть разным,
        # когда маска есть и когда ее нет.

        if self.binary:
            # for binary classification better to use BCE
            loss_fn = nn.BCEWithLogitsLoss
            self.sigmoid = nn.Sigmoid()
        else:
            # for multiclass
            loss_fn = nn.CrossEntropyLoss

        kwargs = {}
        kwargs["label_smoothing"] = label_smoothing or 0.0

        self.loss_mean = loss_fn(reduction="mean", weight=class_weights, **kwargs)

    def forward(self, body_output: torch.Tensor, **kwargs) -> torch.Tensor:
        label_logits = self.label_classifier(self.dropout(body_output))
        return label_logits

    def score(
            self,
            label_logits: torch.Tensor,
            labels: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        """
        Считает лосс.
        :param label_logits: логиты (batch_size, seq_len, n_labels)
        :param labels: правильные метки (batch_size, seq_len, n_labels) or (batch_size, seq_len)
        :param label_mask: маска (batch_size, seq_len) or None
        :return: лосс
        """

        label_logits = label_logits.view(-1, self.num_labels)
        labels = labels.view(-1)  # (batch_size * seq_len)
        loss = self.loss_mean(label_logits, labels)
        return loss

    def predict(
            self,
            label_logits: torch.Tensor,
            *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate predictions - i.e. most probable class based on probabilities.
        Here probabilities are softmaxed logits.
        :param label_logits: logits vector, of size: [bs, *] - ?;
        :type label_logits: torch.Tensor;
        :return:
        :rtype:
        """
        probas = torch.softmax(label_logits, dim=-1)
        # tuple of (max scores, max score's indexes == classes)
        scores, preds = label_logits.max(dim=-1)
        return scores, probas, preds


class PrototypicalHead(HeadAbstract):
    """
    Linear single layer head which operates similar to a nearest neighbor classification.
    Metric-based meta-learning head classify a new example `x` based on some distance
    `dist(x, support)` between x and all elements in the support set.
    The nearest support set barycentre will be a class for current sample `x`.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            n_support: int,
            p_dropout: float = 0.2,
            confidence_level: float = 0.5,
            do_compute_loss: Optional[bool] = False,
            is_predict: bool = False
    ):
        super().__init__(input_size, is_predict)
        self.output_size = output_size
        self.n_support = n_support
        self.confidence_level = confidence_level  # todo: implement query classification with threshold
        self.do_compute_loss = do_compute_loss

        self.dropout = nn.Dropout(p_dropout)
        self.linear = nn.Linear(self.input_size, self.output_size)
        # i.e. prototypes for static classes,
        # for the inference time when we know which users are logged in sysem (for example)
        self.cold_prototypes = None

    def forward(self, body_output: torch.Tensor, **kwargs) -> torch.Tensor:
        embeddings = self.linear(self.dropout(body_output))
        return embeddings

    @staticmethod
    def euclidean_dist(x, y) -> torch.Tensor:
        """
        Compute euclidean distance between two tensors (x and y).
        Equals to: torch.cdist(x, y, p=2)
        :param x: shape N x D
        :param y: shape M x D
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.sqrt(torch.pow(x - y, 2).sum(2))

    def init_prototypes(self, embeddings: List[torch.Tensor], labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Computes prototypes (cold / static) for embeddings of supported set (or for inference time).
        :param embeddings: embeddings of supported set [n_support, embedding_size == output_size];
        :type embeddings: torch.Tensor;
        :param labels: list of corresponding labels;
        :type labels: torch.Tensor;
        :return: prototypes for each class met in labels;
        :rtype: Dict[int, torch.Tensor].
        """

        def index_select(c):
            return labels.eq(c).nonzero().squeeze(1)

        classes = torch.unique(labels)
        n_classes = len(classes)
        # get indexes of support samples from batch
        class_idxs = list(map(index_select, classes))
        class_embeddings = [embeddings[idx_list] for idx_list in class_idxs]
        prototypes = self.calculate_prototypes(class_embeddings)

        self.cold_prototypes = prototypes
        self.logger.info(f"Initialized prototypes for {n_classes} classes.")

    def calculate_prototypes(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Gets embeddings of supported set and computes their the barycentres by averaging.
        :param embeddings: embeddings of supported set [n_support, embedding_size == output_size];
        :type embeddings: torch.Tensor;
        :return: their the barycentres i.e. prototypes;
        :rtype: torch.Tensor.
        """
        # count prototypes for each support class samples
        prototypes = torch.stack([embeddings_.mean(0) for embeddings_ in embeddings])
        return prototypes

    def predict(self, query_embeddings: torch.Tensor,
                support_embeddings: Optional[torch.Tensor] = None,
                support_labels: Optional[torch.Tensor] = None,
                return_dists: Optional[bool] = False,
                return_dict: Optional[bool] = False) -> Dict[str, Any]:
        """
        Calculate predictions - i.e. most probable class based on probabilities.
        :param support_labels:
        :type support_labels:
        :param query_embeddings:
        :type query_embeddings:
        :param support_embeddings:
        :type support_embeddings:
        :param return_dict: to return distances from all queries to all prototypes;
        :type return_dists: torch.Tensor, [len(query_embeddings), len(prototypes)];
        :return: predicted classes and their probabilities, optionally distances.
        :rtype: Dict[str, Any].
        """
        def index_select(c):
            return support_labels.eq(c).nonzero().squeeze(1)

        # If support_embeddings are provided, then use them for prototypes computation
        if (support_embeddings is not None) and (support_labels is not None):
            # get indexes of support samples by classes
            classes = torch.unique(support_labels)
            class_idxs = list(map(index_select, classes))
            support_class_embeddings = [support_embeddings[idx_list] for idx_list in class_idxs]
            prototypes = self.calculate_prototypes(support_class_embeddings)
        elif self.cold_prototypes is not None:
            prototypes = self.cold_prototypes
        else:
            raise AttributeError(
                "Predicting class for query samples requires prototype initialization."
                "For this purpose, you should fulfill one of the following requirements: "
                "\n - compute and initialize `cold_prototypes`;"
                "\n - or provide `support_embeddings` AND `support_labels` as support set "
                "for 'hot' prototypes calculation."
            )

        # count distances -> [n_classes, bs]
        # [bs, embedding_size] * [n_classes, embedding_size].T = [bs, n_classes]
        dists = self.euclidean_dist(query_embeddings, prototypes)
        p_y = nn.functional.softmax(-dists, dim=1)
        # Get predicted labels
        # tuple of two output tensors (max, max_indices)
        probabilities, predictions = p_y.max(1)
        if return_dict:
            output = {
                "probabilities": probabilities,
                "predictions": predictions
            }
            if return_dists:
                output['distances'] = dists
        else:
            output = (dists, probabilities, predictions)
        return output

    def score(self, embeddings: torch.Tensor,
              labels: torch.Tensor) -> Tuple[Any, Any]:
        """
        Compute the barycentres by averaging the features of `self.n_support`
        samples for each class in target, computes then the distances from each
        samples' features to each one of the barycentres, computes the
        log_probability for each n_query samples for each one of the current
        classes, of appartaining to a class c, loss and accuracy are then computed and returned.
        """

        def supp_idxs(c):
            return labels.eq(c).nonzero()[:self.n_support].squeeze(1)

        classes = torch.unique(labels)
        n_classes = len(classes)

        # assuming n_query, n_labels constants for all classes
        n_query = labels.eq(classes[0].item()).sum().item() - self.n_support
        # get indexes of support samples from batch
        support_idxs = list(map(supp_idxs, classes))
        support_embeddings = [embeddings[idx_list] for idx_list in support_idxs]
        prototypes = self.calculate_prototypes(support_embeddings)

        # get indexes of query samples from batch
        query_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[self.n_support:], classes))).view(-1)

        # used: embeddings.to(device)[query_idxs]
        query_samples = embeddings[query_idxs]  # get embeddings of query samples
        dists = self.euclidean_dist(query_samples, prototypes)  # count distances
        log_p_y = nn.functional.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
        target_inds = torch.arange(0, n_classes)  # used: .to(device)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()
        # compute loss value
        if self.do_compute_loss:
            loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        # Get predicted labels
        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        output = {
            'loss': loss,
            'predictions': y_hat,
            'true_tags': target_inds,
            'accuracy': acc_val
        }

        return output
