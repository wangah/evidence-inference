# much code in this module is borrowed from the Eraser Benchmark (DeYoung et al., 2019)
from os.path import join, dirname, abspath
import sys

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), "..", "..")))

from typing import List, Optional

import json

import torch
import torch.nn as nn

from transformers import BertForSequenceClassification, BertTokenizer, PretrainedConfig

# from transformers import RobertaForSequenceClassification, RobertaTokenizer, PretrainedConfig
# from transformers import (
#     AutoModelForSequenceClassification,
#     AutoTokenizer,
#     PretrainedConfig,
# )

from evidence_inference.models.utils import PaddedSequence


def initialize_models(params: dict, unk_token="<unk>"):
    max_length = params["max_length"]
    # tokenizer = AutoTokenizer.from_pretrained(params["bert_vocab"])
    tokenizer = BertTokenizer.from_pretrained(params["bert_vocab"])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    evidence_class_to_id = dict(
        (y, x) for (x, y) in enumerate(params["evidence_classifier"]["classes"])
    )
    if bool(params.get("random_init", 0)):
        with open(params["bert_config"], "r") as inf:
            cfg = inf.read()
            id_config = PretrainedConfig.from_dict(json.loads(cfg), num_labels=2)
            cls_config = PretrainedConfig.from_dict(
                json.loads(cfg), num_labels=len(evidence_class_to_id)
            )
        use_half_precision = bool(
            params["evidence_identifier"].get("use_half_precision", 0)
        )
        evidence_identifier = BertClassifier(
            bert_dir=None,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            num_labels=2,
            max_length=max_length,
            use_half_precision=use_half_precision,
            config=id_config,
        )
        use_half_precision = bool(
            params["evidence_classifier"].get("use_half_precision", 0)
        )
        evidence_classifier = BertClassifier(
            bert_dir=None,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            num_labels=len(evidence_class_to_id),
            max_length=max_length,
            use_half_precision=use_half_precision,
            config=cls_config,
        )
    else:
        bert_dir = params["bert_dir"]
        use_half_precision = bool(
            params["evidence_identifier"].get("use_half_precision", 0)
        )
        evidence_identifier = BertClassifier(
            bert_dir=bert_dir,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            num_labels=2,
            max_length=max_length,
            use_half_precision=use_half_precision,
        )
        use_half_precision = bool(
            params["evidence_classifier"].get("use_half_precision", 0)
        )
        evidence_classifier = BertClassifier(
            bert_dir=bert_dir,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            num_labels=len(evidence_class_to_id),
            max_length=max_length,
            use_half_precision=use_half_precision,
        )
    word_interner = tokenizer.get_vocab()
    de_interner = dict((x, y) for (y, x) in word_interner.items())
    return (
        evidence_identifier,
        evidence_classifier,
        word_interner,
        de_interner,
        evidence_class_to_id,
        tokenizer,
    )


class BertClassifier(nn.Module):
    """Thin wrapper around BertForSequenceClassification"""

    def __init__(
        self,
        bert_dir: Optional[str],
        pad_token_id: int,
        cls_token_id: int,
        sep_token_id: int,
        num_labels: int,
        max_length: int = 512,
        use_half_precision=False,
        config=Optional[PretrainedConfig],
    ):
        super(BertClassifier, self).__init__()
        if bert_dir is None:
            assert config is not None
            assert config.num_labels == num_labels
            bert = BertForSequenceClassification.from_config(config)
        else:
            bert = BertForSequenceClassification.from_pretrained(
                bert_dir, num_labels=num_labels
            )
        if use_half_precision:
            import apex

            bert = bert.half()
        self.bert = bert
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self, query: List[torch.tensor], document_batch: List[torch.tensor]):
        assert len(query) == len(document_batch)
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these params on the input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        cls_token = torch.tensor(
            [self.cls_token_id]
        )  # .to(device=document_batch[0].device)
        sep_token = torch.tensor(
            [self.sep_token_id]
        )  # .to(device=document_batch[0].device)

        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[: (self.max_length - len(q) - 2)]
            input_tensors.append(
                torch.cat([cls_token, q, sep_token, d.to(dtype=q.dtype)])
            )
            position_ids.append(torch.arange(0, input_tensors[-1].size().numel()))

        bert_input = PaddedSequence.autopad(
            input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
            device=target_device,
        )
        positions = PaddedSequence.autopad(
            position_ids, batch_first=True, padding_value=0, device=target_device
        )

        # get output as a transformers.modeling_outputs.SequenceClassifierOutput
        out = self.bert(
            bert_input.data,
            attention_mask=bert_input.mask(
                on=1.0, off=0.0, dtype=torch.float, device=target_device
            ),
            position_ids=positions.data,
        )

        assert not torch.any(torch.isnan(out.logits))
        return out.logits
