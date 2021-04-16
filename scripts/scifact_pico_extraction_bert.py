"""
Train a BERT-based PICO Extraction model on the EBM-NLP corpus and create PIO
predictions for SciFact claims.

The following repositories are required and should be downloaded to `../data`:
- [SciFact](https://github.com/allenai/scifact/)
    - Contains the claims to make predictions for.
- [EBM-NLP 2.0](https://github.com/bepnye/EBM-NLP)
    - contains the training data for the PICO Extraction task.
- [Evidence Extraction](https://github.com/bepnye/evidence_extraction)
    - contains the train/dev/test splits for EBM-NLP.
Run the script `/scripts/scifact_pretraining/download_data.sh` to download
the required files.

This code is largely based off of the HuggingFace token classification example:
https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import classification_report
import torch
import transformers
from transformers import (
    AutoConfig,
    BertForTokenClassification,
    BertTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
)
from evidence_inference.models.utils import load_jsonl, dump_jsonl


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="allenai/scibert_scivocab_uncased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="pico-extraction",
        metadata={"help": "The name of the task (ner, pos...)."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=True,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


class PIODataset(torch.utils.data.Dataset):
    """Torch Dataset to process EBM-NLP encodings and labels.

    Copied from the HuggingFace tutorial:
    https://huggingface.co/transformers/custom_datasets.html#ft-trainer
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class SciFactDataset(torch.utils.data.Dataset):
    """Torch Dataset to process SciFact claim encodings."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def get_ebm_nlp_data(split, params):
    """Preprocesses the EBM-NLP dataset into `token_docs` which is a list of lists of token strings
    and `token_tags` which is a list of lists of tag strings for the given dataset split.

    Parameters
    ----------
    split: {'train', 'dev', 'test'}
        The EBM-NLP dataset split to preprocess.

    params:
        Dictionary containing data and model parameters. Used here to get the
        data directories for EBM-NLP and its id splits, the tags to use to note
        participants, interventions, outcomes, and the negative class.

    Returns
    -------
    token_docs: list[list[str]]
        The tokens for the split's corresponding abstracts.

    token_tags: list[list[str]]
        The tags for the corresponding tokens in `token_docs`.
    """
    ebm_nlp_dir = params["ebm_nlp_dir"]
    id_splits_dir = params["id_splits_dir"]
    pio_to_tag = params["pio_to_tag"]
    neg_tag = params["neg_tag"]

    def read_lines(file_path):
        return open(file_path).read().strip("\n").split("\n")

    def get_annotations_file_path(pmid, pio_category):
        if split == "test":
            sub_dir = os.path.join("test", "gold")
        else:
            sub_dir = "train"
        file_name = f"{pmid}.AGGREGATED.ann"
        return os.path.join(
            ebm_nlp_dir,
            "annotations",
            "aggregated",
            "starting_spans",
            pio_category,
            sub_dir,
            file_name,
        )

    pmids_file_path = os.path.join(id_splits_dir, f"{split}.txt")
    pmids = read_lines(pmids_file_path)
    token_docs = []
    token_tags = []
    for pmid in pmids:
        # get tokens
        tokens_file_path = os.path.join(ebm_nlp_dir, "documents", f"{pmid}.tokens")
        tokens = read_lines(tokens_file_path)
        token_docs.append(tokens)

        # get the per token annotations for participants, interventions, and outcomes and replace
        # the '1' and '0' annotations with custom tags
        pio_tags = []
        for category, tag in pio_to_tag.items():
            annotations_file_path = get_annotations_file_path(pmid, category)
            annotations = read_lines(annotations_file_path)
            tags = [tag if a == "1" else neg_tag for a in annotations]
            pio_tags.append(tags)

        # condense the per token tags into a single tag per token
        final_tags = []
        per_token_tags = zip(*pio_tags)
        for tags in per_token_tags:
            final_tag = neg_tag
            for t in tags:
                if t != neg_tag:
                    final_tag = t
            final_tags.append(final_tag)
        token_tags.append(final_tags)

    return token_docs, token_tags


def tokenize_scifact_claims(tokenizer, claims, max_seq_length):
    """
    Parameters
    ----------
    tokenizer: transformers.PreTrainedTokenizerFast
        The HuggingFace tokenizer corresponding to the desired model.

    claims: list[dict]
        The SciFact 2dataset split extracted as a list of dictionaries.

    max_seq_length: int
        The maximum sequence length to pad/truncate encodings to.

    Returns
    -------
    encodings: transformers.tokenization_utils_base.BatchEncoding
        The tokenized inputs in the HuggingFace tokenizer output format.
    """
    texts = [c["claim"] for c in claims]
    encodings = tokenizer(
        texts,
        is_split_into_words=False,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )
    return encodings


def replace_with_dict(ar, dic):
    """From https://stackoverflow.com/questions/47171356"""
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))
    sidx = k.argsort()
    return v[sidx[np.searchsorted(k, ar, sorter=sidx)]]


def trim_tokens_predtags(tokens, tags):
    """Remove the '[CLS]' token and corresponding tag as well as the everything starting from
    the first '[SEP]' token and corresponding tags.
    """
    sep_idx = tokens.index("[SEP]")
    return tokens[1:sep_idx], tags[1:sep_idx]


def predict_scifact_tags(dataset, trainer, tokenizer, id_to_tag):
    """Creates interventions, participants, and outcomes predictions for the SciFact claims. The
    tokens and tags will be trimmed to remove extra padding added by the tokenizer.

    Parameters
    ----------
    dataset: SciFactDataset
        The SciFact torch dataset containing the encoded claims.

    trainer: transformers.Trainer
        The HuggingFace trainer that has been trained on EBM-NLP.

    tokenizer: transformers.PreTrainedTokenizerFast
        The HuggingFace tokenizer used in the trainer.

    id_to_tag: dict[int, str]
        Maps each label to its corresponding text tag.

    Returns
    -------
    tokens_predtags: list[tuple[list[str], list[str]]]
        The claims tokenized by the `tokenizer` and the corresponding tag predictions which are
        trimmed to remove the extra padding - the '[CLS]' token and everything including and after
        the '[SEP]' token.
    """
    tokens = [
        tokenizer.convert_ids_to_tokens(input_ids)
        for input_ids in dataset.encodings["input_ids"]
    ]

    p = trainer.predict(dataset)
    pred_ids = np.argmax(p.predictions, axis=2)
    pred_tags = replace_with_dict(pred_ids, id_to_tag).tolist()

    tokens_predtags = [
        trim_tokens_predtags(toks, tags) for toks, tags in zip(tokens, pred_tags)
    ]
    return tokens_predtags


def write_tokens_predtags_to_claims(claims, tokens_predtags, out_path):
    assert len(claims) == len(tokens_predtags)
    for claim, (tokens, pred_tags) in zip(claims, tokens_predtags):
        claim["tokens"] = tokens
        claim["pred_tags"] = pred_tags

    dump_jsonl(claims, out_path)


def main():
    hf_parser = HfArgumentParser(
        dataclass_types=(ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    hf_parser.add_argument(
        "--params",
        dest="params",
        required=True,
        help="JSoN file for HuggingFace ModelArguments, DataTrainingArguments, TrainingArguments, and other parameters.",
    )
    args = hf_parser.parse_args()
    with open(args.params, "r") as fp:
        params = json.load(fp)

    model_args, data_args, training_args = hf_parser.parse_dict(params)

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get EBM-NLP data
    train_texts, train_tags = get_ebm_nlp_data("train", params)
    dev_texts, dev_tags = get_ebm_nlp_data("dev", params)
    test_texts, test_tags = get_ebm_nlp_data("test", params)

    unique_tags = set(tag for doc in train_tags for tag in doc)
    tag_to_id = {tag: id for id, tag in enumerate(sorted(unique_tags))}
    id_to_tag = {id: tag for tag, id in tag_to_id.items()}

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=len(unique_tags),
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = BertForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    assert isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), "Tokenizer must be a PreTrainedTokenizerFast"

    # Preprocess data
    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_and_align_labels(
        tokenizer,
        texts,
        tags,
        tag_to_id=tag_to_id,
        max_seq_length=params["max_seq_length"],
        label_all_tokens=params["label_all_tokens"],
    ):
        """Tokenize the given text, convert the tags to labels, and align the labels with the processed
        input ids.

        Parameters
        ----------
        tokenizer: transformers.PreTrainedTokenizerFast
            The HuggingFace fast tokenizer that corresponds to the desired model architecture.

        texts: list[list[str]]
            The tokens for the dataset, which already have been split into lists of words.

        tags: list[list[str]]
            The corresponding NER tags for each token in `text`.

        tag_to_id: dict[str, int]
            Maps each tag to a corresponding non-negative integer id.

        max_seq_length: int
            The maximum sequence length to pad/truncate encodings to.

        label_all_tokens: bool, default=False
            If True, the label of all special tokens will be set to -100 (the ignored index in PyTorch)
            and the labels of all other tokens will be set to the label of the word they come from. If
            False, only the first token obtained from a given word will be set to its original label and
            all other subtokens from the same word will be assigned to the -100 label. This should be
            set to False for validation and test sets.

        Returns
        -------
        encodings: transformers.tokenization_utils_base.BatchEncoding
            The tokenized inputs in the HuggingFace tokenizer output format.

        labels: list[list[str]]
            The resulting aligned labels filled using the strategy defined by the `label_all_tokens`
            param.
        """
        encodings = tokenizer(
            texts,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
        )

        labels = []
        for i, label in enumerate(tags):
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are
                # automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(tag_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100,
                # depending on the label_all_tokens flag.
                else:
                    label_ids.append(
                        tag_to_id[label[word_idx]] if label_all_tokens else -100
                    )
                previous_word_idx = word_idx

            labels.append(label_ids)

        return encodings, labels

    # Only apply the tokenization rule to the training set. The splits used for
    # evaluation should NOT label all tokens.
    train_encodings, train_labels = tokenize_and_align_labels(
        tokenizer,
        train_texts,
        train_tags,
        label_all_tokens=params["label_all_tokens"],
    )
    dev_encodings, dev_labels = tokenize_and_align_labels(
        tokenizer, dev_texts, dev_tags, label_all_tokens=False
    )
    test_encodings, test_labels = tokenize_and_align_labels(
        tokenizer, test_texts, test_tags, label_all_tokens=False
    )
    train_encodings.pop("offset_mapping")
    dev_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")

    train_dataset = PIODataset(train_encodings, train_labels)
    dev_dataset = PIODataset(dev_encodings, dev_labels)
    test_dataset = PIODataset(test_encodings, test_labels)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Metrics
    def compute_metrics(p, id_to_tag=id_to_tag):
        """Compute the PICO extraction metrics while ignoring -100 labels.

        Code is largely sourced from the following tutorial:
        https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb

        Parameters
        ----------
        p: tuple[]
            The result of Trainer.evaluate, which is a namedtuple containing predictions and labels.

        id_to_tag: dict
            Dictionary mapping each label to its corresponding text tag.

        Return
        ------
        metrics: dict
            Dictionary containing per-category and overall precision, recall, f1, and accuracy scores.
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Produce classification report with ignored indexes removed (special tokens)
        true_predictions = [
            id_to_tag[token_prediction]
            for sentence_predictions, sentence_labels in zip(predictions, labels)
            for token_prediction, token_label in zip(
                sentence_predictions, sentence_labels
            )
            if token_label != -100
        ]
        true_tags = [
            id_to_tag[token_label]
            for sentence_predictions, sentence_labels in zip(predictions, labels)
            for token_prediction, token_label in zip(
                sentence_predictions, sentence_labels
            )
            if token_label != -100
        ]
        report = classification_report(true_tags, true_predictions, output_dict=True)
        report.pop("weighted avg")

        # Calculate macro averaged scores for the PIO tags
        report["pio_macro_avg"] = {
            metric: np.mean([report[tag][metric] for tag in ("p", "i", "o")])
            for metric in ("f1-score", "precision", "recall")
        }

        # Flatten the report
        metrics = {}
        for key, value in report.items():
            if isinstance(value, dict):
                for metric, score in value.items():
                    if metric != "support":
                        label_metric_name = f"{key}_{metric}".replace(" ", "_")
                        metrics[label_metric_name] = score
            else:
                metrics[key] = value
        return metrics

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        trainer.train(resume_from_checkpoint=checkpoint)
        train_metrics = trainer.evaluate(train_dataset, metric_key_prefix="train")

        os.makedirs(params["model_save_dir"], exist_ok=True)
        trainer.save_model(params["model_save_dir"])

        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        dev_metrics = trainer.evaluate(dev_dataset, metric_key_prefix="dev")
        trainer.log_metrics("dev", dev_metrics)
        trainer.save_metrics("dev", dev_metrics)

        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="dev")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    if params["do_scifact_predict"]:
        logger.info("Creating PICO predictions on SciFact claims...")
        for name, path in (
            ("claims_train", params["claims_train"]),
            ("claims_dev", params["claims_dev"]),
            ("claims_test", params["claims_test"]),
        ):
            claims = load_jsonl(path)
            claim_encodings = tokenize_scifact_claims(
                tokenizer, claims, params["max_seq_length"]
            )
            claim_dataset = SciFactDataset(claim_encodings)
            tokens_predtags = predict_scifact_tags(
                claim_dataset, trainer, tokenizer, id_to_tag
            )
            outpath = os.path.join(
                params["claims_pred_save_dir"], f"{name}_pio_bert.jsonl"
            )

            logging.info(f"Writing scifact predictions on {name} to {outpath}...")
            write_tokens_predtags_to_claims(claims, tokens_predtags, outpath)


if __name__ == "__main__":
    main()
