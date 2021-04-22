from dataclasses import dataclass
from itertools import groupby
import json
import os
from os.path import join, dirname, abspath
import sys
from typing import List, Dict, Tuple

import torch

# this monstrosity produces the module directory in an environment where this is unpacked
sys.path.insert(0, abspath(join(dirname(abspath(__file__)), "..", "..")))


@dataclass(frozen=False, repr=True, eq=True)
class SciFactAnnotation:
    claim_id: int
    doc_id: int
    sentences: List[List[str]]
    encoded_sentences: List[torch.IntTensor]
    rationale_sentences: List[int]
    i: torch.IntTensor
    c: torch.IntTensor
    o: torch.IntTensor
    rationale_class: str
    rationale_id: int


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    return data


def dump_jsonl(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w+", encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")


def condense_labels(labels, neg_class="0"):
    labels = [str(label) for label in labels]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(labels)]
    tok_spans = []
    i = 0
    for label, length in groups:
        if label != neg_class:
            tok_spans.append((i, i + length, label))
        i += length
    return tok_spans


def extract_ico_prompt_from_predictions(claims):
    """Extracts the Intervention, Comparator, and Outcome tokens from the claims
    predictions.

    The claims are expected to be in the format outputted by the PICO Extraction
    script `scripts/scifact_pico_extraction_bert.py`. Specifically, the claims
    should be a list of dictionaries with keys `id`, `claim`, `evidence`,
    `cited_doc_ids`, `tokens`, and `pred_tags`. This uses the per token
    prediction tags to extract the Intervention, Comparator, and Outcome tokens
    from the claim. Ideally, the claims should be formatted with two
    interventions spans (one acts as the intervention and the other acts as the
    comparator) and one outcomes span. In cases where this does not hold true
    for the predictions, the following heuristics will be used:
    - No Intervention: drop the claim from the dataset.
    - One Intervention: force the use of `placebo` as a comparator.
    - Multiple Interventions: choose the two longest intervention spans.
    - No Outcome: drop the claim from the dataset.
    - Multiple Outcomes: choose the longest outcome span.

    Returns the claims with new keys `i_tokens`, `c_tokens`, `o_tokens`.
    """
    for c in claims:
        # Get pico spans
        tokens = c["tokens"]
        token_spans = condense_labels(c["pred_tags"])

        # Extract the tokens for each interventions and outcomes span along with
        # the span's total character length with stripped wordpiece ##'s
        interventions = []
        outcomes = []
        for (t_start, t_stop, label) in token_spans:
            if label in ("p", "0"):
                continue

            # Extract spans
            span_tokens = tokens[t_start:t_stop]
            length = len("".join(span_tokens).replace("##", ""))
            if label == "i":
                interventions.append((length, span_tokens))
            elif label == "o":
                outcomes.append((length, span_tokens))

        # Choose the final span tokens to use for the intervention, comparator,
        # and outcome
        i_tokens = []
        c_tokens = []
        if len(interventions) == 1:
            i_tokens = interventions[0][1]
            c_tokens = ["placebo"]
        elif len(interventions) == 2:
            i_tokens = interventions[0][1]
            c_tokens = interventions[1][1]
        elif len(interventions) >= 3:
            interventions.sort(key=lambda x: x[0], reverse=True)
            i_tokens = interventions[0][1]
            c_tokens = interventions[1][1]

        o_tokens = []
        if len(outcomes) == 1:
            o_tokens = outcomes[0][1]
        elif len(outcomes) > 1:
            outcomes.sort(key=lambda x: x[0], reverse=True)
            o_tokens = outcomes[0][1]

        # write the tokens to the claims
        c["i_tokens"] = i_tokens
        c["c_tokens"] = c_tokens
        c["o_tokens"] = o_tokens

    return claims


def drop_claims_with_malformed_prompts(claims):
    def is_prompt_complete(claim):
        return (
            len(claim["i_tokens"]) > 0
            and len(claim["c_tokens"]) > 0
            and len(claim["o_tokens"]) > 0
        )

    claims = [claim for claim in claims if is_prompt_complete(claim)]
    return claims


def preprocess_claim_predictions_for_pipeline(claims):
    claims = extract_ico_prompt_from_predictions(claims)
    claims = drop_claims_with_malformed_prompts(claims)
    return claims


def create_scifact_annotations(
    claims, corpus, tokenizer, class_to_id: Dict[str, int], neutral_class: str
) -> List[SciFactAnnotation]:
    """Create a SciFactAnnotation for each claim - evidence/cited document pair."""

    def get_abstract_and_encoding(
        doc_id,
    ) -> Tuple[List[List[str]], List[torch.IntTensor]]:
        doc = [d for d in corpus if d["doc_id"] == int(doc_id)]
        assert len(doc) == 1
        abstract = doc[0]["abstract"]
        encoding = [
            torch.IntTensor(tokenizer.encode(sentence, add_special_tokens=False))
            for sentence in abstract
        ]

        return abstract, encoding

    annotations = []
    for c in claims:
        # Convert Interventions, Comparator, and Outcomes tokens to encodings
        intervention = torch.IntTensor(tokenizer.convert_tokens_to_ids(c["i_tokens"]))
        comparator = torch.IntTensor(tokenizer.convert_tokens_to_ids(c["c_tokens"]))
        outcome = torch.IntTensor(tokenizer.convert_tokens_to_ids(c["o_tokens"]))

        evidence = c["evidence"]

        # Handle claims with no evidence (label is NOT_ENOUGH_INFO)
        if not evidence:
            cited_doc_id = c["cited_doc_ids"][0]
            abstract, encoded_abstract = get_abstract_and_encoding(cited_doc_id)
            rationale_id = class_to_id[neutral_class]

            s_ann = SciFactAnnotation(
                claim_id=int(c["id"]),
                doc_id=int(cited_doc_id),
                sentences=abstract,
                encoded_sentences=encoded_abstract,
                rationale_sentences=[],
                i=intervention,
                c=comparator,
                o=outcome,
                rationale_class=neutral_class,
                rationale_id=rationale_id,
            )
            annotations.append(s_ann)

        # Create a SciFact Annotation for each evidence document
        else:
            for doc_id, doc_rationales in evidence.items():
                abstract, encoded_abstract = get_abstract_and_encoding(doc_id)

                rationale_class = doc_rationales[0]["label"]
                rationale_id = class_to_id[rationale_class]

                # extract all rationale sentence indices from the document
                rationale_sentences = []
                for rationale in doc_rationales:
                    rationale_sentences.extend(rationale["sentences"])

                s_ann = SciFactAnnotation(
                    claim_id=int(c["id"]),
                    doc_id=int(doc_id),
                    sentences=abstract,
                    encoded_sentences=encoded_abstract,
                    rationale_sentences=rationale_sentences,
                    i=intervention,
                    c=comparator,
                    o=outcome,
                    rationale_class=rationale_class,
                    rationale_id=rationale_id,
                )
                annotations.append(s_ann)
    return annotations
