# SciFact Pre-training

This work further investigates whether pre-training on related scientific 'fact
verification' tasks might improve performance for the Evidence Inference
BERT-to-BERT pipeline model. Specifically, we use the [SciFact](https://github.com/allenai/scifact)
claim verification corpus for such pre-training.

See `README.evidence_inference.md` for the original Evidence Inference README.

## Experiment Design

The following steps were performed to evaluate the effectiveness of SciFact pre-training for the Evidence Inference BERT-to-BERT pipeline model:

1. Extract and preprocess PICO spans from SciFact claims.

2. Adapt SciFact data into the format expected by the BERT pipeline and define corresponding samplers.

3. Train the pipeline on SciFact and save the model weights. We optionally converted RoBERTa to SciBERT due to memory constraints.

4. Train the pipeline on Evidence Inference using the pre-trained weights.

Note that this experiment does not change the model architecture and instead forces the SciFact dataset into the same format as the Evidence Inference data via PICO extraction of SciFact claims. The following options may also be considered:

1. Add module to model that learns some representation for prompts and claims first before feeding it to the model.

2. Apply linearization to both the claims and the prompts.
