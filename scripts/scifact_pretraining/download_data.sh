#!/bin/bash

# Move to the directory that this file is located in (scripts/scifact_pretraining)
cd "$(dirname "$0")"

data_dir="../../data"
if [ -d $data_dir ]
then
    echo "Data directory already exists. Skip download."
    exit 0
fi
mkdir $data_dir

# Download SciFact data to data/scifact
scifact_data="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
wget $scifact_data
tar -zxf data.tar.gz -C $data_dir && mv "${data_dir}/data" "${data_dir}/scifact"
rm data.tar.gz

# Download EBM-NLP corpus to data/ebm_nlp_2_00
ebm_nlp_data="https://raw.githubusercontent.com/bepnye/EBM-NLP/master/ebm_nlp_2_00.tar.gz"
wget $ebm_nlp_data
tar -zxf ebm_nlp_2_00.tar.gz -C $data_dir
rm ebm_nlp_2_00.tar.gz

# Download EBM-NLP train/dev/test splits to data/id_splits
train_ids_url="https://raw.githubusercontent.com/bepnye/evidence_extraction/master/data/id_splits/ebm_nlp/train.txt"
dev_ids_url="https://raw.githubusercontent.com/bepnye/evidence_extraction/master/data/id_splits/ebm_nlp/dev.txt"
test_ids_url="https://raw.githubusercontent.com/bepnye/evidence_extraction/master/data/id_splits/ebm_nlp/test.txt"
id_splits_dir="${data_dir}/id_splits"
mkdir $id_splits_dir
wget $train_ids_url -P $id_splits_dir
wget $dev_ids_url -P $id_splits_dir
wget $test_ids_url -P $id_splits_dir