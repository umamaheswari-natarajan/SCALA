# SCALA: Scalable Context-Aware Fault Resolution in Autonomous Networks

This repository contains the implementation and experimental code for
SCALA, a Retrieval-Augmented Generation (RAG)-based framework for
efficient and context-aware fault resolution in autonomous networks.

SCALA combines resolution-aware representation learning with
knowledge-space organization to improve the relevance of retrieved
fault-resolution knowledge while reducing the retrieval search space.

## Repository Structure

``` text
SCALA/
├── public-dataset/
│   └── Code and results for experiments using the public fault datasets.
│
├── cold-start/
│   └── Code and results for experiments under the cold-start setting.
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and install the required Python packages:

``` bash
pip install -r requirements.txt
```

## Experiments

### Public Dataset Experiments

The `public-dataset/` directory contains the implementation and
experimental results for the main evaluation of SCALA, including
representation learning, knowledge-space organization, retrieval,
generation, baseline evaluation, and ablation studies.

### Cold-Start Experiments

The `cold-start/` directory contains the implementation and experimental
results for evaluating SCALA under limited historical fault-resolution
knowledge.

## Data Availability

The public datasets used for the main experiments are included in the
`public-dataset/` directory and for the cold-start evaluation, the
experimental results are provided.

## Results

The repository includes the scripts and experimental outputs used to
obtain the results reported in the paper.

## Citation

Citation information will be added upon publication.
