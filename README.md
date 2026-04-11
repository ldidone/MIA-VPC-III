# MIA-VPC-III

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MobileViT and TensorFlow Flowers experiments for MIA VPC III.

Structured with [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) (v2.3.0).

## Repository

This project’s Git remote is **`https://github.com/ldidone/MIA-VPC-III.git`** (`origin`). Clone with:

```bash
git clone https://github.com/ldidone/MIA-VPC-III.git
cd MIA-VPC-III
```

Install dependencies (including the local package in editable mode) with `pip install -r requirements.txt` from the repository root. When you open the notebooks under `notebooks/`, keep the Jupyter kernel’s working directory at the **project root** so paths like `models/...` resolve correctly.

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Project documentation assets (optional; mkdocs not configured by default)
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mia_vpc_iii and tool configuration (e.g. ruff)
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── mia_vpc_iii        <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mia_vpc_iii a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

