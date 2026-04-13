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

## Flower classifier demo (Streamlit)

### Hugging Face Space

The same MobileViT flower classifier runs as a **Docker Space** on the Hugging Face Hub:

**[https://huggingface.co/spaces/ldidone/flower-classifier-mobilevit](https://huggingface.co/spaces/ldidone/flower-classifier-mobilevit)**

1. Open the Space and wait until the app shows **Running**.
2. **Upload an image** — choose a JPG/PNG/JPEG, then click **Classify** to see the predicted class and probability bar chart.
3. **Use a demo image** — pick a sample from the dropdown, then **Classify**.

The Space bundles its own model and demo images under [`hf_space/`](hf_space/) in this repo (see that folder’s `Dockerfile`, `app.py`, and `README.md` for deployment details).

### Run locally

From the repository root, with a virtual environment activated (Python **3.10–3.12** is recommended for PyTorch wheels; **3.13** may not have a matching `torch` build on all platforms):

```bash
pip install -r requirements.txt
```

Optional: populate `data/demo/` with sample images from the tf_flowers archive (used by the local app’s demo tab):

```bash
python -m mia_vpc_iii.save_demo_images
```

Ensure the fine-tuned checkpoint exists at `models/mobilevit-small-tf-flowers-final-colab/` (including `model.safetensors`, `config.json`, and `preprocessor_config.json`).

Start the Streamlit app:

```bash
streamlit run mia_vpc_iii/app.py
```

Streamlit prints a local URL (typically `http://localhost:8501`). Use the browser to upload images or try demo files from `data/demo/`.

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── demo           <- Sample flower images for the local Streamlit app (`save_demo_images`)
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Project documentation assets (optional; mkdocs not configured by default)
│
├── hf_space           <- Hugging Face Space (Docker): Streamlit app, bundled model, demo JPGs
│   ├── Dockerfile
│   ├── README.md      <- Space card metadata (YAML) and short description
│   ├── app.py         <- Streamlit UI for the Hub deployment
│   ├── requirements.txt
│   ├── .gitattributes <- Optional Git LFS patterns (Hub-style)
│   ├── .streamlit
│   │   └── config.toml <- Server options (e.g. XSRF disabled for Space file uploads)
│   ├── demo           <- Sample images shipped with the Space
│   └── model          <- `config.json`, `preprocessor_config.json`, `model.safetensors`
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
    ├── app.py                  <- Local Streamlit flower classifier
    │
    ├── save_demo_images.py     <- Fills `data/demo/` from the tf_flowers image bundle
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

