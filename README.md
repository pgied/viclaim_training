# **ViClaim Training**

Train and evaluate multi-label sequence classification models (fcw, fnc, opn) on the **ViClaim** dataset using state-of-the-art language models.

## **About the Project**

This repository contains:

- **Training harness** for fine-tuning language models on the ViClaim multi-label classification task
- **Dataset utilities** for preprocessing and splitting the data
- **Evaluation code** for computing metrics and generating performance reports

The code is based on the paper:

> **ViClaim: A Multilingual Multilabel Dataset for Automatic Claim Detection in Videos**  
> [[Paper Link]](https://arxiv.org/abs/2504.12882)

---

## **Getting Started**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/viclaim_training.git
cd viclaim_training
```

### **2. Create a Virtual Environment**

#### Windows

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

#### Mac / Linux

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
# For development install
python -m pip install -e .
```

## **Preparing the Data**

1. Place your dataset CSV file according to the path in `conf/conf.yaml`
2. The code will automatically:
   - Clean and merge sentences into full text
   - Create train/test splits (stratified or k-fold)
   - Convert data into model-ready format

## **Training Models**

The main training pipeline is implemented in **`src/main.py`**.

### **Basic Training**

```bash
# Run with default config
python -m src.main
```

### **Advanced Options**

Override config values via Hydra arguments:

```bash
# Use different model with LoRA
python -m src.main \
  +trainer.hf_model="mistralai/Mistral-7B-v0.3" \
  +trainer.use_lora=true

# Modify training parameters
python -m src.main \
  +trainer.batch_size=8 \
  +trainer.learning_rate=2e-5
```

## **Configuration**

- Edit YAML files in `conf/` directory
- Main config: `conf/conf.yaml`
- Override via command line using Hydra syntax

## **Development Guide**

Key components:

- **Dataset Processing**: [`src/training_module/dataset_manager.py`](src/training_module/dataset_manager.py)
- **Training Logic**: [`src/training_module/trainer.py`](src/training_module/trainer.py)
- **Metrics & Evaluation**: [`src/training_module/compute_metrics.py`](src/training_module/compute_metrics.py)

### **Code Structure**

- `src/main.py` - Entry point
- `src/training_module/` - Core training components
- `conf/` - Configuration files
- `tests/` - Unit tests

---

## **Intended Use**

This repository is provided **for research and educational purposes only**.
The code is designed to support:

- Training claim detection models
- Evaluating multi-label classification performance
- Advancing NLP research in video content analysis

### **Ethical Considerations**

- Use models and data responsibly
- Respect computational resources
- Consider environmental impact of large-scale training
- Follow license terms of pretrained models

### **Disclaimer**

The authors assume no responsibility for misuse of the code or trained models.
Users agree to:

- Use the code responsibly
- Respect applicable laws and guidelines
- Cite the ViClaim paper in derivative work

> For questions about usage or ethical concerns, please contact the maintainers.
