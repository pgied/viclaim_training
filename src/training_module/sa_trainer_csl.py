import os
from omegaconf import DictConfig
import wandb
from huggingface_hub import login
from dotenv import load_dotenv

from src.training_module import utils


def run_sequence_classification_trainig_eval_test(cfg: DictConfig):
    # load environment variables
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    # load config
    cfg.trainer

    # Log in using the token
    login(hf_token)

    expermient_tag_hf_model = utils.make_safe_filename(hf_model)

    is_split_by_topic_text = 'split_normal'
    if split_by_topic and not create_specific_dataset:
        is_split_by_topic_text = 'split_by_topic'

    weight_by_topic_text = 'all_topic_combined'
    if weight_by_topic:
        weight_by_topic_text = 'weight_by_topic'

    soft_label_text = 'hard_labels'
    if do_soft:
        soft_label_text = 'soft_labels'


    experiment_tag = f"{expermient_tag_hf_model}_{is_split_by_topic_text}_{weight_by_topic_text}_{soft_label_text}_{experiment_tag}" 