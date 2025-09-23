import os
import random
from typing import Dict
from omegaconf import DictConfig
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding, SchedulerType, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import wandb
from huggingface_hub import login
from dotenv import load_dotenv

from src.llm_architectures import modeling_mistral
from src.training_module import compute_metrics, dataset_manager, log_manager, utils

model_name_to_lora_config = {
    'mistralai/Mistral-7B-v0.3': LoraConfig(
        r=16,  # the dimension of the low-rank matrices
        lora_alpha=8,  # scaling factor for LoRA activations vs pre-trained weight activations
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,  # dropout probability of the LoRA layers
        bias='none',  # wether to train bias weights, set to 'none' for attention layers
        task_type='SEQ_CLS'
    ),
    'tiiuae/falcon-7b': LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    ),
    'meta-llama/Meta-Llama-3-8B': LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
    'meta-llama/Llama-3.1-8B': LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ),
    'meta-llama/Llama-3.2-3B': LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
}


def topic_fold_cv_df(
    df: pd.DataFrame,
    topic_col: str = "topic",
    keep_order_col: str = "row_index",
) -> Dict[str, pd.DataFrame]:
    """
    Group the sentence/span DataFrame by topic and return a mapping
    topic -> DataFrame (rows belonging to that topic).
    If keep_order_col is provided and exists, each group's rows will be
    sorted by that column using a stable sort (preserves original order).
    """
    if topic_col not in df.columns:
        raise KeyError(f"topic column '{topic_col}' not found in DataFrame")

    grouped = {}
    # groupby(sort=False) preserves first-seen topic order
    for topic, g in df.groupby(topic_col, sort=False):
        grp = g.copy()
        if keep_order_col and keep_order_col in grp.columns:
            # stable sort to preserve ties/order
            grp = grp.sort_values(keep_order_col, kind="mergesort")
        grouped[topic] = grp.reset_index(drop=True)
    return grouped


def setup_model_for_sequence_classification(use_lora, hf_model, tokenizer, num_labels):
    problem_type = "multi_label_classification"
        
    if use_lora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # enable 4-bit quantization
            bnb_4bit_quant_type='nf4',  # information theoretically optimal dtype for normally distributed weights
            bnb_4bit_use_double_quant=True,  # quantize quantized weights //insert xzibit meme
            bnb_4bit_compute_dtype=torch.bfloat16  # optimized fp format for ML
        )
        if 'mistral' in tokenizer.name_or_path:
            model = modeling_mistral.MistralForSequenceClassification.from_pretrained(
                hf_model, num_labels=num_labels, quantization_config=quantization_config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_model, num_labels=num_labels,
                quantization_config=quantization_config,
                problem_type="multi_label_classification"
            )

        lora_config = model_name_to_lora_config[hf_model]

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model, num_labels=num_labels,
            problem_type=problem_type
        )


    return model

def tokenize_inputs(samples, tokenizer, max_length = 1024):
    tokenized_inputs = tokenizer(
        samples['text'], 
        truncation=True,
        max_length=max_length,
    )
    tokenized_inputs['labels'] = [label[:3] for label in samples['labels']]
    tokenized_inputs['clip_id'] = samples['clip_id']
    tokenized_inputs['se_id'] = samples['se_id']
    return tokenized_inputs
    

def run_sequence_classification_trainig_eval_test(cfg: DictConfig):
    # load environment variables
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')

    # load config
    random_seed = cfg.get("random_seed", 42)
    log_wandb = cfg.get("log_wandb", False)
    
    # trainer config
    hf_model = cfg.trainer.get("hf_model", "")
    dataset_path = cfg.trainer.get("dataset_path", "")
    split_by_topic = cfg.trainer.get("split_by_topic", False)
    experiment_tag = cfg.trainer.get("experiment_tag", False)
    experiment_name = cfg.trainer.get("experiment_name", False)
    per_device_train_batch_size = cfg.trainer.get("per_device_train_batch_size", 8)
    per_device_eval_batch_size = cfg.trainer.get("per_device_eval_batch_size", 8)
    gradient_accumulation_steps = cfg.trainer.get("gradient_accumulation_steps", 8)
    gradient_checkpointing = cfg.trainer.get("gradient_checkpointing", 8)
    eval_strategy = cfg.trainer.get("eval_strategy", "steps")
    num_training_steps = cfg.trainer.get("num_training_steps", 1000)
    bf16 = cfg.trainer.get("bf16", False)
    fp16 = cfg.trainer.get("fp16", False)
    eval_steps = cfg.trainer.get("eval_steps", 50)
    save_steps = cfg.trainer.get("save_steps", 50)
    logging_steps = cfg.trainer.get("logging_steps", 10)
    learning_rate = cfg.trainer.get("learning_rate", 2e-05)
    optim = cfg.trainer.get("optim", "adamw_hf")
    warmup_steps = cfg.trainer.get("warmup_steps", 100)
    save_total_limit = cfg.trainer.get("save_total_limit", 2)
    label_smoothing_factor = cfg.trainer.get("label_smoothing_factor", 0.0)
    logging_dir = cfg.trainer.get("logging_dir", "")
    n_folds = cfg.trainer.get("n_folds", 5)
    test_size = cfg.trainer.get("test_size", 0.15)
    use_full_context = cfg.trainer.get("use_full_context", True)
    use_lora = cfg.trainer.get("use_lora", False)
    saved_model_dir = cfg.trainer.get("saved_model_dir", "")
    topic_list = cfg.trainer.get("topic_list", [])

    num_labels = 3

    # Log in using the token
    login(hf_token)
    
    random.seed(random_seed)

    expermient_tag_hf_model = utils.make_safe_filename(hf_model)

    is_split_by_topic_text = 'split_normal'
    if split_by_topic:
        is_split_by_topic_text = 'split_by_topic'


    experiment_tag = f"{expermient_tag_hf_model}_{is_split_by_topic_text}_{experiment_tag}" 

    # load data
    df_dataset = dataset_manager.load_and_filter_data(dataset_path=dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorWithPadding(tokenizer, padding=True)

    logging_path = os.path.join(logging_dir, experiment_name, experiment_tag)
    os.makedirs(logging_path, exist_ok=True)

    model_path = os.path.join('models', experiment_name, experiment_tag)

    print(f'is cuda possible: {torch.cuda.is_available()}')  # Should print True
    print(f'is mps possible: {torch.mps.is_available()}')  # Should print True

    training_args = TrainingArguments(
        output_dir=model_path,
        logging_dir=logging_path,
        group_by_length=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy=eval_strategy,
        max_steps=num_training_steps,
        bf16=bf16,
        fp16=fp16,
        bf16_full_eval=bf16,
        fp16_full_eval=fp16,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        optim=optim,
        max_grad_norm=1.0,
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_steps=warmup_steps,
        ddp_find_unused_parameters=True,
        ignore_data_skip=True,
        label_smoothing_factor=label_smoothing_factor
    )
    
    if split_by_topic:
        for selected_topic in topic_list:
            sentence_for_topic = topic_fold_cv_df(df_dataset)
            
            for topic, test_df in sentence_for_topic.items():
                if selected_topic and topic != selected_topic:
                    continue
                
                eval_step = 0
                logging_path_fold = os.path.join(logging_path, f'fold-{topic}')
                os.makedirs(logging_path_fold, exist_ok=True)

                train_set_folds = [fold for _it, fold in sentence_for_topic.items() if _it != topic]
                train_set = pd.concat(train_set_folds, ignore_index=True)

                train_df, eval_df, _ = dataset_manager.create_stratified_train_test_split_df(train_set, test_size=0.0, eval_size=0.15, split_by_se_id=True, random_seed=random_seed)
                
                clip_ids_train, se_ids_train, x_train, y_train = dataset_manager.process_sentences(train_df, use_full_context=use_full_context)
                clip_ids_eval, se_ids_eval, x_eval, y_eval = dataset_manager.process_sentences(eval_df, use_full_context=use_full_context)
                clip_ids_test, se_ids_test, x_test, y_test = dataset_manager.process_sentences(test_df, use_full_context=use_full_context)

                ds = DatasetDict({
                    'train': Dataset.from_dict({'text': x_train, 'labels': y_train, 'clip_id': clip_ids_train, 'se_id': se_ids_train}),
                    'eval': Dataset.from_dict({'text': x_eval, 'labels': y_eval, 'clip_id': clip_ids_eval, 'se_id': se_ids_eval}),
                    'test': Dataset.from_dict({'text': x_test, 'labels': y_test, 'clip_id': clip_ids_test, 'se_id': se_ids_test})
                })
                tokenized_datasets = ds.map(tokenize_inputs, batched=True, fn_kwargs={"tokenizer": tokenizer})
                
                model = setup_model_for_sequence_classification(use_lora=use_lora, hf_model=hf_model, tokenizer=tokenizer, num_labels=num_labels)
                
                print(f'experiment-tag is: {experiment_tag}')

                if log_wandb:
                    wandb.init(project="Hamison-SA-RUN-TOPIC", group=f'{experiment_name}/{experiment_tag}', name=f'{n_folds}-topic-{topic}')

                pred_trainer = Trainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=None
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_datasets["train"],
                    eval_dataset=tokenized_datasets["eval"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics.prepare_compute_metrics(
                        model, 
                        tokenizer, 
                        tokenized_datasets, 
                        logging_path_fold, 
                        split_by_topic=split_by_topic,
                        pred_trainer=pred_trainer,
                        eval_step=eval_step,
                        save_best_model_path=saved_model_dir
                    ),
                    callbacks=[log_manager.LogToFileCallback(logging_path)]
                )

                trainer.train()
                if log_wandb:
                    wandb.finish()

    else:
        # normal processing
        train_df, _, test_set_df = dataset_manager.create_stratified_train_test_split_df(df_dataset, test_size=0.15, eval_size=0.0, split_by_se_id=True, random_seed=random_seed)
    
        folds = dataset_manager.k_fold_cv_df(train_df, n_folds=n_folds)
        for idx in range(n_folds):
            eval_step = 0
            logging_path_fold = os.path.join(logging_path, f'fold-{idx}')
            os.makedirs(logging_path_fold, exist_ok=True)
            eval_set_df = folds[idx]
            train_set_folds = [fold for _i, fold in enumerate(folds) if not _i == idx]
            # train_set = [x for xs in train_set_folds for x in xs]
    
            # merge (ignore_index to reset indices; sort=False to avoid sorting columns)
            train_set_folds_merged_df = pd.concat(train_set_folds, ignore_index=True, sort=False)

            # shuffle rows and reset index
            shuffled_train_set_folds_merged_df = train_set_folds_merged_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

            clip_ids_train, se_ids_train, x_train, y_train = dataset_manager.process_sentences(shuffled_train_set_folds_merged_df, use_full_context=use_full_context)
            clip_ids_eval, se_ids_eval, x_eval, y_eval = dataset_manager.process_sentences(eval_set_df, use_full_context=use_full_context)
            clip_ids_test, se_ids_test, x_test, y_test = dataset_manager.process_sentences(test_set_df, use_full_context=use_full_context)

            ds = DatasetDict({
                'train': Dataset.from_dict({'text': x_train, 'labels': y_train, 'clip_id': clip_ids_train, 'se_id': se_ids_train}),
                'eval': Dataset.from_dict({'text': x_eval, 'labels': y_eval, 'clip_id': clip_ids_eval, 'se_id': se_ids_eval}),
                'test': Dataset.from_dict({'text': x_test, 'labels': y_test, 'clip_id': clip_ids_test, 'se_id': se_ids_test})
            })
            tokenized_datasets = ds.map(tokenize_inputs, batched=True, fn_kwargs={"tokenizer": tokenizer})

            # Print the size of the tokenized dataset
            print(f"Tokenized Train Dataset Size: {len(tokenized_datasets['train'])}")
            print(f"Tokenized Eval Dataset Size: {len(tokenized_datasets['eval'])}")
            print(f"Tokenized Test Dataset Size: {len(tokenized_datasets['test'])}")

            model = setup_model_for_sequence_classification(use_lora=use_lora, hf_model=hf_model, tokenizer=tokenizer, num_labels=num_labels)
            
            print(f'experiment-tag is: {experiment_tag}')
            
            if log_wandb:
                wandb.init(project="Hamison_SA-SC-RUN", group=f'{experiment_name}/{experiment_tag}', name=f'{n_folds}-fold-{idx}')

            pred_trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=None
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["eval"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics.prepare_compute_metrics(
                    model, 
                    tokenizer, 
                    tokenized_datasets, 
                    logging_path_fold, 
                    split_by_topic=split_by_topic,
                    pred_trainer=pred_trainer,                
                    eval_step=eval_step,
                    save_best_model_path=saved_model_dir
                ),
                callbacks=[log_manager.LogToFileCallback(logging_path)]
            )

            trainer.train()
            if log_wandb:
                wandb.finish()