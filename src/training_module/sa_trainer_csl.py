import os
import random
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, SchedulerType, TrainingArguments
import wandb
from huggingface_hub import login
from dotenv import load_dotenv

from src.training_module import dataset_manager, utils


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
    do_soft_labeling = cfg.trainer.get("do_soft_labeling", False)
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


    # Log in using the token
    login(hf_token)
    
    random.seed(random_seed)

    expermient_tag_hf_model = utils.make_safe_filename(hf_model)

    is_split_by_topic_text = 'split_normal'
    if split_by_topic:
        is_split_by_topic_text = 'split_by_topic'


    soft_label_text = 'hard_labels'
    if do_soft_labeling:
        soft_label_text = 'soft_labels'

    experiment_tag = f"{expermient_tag_hf_model}_{is_split_by_topic_text}_{soft_label_text}_{experiment_tag}" 

    # load data
    df_dataset = dataset_manager.load_and_filter_data(dataset_path=dataset_path, do_soft_labeling=do_soft_labeling)

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
        sentence_for_topic = topic_fold_cv(se_dataset)
        
        for topic, test_se in sentence_for_topic.items():
            if selected_topic and topic != selected_topic:
                continue
            
            eval_step = 0
            logging_path_fold = os.path.join(logging_path, f'fold-{topic}')
            os.makedirs(logging_path_fold, exist_ok=True)
            train_set_folds = [fold for _it, fold in sentence_for_topic.items() if not _it == topic]
            train_set = [x for xs in train_set_folds for x in xs]
            train, eval, _ = create_stratifies_train_test_split(train_set, test_size=0.0, eval_size=0.15, split_by_se_id=True)
            
            clip_ids_train, se_ids_train, x_train, y_train = process_sentences(train, use_multi_label=use_multi_label, use_full_context=use_full_context)
            clip_ids_eval, se_ids_eval, x_eval, y_eval = process_sentences(eval, use_multi_label=use_multi_label, use_full_context=use_full_context)
            clip_ids_test, se_ids_test, x_test, y_test = process_sentences(test_se, use_multi_label=use_multi_label, use_full_context=use_full_context)

            ds = DatasetDict({
                'train': Dataset.from_dict({'text': x_train, 'labels': y_train, 'clip_id': clip_ids_train, 'se_id': se_ids_train}),
                'eval': Dataset.from_dict({'text': x_eval, 'labels': y_eval, 'clip_id': clip_ids_eval, 'se_id': se_ids_eval}),
                'test': Dataset.from_dict({'text': x_test, 'labels': y_test, 'clip_id': clip_ids_test, 'se_id': se_ids_test})
            })
            tokenized_datasets = ds.map(model_manager.tokenize_and_extract_sentences_for_sequence_classification_multi_label, batched=True, fn_kwargs={"tokenizer": tokenizer})
            
            model = setup_model_for_sequence_classification(use_lora=use_lora, hf_model=hf_model, tokenizer=tokenizer, num_labels=num_labels, use_multi_label=use_multi_label)
            
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
                # compute_metrics=compute_metrics,
                compute_metrics=compute_metrics.prepare_compute_sa_metrics(
                    model, 
                    tokenizer, 
                    tokenized_datasets, 
                    log_wandb, 
                    logging_path_fold, 
                    split_by_topic=split_by_topic,
                    pred_trainer=pred_trainer,
                    eval_step=eval_step,
                    do_soft=do_soft,
                    save_best_model_path=saved_best_model_dir
                ),
                callbacks=[log_manager.LogToFileCallback(logging_path)]
            )

            trainer.train()
            if log_wandb:
                wandb.finish()

    else:
        train = []
        test_set = []

        # normal processing
        train, _, test_set = dataset_manager.create_stratified_train_test_split_df(df_dataset, test_size=0.15, eval_size=0.0, split_by_se_id=True)
    
        # train_set, eval_set, _ = create_stratifies_train_test_split(train, test_size=0.0, eval_size=0.2)
        folds = dataset_manager.k_fold_cv_df(train, n_folds=n_folds, use_se_id=True)
        # cid_test, x_test, y_test = process_spans(test, id2tag, tag2id, merge_dict)
        for idx in range(n_folds):
            eval_step = 0
            logging_path_fold = os.path.join(logging_path, f'fold-{idx}')
            # logging_path_fold = os.path.join('logging', experiment_name, experiment_tag, f'fold-{idx}')
            os.makedirs(logging_path_fold, exist_ok=True)
            eval_set = folds[idx]
            train_set_folds = [fold for _i, fold in enumerate(folds) if not _i == idx]
            train_set = [x for xs in train_set_folds for x in xs]

            random.shuffle(train_set)

            clip_ids_train, se_ids_train, x_train, y_train = process_sentences(train_set, use_multi_label=use_multi_label, use_full_context=use_full_context)
            clip_ids_eval, se_ids_eval, x_eval, y_eval = process_sentences(eval_set, use_multi_label=use_multi_label, use_full_context=use_full_context)
            clip_ids_test, se_ids_test, x_test, y_test = process_sentences(test_set, use_multi_label=use_multi_label, use_full_context=use_full_context)

            ds = DatasetDict({
                'train': Dataset.from_dict({'text': x_train, 'labels': y_train, 'clip_id': clip_ids_train, 'se_id': se_ids_train}),
                'eval': Dataset.from_dict({'text': x_eval, 'labels': y_eval, 'clip_id': clip_ids_eval, 'se_id': se_ids_eval}),
                'test': Dataset.from_dict({'text': x_test, 'labels': y_test, 'clip_id': clip_ids_test, 'se_id': se_ids_test})
            })
            tokenized_datasets = ds.map(model_manager.tokenize_and_extract_sentences_for_sequence_classification_multi_label, batched=True, fn_kwargs={"tokenizer": tokenizer})

            # Print the size of the tokenized dataset
            print(f"Tokenized Train Dataset Size: {len(tokenized_datasets['train'])}")
            print(f"Tokenized Eval Dataset Size: {len(tokenized_datasets['eval'])}")
            print(f"Tokenized Test Dataset Size: {len(tokenized_datasets['test'])}")

            model = setup_model_for_sequence_classification(use_lora=use_lora, hf_model=hf_model, tokenizer=tokenizer, num_labels=num_labels, use_multi_label=use_multi_label)
            
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
                # compute_metrics=compute_metrics,
                compute_metrics=compute_metrics.prepare_compute_sa_metrics(
                    model, 
                    tokenizer, 
                    tokenized_datasets, 
                    log_wandb, 
                    logging_path_fold, 
                    split_by_topic=split_by_topic,
                    pred_trainer=pred_trainer,                
                    eval_step=eval_step,
                    do_soft=do_soft,
                    save_best_model_path=saved_best_model_dir
                ),
                callbacks=[log_manager.LogToFileCallback(logging_path)]
            )

            trainer.train()
            if log_wandb:
                wandb.finish()