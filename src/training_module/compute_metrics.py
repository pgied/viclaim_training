import json
import os
import numpy as np
from sklearn import metrics

current_best_avg_score = 0

def _col_to_list(col):
    # convert HF dataset Column / pandas Series / np.ndarray / iterable -> Python list
    if hasattr(col, "tolist"):
        try:
            return col.tolist()
        except Exception:
            # some Arrow columns raise on tolist(); fall back to list()
            return list(col)
    return list(col)

def log_eval_to_file(log_dir, log_object, eval_step):
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    safe_object = ensure_serializable(log_object)
    with open(os.path.join(log_dir, f'outs-{eval_step}.json'), 'wt', encoding='utf-8') as ofile:
        json.dump(safe_object, ofile, ensure_ascii=False, indent=4)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_best_threshold(y_true, y_score, sample_weight=None):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_score, sample_weight=sample_weight)
    J = tpr - fpr #YOuden J stat
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh


def generate_metrics_report(pred_ids_list, label_ids_list, thresholds=np.array([0.5, 0.5, 0.5])):
    prediction_scores = sigmoid(pred_ids_list)
    predictions = (prediction_scores > thresholds).astype(int)

    sample_weights = np.copy(label_ids_list)
    sample_weights[sample_weights == 0] = 1

    labels = np.zeros_like(label_ids_list)
    labels[label_ids_list > 0] = 1.0
    labels = labels.astype(int)

    accuracy = metrics.accuracy_score(y_true=labels.reshape(-1), y_pred=predictions.reshape(-1), sample_weight=sample_weights.reshape(-1), normalize=True)
    fcw_f1 = metrics.f1_score(y_true=labels[:, 0], y_pred=predictions[:, 0], sample_weight=sample_weights[:, 0], pos_label=1, average='binary', zero_division=0)
    fcw_precision = metrics.precision_score(y_true=labels[:, 0], y_pred=predictions[:, 0], sample_weight=sample_weights[:, 0], pos_label=1, average='binary', zero_division=0)
    fcw_recall = metrics.recall_score(y_true=labels[:, 0], y_pred=predictions[:, 0], sample_weight=sample_weights[:, 0], pos_label=1, average='binary', zero_division=0)
    fcw_roc_auc = metrics.roc_auc_score(y_true=labels[:, 0], y_score=prediction_scores[:, 0], sample_weight=sample_weights[:, 0])
    fcw_best_threshold = compute_best_threshold(y_true=labels[:, 0], y_score=prediction_scores[:, 0], sample_weight=sample_weights[:, 0])

    fnc_f1 = metrics.f1_score(y_true=labels[:, 1], y_pred=predictions[:, 1], sample_weight=sample_weights[:, 1], pos_label=1, average='binary', zero_division=0)
    fnc_precision = metrics.precision_score(y_true=labels[:, 1], y_pred=predictions[:, 1], sample_weight=sample_weights[:, 1], pos_label=1, average='binary', zero_division=0)
    fnc_recall = metrics.recall_score(y_true=labels[:, 1], y_pred=predictions[:, 1], sample_weight=sample_weights[:, 1], pos_label=1, average='binary', zero_division=0)
    fnc_roc_auc = metrics.roc_auc_score(y_true=labels[:, 1], y_score=prediction_scores[:, 1], sample_weight=sample_weights[:, 1])
    fnc_best_threshold = compute_best_threshold(y_true=labels[:, 1], y_score=prediction_scores[:, 1],  sample_weight=sample_weights[:, 1])

    opn_f1 = metrics.f1_score(y_true=labels[:, 2], y_pred=predictions[:, 2], sample_weight=sample_weights[:, 2], pos_label=1, average='binary', zero_division=0)
    opn_precision = metrics.precision_score(y_true=labels[:, 2], y_pred=predictions[:, 2], sample_weight=sample_weights[:,2], pos_label=1, average='binary', zero_division=0)
    opn_recall = metrics.recall_score(y_true=labels[:, 2], y_pred=predictions[:, 2], sample_weight=sample_weights[:, 2], pos_label=1, average='binary', zero_division=0)
    opn_roc_auc = metrics.roc_auc_score(y_true=labels[:, 2], y_score=prediction_scores[:, 2],  sample_weight=sample_weights[:, 2])
    opn_best_threshold = compute_best_threshold(y_true=labels[:, 2], y_score=prediction_scores[:, 2], sample_weight=sample_weights[:, 2])

    metrics_object = {
        "fcw": {
            "precision": fcw_precision,
            "recall": fcw_recall,
            "f1": fcw_f1,
            "roc_auc": fcw_roc_auc,
            "best_threshold": float(fcw_best_threshold),
        },
        "fnc": {
            "precision": fnc_precision,
            "recall": fnc_recall,
            "f1": fnc_f1,
            "roc_auc": fnc_roc_auc,
            "best_threshold": float(fnc_best_threshold),
        },
        "opn": {
            "precision": opn_precision,
            "recall": opn_recall,
            "f1": opn_f1,
            "roc_auc": opn_roc_auc,
            "best_threshold": float(opn_best_threshold),
        },
        "accuracy": accuracy,
    }

    return metrics_object


def ensure_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    return obj  # Return the object as-is if it's already serializable

def prepare_compute_metrics(model, tokenizer, tokenized_dataset, log_dir, split_by_topic, pred_trainer, eval_step, save_best_model_path):
    def compute_metrics(pred):
        nonlocal model
        nonlocal tokenizer
        nonlocal tokenized_dataset
        nonlocal log_dir
        nonlocal split_by_topic
        nonlocal pred_trainer
        nonlocal eval_step

        global current_best_avg_score

        eval_pred_logits = pred.predictions
        eval_label_ids_list = pred.label_ids

        full_log_object = {
            "split_by_topic": split_by_topic
        }

        eval_metrics_object = generate_metrics_report(pred_ids_list=eval_pred_logits, label_ids_list=eval_label_ids_list)
        
        eval_log_object = {
            "type": "eval"
        }
        eval_log_object['metrics'] = eval_metrics_object
        eval_log_object['output'] = {
            'pred_logits': ensure_serializable(eval_pred_logits),
            'label_ids': ensure_serializable(eval_label_ids_list),
            'clip_id': _col_to_list(tokenized_dataset['eval']['clip_id']),
            'se_id': _col_to_list(tokenized_dataset['eval']['se_id'])  
        }
        full_log_object["eval"] = eval_log_object

        local_avg_score = np.array([scores['roc_auc'] for label, scores in eval_metrics_object.items() if type(scores) == dict]).mean()
        if current_best_avg_score < local_avg_score:
            current_best_avg_score = local_avg_score
            model.save_pretrained(save_best_model_path)
            tokenizer.save_pretrained(save_best_model_path)

        # do testing
        test_pred = pred_trainer.predict(test_dataset=tokenized_dataset["test"])
        
        test_pred_logits = test_pred.predictions
        test_pred_label_ids_list = test_pred.label_ids
        test_thresholds = np.array([eval_metrics_object[x]['best_threshold'] for x in ['fcw', 'fnc', 'opn']]) #hardcoded sequence of 'fcw', 'fnc', 'opn' could be done nicer :)
        test_pred_metrics_object = generate_metrics_report(pred_ids_list=test_pred_logits, label_ids_list=test_pred_label_ids_list, thresholds=test_thresholds)

        test_pred_log_object = {
            "type": "test"
        }
        test_pred_log_object['metrics'] = test_pred_metrics_object
        test_pred_log_object['output'] = {
            'pred_logits': ensure_serializable(test_pred_logits),
            'label_ids': ensure_serializable(test_pred_label_ids_list),
            'clip_id': _col_to_list(tokenized_dataset['test']['clip_id']),  
            'se_id': _col_to_list(tokenized_dataset['test']['se_id'])   
        }
        full_log_object["test"] = test_pred_log_object

        log_eval_to_file(log_dir=log_dir, log_object=full_log_object, eval_step=eval_step)

        return_object = {
            'eval/': eval_metrics_object,
            'test/': test_pred_metrics_object,
        }

        eval_step += 1

        return return_object
    
    return compute_metrics