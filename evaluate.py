import os
import json
import yaml
import argparse
import numpy as np
from src.utils import MultithreadManager
from src.evaluator import build_evaluator


def dict_operation(src_dict: dict, tgt_dict: dict = None, op: str = "append"):
    assert op in ["append", "inc", "mean", "sum", "extend"]
    for key in src_dict:
        if (tgt_dict is None and op not in ["mean", "sum"]) or (
            isinstance(tgt_dict, dict) and key not in tgt_dict
        ):
            continue

        if isinstance(src_dict[key], dict):
            if tgt_dict is not None:
                src_dict[key] = dict_operation(src_dict[key], tgt_dict[key], op)
            else:
                src_dict[key] = dict_operation(src_dict[key], None, op)
        else:
            if op == "append":
                src_dict[key].append(tgt_dict[key])
            elif op == "inc":
                src_dict[key] += tgt_dict[key]
            elif op == "mean":
                src_dict[key] = np.mean(src_dict[key]).tolist()
            elif op == "sum":
                src_dict[key] = np.sum(src_dict[key]).tolist()
            elif op == "extend":
                src_dict[key].extend(tgt_dict[key])
    return src_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarize-log-dir', type=str, required=True)
    parser.add_argument('--config-file', type=str, default='./src/config/evaluate_custom.yaml')
    args = parser.parse_args()
    
    samples = []
    for log_file in os.listdir(args.summarize_log_dir):
        with open(os.path.join(args.summarize_log_dir, log_file), 'r+', encoding='utf-8') as f:
            log = json.load(f)
            if not log['success']:
                continue
            
            sample = log['result']
            if sample is None:
                continue
            
            sample['query_id'] = log['id']
            samples.append(sample)
    
    # load config
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    evaluator = build_evaluator(config['evaluator_config'])
    manager = MultithreadManager(**config['multi_thread_config'])
    
    summarize_log_dir = args.summarize_log_dir
    while summarize_log_dir.endswith('/'):
        summarize_log_dir = summarize_log_dir[:-1]
    
    summarizer_id = os.path.basename(summarize_log_dir)

    for summary in samples:
        manager.add_task(
            evaluator.evaluate,
            None,
            f"{summary['query_id']}-{summarizer_id}",
            query_id=summary['query_id'],
            summarizer_id=summarizer_id,
            query=summary['user_query'],
            webpages=summary['webpages'],
            aux_images=summary['aux_images'],
            raw_summary=summary['placeholder_response'],
            output_images=summary['output_images'],
            metrics=None
        )
    
    results = manager.execute_tasks()
    manager.clear_tasks()
    
    score_mapper = {
        "text": {
            "fluency": [],
            "response_relevancy": [],
            "context_precision": [],
            "faithfulness": [],
        },
        "multi_modal": {
            "image_coherence": [],
            "image_helpfulness": [],
            "image_reference": [],
            "image_recall": [],
        },
        "overall": [],
    }
    
    text_metrics = list(score_mapper['text'].keys())
    image_metrics = list(score_mapper['multi_modal'].keys())
    image_score_threshold = 8
    
    for result in results:
        eval_scores = result['result']["evaluate_scores"]
        eval_result = result['result']["evaluate_result"]

        multi_modal_scores = {
            metric_name: (
                [
                    _result["normed_score"]
                    for _result in eval_result["multi_modal"][metric_name]
                ]
                if metric_name in eval_result["multi_modal"]
                else []
            )
            for metric_name in image_metrics[:-1]
        }

        positive_images = []
        for webpage in result['kwargs']["webpages"]:
            for image in webpage["images"]:
                if image["final_score"] >= image_score_threshold:
                    positive_images.append(image["cached_image_url"])

        for aux_image in result['kwargs']["aux_images"]:
            if aux_image["final_score"] >= image_score_threshold:
                positive_images.append(aux_image["cached_image_url"])

        output_images = [
            image["cached_image_url"] for image in result['kwargs']["output_images"]
        ]

        if len(positive_images) == 0:
            multi_modal_scores["image_recall"] = [1.0]
        else:
            multi_modal_scores["image_recall"] = [
                len(list(set(positive_images) & set(output_images)))
                / len(positive_images)
            ]

        score_mapper["text"] = dict_operation(
            score_mapper["text"], eval_scores["text"]
        )
        score_mapper["multi_modal"] = dict_operation(
            score_mapper["multi_modal"], multi_modal_scores, op="extend"
        )

    score_mapper = dict_operation(score_mapper, op="mean")
    overall_score = sum(
        [score_mapper["text"][metric_name] for metric_name in text_metrics]
        + [
            score_mapper["multi_modal"][metric_name]
            for metric_name in image_metrics
        ]
    ) / (len(text_metrics) + len(image_metrics))
    score_mapper['overall'] = overall_score
    
    print(f"Scores")
    print(json.dumps(score_mapper, indent=4))
    print(f"Overall score: {score_mapper['overall']}")
    print(
        f"Text Avg.: {np.mean([value for key, value in score_mapper['text'].items() if key in text_metrics[:4]])}"
    )
    print(
        f"Multi-modal P. Avg.: {np.mean([value for key, value in score_mapper['multi_modal'].items() if key in image_metrics[:3]])}"
    )
