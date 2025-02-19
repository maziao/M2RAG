import os
import json
import copy
import pymongo
import argparse
import numpy as np


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
    parser.add_argument('--database-name', type=str, default='dev_set')
    parser.add_argument('--summarizer-id', type=str, required=True)
    args = parser.parse_args()
    
    mongodb_url = os.environ.get('MONGODB_URL')
    assert mongodb_url is not None
    mongodb_client = pymongo.MongoClient(mongodb_url)

    m2rag_db = mongodb_client["m2rag"][args.database_name]

    table_queries = m2rag_db["queries"]
    table_webpages = m2rag_db["webpages"]
    table_aux_images = m2rag_db["aux_images"]
    table_images = m2rag_db["images"]
    table_summaries = m2rag_db["summaries"]

    # score calculator
    sample_counter = 0
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

    summarizer_id = args.summarizer_id
    summarizer_id_list = [summarizer_id]

    text_metrics = list(score_mapper['text'].keys())
    image_metrics = list(score_mapper['multi_modal'].keys())

    image_score_threshold = 8

    score_mapper_list = copy.deepcopy(score_mapper)

    query_ids = [query["query_id"] for query in table_queries.find({"level": 1})]

    for _summ_id in summarizer_id_list:
        summarizer_score_mapper = copy.deepcopy(score_mapper)

        for summary in table_summaries.find(
            {"summarizer_id": _summ_id, "evaluate_label": True}
        ):
            if summary["query_id"] not in query_ids:
                continue

            scores = summary["evaluate_scores"]
            results = summary["evaluate_result"]

            multi_modal_scores = {
                metric_name: (
                    [
                        result["normed_score"]
                        for result in results["multi_modal"][metric_name]
                    ]
                    if metric_name in results["multi_modal"]
                    else []
                )
                for metric_name in image_metrics[:-1]
            }

            positive_images = []
            for webpage in summary["webpages"]:
                for image in webpage["images"]:
                    if image["final_score"] >= image_score_threshold:
                        positive_images.append(image["cached_image_url"])

            for aux_image in summary["aux_images"]:
                if aux_image["final_score"] >= image_score_threshold:
                    positive_images.append(aux_image["cached_image_url"])

            output_images = [
                image["cached_image_url"] for image in summary["output_images"]
            ]

            if len(positive_images) == 0:
                multi_modal_scores["image_recall"] = [1.0]
            else:
                multi_modal_scores["image_recall"] = [
                    len(list(set(positive_images) & set(output_images)))
                    / len(positive_images)
                ]

            sample_counter += 1

            summarizer_score_mapper["text"] = dict_operation(
                summarizer_score_mapper["text"], scores["text"]
            )
            summarizer_score_mapper["multi_modal"] = dict_operation(
                summarizer_score_mapper["multi_modal"], multi_modal_scores, op="extend"
            )

        summarizer_score_mapper = dict_operation(summarizer_score_mapper, op="mean")
        overall_score = sum(
            [summarizer_score_mapper["text"][metric_name] for metric_name in text_metrics]
            + [
                summarizer_score_mapper["multi_modal"][metric_name]
                for metric_name in image_metrics
            ]
        ) / (len(text_metrics) + len(image_metrics))
        summarizer_score_mapper["overall"] = overall_score

        score_mapper_list = dict_operation(
            score_mapper_list, summarizer_score_mapper, op="append"
        )

    score_mapper_list = dict_operation(score_mapper_list, op="mean")

    print(summarizer_id_list)
    print(sample_counter)
    print(f"## scores")
    print(json.dumps(score_mapper_list, indent=4))
    print(f"## overall score: {score_mapper_list['overall']}")
    score_tex = ""
    for score in score_mapper_list["text"].values():
        score_tex += f" & {score * 100:>.1f}"

    for score in score_mapper_list["multi_modal"].values():
        score_tex += f" & {score * 100:>.1f}"

    score_tex += f" & {score_mapper_list['overall'] * 100:>.1f} \\\\"
    print(score_tex)

    print(
        f"Text Avg.: {np.mean([value for key, value in score_mapper_list['text'].items() if key in text_metrics[:4]])}"
    )
    print(
        f"Multi-modal P. Avg.: {np.mean([value for key, value in score_mapper_list['multi_modal'].items() if key in image_metrics[:3]])}"
    )
