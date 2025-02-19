import os
import pymongo
import argparse
import tabulate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database-name', type=str, default='dev_set')
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

    # query statistics
    num_queries = table_queries.count_documents({})
    no_search_result = table_queries.count_documents({"search_result": None})
    no_image_result = table_queries.count_documents({"image_search_result": None})
    query_table = [
        ["", "Num"],
        ["Num Queries", num_queries],
        ["Queries With Webpages", num_queries - no_search_result],
        ["Queries With Images", num_queries - no_image_result],
    ]
    print(tabulate.tabulate(query_table))

    # webpage statistics
    num_webpages = table_webpages.count_documents({})
    downloaded_webpages = table_webpages.count_documents({"webpage_label": True})
    cleaned_webpages = table_webpages.count_documents({"clean_label": True})
    dedup_webpages = table_webpages.count_documents({"image_dedup_label": True})
    score_webpages = table_webpages.count_documents({"image_score_label": True})
    total_valid_images = sum(
        [
            webpage["num_valid_images"]
            for webpage in table_webpages.find(
                {"image_score_label": True}, {"num_valid_images": 1}
            )
        ]
    )
    webpage_table = [
        ["", "Num"],
        ["Num Webpages", num_webpages],
        ["Downloaded", downloaded_webpages],
        ["Cleaned", cleaned_webpages],
        ["Deduped", dedup_webpages],
        ["Scored", score_webpages],
        ["Total Images", total_valid_images],
    ]
    print(tabulate.tabulate(webpage_table))

    # webpage char len statistics
    char_len_counter = {"raw": 0, "image_replaced": 0, "link_replaced": 0}
    for webpage in table_webpages.find({"clean_label": True}, {"char_len_trend": 1}):
        for key, value in webpage["char_len_trend"].items():
            char_len_counter[key] += value
    char_len_table = [
        ["", "Char Len"],
        ["Raw", char_len_counter["raw"] // cleaned_webpages],
        ["Image Replaced", char_len_counter["image_replaced"] // cleaned_webpages],
        ["Link Replaced", char_len_counter["link_replaced"] // cleaned_webpages],
    ]
    print(tabulate.tabulate(char_len_table))

    # image statistics
    num_images = table_images.count_documents({})
    cached_images = table_images.count_documents({"cache_label": True})
    invalid_images = table_images.count_documents({"valid": False})
    non_duplicate_images = table_images.count_documents(
        {"cache_label": True, "duplicate_label": False}
    )
    score_images = table_images.count_documents({"scoring_label": True})
    valid_images = table_images.count_documents(
        {"scoring_label": True, "final_score": {"$ne": None}}
    )
    caption_images = table_images.count_documents({"caption_label": True})
    image_table = [
        ["", "Num"],
        ["Num Images", num_images],
        ["Visited", cached_images + invalid_images],
        ["Cached", cached_images],
        ["Deduped", non_duplicate_images],
        ["Scored", score_images],
        ["Valid", valid_images],
        ["Caption", caption_images],
    ]
    print(tabulate.tabulate(image_table))

    # auxiliary image statistics
    num_aux_images = table_aux_images.count_documents({})
    cached_aux_images = table_aux_images.count_documents({"cache_label": True})
    invalid_aux_images = table_aux_images.count_documents({"valid": False})
    score_aux_images = table_aux_images.count_documents({"scoring_label": True})
    valid_aux_images = table_aux_images.count_documents(
        {"scoring_label": True, "final_score": {"$ne": None}}
    )
    caption_aux_images = table_aux_images.count_documents({"caption_label": True})
    aux_image_table = [
        ["", "Num"],
        ["Num Aux Images", num_aux_images],
        ["Visited", cached_aux_images + invalid_aux_images],
        ["Cached", cached_aux_images],
        ["Scored", score_aux_images],
        ["Valid", valid_aux_images],
        ["Caption", caption_aux_images],
    ]
    print(tabulate.tabulate(aux_image_table))

    # summary and evaluation statistics
    num_summary = table_summaries.count_documents({})
    summarized = table_summaries.count_documents({"summarize_label": True})
    evaluated = table_summaries.count_documents({"evaluate_label": True})
    summary_table = [
        ["", "Num"],
        ["Num Summary", num_summary],
        ["Summarized", summarized],
        ["Evaluated", evaluated],
    ]
    print(tabulate.tabulate(summary_table))
