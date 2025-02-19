import os
from typing import Optional
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.search_engine import build_search_engine


@TASK_WATCHER.register_module
class SearchWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: str,
        database_name: str,
        batch_size: int = 0,
        search_engine_config: dict = {},
        log_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            mongodb_url=mongodb_url, database_name=database_name, batch_size=batch_size
        )
        self.query_table = self.database["queries"]
        self.webpage_table = self.database["webpages"]
        self.search_engine = build_search_engine(search_engine_config)

        if log_dir is not None:
            self.log_dir = os.path.join(log_dir, "search")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = None

    def _get_query_ids_and_contents(self):
        query_ids_and_contents = []
        for query in self.query_table.find(
            filter={"search_result": None},
            projection={"query_id": 1, "query_content": 1},
            limit=self.batch_size,
        ):
            query_ids_and_contents.append({"query_id": query["query_id"], "query_content": query["query_content"]})
        return query_ids_and_contents

    def _get_query_ids_and_contents_for_given_query_ids(self, query_ids):
        query_ids_and_contents = []
        for query_id in query_ids:
            query_content = self.query_table.find_one({"query_id": query_id}, {"query_content": 1})["query_content"]
            query_ids_and_contents.append({"query_id": query_id, "query_content": query_content})
        return query_ids_and_contents

    def _watch_single_cycle(self, *args, **kwargs):
        query_ids = kwargs.pop("query_ids", None)
        if query_ids is not None:
            query_ids_and_contents = self._get_query_ids_and_contents_for_given_query_ids(query_ids=query_ids)
        else:
            query_ids_and_contents = self._get_query_ids_and_contents()

        for query in query_ids_and_contents:
            self.search_engine.manager.add_task(
                self.search_engine.search_single,
                os.path.join(self.log_dir, f"{query['query_id']}.json") if self.log_dir is not None else None,
                query["query_id"],
                content=query["query_content"],
            )

        results = self.search_engine.manager.execute_tasks()
        self.search_engine.manager.clear_tasks()

        for result in results:
            if (
                result["result"] is not None
                and len(result["result"]) > 0
                and result["success"]
            ):
                self.query_table.update_one(
                    filter={"query_id": result["id"]},
                    update={"$set": {"search_result": result["result"]}},
                )
                for webpage in result["result"]:
                    self.webpage_table.insert_one(
                        {
                            "query_id": result["id"],
                            "webpage_id": webpage["webpage_id"],
                            "webpage_url": webpage["url"],
                            "retry": 0,
                            "webpage_label": False,
                            "webpage_content": None,
                            "clean_label": False,
                            "char_len_trend": None,
                            "cleaned_webpage_splits": None,
                            "images": None,
                            "image_dedup_label": False,
                            "image_score_label": False,
                            "num_valid_images": None,
                        }
                    )
        
        return sum(
            [
                result["success"]
                and result["result"] is not None
                and len(result["result"]) > 0
                for result in results
            ]
        ), len(results)
