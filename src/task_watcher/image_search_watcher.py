import os
from typing import List, Any, Tuple, Optional
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.search_engine import build_search_engine


@TASK_WATCHER.register_module
class ImageSearchWatcher(TaskWatcher):
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
        self.aux_image_table = self.database["aux_images"]
        self.search_engine = build_search_engine(search_engine_config)

        if log_dir is not None:
            self.log_dir = os.path.join(log_dir, "image_search")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
        else:
            self.log_dir = None

    def _watch_single_cycle(self, *args, **kwargs):
        queries = kwargs.pop("queries", None)
        if queries is not None:
            return self._watch_single_cycle_with_queries(queries=queries)

        for query in self.query_table.find(
            filter={"image_search_result": None},
            projection={"query_id": 1, "query_content": 1},
            limit=self.batch_size,
        ):
            self.search_engine.manager.add_task(
                self.search_engine.search_image_single,
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
                    update={"$set": {"image_search_result": result["result"]}},
                )
                for image in result["result"]:
                    self.aux_image_table.insert_one(
                        {
                            "query_id": result["id"],
                            "image_id": image["image_id"],
                            "image_url": image["image_url"],
                            "valid": None,
                            "retry": 0,
                            "cache_label": False,
                            "cached_image_url": None,
                            "relative_path": None,
                            "scoring_label": False,
                            "scores": None,
                            "final_score": None,
                            "caption_label": False,
                            "caption": None,
                            "all_captions": {},
                        }
                    )
        return sum(
            [result["success"] and len(result["result"]) > 0 for result in results]
        ), len(results)

    def _watch_single_cycle_with_queries(self, queries: List[Any]) -> Tuple[int, int]:
        if self.batch_size is not None:
            lower_bound = self.batch_size * self.cycle_id
            higher_bound = self.batch_size * (self.cycle_id + 1)
        else:
            lower_bound = None
            higher_bound = None

        for query in queries[lower_bound:higher_bound]:
            self.search_engine.manager.add_task(
                self.search_engine.search_image_single,
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
                    update={"$set": {"image_search_result": result["result"]}},
                )
                for image in result["result"]:
                    self.aux_image_table.insert_one(
                        {
                            "query_id": result["id"],
                            "image_id": image["image_id"],
                            "image_url": image["image_url"],
                            "valid": None,
                            "retry": 0,
                            "cache_label": False,
                            "cached_image_url": None,
                            "relative_path": None,
                            "scoring_label": False,
                            "scores": None,
                            "final_score": None,
                            "caption_label": False,
                            "caption": None,
                            "all_captions": {},
                        }
                    )
        
        return sum(
            [result["success"] and len(result["result"]) > 0 for result in results]
        ), len(results)
