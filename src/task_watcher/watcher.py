import os
import time
import pymongo
import logging
from tqdm import tqdm
from abc import abstractmethod
from typing import Tuple, Optional
from src.task_watcher import TASK_WATCHER


@TASK_WATCHER.register_module
class TaskWatcher:
    def __init__(
        self,
        mongodb_url: Optional[str],
        database_name: Optional[str],
        batch_size: int = 0,
        interval_on_empty_cycle: int = 60,
        max_empty_cycles_before_exit: int = -1
    ) -> None:
        if mongodb_url is None:
            mongodb_url = os.environ.get('MONGODB_URL', None)
        assert mongodb_url is not None, f"`MONGODB_URL` is not set"
        if database_name is None:
            database_name = os.environ.get('MONGODB_NAME', None)
        assert database_name is not None, f"`MONGODB_NAME` is not set"
        self.client = pymongo.MongoClient(mongodb_url)
        self.database = self.client['m2rag'][database_name]
        
        assert isinstance(batch_size, int) and batch_size >= 0, f"`batch_size` for task watchers must be a non-negative integer, but got {batch_size}"
        self.batch_size = batch_size
        
        assert isinstance(interval_on_empty_cycle, int) or isinstance(interval_on_empty_cycle, float), f"Invalid interval: {interval_on_empty_cycle}"
        self.interval_on_empty_cycle = interval_on_empty_cycle
        
        self.max_empty_cycles_before_exit = max_empty_cycles_before_exit
        
        self.cycle_id = 0

    @abstractmethod
    def _watch_single_cycle(self, *args, **kwargs) -> Tuple[int, int]:
        raise NotImplementedError
    
    def watch_forever(self, terminate_on_cycle_end: bool = False, **kwargs):
        empty_counter = 0
        while True:
            success, overall = self._watch_single_cycle(**kwargs)
            
            if terminate_on_cycle_end:
                break
            
            if overall == 0:
                empty_counter += 1
                if self.max_empty_cycles_before_exit > 0 and empty_counter == self.max_empty_cycles_before_exit:
                    logging.warning(f"max empty cycles reached preset threshold {self.max_empty_cycles_before_exit}, exit")
                    break
                
                tqdm.write(f"[!] Empty cycle | cycle id: {self.cycle_id}")
                if self.interval_on_empty_cycle > 0:
                    time.sleep(self.interval_on_empty_cycle)
            else:
                tqdm.write(f"[!] Overall: {overall}; Success: {success}; Success Rate: {round(100 * success / overall)}%")
            
            self.cycle_id += 1
            