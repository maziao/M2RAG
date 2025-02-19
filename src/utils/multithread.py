import os
import json
import copy
import inspect
import threading
from tqdm import tqdm
from typing import List, Union, Callable, Optional, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging

logger = logging.getLogger(__name__)


class MultithreadManager:
    """A class used to handle simple multithread tasks.

    Args:
            num_threads (int, optional): max number of concurrent threads. Defaults to 1.
            max_retry_on_failure (int, optional): max retry times if `fn` fails. Defaults to 0.
            log_warning_on_failure (bool, optional): whether to log a warining when a task finally failed. Defaults to True.
            mark_as_completed_on_failure (bool, optional): whether to mark a failed task as completed. Defaults to True.
            timeout (int, optional): max number of seconds for a single task to complete. Defaults to 60.
    """

    def __init__(
        self,
        num_threads: int = 1,
        max_retry_on_failure: int = 0,
        log_warning_on_failure: bool = True,
        mark_as_completed_on_failure: bool = True,
        timeout: int = None,
    ) -> None:
        self.num_threads = num_threads
        self.max_retry_on_failure = max_retry_on_failure
        self.log_warning_on_failure = log_warning_on_failure
        self.mark_as_completed_on_failure = mark_as_completed_on_failure
        self.timeout = timeout

        self.tasks = []
        self.progress_cache = {}
        self.task_result_cache = {}

    def add_tasks(
        self,
        tasks: Union[Dict, List[Dict]]
    ):
        """Add a single task or a list of tasks.

        Args:
            tasks (Union[dict, List[dict]]): Each dict must contain keys `fn`, as your function to execute.
            If your function has arguments or keyword arguments, the dict should also contain `args` and `kwargs`.

        Returns:
            int: Number of tasks added successfully. If error occurred, return -1.
        """
        assert isinstance(tasks, dict) or isinstance(tasks, list)
        if isinstance(tasks, dict):
            tasks = [tasks]

        for task in tasks:
            # Build a valid task
            assert "fn" in task
            assert "task_id" in task
            fn = task.get('fn')
            args = task.get('args', ())
            kwargs = task.get('kwargs', {})
            log_file = task.get('log_file', None)
            task_id = task.get('task_id')

            # Add a new task to task list
            self.add_task(fn, log_file, task_id, *args, **kwargs)
    
    def add_task(
        self,
        fn: Callable,
        log_file: Optional[os.PathLike],
        task_id: Any,
        *args,
        **kwargs
    ):
        assert isinstance(fn, Callable)
        
        completed = False
        if log_file is None:
            logger.debug(f"`log_file` is set to None for task {task_id}, the task will not save its result to log file during task running and have to rerun if the process crushes.")
        elif isinstance(log_file, str):
            assert log_file.endswith('.json') or log_file.endswith(".jsonl")
            
            if os.path.exists(log_file):
                if log_file not in self.progress_cache:
                    with open(log_file, 'r+', encoding='utf-8') as f:
                        if log_file.endswith('.json'):
                            try:
                                log = json.load(f)
                            except json.decoder.JSONDecodeError:
                                log = {'success': False}
                            
                            if log['success']:
                                completed_ids = [log['id']]
                            else:
                                completed_ids = []
                        else:
                            log = []
                            for line in f.readlines():
                                try:
                                    _log = json.loads(line)
                                    log.append(_log)
                                except json.decoder.JSONDecodeError:
                                    continue
                            
                            completed_ids = [_log['id'] for _log in log if _log['success']]
                    self.progress_cache[log_file] = completed_ids
                else:
                    completed_ids = self.progress_cache[log_file]
                        
                if task_id in completed_ids:
                    completed = True
            else:
                log_dir = os.path.dirname(log_file)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
        else:
            raise ValueError(f"`log_file` should be None or a string, but got {type(log_file)}.")
            
        self.tasks.append({
            "id": task_id,
            "log_file": log_file,
            "fn": fn,
            "completed": completed,
            "args": args,
            "kwargs": kwargs
        })

    def execute_tasks(
        self,
        verbose: bool = False,
        ignore_keys_on_save: List[str] = [],
        overwrite: bool = False,
        skip_execution: bool = False
    ) -> List[dict]:
        """Excute all tasks in the queue and get their results in order.

        Returns:
            List[dict]: Results of tasks in the same order. Each item contains 3 keys: `id` (int), `success` (bool) and `result` (defined by the task function).
        """
        result_list = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            uncomleted_tasks = []
            for task in self.tasks:                
                if not task["completed"]:
                    uncomleted_tasks.append(task)
                else:
                    if not overwrite:
                        if task['log_file'] not in self.task_result_cache:
                            with open(task['log_file'], 'r+', encoding='utf-8') as f:
                                if task['log_file'].endswith('.json'):
                                    results = [json.load(f)]
                                else:
                                    results = [json.loads(line) for line in f.readlines()]
                            self.task_result_cache[task['log_file']] = {result['id']: result for result in results}
                        
                        completed = False
                        result = None
                        if task['id'] in self.task_result_cache[task['log_file']]:
                            result = self.task_result_cache[task['log_file']][task['id']]
                            if result['success'] is True:
                                completed = True
                        
                        if not completed:
                            uncomleted_tasks.append(task)
                        else:
                            assert result is not None
                            result_list.append(result)
                    else:
                        task['completed'] = False
                        uncomleted_tasks.append(task)

            # Setup progress bar
            bar = tqdm(total=len(self.tasks))
            bar.update(len(self.tasks) - len(uncomleted_tasks))
            
            tasks = [executor.submit(self._task_fn, task, skip_execution=skip_execution) for task in uncomleted_tasks]
            for task in as_completed(tasks):
                result = task.result(timeout=self.timeout)

                # Logging
                log_file = result.pop("log_file")
                if log_file is not None:
                    result_to_save = copy.deepcopy(result)
                    for key in ignore_keys_on_save:
                        result_to_save['result'].pop(key, None)
                        
                    log = json.dumps(result_to_save, ensure_ascii=False)
                    if log_file.endswith('.json'):
                        with open(log_file, 'w+', encoding='utf-8') as f:
                            json.dump(result_to_save, fp=f, ensure_ascii=False, indent=4)
                    else:
                        with open(log_file, "a+", encoding="utf-8") as f:
                            f.write(log + "\n")
                    if verbose:
                        tqdm.write(log)

                # Mark task as completed
                if not self.mark_as_completed_on_failure and not result.get("success"):
                    pass
                else:
                    for task in self.tasks:
                        if task['id'] == result.get('id'):
                            task["completed"] = True

                result_list.append(result)
                
                bar.update(1)
                
        return sorted(result_list, key=lambda result: result["id"])
    
    def clear_tasks(self) -> None:
        """Clear all tasks added into the queue (including uncompleted ones).
        The indices of newly added tasks will start from 0.
        """
        self.tasks = []
        self.progress_cache = {}
        self.task_result_cache = {}

    def _task_fn(
        self,
        task: Dict,
        skip_execution: bool = False
    ) -> dict:
        success = False
        retry_counter = 0
        while not success and retry_counter <= self.max_retry_on_failure:
            try:
                fn = task.get('fn')
                fn_params = list(inspect.signature(fn).parameters.keys())
                if 'kwargs' in fn_params:
                    fn_kwargs = task.get('kwargs')
                else:
                    fn_kwargs = {}
                    for key, value in task.get('kwargs').items():
                        if key in fn_params:
                            fn_kwargs[key] = value
                if not skip_execution:
                    result = fn(*task.get('args'), **fn_kwargs)
                    success = True
                else:
                    result = None
                    break
            except Exception as e:
                tqdm.write(f"[!] Exception encountered when executing task {task.get('id')}, retry: {retry_counter} / {self.max_retry_on_failure}.\nError info: {e}\n[!]")
                retry_counter += 1
                logging.warning('=' * 200)
                logging.exception(e)
                logging.warning('=' * 200)
                
        if success:
            return {
                "id": task.get('id'),
                "success": success,
                "result": result,
                "log_file": task.get('log_file'),
                "args": task.get('args'),
                "kwargs": task.get('kwargs')
            }
        else:
            if self.log_warning_on_failure:
                logger.warning(
                    f"Function `{fn.__name__}` in thread {threading.get_ident()} aborted unexpectedly."
                )
            return {
                "id": task.get('id'),
                "success": success,
                "result": None,
                "log_file": task.get('log_file'),
                "args": task.get('args'),
                "kwargs": task.get('kwargs')
            }
