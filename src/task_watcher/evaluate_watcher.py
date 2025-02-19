from typing import Optional, Tuple, List, Any
from src.utils import MultithreadManager
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher
from src.evaluator import build_evaluator


@TASK_WATCHER.register_module
class EvaluateWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: Optional[str] = None,
        database_name: Optional[str] = None,
        batch_size: int = 0,
        evaluator_config: dict = {},
        metrics: List[str] = None,
        summarizer_id: str = None,
        summarizer_regex: str = None,
        force_rerun: bool = False,
        multi_thread_config: dict = {}
    ) -> None:
        super().__init__(mongodb_url, database_name, batch_size)
        self.query_table = self.database['queries']
        self.summary_table = self.database['summaries']
        self.evaluator = build_evaluator(evaluator_config)
        self.metrics = metrics
        
        assert summarizer_id is None or summarizer_regex is None, f"`summarizer_id` and `summarizer_regex` cannot be set simultaneously."
        self.summarizer_id = summarizer_id
        self.summarizer_regex = summarizer_regex
        self.force_rerun = force_rerun
        self.completed_cache = {}
        self.manager = MultithreadManager(**multi_thread_config)
        
    def _get_evaluate_samples(self):
        filter_dict = {'summarize_label': True}
        if self.summarizer_id is not None:
            filter_dict['summarizer_id'] = self.summarizer_id
        elif self.summarizer_regex is not None:
            filter_dict['summarizer_id'] = {'$regex': self.summarizer_regex}
        
        samples = []
        if self.metrics is not None:
            for summary in self.summary_table.find(filter_dict):
                completed_metrics = []
                if 'evaluate_scores' in summary and summary['evaluate_scores'] is not None:
                    if 'text' in summary['evaluate_scores']:
                        completed_metrics += list(summary['evaluate_scores']['text'].keys())
                    if 'multi_modal' in summary['evaluate_scores']:
                        completed_metrics += list(summary['evaluate_scores']['multi_modal'].keys())
                
                uncompleted_metrics = list(set(self.metrics) - set(completed_metrics))
                
                completed = summary['summarizer_id'] in self.completed_cache and summary['query_id'] in self.completed_cache[summary['summarizer_id']]
                if len(uncompleted_metrics) > 0 or (self.force_rerun and not completed):
                    samples.append(summary)
                    if len(samples) >= self.batch_size:
                        break
        else:
            filter_dict['evaluate_label'] = False
            for summary in self.summary_table.find(filter_dict, limit=self.batch_size):
                samples.append(summary)
            
        return samples
        
    def _watch_single_cycle(self, *args, **kwargs) -> Tuple[int, int]:
        samples = self._get_evaluate_samples()
        
        for summary in samples:
            query_content = self.query_table.find_one({'query_id': summary['query_id']}, {'query_content': 1})['query_content']
            self.manager.add_task(
                self.evaluator.evaluate,
                None,
                f"{summary['query_id']}-{summary['summarizer_id']}",
                query_id=summary['query_id'],
                summarizer_id=summary['summarizer_id'],
                query=query_content,
                webpages=summary['webpages'],
                aux_images=summary['aux_images'],
                raw_summary=summary['placeholder_response'],
                output_images=summary['output_images'],
                metrics=self.metrics
            )
        
        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        for result in results:
            if result['success'] and result['result'] is not None:
                filter_dict = {'query_id': result['kwargs']['query_id'], 'summarizer_id': result['kwargs']['summarizer_id']}
                
                if result['kwargs']['summarizer_id'] not in self.completed_cache:
                    self.completed_cache[result['kwargs']['summarizer_id']] = []
                self.completed_cache[result['kwargs']['summarizer_id']].append(result['kwargs']['query_id'])
                
                summary = self.summary_table.find_one(filter_dict, {'evaluate_result': 1, 'evaluate_scores': 1})
                orig_result = summary['evaluate_result'] if summary['evaluate_result'] is not None else {'text': {}, 'multi_modal': {}}
                orig_scores = summary['evaluate_scores'] if summary['evaluate_scores'] is not None else {'text': {}, 'multi_modal': {}}
                new_result = result['result']['evaluate_result']
                new_scores = result['result']['evaluate_scores']
                
                orig_result['text'].update(new_result['text'])
                orig_result['multi_modal'].update(new_result['multi_modal'])
                orig_scores['text'].update(new_scores['text'])
                orig_scores['multi_modal'].update(new_scores['multi_modal'])
                
                self.summary_table.update_one(
                    filter=filter_dict,
                    update={
                        '$set': {
                            'evaluate_label': True,
                            'evaluate_result': orig_result,
                            'evaluate_scores': orig_scores
                        }
                    }
                )
        
        return sum([int(result['result'] is not None and result['success']) for result in results]), len(results)
    