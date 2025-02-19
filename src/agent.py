import os
import json
import logging
from tqdm import tqdm
from typing import List, Dict, Optional
from src.crawler import build_crawler
from src.image_handler import build_image_handler
from src.search_engine import build_search_engine
from src.summarizer import build_summarizer
from src.webpage_cleaner import build_webpage_cleaner
from src.utils import MultithreadManager


class M2RAGAgent:
    def __init__(
        self,
        cache_dir: str,
        image_dir: str,
        search_engine_config: Dict,
        crawler_config: Dict,
        webpage_cleaner_config: Dict,
        image_handler_config: Dict,
        aux_image_handler_config: Dict,
        summarizer_config: Dict,
        multi_thread_config: Dict
    ) -> None:
        self.cache_dir = cache_dir
        self.image_dir = image_dir
        
        self.search_engine = build_search_engine(search_engine_config)
        self.crawler = build_crawler(crawler_config)
        self.webpage_cleaner = build_webpage_cleaner(webpage_cleaner_config)
        self.image_handler = build_image_handler(image_handler_config)
        self.aux_image_handler = build_image_handler(aux_image_handler_config)
        self.summarizer = build_summarizer(summarizer_config)
        self.manager = MultithreadManager(**multi_thread_config)
        
        self.session_counter = 0
        
    def _get_search_results(self, query_list: List[Dict], cache_dir: str, verbose: bool = False):
        logging.info(f"Step 1: searching webpages ...")
        for query in query_list:
            query_log_file = os.path.join(cache_dir, f"query-{query['query_id']}", "search_result.json")
            self.search_engine.manager.add_task(
                self.search_engine.search_single,
                query_log_file,
                query['query_id'],
                content=query['content']
            )

        search_results = self.search_engine.manager.execute_tasks()
        self.search_engine.manager.clear_tasks()
        logging.info(f"\tsearch completed: {[len(result['result']) if result['result'] is not None else 0 for result in search_results]} webpages")
        if verbose:
            for result in search_results:
                logging.info(f"\t\tsearch result for query '{result['kwargs']['content']}':")
                if result is not None:
                    for webpage in result['result']:
                        logging.info(f"\t\t\t{webpage['webpage_id']}. {webpage['title']} ({webpage['url']})")
                else:
                    logging.warning(f"\t\t\tno result webpages")
        
        logging.info(f"Step 2: searching auxiliary images ...")
        for query in query_list:
            aux_image_log_file = os.path.join(cache_dir, f"query-{query['query_id']}", "image_search_result.json")
            self.search_engine.manager.add_task(
                self.search_engine.search_image_single,
                aux_image_log_file,
                query['query_id'],
                content=query['content']
            )

        image_search_results = self.search_engine.manager.execute_tasks()
        self.search_engine.manager.clear_tasks()
        logging.info(f"\timage search completed: {[len(result['result']) if result['result'] is not None else 0 for result in image_search_results]} auxiliary images")
        if verbose:
            for result in image_search_results:
                logging.info(f"\t\timage search result for query '{result['kwargs']['content']}':")
                for image in result['result']:
                    logging.info(f"\t\t\t{image['image_id']}. {image['image_url']}")
        
        return search_results, image_search_results
    
    def _get_webpages(self, search_results: List[Dict], cache_dir: str) -> Dict:
        """Download webpages based on the given search_results (index).

        Args:
            search_results (List[Dict]): [
                {
                    'id': Any (query_id),
                    'success': bool,
                    'result': [
                        {
                            'webpage_id': Any,
                            'url': str,
                            ...
                        }
                        ...
                    ]
                },
                ...
            ]
            cache_dir (str): cache directory for this session

        Returns:
            Dict: {
                query_id_0: {
                    webpage_id_0: webpage_0,
                    ...
                },
                ...
            }
        """
        logging.info(f"Step 3: downloading webpages ...")
        for search_result in search_results:
            if search_result['success']:
                query_log_dir = os.path.join(cache_dir, f"query-{search_result['id']}")
                for webpage_meta in search_result['result']:
                    webpage_log_dir = os.path.join(query_log_dir, f"webpage-{webpage_meta['webpage_id']}")
                    webpage_raw_file = os.path.join(webpage_log_dir, 'raw.json')
                    self.crawler.manager.add_task(
                        self.crawler.download_webpage,
                        webpage_raw_file,
                        webpage_meta['webpage_id'],
                        url=webpage_meta['url'],
                        query_id=search_result['id']
                    )
        
        webpage_results = self.crawler.manager.execute_tasks()
        self.crawler.manager.clear_tasks()
        logging.info(f"\twebpage downloading completed: {sum([webpage['success'] and webpage['result'] is not None for webpage in webpage_results])} / {len(webpage_results)}")

        webpage_dict = {}
        for webpage in webpage_results:
            query_id = webpage['kwargs']['query_id']
            if query_id in webpage_dict:
                webpage_dict[query_id][webpage['id']] = webpage
            else:
                webpage_dict[query_id] = {webpage['id']: webpage}
        
        return webpage_dict
    
    def _get_search_results_and_webpages(self, query_list: List[Dict], cache_dir: str):
        """_summary_

        Args:
            query_list: [
                {
                    "query_id": Any,
                    "content": str
                },
                ...
            ]

        Return:
            [
                {
                    "query_id": Any,
                    "content": str,
                    "success": bool,
                    "webpages": [
                        {
                            "webpage_id": Any,
                            "url": str,
                            "success": bool,
                            "title": str,
                            "content": str
                        },
                        ...
                    ],
                    "aux_images": [
                        {
                            "image_id": Any,
                            "image_url": str,
                            "cached_image_url": str,
                            "valid": bool,
                            ...
                        }
                    ]
                },
                ...
            ]
        """
        search_results, image_search_results = self._get_search_results(query_list=query_list, cache_dir=cache_dir)
        crawled_webpages = self._get_webpages(search_results=search_results, cache_dir=cache_dir)
        
        results = []
        for search_result, image_search_result in zip(search_results, image_search_results):
            query_id = search_result['id']
            query_result = {
                "query_id": query_id,
                "content": search_result['kwargs']['content'],
                "success": search_result['success'],
            }
            if search_result['success']:
                for webpage_meta in search_result['result']:
                    crawled_webpage = crawled_webpages[query_id][webpage_meta['webpage_id']]
                    if crawled_webpage['success'] and crawled_webpage['result'] is not None:
                        assert 'title' in crawled_webpage['result'] and 'content' in crawled_webpage['result']
                        assert isinstance(crawled_webpage['result']['title'], str) and isinstance(crawled_webpage['result']['content'], str)
                        webpage_meta.update({
                            "success": True,
                            **crawled_webpage['result']
                        })
                    else:
                        webpage_meta.update({
                            "success": False,
                            "title": None,
                            "content": None
                        })
                query_result['webpages'] = search_result['result']
            else:
                query_result['webpages'] = []
                
            if image_search_result['success']:
                query_result['aux_images'] = image_search_result['result']
            else:
                query_result['aux_images'] = []
            results.append(query_result)
        
        return results
        
    def _clean_text(self, search_results: List[Dict], session_cache_dir: str, verbose: bool = False):
        logging.info(f"Step 4: cleaning webpages ...")
        for query in search_results:
            for webpage in query['webpages']:
                if webpage['success']:
                    self.manager.add_task(
                        self.webpage_cleaner.clean,
                        os.path.join(session_cache_dir, f"query-{query['query_id']}", f"webpage-{webpage['webpage_id']}", 'clean_webpage.json'),
                        f"{query['query_id']}-{webpage['webpage_id']}",
                        text=webpage['content'],
                        ref_text=webpage['title'],
                        cache_file=os.path.join(session_cache_dir, f"query-{query['query_id']}", f"webpage-{webpage['webpage_id']}", 'clean_webpage_pieces.jsonl')
                    )
        
        image_counter = 0
        webpage_piece_counter = 0
        
        results = self.manager.execute_tasks()
        self.manager.clear_tasks()
        
        for query in search_results:
            for webpage in query['webpages']:
                task_id = f"{query['query_id']}-{webpage['webpage_id']}"
                for result in results:
                    if result['id'] == task_id:
                        if result['success']:
                            webpage.update(**result['result'])
                            image_counter += len(result['result']['images'])
                            webpage_piece_counter += len(result['result']['cleaned_webpage_splits'])
                        else:
                            webpage.update({
                                "char_len_trend": None,
                                "cleaned_webpage_splits": [],
                                "images": []
                            })
                        break
        
        if verbose:
            logging.info(f"\twebpage cleaning complted, success: {sum([result['success'] for result in results]) / {len(results)}}, text pieces: {webpage_piece_counter}, extracted images: {image_counter}")
        return search_results
    
    def _handle_images(self, search_results: List[Dict], session_cache_dir: str, session_image_dir: str):
        logging.info(f"Step 5: handling images ...")
        # handle images in the webpages
        for query in search_results:
            for webpage in query['webpages']:
                cleaned_webpage_content = ''
                for split in webpage['cleaned_webpage_splits']:
                    cleaned_webpage_content += split['text']
                
                webpage['images'] = self.image_handler.process(
                    cleaned_webpage_content=cleaned_webpage_content,
                    images=webpage['images'],
                    image_relative_dir=os.path.join(session_image_dir, f"query-{query['query_id']}", f"webpage-{webpage['webpage_id']}"),
                    ref_text=query['content'],
                    cache_file=os.path.join(session_cache_dir, f"query-{query['query_id']}", f"webpage-{webpage['webpage_id']}", 'handle_images.jsonl')
                )
        
        # handle auxiliary images
        for query in search_results:
            query['aux_images'] = self.aux_image_handler.process(
                images=query['aux_images'],
                image_relative_dir=os.path.join(session_image_dir, f"query-{query['query_id']}", f"aux_images"),
                ref_text=query['content'],
                cache_file=os.path.join(session_cache_dir, f"query-{query['query_id']}", 'handle_aux_images.jsonl')
            )
        
        return search_results
    
    def search_pipeline(self, topics: List[Dict], session_id: Optional[int] = None):
        if session_id is not None:
            session_cache_dir = os.path.join(self.cache_dir, f"session-{session_id}")
            session_image_dir = os.path.join(self.image_dir, f"session-{session_id}")
        else:
            session_cache_dir = os.path.join(self.cache_dir, f"session-{self.session_counter}")
            session_image_dir = os.path.join(self.image_dir, f"session-{self.session_counter}")
            self.session_counter += 1
        
        # search
        search_results = self._get_search_results_and_webpages(query_list=topics, cache_dir=session_cache_dir)    
        if all([not query['success'] for query in search_results]):
            raise ValueError(f"No valid search result.")
        
        # clean text
        cleaned_search_results = self._clean_text(search_results=search_results, session_cache_dir=session_cache_dir)
        
        # handle_images
        image_handled_search_results = self._handle_images(
            search_results=cleaned_search_results,
            session_cache_dir=session_cache_dir,
            session_image_dir=session_image_dir
        )
        
        # summarize
        logging.info(f"Step 6: summarizing ...")
        session_result_dir = os.path.join(session_cache_dir, self.summarizer.name)
        if not os.path.exists(session_result_dir):
            os.makedirs(session_result_dir, exist_ok=True)
        summarize_log_file = os.path.join(session_result_dir, 'result.json')
        summary_markdown_file = os.path.join(session_result_dir, 'result.md')
        
        if not os.path.exists(summarize_log_file):
            summary = self.summarizer.summarize(query_list=image_handled_search_results)
            
            with open(summarize_log_file, 'w+', encoding='utf-8') as f:
                json.dump(obj=summary, fp=f, ensure_ascii=False, indent=4)
                
            with open(summary_markdown_file, 'w+', encoding='utf-8') as f:
                f.write(summary['processed_response'])
        else:
            with open(summarize_log_file, 'r+', encoding='utf-8') as f:
                log = json.load(f)
            summary = log['processed_response']
        
        return summary['processed_response']
