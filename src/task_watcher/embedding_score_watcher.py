import logging
from tqdm import tqdm
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
    logging.warning(f"module `sentence_transformers` is not currently imported, `SentenceTransformer` cannot be used.")
from src.task_watcher import TASK_WATCHER
from src.task_watcher.watcher import TaskWatcher


@TASK_WATCHER.register_module
class EmbeddingScoreWatcher(TaskWatcher):
    def __init__(
        self,
        mongodb_url: str,
        database_name: str,
        embedding_model_path_or_id: str
    ) -> None:
        super().__init__(mongodb_url=mongodb_url, database_name=database_name)
        self.query_table = self.database['queries']
        self.webpage_table = self.database['webpages']
        
        self.model = SentenceTransformer(embedding_model_path_or_id)
        
    def _watch_single_cycle(self, *args, **kwargs):
        num_webpages_to_be_scored = 0
        for webpage in self.webpage_table.find({'webpage_label': True, 'clean_label': True}, {'query_id': 1, 'cleaned_webpage_splits': 1}):
            webpage_pieces = webpage['cleaned_webpage_splits']
            if not all(['embedding_score' in piece and piece['embedding_score'] is not None for piece in webpage_pieces]):
                num_webpages_to_be_scored += 1
        
        counter = 0
        for webpage in tqdm(self.webpage_table.find({'webpage_label': True, 'clean_label': True}, {'query_id': 1, 'cleaned_webpage_splits': 1}), total=num_webpages_to_be_scored):
            webpage_pieces = webpage['cleaned_webpage_splits']
            if not all(['embedding_score' in piece and piece['embedding_score'] is not None for piece in webpage_pieces]):
                query_content = self.query_table.find_one({'query_id': webpage['query_id']}, {'query_content': 1})['query_content']
                
                pieces = [piece['text'] for piece in webpage_pieces]
    
                embeddings = self.model.encode([query_content] + pieces)
                sim = self.model.similarity(embeddings[0:1], embeddings[1:]).numpy().tolist()[0]
                
                for i in range(len(pieces)):
                    webpage_pieces[i]['embedding_score'] = sim[i]
                
                self.webpage_table.update_one({'_id': webpage['_id']}, {'$set': {'cleaned_webpage_splits': webpage_pieces}})
                counter += 1
        return counter, counter
