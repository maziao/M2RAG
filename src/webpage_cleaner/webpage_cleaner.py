import re
import logging
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.webpage_cleaner import WEBPAGE_CLEANER
from src.utils import MultithreadManager
from src.model import build_model


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"


CLEAN_WEBPAGE_PROMPT_TEMPLATE = """# Task Description
You are a useful text evaluation assistant. You will be provided with a user query and a piece of text extracted from a relevant webpage. Please evaluate whether this text contains useful information to address the user query.

# Input Data
1. User Query
2. Text from a Webpage

# Evaluation Guidelines
Assign a score from 0 to 10, where a higher score indicates better alignment:
- 0: Entirely irrelevant to the query.
- 1-3: Minimal relevance; most content deviates from the query.
- 4-6: Moderately relevant; some content aligns with the query, but includes notable digressions.
- 7-9: Highly relevant; most content relates directly to the query with minor digressions.
- 10: Perfectly relevant; fully aligns with the query without any digressions.

# User Query
{query}

# Webpage Text
```markdown
{webpage_piece}
```

# Precautions
If there is an <IMAGE_PLACEHOLDER> in the webpage text, ignore it in your evaluation.

# Output Format
Your output should consist of two lines:
1. A brief analysis of the text's relevance to the query.
2. An integer score from 0 to 10.

# Output
"""


@WEBPAGE_CLEANER.register_module
class WebpageCleaner:
    def __init__(
        self,
        model_config: Dict = {},
        chunk_size: int = 1000,
        max_chunks: int = -1,
        multi_thread_config: Dict = {}
    ) -> None:
        self.model = build_model(model_config)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        self.max_chunks = max_chunks
        self.multi_thread_config = multi_thread_config
    
    def extract_images_from_text(self, text: str) -> Dict:
        candidate_images = re.findall(r"\!\[(.*?)(\d+)(.*?)\]\((.*?)\)", text)
        reserved_images = []
        for image in candidate_images:
            if image[0].strip() == 'Image':
                try:
                    image_index = int(image[1])
                    image_url = image[3]
                    reserved_images.append({
                        'image_id': image_index,
                        'image_url': image_url
                    })
                    orig_text = f"![{image[0]}{image[1]}{image[2]}]({image[3]})"
                    replace_text = f"{IMAGE_PLACEHOLDER}[{image_index}]"
                    text = text.replace(orig_text, replace_text)
                except Exception:
                    continue
        return {
            'text': text,
            'images': reserved_images
        }
        
    def delete_links_from_text(self, text: str) -> Dict:
        candidate_links = re.findall(r"\[(.*?)\]\((.*?)\)", text)

        for link in candidate_links:
            orig_text = f"[{link[0]}]({link[1]})"
            replace_text = link[0]
            text = text.replace(orig_text, replace_text)
        return text
    
    def get_image_indices_from_text(self, text: str) -> List:
        image_indices = re.findall(r"<IMAGE_PLACEHOLDER>\[(\d+)\]", text)
        return sorted([int(index) for index in image_indices])
    
    def score_webpage_text(self, text: str, ref_text: str):
        def _get_score(text: str):
            scores = [int(score) for score in re.findall(r"\d+", text)]
            if len(scores) == 0:
                return 0
            return max(0, min(10, scores[-1]))
        
        prompt = CLEAN_WEBPAGE_PROMPT_TEMPLATE.format(query=ref_text, webpage_piece=text)
        result = self.model.chat(
            messages=[{'type': 'text', 'text': prompt}]
        )
        result['score'] = _get_score(text=result['response'])
        return result
        
    def clean(self, text: str, ref_text: str, cache_file: str = None, **kwargs):
        """Clean raw text with a title

        Args:
            text (str): text to be cleaned
            ref_text (str): reference text for webpage cleaning
        """
        temp_manager = MultithreadManager(**self.multi_thread_config)
        
        char_len_trend = {'raw': len(text)}
        
        extract_result = self.extract_images_from_text(text=text)
        char_len_trend['image_replaced'] = len(extract_result['text'])
        
        text_without_link = self.delete_links_from_text(text=extract_result['text'])
        char_len_trend['link_replaced'] = len(text_without_link)
        
        text_splits = self.text_splitter.split_text(text=text_without_link)
        if self.max_chunks > 0 and len(text_splits) > self.max_chunks:
            logging.warning(f"number of webpage pieces is {len(text_splits)}, exceeded the maximum number {self.max_chunks}, truncate and retain the first {self.max_chunks} pieces.")
            text_splits = text_splits[:self.max_chunks]
        
        for i, text_split in enumerate(text_splits):
            temp_manager.add_task(
                self.score_webpage_text,
                cache_file,
                i,
                text=text_split,
                ref_text=ref_text
            )
        
        results = temp_manager.execute_tasks()
        temp_manager.clear_tasks()
        del temp_manager
        
        sorted_results = sorted(results, key=lambda s: s['id'])
        scored_text_splits = []
        images = extract_result['images']
        for result in sorted_results:
            text_split_image_indices = self.get_image_indices_from_text(text=result['kwargs']['text'])
            scored_text_splits.append({
                'split_id': result['id'],
                'text': result['kwargs']['text'],
                'score': result['result']['score'],
                'evaluation': result['result']['response'],
                'image_ids': text_split_image_indices
            })
            
            for image in images:
                if image['image_id'] in text_split_image_indices:
                    image['split_id'] = result['id']
        
        for image in images:
            if 'split_id' not in image:
                image['split_id'] = None
                logging.warning(f"image does not have corresponding text split, set it to None")
        
        return {
            "char_len_trend": char_len_trend,
            "cleaned_webpage_splits": scored_text_splits,
            "images": images
        }
