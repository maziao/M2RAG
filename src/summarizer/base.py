import re
from itertools import accumulate
from abc import abstractmethod
from typing import Dict, List, Tuple
from src.model import build_model
from src.summarizer import SUMMARIZER


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"
TEMP_IMAGE_PLACEHOLDER = "<TEMP_IMAGE_PLACEHOLDER>"


@SUMMARIZER.register_module
class BaseSummarizer:
    def __init__(
        self,
        name: str = None,
        tag: str = None,
        model_config: Dict = {},
        max_samples: int = -1,
        max_pieces: int = -1,
        max_pieces_per_sample: int = -1,
        min_piece_score: int = 0,
        max_tokens_per_sample: int = -1
    ) -> None:
        self.model = build_model(model_config)
        if name is not None:
            self.name = f"summarizer-{name}-{self.model.name}"
        else:
            self.name = f"summarizer-{self.model.name}"
            
        if tag is not None:
            self.name += f"-{tag}"
            
        # max number of webpages
        self.max_samples = max_samples
        # max number of text pieces
        self.max_pieces = max_pieces
        # max number of text pieces (per webpage)
        self.max_pieces_per_sample = max_pieces_per_sample
        # min score of reserved text pieces
        self.min_piece_score = min_piece_score
        
        # max number of tokens of each webpage
        self.max_tokens_per_sample = max_tokens_per_sample
    
    def _get_accum_tokens(self, text_list: List[str], reverse: bool = False) -> List[int]:
        num_tokens = [self.model.num_tokens_from_text(text) for text in text_list]
        if not reverse:
            num_tokens_accum = list(accumulate(num_tokens))
        else:
            num_tokens_accum = list(reversed(list(accumulate(list(reversed(num_tokens))))))
        return num_tokens_accum
    
    def filter_webpage_pieces(self, query_list: List[Dict]) -> List[Dict]:
        """Remove redundant webpage pieces and their corresponding images.

        Args:
            query_list (List[Dict]): query list sent to summarize

        Returns:
            List[Dict]: [
                {
                    'query_id': Any,
                    'webpage_id': Any,
                    'title': str,
                    'url': str,
                    'pieces': List[Dict],
                    'images': List[Dict],
                    'content': str
                },
                ...
            ]
        """
        # trim webpage pieces
        filtered_webpages = []
        for query in query_list:
            if self.max_samples is not None and self.max_samples > 0:
                webpages = query['webpages'][:self.max_samples]
            else:
                webpages = query['webpages']
            
            for webpage in webpages:
                webpage_pieces = webpage['cleaned_webpage_splits']
                if not isinstance(webpage_pieces, list):
                    continue
                
                # threshold filtering
                above_threshold_pieces = [piece for piece in webpage_pieces if piece['score'] >= self.min_piece_score]
                
                # top-k filtering
                if self.max_pieces_per_sample > 0:
                    top_k_indices = sorted([piece['split_id'] for piece in sorted(above_threshold_pieces, key=lambda s: s['score'], reverse=True)[:self.max_pieces_per_sample]])
                else:
                    top_k_indices = [piece['split_id'] for piece in above_threshold_pieces]
                top_k_pieces = [piece for piece in above_threshold_pieces if piece['split_id'] in top_k_indices]
                
                # remove images whose corresponding text pieces are removed
                webpage_images = webpage['images']
                for image in webpage_images:
                    if image['split_id'] not in top_k_indices:
                        image['valid'] = False
                reserved_images = [image for image in webpage_images if image['valid']]
                
                if len(top_k_pieces) > 0:
                    filtered_webpages.append({
                        'query_id': query['query_id'],
                        'webpage_id': webpage['webpage_id'],
                        'title': webpage['title'],
                        'url': webpage['url'],
                        'pieces': top_k_pieces,
                        'images': reserved_images,
                        'content': ''.join([piece['text'] for piece in top_k_pieces])
                    })
        
        # remove redundant pieces
        if self.max_pieces is not None and self.max_pieces > 0:
            piece_score_list = [(i, j, piece['score']) for i, webpage in enumerate(filtered_webpages) for j, piece in enumerate(webpage['pieces'])]
            sorted_piece_score_list = sorted(piece_score_list, key=lambda s:s[2], reverse=True)[:self.max_pieces]
            
            for i, webpage in enumerate(filtered_webpages):
                piece_id_list = sorted([j for (_i, j, _) in sorted_piece_score_list if _i == i])
                if len(piece_id_list) == 0:
                    continue
                webpage['pieces'] = [piece for j, piece in enumerate(webpage['pieces']) if j in piece_id_list]
                webpage['content'] = ''.join([piece['text'] for piece in webpage['pieces']])
        
        return filtered_webpages
    
    def remove_invalid_auxiliary_images(self, query_list: List[Dict]) -> List[Dict]:
        images = []
        for query in query_list:
            for aux_image in query['aux_images']:
                aux_image['aux_image_id'] = f"{query['query_id']}-{aux_image['image_id']}"
                if aux_image['valid']:
                    images.append(aux_image)
        return images
    
    @abstractmethod
    def filter_images_and_reorganize_webpages(self, webpages: List[Dict], aux_images: List[Dict] = []) -> Tuple:
        """Filter images and remove redundant placeholders in webpages.

        Args:
            webpages (List[Dict]): filtered webpages with only valid text and images
            aux_images (List[Dict], optional): valid auxiliary images. Defaults to [].

        Raises:
            NotImplementedError: _description_

        Returns:
            Tuple: webpages, aux_images
        """
        raise NotImplementedError
    
    @abstractmethod
    def construct_summarize_message(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = []) -> List[Dict]:
        """Construct summarize sample from query list for LLM

        Args:
            user_query (str): concatenated query str from user
            webpages (List[Dict]): webpage list
            aux_images (List[Dict]): list of auxiliary images

        Returns:
            messages: List[Dict]
        """
        raise NotImplementedError
    
    def handle_output_references(self, text: str, webpages: List[Dict]) -> str:
        text_splits = text.split('References')
        reference_str = text_splits[-1]
        reference_indices = [int(index) for index in re.findall(r"\d+", reference_str)]
        reference_indices = list(set(reference_indices))
        
        new_reference_str = ""
        reference_id = 1
        for index in reference_indices:
            if index < len(webpages):
                new_reference_str += f"{reference_id}. {webpages[index]['title']} - {webpages[index]['url']}\n"
                reference_id += 1
                
        # handle samples without references
        if len(text_splits) == 1:
            output = text_splits[0]
        else:
            output = 'References'.join(text_splits[:-1]).strip()
            new_splits = output.split('\n')
            if new_splits[-1].startswith('*') or new_splits[-1].startswith('#'):
                output = '\n'.join(new_splits[:-1])

            output += "\n## References\n" + new_reference_str
        return output, reference_id - 1
    
    def unwrap_markdown(self, text: str) -> str:
        start_index = text.find('```markdown')
        end_index = text[::-1].find('```')
        if start_index == -1:
            return text
        else:
            start_index += 11
        
        if end_index == -1 or end_index > 0.5 * len(text):
            end_index = None
        else:
            end_index = - (end_index + 3)
        
        return text[start_index: end_index].strip()
    
    def delete_images_from_text(self, text: str) -> str:
        candidate_images = re.findall(r"\!\[(.*?)\]\((.*?)\)", text)

        for image in candidate_images:
            orig_text = f"![{image[0]}]({image[1]})"
            replace_text = image[0]
            text = text.replace(orig_text, replace_text)
        return text
    
    def delete_links_from_text(self, text: str) -> str:
        candidate_links = re.findall(r"\[(.*?)\]\((.*?)\)", text)

        for link in candidate_links:
            orig_text = f"[{link[0]}]({link[1]})"
            replace_text = link[0]
            text = text.replace(orig_text, replace_text)
        return text
    
    def delete_image_placeholders_from_text(self, text: str) -> str:
        images_with_index = [int(index) for index in re.findall(r"<IMAGE_PLACEHOLDER>\[(\d+)\]", text)]
        for index in images_with_index:
            text = text.replace(f"{IMAGE_PLACEHOLDER}[{index}]", '')
        text = text.replace(IMAGE_PLACEHOLDER, '')
        return text
    
    @abstractmethod
    def summarize_core(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = [], **kwargs) -> Dict:
        """Get summarize result from user query, a list of cleaned webpages and valid auxiliary images.

        Args:
            user_query (str): concatenated user query string.
            webpages (List[Dict]): a list of cleaned webpages with valid images.
            aux_images (List[Dict], optional): valid auxiliary images. Defaults to [].

        Raises:
            NotImplementedError: This method should be implemented by subclasses.

        Returns:
            Dict: {
                "model_input": List[Dict],
                "model_response": str,
                "usage": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int
                },
                "processed_response": str,
                "placeholder_response": str,
                "output_images": List[Dict],
                "num_output_images": int,
                "num_reference_webpages": int
            }
        """
        raise NotImplementedError
    
    def summarize(self, query_list: List[Dict], **kwargs):
        """Get summary based on structured information.

        Args:
            query_list: [
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
                            "webpage_title": str,
                            "webpage_url": str,
                            "webpage_content": str,
                            ...
                        }
                    ]
                },
                ...
            ]
        """
        user_query = ' '.join([query['content'] for query in query_list])
        webpages = self.filter_webpage_pieces(query_list=query_list)
        aux_images = self.remove_invalid_auxiliary_images(query_list=query_list)
        webpages, aux_images = self.filter_images_and_reorganize_webpages(webpages=webpages, aux_images=aux_images)
        
        result = {
            "user_query": user_query,
            "webpages": webpages,
            "aux_images": aux_images,
            "model_input": None,
            "model_response": None,
            "usage": None,
            "processed_response": None,
            "placeholder_response": None,
            "output_images": None,
            "num_input_webpages": {
                "orig": sum([len(query['webpages']) for query in query_list]),
                "filtered": len(webpages),
                "total_char_len": sum([len(webpage['content']) for webpage in webpages])
            },
            "num_input_images": {
                "webpage_images": sum([len(webpage['images']) for webpage in webpages]),
                "aux_images": len(aux_images),
                "total_images": sum([len(webpage['images']) for webpage in webpages]) + len(aux_images)
            },
            "num_output_images": None,
            "num_reference_webpages": None
        }
        summarize_result = self.summarize_core(user_query=user_query, webpages=webpages, aux_images=aux_images, **kwargs)
        result.update(summarize_result)
        
        return result
