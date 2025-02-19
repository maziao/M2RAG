import re
from typing import List, Dict, Optional
from src.model import build_model
from src.image_handler.image_scorer import IMAGE_SCORER
from src.image_handler.image_scorer.image_scorer import ImageScorer


VLM_SCORE_PROMPT_TEMPLATE = """# Task Description
You are an assistant for evaluating the correlation between image and text. You will be provided with an image and a question. The image comes from a webpage, so it may be a highly relevant image to the provided text, a slightly relevant image, or a completely unrelated image, or even an advertising image.

# Scoring Strategy
Score from 0 to 10 based on the correlation between the image content and the problem, with higher scores indicating higher correlation.
- 0 represents that the image is completely unrelated to the question, or the image is an advertising image.
- 1-3 represents that the image is slightly related to the question, and is dispensable for answering the question.
- 4-6 represents that the image is moderately related to the question, and it is helpful for answering some of the content in the question.
- 7-9 represents that the image is highly related to the question, and it is very helpful for answering the question.
- 10 represents that the image is the key to the question, and it is essential for answering the question.

# Precautions
You are not expected to generate any other information other than the score (an integer).

# Question
{question}

# Score
"""


@IMAGE_SCORER.register_module
class VLMImageScorer(ImageScorer):
    def __init__(
        self,
        filter_config: dict = {},
        enable_cache: bool = False,
        name: Optional[str] = None,
        image_url: Optional[str] = None,
        image_root: Optional[str] = None,
        model_config: Dict = {}
    ) -> None:
        self.model = build_model(model_config)
        assert self.model.name.startswith('vlm'), f"the model for image scoring must be a VLM, but got {self.model.name}"
        
        super().__init__(
            filter_config=filter_config,
            enable_cache=enable_cache,
            name=self.model.name if name is None else name,
            image_url=image_url,
            image_root=image_root
        )
        
    def _get_score(self, text: str):
        indices = [int(index) for index in re.findall(r"\d+", text)]
        if len(indices) == 0:
            return None
        score = indices[0]
        score = max(0, min(10, score))
        return score
    
    def get_image_score_single(self, image_url: str, reference_text: Optional[str] = None, **kwargs) -> float:
        messages = [
            {
                'type': 'image_url',
                'image_url': {
                    'url': image_url
                }
            },
            {
                'type': 'text',
                'text': VLM_SCORE_PROMPT_TEMPLATE.format(question=reference_text)
            }
        ]
        response = self.model.chat(messages=messages)

        if response is not None:
            return {
                "score": self._get_score(text=response['response']),
                "usage": response['usage']
            }
        else:
            return None
    
    def _get_image_scores(self, image_urls: List[str], reference_text: Optional[str] = None) -> List[float]:
        return [self.get_image_score_single(image_url=image_url, reference_text=reference_text) for image_url in image_urls]
