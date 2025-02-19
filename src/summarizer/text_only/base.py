
import logging
from typing import Dict, List, Tuple
from src.summarizer import SUMMARIZER
from src.summarizer.base import BaseSummarizer


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"
TEMP_IMAGE_PLACEHOLDER = "<TEMP_IMAGE_PLACEHOLDER>"


LLM_TEXT_ONLY_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a Q&A assistant. Your role is to answer a user's question using information from relevant webpages. These resources are derived from search engine results and are meant to provide a comprehensive summary.

# Input Data
1. **Question**: This is the user's query and serves as the focus of your summary.
2. **Webpages**: A list of webpages provided in a markdown-format code block, each starting with a title "## Webpage k" (k indicates the index starting from 0).

# Guidelines
1. **Understand the Question**: Determine which content from the webpages best answers the user's question.
2. **Text Summary**: Create a coherent markdown-formatted text document, organizing it into chapters and paragraphs as necessary.
3. **References**: Conclude the document with a 'References' section listing the indexes of webpages used, separated by spaces.
    - Example:
        ```markdown
        # References
        1 4 5
        ```

# Precautions
1. Ensure the final output is in markdown format for proper parsing.
2. Use authoritative expressions from the webpages as necessary but summarize and exclude irrelevant information.

# Question
{text_prompt}

# Webpages
"""


@SUMMARIZER.register_module
class TextOnlySummarizer(BaseSummarizer):
    def __init__(
        self,
        name: str = 'text_only',
        tag: str = None,
        model_config: Dict = {},
        max_samples: int = -1,
        max_pieces: int = -1,
        max_pieces_per_sample: int = -1,
        min_piece_score: int = 0,
        max_tokens_per_sample: int = -1
    ) -> None:
        super().__init__(
            name=name,
            tag=tag,
            model_config=model_config,
            max_samples=max_samples,
            max_pieces=max_pieces,
            max_pieces_per_sample=max_pieces_per_sample,
            min_piece_score=min_piece_score,
            max_tokens_per_sample=max_tokens_per_sample
        )
        
    def filter_images_and_reorganize_webpages(self, webpages: List[Dict], aux_images: List[Dict] = []) -> Tuple:
        for webpage in webpages:
            webpage['images'] = []
            webpage['content'] = self.delete_image_placeholders_from_text(text=webpage['content'])
            del webpage['pieces']
        
        return webpages, []
    
    def construct_summarize_message(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = []) -> List[Dict]:
        if len(aux_images) > 0:
            logging.warning(f"`TextOnlySummarizer` does not take images as input, but got {len(aux_images)} auxiliary images. Ignore these images ...")
        
        messages = [
            {
                "type": "text",
                "text": LLM_TEXT_ONLY_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query)
            }
        ]
        for i, webpage in enumerate(webpages):
            text = webpage['content'].replace(IMAGE_PLACEHOLDER, '')
            messages.append({
                "type": "text",
                "text": f"## Webpage {i}\n```markdown\n{text}\n```"
            })
        
        messages.append({
            "type": "text",
            "text": "# Generated Document"
        })
        
        return messages
    
    def summarize_core(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = [], dry_run: bool = False, **kwargs) -> Dict:
        messages = self.construct_summarize_message(user_query=user_query, webpages=webpages, aux_images=aux_images)
        
        if dry_run:
            output = {
                "model_input": messages,
                "model_response": None,
                "usage": sum([self.model.num_tokens_from_text(text=message['text']) for message in messages if message['type'] == 'text']),
                "processed_response": None,
                "placeholder_response": None,
                "output_images": None,
                "num_output_images": None,
                "num_reference_webpages": None
            }
        else:
            result = self.model.chat(messages=messages)

            placeholder_response = self.unwrap_markdown(text=result['response'])
            processed_response, num_references = self.handle_output_references(text=placeholder_response, webpages=webpages)
            
            output = {
                "model_input": messages,
                "model_response": result['response'],
                "usage": result['usage'],
                "processed_response": processed_response,
                "placeholder_response": placeholder_response,
                "output_images": None,
                "num_output_images": None,
                "num_reference_webpages": num_references
            }
        return output
