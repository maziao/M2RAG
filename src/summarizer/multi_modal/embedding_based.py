import os
import re
import logging
import requests
import numpy as np
from typing import Dict, List, Tuple
from src.summarizer import SUMMARIZER
from src.summarizer.multi_modal.base import MultiModalSummarizer, IMAGE_PLACEHOLDER


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"
TEMP_IMAGE_PLACEHOLDER = "<TEMP_IMAGE_PLACEHOLDER>"


MURAR_SUMMARY_PROMPT_TEMPLATE = """# Task Description
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


MURAR_REFINE_PROMPT_TEMPLATE = """# Task Description
You are a skilled text refinement assistant. Your task is to enhance a textual summary that includes image placeholders, aligning it with a user's question for a cohesive multi-modal presentation.

# Input Details
1. **Question**: This is the user's query, guiding the focus of your refined summary.
2. **Summary**: This original summary includes image placeholders formatted as:
    ```markdown
    ![DETAILED_IMAGE_CAPTION](IMAGE_URL)
    ```
    - **DETAILED_IMAGE_CAPTION**: Describes the image content.
    - **IMAGE_URL**: The URL where the image is hosted.

# Guidelines
1. Present the refined output in markdown format to maintain a consistent structure.
2. Retain the essential content of the original summary while smoothly integrating and referencing all images.
3. Ensure each image placeholder is preserved, adhering strictly to its original format without alterations.
4. Ensure the text references or connects to the image explicitly.
5. If no image placeholders are present in the input, do not create any new ones.

# Question
{text_prompt}

# Summary
"""


IMAGE_STRUCT = """

<figure align=center>
    <img
        src="{image_url}"
        width="{width}%"
    />
</figure>

"""


IMAGE_STRUCT_WITH_CAPTION = """

<figure align=center>
    <img
        src="{image_url}"
        width="{width}%"
    />
    <figcaption>
        {image_caption}
    </figcaption>
</figure>

"""


@SUMMARIZER.register_module
class MuRARSummarizer(MultiModalSummarizer):
    def __init__(
        self,
        tag: str = None,
        model_config: Dict = {},
        max_samples: int = -1,
        max_pieces: int = -1,
        max_pieces_per_sample: int = -1,
        min_piece_score: int = 0,
        max_tokens_per_sample: int = -1,
        max_images: int = -1,
        max_images_per_sample: int = -1,
        max_aux_images: int = -1,
        max_image_score: int = None,
        min_image_score: int = 0,
        output_detailed_caption: bool = False,
        embedding_service_url: str = None,
        embedding_sim_threshold: float = 0.0
    ) -> None:
        super().__init__(
            name='murar',
            tag=tag,
            model_config=model_config,
            max_samples=max_samples,
            max_pieces=max_pieces,
            max_pieces_per_sample=max_pieces_per_sample,
            min_piece_score=min_piece_score,
            max_images_per_sample=max_images_per_sample,
            max_images=max_images,
            max_image_score=max_image_score,
            min_image_score=min_image_score,
            max_tokens_per_sample=max_tokens_per_sample,
            max_aux_images=max_aux_images,
            output_detailed_caption=output_detailed_caption
        )
        
        if embedding_service_url is None:
            embedding_service_url = os.environ.get('EMBEDDING_SERVICE_URL')
        assert embedding_service_url is not None
        self.embedding_service_url = embedding_service_url
        self.embedding_sim_threshold = embedding_sim_threshold
    
    def filter_images_and_reorganize_webpages(self, webpages: List[Dict], aux_images: List[Dict] = []) -> Tuple:
        if len(aux_images) > 0:
            logging.warning(f"`MuRARSummarizer` does not take auxiliary images as input, but got {len(aux_images)} auxiliary images. Ignore these images ...")
        for webpage in webpages:
            webpage['content'] = self.delete_image_placeholders_from_text(text=webpage['content'])
            webpage_image_indices = [image['image_id'] for image in webpage['images']]
            for piece in webpage['pieces']:
                image_indices = [int(index) for index in re.findall(r"<IMAGE_PLACEHOLDER>\[(\d+)\]", piece['text'])]
                valid_image_indices = [index for index in image_indices if index in webpage_image_indices]
                piece['image_indices'] = valid_image_indices
        return webpages, []
    
    def construct_summarize_message_stage_1(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = []) -> List[Dict]:
        if len(aux_images) > 0:
            logging.warning(f"`MuRARSummarizer` does not take images as input in stage 1, but got {len(aux_images)} auxiliary images. Ignore these images ...")
        
        messages = [
            {
                "type": "text",
                "text": MURAR_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query)
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
    
    def retrieve_relevant_images_stage_2(self, summary_pieces: List[str], webpages: List[str]) -> List[Dict]:
        selection_result_list = [{'text': piece, 'image': None} for piece in summary_pieces]
        
        non_empty_summary_pieces = []
        for i, piece in enumerate(summary_pieces):
            if len(piece.strip()) > 0:
                non_empty_summary_pieces.append({
                    '_piece_id': i,
                    'text': piece
                })
        
        webpage_pieces_with_images = []
        for i, webpage in enumerate(webpages):
            for j, piece in enumerate(webpage['pieces']):
                if len(piece['image_indices']) > 0:
                    webpage_pieces_with_images.append({
                        '_webpage_id': i,
                        '_split_id': j,
                        'text': piece['text']
                    })
        if len(webpage_pieces_with_images) == 0:
            return selection_result_list
        
        request_json = {
            'sentence_list_1': [piece['text'] for piece in non_empty_summary_pieces],
            'sentence_list_2': [piece['text'] for piece in webpage_pieces_with_images]
        }
        
        response = requests.post(
            self.embedding_service_url,
            json=request_json,
            headers={'content-type': 'application/json'}
        ).json()
        assert response['success']
        
        sim_matrix = np.array(response['scores'])
        best_match_indices = np.argmax(sim_matrix, axis=1).tolist()
        assert len(non_empty_summary_pieces) == len(best_match_indices)
        
        # ignore summary pieces below threshold
        for i in range(len(best_match_indices)):
            if sim_matrix[i][best_match_indices[i]] < self.embedding_sim_threshold:
                best_match_indices[i] = -1
        
        for i in range(len(webpage_pieces_with_images)):
            matched_summary_pieces = [non_empty_summary_pieces[summary_index] for summary_index, piece_index in enumerate(best_match_indices) if piece_index == i]
            if len(matched_summary_pieces) == 0:
                continue
            
            webpage_index = webpage_pieces_with_images[i]['_webpage_id']
            split_index = webpage_pieces_with_images[i]['_split_id']
            context = webpage_pieces_with_images[i]['text']
            webpage_images = webpages[webpage_index]['images']
            split_image_indices = webpages[webpage_index]['pieces'][split_index]['image_indices']
            
            images = []
            for index in split_image_indices:
                success = False
                for image in webpage_images:
                    if image['image_id'] == index:
                        images.append(image)
                        success = True
                        break
                assert success is True
            
            request_json = {
                'sentence_list_1': [piece['text'] for piece in matched_summary_pieces],
                'sentence_list_2': [f"Image caption: {image['detailed_image_caption']}\nImage context: {context}" for image in images]
            }
            
            response = requests.post(
                self.embedding_service_url,
                json=request_json,
                headers={'content-type': 'application/json'}
            ).json()
            assert response['success']
            
            sim_matrix = np.array(response['scores'])
            
            # assign match result
            best_match_indices = np.argmax(sim_matrix, axis=1).tolist()
            best_match_sims = np.max(sim_matrix, axis=1).tolist()
            for j, image in enumerate(images):
                image_match_pairs = [(k, index, sim) for k, (index, sim) in enumerate(zip(best_match_indices, best_match_sims)) if index == j]
                if len(image_match_pairs) > 0:
                    sorted_image_match_pairs = sorted(image_match_pairs, key=lambda s: s[-1], reverse=True)
                    piece_index = matched_summary_pieces[sorted_image_match_pairs[0][0]]['_piece_id']
                    selection_result_list[piece_index]['image'] = image
        
        return selection_result_list

    def construct_summarize_message_stage_3(self, user_query: str, text_summary_pieces_with_retrieved_images: List[Dict]) -> List[Dict]:
        messages = [
            {
                "type": "text",
                "text": MURAR_REFINE_PROMPT_TEMPLATE.format(text_prompt=user_query) + '```markdown\n'
            }
        ]
        global_counter = 0
        for text_summary_piece_with_retrieved_image in text_summary_pieces_with_retrieved_images:
            image = text_summary_piece_with_retrieved_image['image']
            if image is not None:
                messages.extend([
                    {
                        "type": "image_caption",
                        "image_caption": {
                            "caption": f"\n![{image['detailed_image_caption']}]({self.get_pseudo_image_url(image_id=global_counter, cached_image_url=image['cached_image_url'])})\n"
                        },
                    },
                    {
                        "type": "text",
                        "text": text_summary_piece_with_retrieved_image['text'] + '\n'
                    }
                ])
            else:
                messages[-1]['text'] += text_summary_piece_with_retrieved_image['text'] + '\n'
        
        messages.append({
            "type": "text",
            "text": "```\n# Generated Document"
        })
        
        return messages
    
    def summarize_core(self, user_query: str, webpages: List[Dict], **kwargs) -> Dict:
        # images for global index
        all_images = []
        for webpage in webpages:
            for image in webpage['images']:
                image['is_aux_image'] = False
                all_images.append(image)
        
        stage_1_messages = self.construct_summarize_message_stage_1(user_query=user_query, webpages=webpages, aux_images=[])
        stage_1_result = self.model.chat(messages=stage_1_messages)
        
        orig_text_summary = self.unwrap_markdown(text=stage_1_result['response'])
        orig_text_summary_lines = orig_text_summary.split('\n')

        selection_result_list = self.retrieve_relevant_images_stage_2(
            summary_pieces=orig_text_summary_lines,
            webpages=webpages
        )
        selected_images = [result['image'] for result in selection_result_list if result['image'] is not None]
        
        stage_3_messages = self.construct_summarize_message_stage_3(user_query=user_query, text_summary_pieces_with_retrieved_images=selection_result_list)
        stage_3_result = self.model.chat(messages=stage_3_messages)
        stage_3_response = self.unwrap_markdown(stage_3_result['response'])
        
        handle_image_result = self.handle_output_images(text=stage_3_response, images=selected_images)
        refined_summary, num_references = self.handle_output_references(text=handle_image_result['processed_response'], webpages=webpages)
        
        overall_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        for key in overall_usage:
            overall_usage[key] = stage_1_result['usage'][key] + stage_3_result['usage'][key]
        
        output = {
            "stage_1_model_input": stage_1_messages,
            "stage_1_model_response": stage_1_result['response'],
            "stage_1_usage": stage_1_result['usage'],
            "stage_2_match_results": selection_result_list,
            "stage_3_model_input": stage_3_messages,
            "stage_3_model_response": stage_3_result,
            "stage_3_usage": stage_3_result['usage'],
            "model_input": None,
            "model_response": None,
            "usage": overall_usage,
            "processed_response": refined_summary,
            "placeholder_response": handle_image_result['placeholder_response'],
            "output_images": handle_image_result['output_images'],
            "num_output_images": handle_image_result['num_images'],
            "num_reference_webpages": num_references
        }
        return output
