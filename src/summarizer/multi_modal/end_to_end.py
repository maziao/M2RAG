import re
import copy
from abc import abstractmethod
from typing import Dict, List, Tuple
from src.summarizer import SUMMARIZER
from src.summarizer.multi_modal.base import MultiModalSummarizer, IMAGE_PLACEHOLDER


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"
TEMP_IMAGE_PLACEHOLDER = "<TEMP_IMAGE_PLACEHOLDER>"


LLM_MULTI_MODAL_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a multi-modal Q&A assistant. Your role is to answer a user's question using information from relevant webpages that include both text and detailed image captions. These resources are derived from search engine results and are meant to provide a comprehensive multi-modal summary.

# Input Data
1. **Question**: This is the user's query and serves as the focus of your multi-modal summary.
2. **Webpages**: A list of webpages provided in a markdown-format code block, each starting with a title "## Webpage k" (k indicates the index starting from 0). Each image in the webpage is formatted as:
    ```markdown
    ![DETAILED_IMAGE_CAPTION](IMAGE_URL)
    ```
    - *DETAILED_IMAGE_CAPTION*: Describes the image content.
    - *IMAGE_URL*: Use this exact URL if incorporating the image into your output.
3. **Auxiliary Images**: Additional images following the same format as above.

# Guidelines
1. **Understand the Question**: Determine which content from the webpages best answers the user's question.
2. **Text Summary**: Create a coherent markdown-formatted text document to fully address the user's question.
3. **Image Integration**:
    - Integrate images appropriately into the text document.
    - Follow this format for each image:
        ```markdown
        ![IMAGE_CAPTION](IMAGE_URL)
        ```
        - *IMAGE_CAPTION*: A concise caption describing the image.
        - *IMAGE_URL*: Ensure no changes are made to the original URL.
    - Example:
        ```markdown
        ![Example of ...](images/0.jpeg)
        ```
4. **Image Relevance**: Only include images that are highly relevant to the question and enhance the text summary. Ensure the text references or connects to the image explicitly.

# Precautions
1. Ensure the final output is in markdown format for proper parsing.
2. Use authoritative expressions from the webpages as necessary but summarize and exclude irrelevant information.
3. Adhere strictly to the image format, ensuring each image is presented on a single line without additional information.

# Question
{text_prompt}

# Webpages
"""


VLM_MULTI_MODAL_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a multi-modal Q&A assistant. Your role is to answer a user's question using information from relevant webpages that include both text and images. These resources are derived from search engine results and are meant to provide a comprehensive multi-modal summary.

# Input Data
1. **Question**: This is the user's query and serves as the focus of your multi-modal summary.
2. **Webpages**: A list of webpages provided in a markdown-format code block, each starting with a title "## Webpage k" (k indicates the index starting from 0). Each image in the webpage is formatted as:
    ```markdown
    ![IMAGE_TOKENS](IMAGE_URL)
    ```
    - *IMAGE_TOKENS*: Tokens of the image.
    - *IMAGE_URL*: Use this exact URL if incorporating the image into your output.
3. **Auxiliary Images**: Additional images following the same format as above.

# Guidelines
1. **Understand the Question**: Determine which content from the webpages best answers the user's question.
2. **Text Summary**: Create a coherent markdown-formatted text document to fully address the user's question.
3. **Image Integration**:
    - Integrate images appropriately into the text document.
    - Follow this format for each image:
        ```markdown
        ![IMAGE_CAPTION](IMAGE_URL)
        ```
        - *IMAGE_CAPTION*: A concise caption describing the image.
        - *IMAGE_URL*: Ensure no changes are made to the original URL.
    - Example:
        ```markdown
        ![Example of ...](images/0.jpeg)
        ```
4. **Image Relevance**: Only include images that are highly relevant to the question and enhance the text summary. Ensure the text references or connects to the image explicitly.

# Precautions
1. Ensure the final output is in markdown format for proper parsing.
2. Use authoritative expressions from the webpages as necessary but summarize and exclude irrelevant information.
3. Adhere strictly to the image format, ensuring each image is presented on a single line without additional information.

# Question
{text_prompt}

# Webpages
"""


LLM_MULTI_MODAL_MULTI_STAGE_1_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a multi-modal Q&A assistant. Your role is to answer a user's question using information from relevant webpages that include both text and detailed image captions. These resources are derived from search engine results and are meant to provide a comprehensive text summary.

# Input Data
1. **Question**: This is the user's query and serves as the focus of your multi-modal summary.
2. **Webpages**: A list of webpages provided in a markdown-format code block, each starting with a title "## Webpage k" (k indicates the index starting from 0). Each image in the webpage is formatted as:
    ```markdown
    ![DETAILED_IMAGE_CAPTION](IMAGE_URL)
    ```
    - *DETAILED_IMAGE_CAPTION*: Describes the image content.
    - *IMAGE_URL*: The URL of the input image.
3. **Auxiliary Images**: Additional images following the same format as above.

# Guidelines
1. **Understand the Question**: Determine which content from the webpages best answers the user's question.
2. **Text Summary**: Create a coherent markdown-formatted text document to fully address the user's question.
3. **Image Integration**: Include the information of images that are highly relevant to the question in the text summary. Do not embed images using markdown syntax such as ![IMAGE_CAPTION](IMAGE_URL).

# Precautions
1. Ensure the final output is in markdown format for proper parsing.
2. Use authoritative expressions from the webpages as necessary but summarize and exclude irrelevant information.

# Question
{text_prompt}

# Webpages
"""


VLM_MULTI_MODAL_MULTI_STAGE_1_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a multi-modal Q&A assistant. Your role is to answer a user's question using information from relevant webpages that include both text and images. These resources are derived from search engine results and are meant to provide a comprehensive text summary.

# Input Data
1. **Question**: This is the user's query and serves as the focus of your multi-modal summary.
2. **Webpages**: A list of webpages provided in a markdown-format code block, each starting with a title "## Webpage k" (k indicates the index starting from 0). Each image in the webpage is formatted as:
    ```markdown
    ![IMAGE_TOKENS](IMAGE_URL)
    ```
    - *IMAGE_TOKENS*: Tokens of the image.
    - *IMAGE_URL*: The URL of the input image.
3. **Auxiliary Images**: Additional images following the same format as above.

# Guidelines
1. **Understand the Question**: Determine which content from the webpages best answers the user's question.
2. **Text Summary**: Create a coherent markdown-formatted text document to fully address the user's question.
3. **Image Integration**: Include the information of images that are highly relevant to the question in the text summary. Do not embed images using markdown syntax such as ![IMAGE_CAPTION](IMAGE_URL).

# Precautions
1. Ensure the final output is in markdown format for proper parsing.
2. Use authoritative expressions from the webpages as necessary but summarize and exclude irrelevant information.

# Question
{text_prompt}

# Webpages
"""


LLM_MULTI_MODAL_MULTI_STAGE_2_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a multi-modal Q&A assistant. Your task is to determine the most suitable position for an image within a text summary to maximize reader comprehension and engagement.

# Input Data
1. **User Query**: The user's inquiry which guides the focus of the summary.
2. **Summary Lines**: Each line of the summary, with indices starting from [0].
3. **Image Caption**: A detailed description of the image content.

# Guidelines
1. **Content Review**: Examine both the summary and the image caption thoroughly.
2. **Placement Decision**: Assess where the image adds the most value. If integration is appropriate, provide a concise caption and specify the summary line index to place it before.
3. **Relevance Check**: Ensure the image is directly relevant to the content of the line it precedes, and the preceding context does not already cover it.

# Output Format
- First Line: A concise caption for the image, without any additional formatting or labels.
- Second Line: The index of the summary line to precede with the image, or -1 if not suitable.

# User Query
{text_prompt}

# Summary Lines
```markdown
{text_summary_lines}
```

# Image Caption
"""


VLM_MULTI_MODAL_MULTI_STAGE_2_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a multi-modal Q&A assistant. Your task is to determine the most suitable position for an image within a text summary to maximize reader comprehension and engagement.

# Input Data
1. **User Query**: The user's inquiry which guides the focus of the summary.
2. **Summary Lines**: Each line of the summary, with indices starting from [0].
3. **Image**: The input image.

# Guidelines
1. **Content Review**: Examine both the summary and the image thoroughly.
2. **Placement Decision**: Assess where the image adds the most value. If integration is appropriate, provide a concise caption and specify the summary line index to place it before.
3. **Relevance Check**: Ensure the image is directly relevant to the content of the line it precedes, and the preceding context does not already cover it.

# Output Format
- First Line: A concise caption for the image, without any additional formatting or labels.
- Second Line: The index of the summary line to precede with the image, or -1 if not suitable.

# User Query
{text_prompt}

# Summary Lines
```markdown
{text_summary_lines}
```

# Image
"""


LLM_MULTI_MODAL_MULTI_STAGE_3_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a skilled text refinement assistant. Your role is to receive a user's question, a piece of a textual summary (as an answer to the question), and a detailed caption of an image that will precede the text in the final summary. Your job is to refine the textual summary to ensure it seamlessly complements the image.

# Guidelines
1. The final output must be in markdown format for consistency.
2. Maintain the original meaning and core information of the summary while smoothly incorporating references or connections to the image.
3. Explicitly reference or describe elements of the image within the text without using markdown image syntax such as ![IMAGE_CAPTION](IMAGE_URL).
4. Consider the summary piece as part of a larger document; refine the content without altering its structure to stand alone or appear as an isolated document.
5. Ensure the language flows naturally and logically links the image to the text.
6. Only output the refined summary, without any additional commentary, explanations, or chain-of-thought steps. If unable to perform the task, respond with "I can't ...".

# Question
{text_prompt}

# Summary Piece
```markdown
{text_summary_piece}
```

# Image Caption
"""


VLM_MULTI_MODAL_MULTI_STAGE_3_SUMMARY_PROMPT_TEMPLATE = """# Task Description
You are a skilled text refinement assistant. Your role is to receive a user's question, a piece of a textual summary (as an answer to the question), and an image that will precede the text in the final summary. Your job is to refine the textual summary to ensure it seamlessly complements the image.

# Guidelines
1. The final output must be in markdown format for consistency.
2. Maintain the original meaning and core information of the summary while smoothly incorporating references or connections to the image.
3. Explicitly reference or describe elements of the image within the text without using markdown image syntax such as ![IMAGE_CAPTION](IMAGE_URL).
4. Consider the summary piece as part of a larger document; refine the content without altering its structure to stand alone or appear as an isolated document.
5. Ensure the language flows naturally and logically links the image to the text.
6. Only output the refined summary, without any additional commentary, explanations, or chain-of-thought steps. If unable to perform the task, respond with "I can't ...".

# Question
{text_prompt}

# Summary Piece
```markdown
{text_summary_piece}
```

# Image
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
class M2RAGEndToEndSummarizer(MultiModalSummarizer):
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
        max_image_score: int = 1e10,
        min_image_score: int = 0,
        output_detailed_caption: bool = False,
        multi_stage: bool = False
    ) -> None:
        super().__init__(
            name='multi_stage' if multi_stage else 'single_stage',
            tag=tag,
            model_config=model_config,
            max_samples=max_samples,
            max_pieces=max_pieces,
            max_pieces_per_sample=max_pieces_per_sample,
            min_piece_score=min_piece_score,
            max_tokens_per_sample=max_tokens_per_sample,
            max_images=max_images,
            max_images_per_sample=max_images_per_sample,
            max_aux_images=max_aux_images,
            max_image_score=max_image_score,
            min_image_score=min_image_score,
            output_detailed_caption=output_detailed_caption
        )
        self.multi_stage = multi_stage
        
    def _reorganize_webpage_pieces_and_images(self, webpage: Dict) -> Dict:
        """Concatenate webpage pieces into a single string and replace indexed image placeholders with plain ones,
        and sort reserved images according to the sequence of images in the text.

        Args:
            webpage (Dict): a cleaned webpage with valid text pieces and images. 

        Returns:
            Dict: reorganized webpage
        """
        # remove empty webpages
        if len(webpage['pieces']) == 0:
            return None
        
        text = ''.join([piece['text'] for piece in webpage['pieces']])
        
        images = []
        image_indices = [int(index) for index in re.findall(r"<IMAGE_PLACEHOLDER>\[(\d+)\]", text)]
        reserved_image_indices = [image['image_id'] for image in webpage['images']]
        for index in image_indices:
            if index not in reserved_image_indices:
                text = text.replace(f"{IMAGE_PLACEHOLDER}[{index}]", '')
            else:
                text = text.replace(f"{IMAGE_PLACEHOLDER}[{index}]", TEMP_IMAGE_PLACEHOLDER)
                success = False
                for image in webpage['images']:
                    if image['image_id'] == index:
                        images.append(image)
                        success = True
                assert success is True
        text = text.replace(IMAGE_PLACEHOLDER, '').replace(TEMP_IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER)
        webpage['content'] = text
        webpage['images'] = images
        del webpage['pieces']
        return webpage
        
    def filter_images_and_reorganize_webpages(self, webpages: List[Dict], aux_images: List[Dict] = ...) -> Tuple:
        global_counter = 0
        for webpage in webpages:
            # range filtering in webpages
            webpage_images = [image for image in webpage['images'] if image['final_score'] >= self.min_image_score and image['final_score'] < self.max_image_score]
            
            # top-k filtering in webpages
            if self.max_images_per_sample > 0:
                image_top_k_indices = sorted([image['image_id'] for image in sorted(webpage_images, key=lambda s: s['final_score'], reverse=True)[:self.max_images_per_sample]])
            else:
                image_top_k_indices = [image['image_id'] for image in webpage_images]
            top_k_images = [image for image in webpage_images if image['image_id'] in image_top_k_indices]
        
            for image in top_k_images:
                image['_global_sort_id'] = global_counter
                global_counter += 1
                
            webpage['images'] = top_k_images
        
        # trim auxiliary images
        aux_images = [image for image in aux_images if image['final_score'] >= self.min_image_score and image['final_score'] < self.max_image_score]
        
        if self.max_aux_images >= 0:
            aux_image_top_k_indices = [image['aux_image_id'] for image in sorted(aux_images, key=lambda s: s['final_score'], reverse=True)[:self.max_aux_images]]
        else:
            aux_image_top_k_indices = [image['aux_image_id'] for image in aux_images]
        top_k_aux_images = [aux_image for aux_image in aux_images if aux_image['aux_image_id'] in aux_image_top_k_indices]
        
        for image in top_k_aux_images:
            image['_global_sort_id'] = global_counter
            global_counter += 1
            
        # trim images globally
        all_images = []
        for webpage in webpages:
            for image in webpage['images']:
                all_images.append({
                    '_global_sort_id': image['_global_sort_id'],
                    'is_aux_image': False,
                    'final_score': image['final_score']
                })
        for aux_image in top_k_aux_images:
            all_images.append({
                '_global_sort_id': aux_image['_global_sort_id'],
                'is_aux_image': True,
                'final_score': aux_image['final_score']
            })
        
        if self.max_images > 0:
            global_top_k_image_indices = [image['_global_sort_id'] for image in sorted(all_images, key=lambda s: s['final_score'], reverse=True)[:self.max_images]]
        else:
            global_top_k_image_indices = [image['_global_sort_id'] for image in all_images]

        for webpage in webpages:
            final_webpage_images = [image for image in webpage['images'] if image['_global_sort_id'] in global_top_k_image_indices]
            webpage['images'] = final_webpage_images
            
        final_webpages = []
        for webpage in webpages:
            final_webpage = self._reorganize_webpage_pieces_and_images(webpage=webpage)
            if final_webpage is not None:
                final_webpages.append(final_webpage)
            
        final_aux_images = [image for image in top_k_aux_images if image['_global_sort_id'] in global_top_k_image_indices]
        
        return final_webpages, final_aux_images
    
    @abstractmethod
    def construct_summarize_message_stage_1(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = []) -> List[Dict]:
        """Construct summarize sample from query, text summary and images from webpages.

        Args:
            user_query (str): concatenated query str from user
            webpages (List[Dict]): webpage list
            aux_images (List[Dict]): list of auxiliary images

        Returns:
            messages: List[Dict]
        """
        raise NotImplementedError
    
    @abstractmethod
    def construct_summarize_message_stage_2(self, user_query: str, text_summary_lines: List[str], image: Dict) -> List[Dict]:
        """Construct image selection sample from query, text summary and images from webpages.

        Args:
            user_query (str): concatenated query str from user
            text_summary_lines (List[str]): generated text summary based on textual search results
            image (Dict): image to be inserted

        Returns:
            messages: List[Dict]
        """
        raise NotImplementedError
    
    def parse_image_selection_result(self, text: str, max_index: int):
        splits = text.split('\n')
        integers = re.findall(r"\d+", splits[-1])
        if len(integers) == 0 or int(integers[-1]) > max_index or int(integers[-1]) < 0:
            return {
                'summary_line_id': -1,
                'image_caption': ' '.join(splits[:-1]).strip()
            }
        else:
            return {
                'summary_line_id': int(integers[-1]),
                'image_caption': ' '.join(splits[:-1]).strip()
            }
    
    @abstractmethod
    def construct_summarize_message_stage_3(self, user_query: str, text_summary_piece: str, image: Dict) -> List[Dict]:
        """Construct refinement sample from query, text summary and images from webpages.

        Args:
            user_query (str): concatenated query str from user
            text_summary_piece (str): text summary piece in the front of which images will be inserted
            image (Dict): chosen image

        Returns:
            messages: List[Dict]
        """
        raise NotImplementedError
    
    def summarize_core_single_stage(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = [], dry_run: bool = False):
        # images for global index
        all_images = []
        for webpage in webpages:
            for image in webpage['images']:
                image['is_aux_image'] = False
                all_images.append(image)
        for aux_image in aux_images:
            aux_image['is_aux_image'] = True
            all_images.append(aux_image)
        
        messages = self.construct_summarize_message(user_query=user_query, webpages=webpages, aux_images=aux_images)
        if dry_run:
            messages = self.replace_image_url_in_messages(messages=messages, images=all_images)
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
            messages = self.replace_image_url_in_messages(messages=messages, images=all_images)

            handle_image_result = self.handle_output_images(text=self.unwrap_markdown(result['response']), images=all_images)
            processed_response = handle_image_result['processed_response']
            num_references = 0
            
            output = {
                "model_input": messages,
                "model_response": result['response'],
                "usage": result['usage'],
                "processed_response": processed_response,
                "placeholder_response": handle_image_result['placeholder_response'],
                "output_images": handle_image_result['output_images'],
                "num_output_images": handle_image_result['num_images'],
                "num_reference_webpages": num_references
            }
        return output
    
    def summarize_core_multi_stage(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = [], dry_run: bool = False):
        # images for global index
        all_images = []
        for webpage in webpages:
            for image in webpage['images']:
                image['is_aux_image'] = False
                all_images.append(image)
        for aux_image in aux_images:
            aux_image['is_aux_image'] = True
            all_images.append(aux_image)
        
        stage_1_messages = self.construct_summarize_message_stage_1(user_query=user_query, webpages=webpages, aux_images=aux_images)
        stage_1_result = self.model.chat(messages=stage_1_messages)
        stage_1_messages = self.replace_image_url_in_messages(messages=stage_1_messages, images=all_images)
        
        orig_text_summary = self.unwrap_markdown(text=stage_1_result['response'])
        orig_text_summary_lines = orig_text_summary.split('\n')

        stage_2_messages_list = []
        stage_2_result_list = []
        stage_2_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        selection_result_list = []
        for i, image in enumerate(all_images):
            stage_2_messages = self.construct_summarize_message_stage_2(user_query=user_query, text_summary_lines=orig_text_summary_lines, image=image)
            stage_2_result = self.model.chat(messages=stage_2_messages)
            stage_2_messages = self.replace_image_url_in_messages(messages=stage_2_messages, images=[image])
            
            selection_result = self.unwrap_markdown(text=stage_2_result['response'])
            selection_result = self.parse_image_selection_result(text=selection_result, max_index=len(orig_text_summary_lines) - 1)
            
            stage_2_messages_list.append({
                '_image_id': i,
                'messages': stage_2_messages
            })
            stage_2_result_list.append({
                '_image_id': i,
                'result': stage_2_result
            })
            selection_result_list.append({
                '_image_id': i,
                **selection_result
            })
            for key, value in stage_2_result['usage'].items():
                stage_2_usage[key] += value
        
        selection_result_list = [result for result in sorted(selection_result_list, key=lambda s: s['summary_line_id']) if result['summary_line_id'] != -1]
        if len(selection_result_list) > 0:
            refined_summary = '\n'.join(orig_text_summary_lines[:selection_result_list[0]['summary_line_id']])
            refined_summary_with_placeholders = '\n'.join(orig_text_summary_lines[:selection_result_list[0]['summary_line_id']])
        else:
            refined_summary = orig_text_summary
            refined_summary_with_placeholders = orig_text_summary
        
        num_images = len(selection_result_list)
        output_images = []
        
        stage_3_messages_list = []
        stage_3_result_list = []
        stage_3_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        for i, selected_image in enumerate(selection_result_list):
            start_index = selected_image['summary_line_id']
            if i != len(selection_result_list) - 1:
                end_index = selection_result_list[i + 1]['summary_line_id']
            else:
                end_index = None

            summary_piece = '\n'.join(orig_text_summary_lines[start_index: end_index])
            
            image = all_images[selected_image['_image_id']]
            if selected_image['image_caption'] is not None and len(selected_image['image_caption']) > 0:
                refined_summary += IMAGE_STRUCT_WITH_CAPTION.format(image_url=image['image_url'], width=60, image_caption=selected_image['image_caption'])
            else:
                refined_summary += IMAGE_STRUCT.format(image_url=image['image_url'], width=60)
            
            if self.output_detailed_caption:
                refined_summary += f"\n[!] Image {selected_image['_image_id']}{' - aux' if image['is_aux_image'] else ''}; Original image caption: {image['detailed_image_caption']}[!]\n"
            
            refined_summary_with_placeholders += f"\n{IMAGE_PLACEHOLDER}\n"
            
            stage_3_messages = self.construct_summarize_message_stage_3(user_query=user_query, text_summary_piece=summary_piece, image=image)
            stage_3_result = self.model.chat(messages=stage_3_messages)
            stage_3_messages = self.replace_image_url_in_messages(messages=stage_3_messages, images=[image])
            
            stage_3_messages_list.append({
                'split_id': i,
                'messages': stage_3_messages
            })
            stage_3_result_list.append({
                'split_id': i,
                'result': stage_3_result
            })
            for key, value in stage_3_result['usage'].items():
                stage_3_usage[key] += value
            
            stage_3_response = self.delete_links_from_text(self.delete_images_from_text(self.unwrap_markdown(stage_3_result['response'])))
            refined_summary += stage_3_response
            refined_summary_with_placeholders += stage_3_response
            
            output_images.append({
                'id': selected_image['_image_id'],
                'image_caption': selected_image['image_caption'],
                'detailed_image_caption': image['detailed_image_caption'],
                'image_width': 60,
                'image_url': image['image_url'],
                'cached_image_url': image['cached_image_url']
            })
        
        # refined_summary, num_references = self.handle_output_references(text=refined_summary, webpages=webpages)
        num_references = 0
        
        overall_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
        for key in overall_usage:
            overall_usage[key] = stage_1_result['usage'][key] + stage_2_usage[key] + stage_3_usage[key]
        
        output = {
            "stage_1_model_input": stage_1_messages,
            "stage_1_model_response": stage_1_result['response'],
            "stage_1_usage": stage_1_result['usage'],
            "stage_2_model_inputs": stage_2_messages_list,
            "stage_2_model_responses": stage_2_result_list,
            "stage_2_usage": stage_2_usage,
            "stage_3_model_inputs": stage_3_messages_list,
            "stage_3_model_responses": stage_3_result_list,
            "stage_3_usage": stage_3_usage,
            "model_input": None,
            "model_response": None,
            "usage": overall_usage,
            "processed_response": refined_summary,
            "placeholder_response": refined_summary_with_placeholders,
            "output_images": output_images,
            "num_output_images": num_images,
            "num_reference_webpages": num_references
        }
        return output
    
    def summarize_core(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = [], dry_run: bool = False) -> Dict:
        if not self.multi_stage:
            return self.summarize_core_single_stage(user_query=user_query, webpages=webpages, aux_images=aux_images, dry_run=dry_run)
        else:
            return self.summarize_core_multi_stage(user_query=user_query, webpages=webpages, aux_images=aux_images, dry_run=dry_run)


@SUMMARIZER.register_module
class M2RAGLLMSummarizer(M2RAGEndToEndSummarizer):
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
        max_image_score: int = 1e10,
        min_image_score: int = 0,
        output_detailed_caption: bool = False,
        multi_stage: bool = False
    ) -> None:
        super().__init__(
            tag=tag,
            model_config=model_config,
            max_samples=max_samples,
            max_pieces=max_pieces,
            max_pieces_per_sample=max_pieces_per_sample,
            min_piece_score=min_piece_score,
            max_tokens_per_sample=max_tokens_per_sample,
            max_images=max_images,
            max_images_per_sample=max_images_per_sample,
            max_aux_images=max_aux_images,
            max_image_score=max_image_score,
            min_image_score=min_image_score,
            output_detailed_caption=output_detailed_caption,
            multi_stage=multi_stage
        )

    def construct_summarize_message(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = []) -> List[Dict]:
        messages = [
            {
                "type": "text",
                "text": LLM_MULTI_MODAL_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query)
            }
        ]
        global_counter = 0
        for i, webpage in enumerate(webpages):
            text_splits = webpage['content'].split(IMAGE_PLACEHOLDER)
            assert len(text_splits) == len(webpage['images']) + 1
            current_chunk = f"## Webpage {i}\n```markdown\n"
            for text_split, image in zip(text_splits[:-1], webpage['images']):
                current_chunk += text_split
                detailed_image_caption = image['detailed_image_caption'].replace('\n', ' ')
                messages.extend([
                    {
                        "type": "text",
                        "text": current_chunk
                    },
                    {
                        "type": "image_caption",
                        "image_caption": {
                            "caption": f"\n![{detailed_image_caption}]({self.get_pseudo_image_url(image_id=global_counter, cached_image_url=image['cached_image_url'])})\n"
                        }
                    }
                ])
                current_chunk = ""
                global_counter += 1
            if len(webpage['images']) != 0:
                messages.append({
                    "type": "text",
                    "text": text_splits[-1] + '\n```\n'
                })
            else:
                messages.append({
                    "type": "text",
                    "text": f"## Webpage {i}\n```markdown\n{text_splits[-1]}\n```\n"
                })
        
        if isinstance(aux_images, list) and len(aux_images) > 0:
            messages.append({
                "type": "text",
                "text": f"# Auxiliary Images\n"
            })
            for image in aux_images:
                detailed_image_caption = image['detailed_image_caption'].replace('\n', ' ')
                messages.append({
                    "type": "image_caption",
                    "image_caption": {
                        "caption": f"\n![{detailed_image_caption}]({self.get_pseudo_image_url(image_id=global_counter, cached_image_url=image['cached_image_url'])})\n"
                    }
                })
                global_counter += 1
        
        messages.append({
            "type": "text",
            "text": "# Generated Document"
        })
        
        return messages
    
    def construct_summarize_message_stage_1(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = []) -> List[Dict]:
        summarize_sample_stage_1 = self.construct_summarize_message(user_query=user_query, webpages=webpages, aux_images=aux_images)
        summarize_sample_stage_1[0]['text'] = LLM_MULTI_MODAL_MULTI_STAGE_1_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query)
        return summarize_sample_stage_1
    
    def construct_summarize_message_stage_2(self, user_query: str, text_summary_lines: List[str], image: Dict) -> List[Dict]:
        concat_text_summary_lines = ''
        for i, line in enumerate(text_summary_lines):
            concat_text_summary_lines += f"[{i}] {line}\n"
        
        messages = [
            {
                "type": "text",
                "text": LLM_MULTI_MODAL_MULTI_STAGE_2_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query, text_summary_lines=concat_text_summary_lines)
            },
            {
                "type": "image_caption",
                "image_caption": {
                    "caption": f"{image['detailed_image_caption']}\n"
                }
            },
            {
                "type": "text",
                "text": "# Your Output"
            }
        ]
        
        return messages
    
    def construct_summarize_message_stage_3(self, user_query: str, text_summary_piece: str, image: Dict) -> List[Dict]:
        messages = [
            {
                'type': 'text',
                'text': LLM_MULTI_MODAL_MULTI_STAGE_3_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query, text_summary_piece=text_summary_piece)
            },
            {
                'type': 'image_caption',
                'image_caption': {
                    'caption': f"{image['detailed_image_caption']}\n"
                }
            },
            {
                "type": "text",
                "text": "# Your Output"
            }
        ]
        
        return messages


@SUMMARIZER.register_module
class M2RAGVLMSummarizer(M2RAGEndToEndSummarizer):
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
        max_image_score: int = 1e10,
        min_image_score: int = 0,
        output_detailed_caption: bool = False,
        multi_stage: bool = False,
        use_orig_image: bool = False
    ) -> None:
        super().__init__(
            tag=tag,
            model_config=model_config,
            max_samples=max_samples,
            max_pieces=max_pieces,
            max_pieces_per_sample=max_pieces_per_sample,
            min_piece_score=min_piece_score,
            max_tokens_per_sample=max_tokens_per_sample,
            max_images=max_images,
            max_images_per_sample=max_images_per_sample,
            max_aux_images=max_aux_images,
            max_image_score=max_image_score,
            min_image_score=min_image_score,
            output_detailed_caption=output_detailed_caption,
            multi_stage=multi_stage
        )
        self.use_orig_image = use_orig_image
        assert self.model.name.startswith('vlm'), f"the model for multi-modal summarizer must be a VLM, but got {self.model.name}"

    def construct_summarize_message(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = []) -> List[Dict]:
        messages = [
            {
                "type": "text",
                "text": VLM_MULTI_MODAL_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query)
            }
        ]
        global_counter = 0
        for i, webpage in enumerate(webpages):
            text_splits = webpage['content'].split(IMAGE_PLACEHOLDER)
            assert len(text_splits) == len(webpage['images']) + 1
            current_chunk = f"## Webpage {i}\n```markdown\n"
            for text_split, image in zip(text_splits[:-1], webpage['images']):
                current_chunk += text_split
                current_chunk += f"\n!["
                messages.extend([
                    {
                        "type": "text",
                        "text": current_chunk
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.get_image_url_from_image(image=image)
                        }
                    },
                    {
                        "type": "text",
                        "text": f"]({self.get_pseudo_image_url(image_id=global_counter, cached_image_url=image['cached_image_url'])})\n"
                    }
                ])
                current_chunk = ""
                global_counter += 1
            if len(webpage['images']) != 0:
                messages.append({
                    "type": "text",
                    "text": text_splits[-1] + '\n```\n'
                })
            else:
                messages.append({
                    "type": "text",
                    "text": f"## Webpage {i}\n```markdown\n{text_splits[-1]}\n```\n"
                })
        
        if isinstance(aux_images, list) and len(aux_images) > 0:
            messages.append({
                "type": "text",
                "text": f"# Auxiliary Images\n"
            })
            for image in aux_images:
                messages.extend([
                    {
                        "type": "text",
                        "text": f"\n!["
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self.get_image_url_from_image(image=image)
                        }
                    },
                    {
                        "type": "text",
                        "text": f"]({self.get_pseudo_image_url(image_id=global_counter, cached_image_url=image['cached_image_url'])})\n"
                    }
                ])
                global_counter += 1
                
        messages.append({
            "type": "text",
            "text": "# Generated Document"
        })
        
        return messages
    
    def construct_summarize_message_stage_1(self, user_query: str, webpages: List[Dict], aux_images: List[Dict] = []) -> List[Dict]:
        summarize_sample_stage_1 = self.construct_summarize_message(user_query=user_query, webpages=webpages, aux_images=aux_images)
        summarize_sample_stage_1[0]['text'] = VLM_MULTI_MODAL_MULTI_STAGE_1_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query)
        return summarize_sample_stage_1
    
    def construct_summarize_message_stage_2(self, user_query: str, text_summary_lines: List[str], image: Dict) -> List[Dict]:
        concat_text_summary_lines = ''
        for i, line in enumerate(text_summary_lines):
            concat_text_summary_lines += f"[{i}] {line}\n"
        
        messages = [
            {
                "type": "text",
                "text": VLM_MULTI_MODAL_MULTI_STAGE_2_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query, text_summary_lines=concat_text_summary_lines)
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": self.get_image_url_from_image(image=image)
                }
            },
            {
                "type": "text",
                "text": "\n# Your Output"
            }
        ]
        
        return messages
    
    def construct_summarize_message_stage_3(self, user_query: str, text_summary_piece: str, image: Dict) -> List[Dict]:
        messages = [
            {
                'type': 'text',
                'text': VLM_MULTI_MODAL_MULTI_STAGE_3_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=user_query, text_summary_piece=text_summary_piece)
            },
            {
                'type': 'image_url',
                'image_url': {
                    'url': self.get_image_url_from_image(image=image)
                }
            },
            {
                "type": "text",
                "text": "\n# Your Output"
            }
        ]
        
        return messages
