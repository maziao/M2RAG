import os
import json
import argparse
import pymongo
import logging
import numpy as np
from typing import Optional


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


class M2RAGTrainingSetConstructor:
    def __init__(
        self,
        mongodb_url: Optional[str],
        database_name: Optional[str],
        output_dir: str
    ):
        if mongodb_url is None:
            mongodb_url = os.environ.get('MONGODB_URL', None)
        assert mongodb_url is not None, f"`MONGODB_URL` is not set"
        if database_name is None:
            database_name = os.environ.get('MONGODB_NAME', None)
        assert database_name is not None, f"`MONGODB_NAME` is not set"
        self.client = pymongo.MongoClient(mongodb_url)
        self.database = self.client[database_name]
        
        # tables
        self.table_query = self.database["queries"]
        self.table_webpage = self.database["webpages"]
        self.table_aux_images = self.database["aux_images"]
        self.table_image = self.database["images"]
        self.table_summary = self.database["summaries"]
        
        # prepare output_dir
        os.makedirs(output_dir, exist_ok=True)
        llm_output_dir = os.path.join(output_dir, 'llm')
        os.makedirs(llm_output_dir, exist_ok=True)
        mllm_output_dir = os.path.join(output_dir, 'mllm')
        os.makedirs(mllm_output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        self.image_root = os.environ.get('IMAGE_ROOT')
        assert self.image_root is not None
        self.image_url = os.environ.get('IMAGE_URL')
        assert self.image_url is not None
        
    def _get_images(
        self,
        webpages,
        aux_images
    ):
        images = []
        for webpage in webpages:
            webpage_images = webpage['images']
            for image in webpage_images:
                image['id'] = len(images)
                images.append(image)
        
        for image in aux_images:
            image['id'] = len(images)
            images.append(image)
        
        return images
    
    def _get_sample(
        self,
        query_id,
        summarizer_id
    ):
        summary_metadata = self.table_summary.find_one({'query_id': query_id, 'summarizer_id': summarizer_id})
        if 'multi_stage' in summarizer_id:
            model_input = summary_metadata['stage_1_model_input']
        else:
            model_input = summary_metadata['model_input']
        
        query_content = self.table_query.find_one({'query_id': query_id}, {'query_content': 1})['query_content']
        images = self._get_images(webpages=summary_metadata['webpages'], aux_images=summary_metadata['aux_images'])
        
        model_input[0]['text'] = LLM_MULTI_MODAL_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=query_content)
        llm_model_input = self.merge_messages(model_input, images=images)
        
        model_input[0]['text'] = VLM_MULTI_MODAL_SINGLE_STAGE_SUMMARY_PROMPT_TEMPLATE.format(text_prompt=query_content)
        mllm_model_input = self.merge_messages(model_input, images=images)
        
        if 'multi_stage' in summarizer_id:
            model_response = self.get_model_response_from_multi_stage_result(
                placeholder_response=summary_metadata['placeholder_response'],
                output_images=summary_metadata['output_images']
            )
        else:
            model_response = summary_metadata['model_response']
        
        return {
            'llm': {
                'model_input': llm_model_input['text'],
                'model_response': model_response
            },
            'mllm': {
                'model_input': mllm_model_input['placeholder_text'],
                'model_response': model_response,
                'images': [image['cached_image_url'].replace(self.image_url, '') for image in images]
            }
        }
        
    def _get_query_ids(self, summarizer_id: str):
        image_metrics = ['image_coherence', 'image_helpfulness', 'image_reference', 'image_recall']
        image_score_threshold = 8
        
        query_ids = []
        for summary in self.table_summary.find({'summarizer_id': summarizer_id, 'evaluate_label': True}):
            # quality check
            scores = summary['evaluate_scores']
            results = summary['evaluate_result']
            
            multi_modal_scores = {
                metric_name: [result['normed_score'] for result in results['multi_modal'][metric_name]] if metric_name in results['multi_modal'] else [] for metric_name in image_metrics[:-1]
            }
            
            positive_images = []
            for webpage in summary["webpages"]:
                for image in webpage["images"]:
                    if image["final_score"] >= image_score_threshold:
                        positive_images.append(image["cached_image_url"])

            for aux_image in summary["aux_images"]:
                if aux_image["final_score"] >= image_score_threshold:
                    positive_images.append(aux_image["cached_image_url"])

            output_images = [
                image["cached_image_url"] for image in summary["output_images"]
            ]

            if len(positive_images) == 0:
                multi_modal_scores["image_recall"] = [1.0]
            else:
                multi_modal_scores["image_recall"] = [
                    len(list(set(positive_images) & set(output_images)))
                    / len(positive_images)
                ]
            
            valid = True
            for metric_name in multi_modal_scores:
                score = np.mean(multi_modal_scores[metric_name])
                if metric_name == 'image_coherence':
                    threshold = 0.7
                elif metric_name == 'image_helpfulness':
                    threshold = 0.7
                elif metric_name == 'image_reference':
                    threshold = 0.7
                else:
                    threshold = 0.7
                if score < threshold:
                    valid = False
            
            if valid:
                query_ids.append(summary['query_id'])
        return query_ids
    
    def construct_dataset(
        self,
        summarizer_id,
        num_samples
    ):
        samples = {'llm': [], 'mllm': []}
        query_ids = self._get_query_ids(summarizer_id=summarizer_id)
        logging.info(f"Found {len(query_ids)} samples.")
        for query_id in query_ids[:num_samples]:
            sample = self._get_sample(query_id=query_id, summarizer_id=summarizer_id)
            samples['llm'].append(sample['llm'])
            samples['mllm'].append(sample['mllm'])
        self.dump_data(samples['llm'], file=os.path.join(self.output_dir, 'llm', 'raw.jsonl'))
        self.dump_data(samples['mllm'], file=os.path.join(self.output_dir, 'mllm', 'raw.jsonl'))

    def dump_data(self, data, file):
        with open(file, 'w+', encoding='utf-8') as f:
            for i, item in enumerate(data):
                if item is None:
                    continue
                sample = {
                    'id': f"{i}",
                    **item
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    @staticmethod
    def merge_messages(messages: list, images: list) -> str:
        text = ''
        placeholder_text = ''
        num_images = 0
        for message in messages:
            if message['type'] == 'text':
                text += message['text']
                placeholder_text += message['text']
            elif message['type'] == 'image_caption':
                text += message['image_caption']['caption']
                placeholder_text += '![<image>' + message['image_caption']['caption'][message['image_caption']['caption'].find(']'):]
                num_images += 1
            elif message['type'] == 'image_url':
                text += images[num_images]['detailed_image_caption']
                placeholder_text += '<image>'
                num_images += 1
        return {
            'text': text,
            'placeholder_text': placeholder_text,
            'num_images': num_images
        }

    @staticmethod
    def get_model_response_from_multi_stage_result(placeholder_response: str, output_images: list):
        splits = placeholder_response.split('<IMAGE_PLACEHOLDER>')
        if len(splits) != len(output_images) + 1:
            splits = splits[:len(output_images)] + [' '.join(splits[len(output_images):])]
        
        model_response = splits[0]
        for split, image in zip(splits[1:], output_images):
            image_caption = image['image_caption']
            image_id = image['id']
            image_format = image['cached_image_url'].split('.')[-1]
            pseudo_url = f"images/{image_id}.{image_format}"
            
            model_response += f"\n![{image_caption}]({pseudo_url})\n"
            model_response += split
        return model_response
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongodb-url', type=str, default='mongodb://0.0.0.0:27017')
    parser.add_argument('--database-name', type=str, default='training_set')
    parser.add_argument('--summarizer-id', type=str, default='summarizer-multi_stage-llm-openai-gpt-4o-2024-08-06')
    parser.add_argument('--num-samples', type=int, default=1605)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    constructor = M2RAGTrainingSetConstructor(
        mongodb_url=args.mongodb_url,
        database_name=args.database_name,
        output_dir=args.output_dir
    )
    
    constructor.construct_dataset(
        summarizer_id=args.pos_summarizer_id,
        num_samples=args.num_samples,
    )
    