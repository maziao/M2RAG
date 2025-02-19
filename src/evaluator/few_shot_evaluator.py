import os
import re
import json
import logging
import asyncio
import nest_asyncio
from typing import List, Dict, Any
from langchain_openai.chat_models import ChatOpenAI
try:
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithoutReference, Faithfulness
except Exception:
    SingleTurnSample = None
    LangchainLLMWrapper = None
    LLMContextPrecisionWithoutReference = None
    Faithfulness = None
    logging.warning(f"ragas is not currently imported, evaluation with ragas cannot be used.")
from src.model import build_model
from src.evaluator import EVALUATOR
from src.utils import MultithreadManager


FLUENCY_PROMPT = """# Task Description
You are a document evaluation assistant. Your task is to evaluate the fluency of the provided text written in markdown.

# Input Text
```markdown
{text}
```

# Scoring Criteria
Assign a score from 0 to 10 based on the fluency of the text:
- **0:** The text is incoherent, lacks logical flow, and deviates significantly from correct markdown syntax.
- **1-3:** The text has potential with partial fluency and logic but is plagued by multiple markdown errors, such as unhandled duplicate headings or incorrect formatting structures.
- **4-6:** The text demonstrates general fluency with minor grammatical or logical inconsistencies, though the markdown structure lacks clarity or uniformity, like having redundant sections.
- **7-9:** The text is well-written and logically coherent, with proper markdown usage, forming a mostly seamless document that maintains consistent quality.
- **10:** The text is exemplary, demonstrating perfect fluency, logical progression, and impeccable markdown syntax, representing an ideal markdown document.

Be rigorous and discerning when assigning your score.

# Output Instructions
Your output should contain only 2 lines:
1. A brief explanation justifying the assigned score.
2. The score as an integer value.

Do not provide any additional information beyond the explanation and score.
"""


RELEVANCE_PROMPT = """# Task Description
You are a document evaluation assistant. Your task is to evaluate the relevance between the query and provided text written in markdown.

# Query
{query}

# Input Text
```markdown
{text}
```

# Scoring Criteria
Assign a score from 0 to 10 based on the degree of relevance between the input text and the given query:
- **0:** The text is entirely irrelevant to the given query.
- **1-3:** The text has minimal relevance, with most content deviating from the intended query.
- **4-6:** The text is moderately relevant, with some content aligned with the intended query, but contains noticeable digressions.
- **7-9:** The text is highly relevant, with most content directly related to the intended query and only minor digressions.
- **10:** The text is perfectly relevant, fully aligning with the intended query without any digressions.

Be rigorous and discerning when assigning your score.

# Output Instructions
Your output should contain only 2 lines:
1. A brief explanation justifying the assigned score.
2. The score as an integer value.

Do not provide any additional information beyond the explanation and score.
"""


IMAGE_COHERENCE_PROMPT = """# Task Description
You are a multimodal document evaluation assistant. You will receive an image and its textual context written in markdown. Your task is to evaluate the coherence between the image and the text (context above and below) it accompanies.

# Context Above
```markdown
{context_above}
```

# Image
<IMAGE_PLACEHOLDER>

# Context Below
```markdown
{context_below}
```

# Scoring Criteria and Examples
Evaluate how well the image complements its accompanying text and assign a score from 0 to 10. A higher score reflects stronger coherence between the image and the text. Be precise in your evaluation.

- **Score 0-3**: Minimal or no coherence between the image and text.
  - *Example 1*: The image shows a video game controller, but the text discusses supply and demand in virtual goods on platforms like Steam. These topics are unrelated, and the image adds no context to the discussion of virtual goods. The score should be 0-3.
  - *Example 2*: The image depicts a luxury car, but the text is about the environmental impact of electric vehicles. While both relate to automobiles, the image doesn’t contribute to the discussion of electric vehicles’ environmental effects. Thus, it merits a score of 0-3.
  - *Example 3*: A photo of a pizza is shown, but the text is a deep dive into the history of ancient Greece. There is no connection between the image and the historical content, making the coherence score 0-3.

- **Score 4-6**: Some coherence, but with unrelated elements.
  - *Example 1*: The image shows various components like a remote control, cables, and a media player, which are related to the media player mentioned in the text. However, the context focuses on video formats, which is not addressed by the image. The coherence score should be between 4 and 6.
  - *Example 2*: The image shows graduates from Harvard University, and the text is about the cost of university education and the perceived benefits of attending prestigious institutions. While the image relates to the general topic of education, it doesn’t directly support the discussion about the financial costs or value of education, meriting a score of 4-6.
  - *Example 3*: The image of a crowded subway is referenced in the text discussing urbanization and its effects on public transport. However, the text focuses more on the environmental impact of urbanization rather than directly linking to the image of subway crowds, leading to a coherence score of 4-6.

- **Score 7-9**: High coherence, with the image closely aligning with the text.
  - *Example 1*: The image is a photo of a video game controller, and the text describes features of the latest gaming console. The image directly aligns with the content, visually representing the technology described in the text, earning a score of 7-9.
  - *Example 2*: The image depicts a close-up of a laptop keyboard, and the text discusses advancements in laptop design. The image complements the text well, helping the reader understand the specific design features mentioned, justifying a score of 7-9.
  - *Example 3*: A chart showing sales data for various smartphone models is paired with text explaining smartphone market trends. The image is directly relevant to the discussion of market share and trends, aiding comprehension, and warrants a score of 7-9.

- **Score 10**: Perfect coherence, where the image completely corresponds to and enhances the text.
  - *Example 1*: The image is a detailed picture of a video game controller, and the text provides a comprehensive review of the controller’s features, design, and performance. The image fully complements the text, making the reader’s understanding more complete and vivid, deserving a score of 10.
  - *Example 2*: The image shows a close-up of a public toilet seat, which is relevant to the context below that discusses the design differences between public and home toilet seats. The image is completely aligned with the text, directly aiding the discussion. The score should be 10.
  - *Example 3*: The image is an anatomical diagram of the human heart, and the text explains how blood circulates through the body. The visual directly supports the explanation, making the text easier to understand and reinforcing key points, justifying a score of 10.

**Note**: The image needs to be coherent with the context above OR below it, not necessarily both. Be rigorous and objective in your assessment.

# Output Instructions
Your output should contain only 2 lines:
1. A brief explanation justifying the assigned score.
2. The score as an integer value.

Do not provide any additional information beyond the explanation and score.
"""


IMAGE_HELPFULNESS_PROMPT = """# Task Description
You are a multimodal document evaluation assistant. You will receive an image and its textual context written in markdown. Your task is to evaluate the helpfulness of the image in enabling human readers to comprehend the text (context above and below) it accompanies.

# Context Above
```markdown
{context_above}
```

# Image
<IMAGE_PLACEHOLDER>

# Context Below
```markdown
{context_below}
```

# Scoring Criteria and Examples
Assess the image's helpfulness in improving comprehension of the text, assigning a score from 0 to 10. A higher score indicates the image significantly enhances understanding. Be precise and rigorous when assigning the score.

- **Score 0-3**: The image is minimally or not at all helpful for comprehension.
  - *Example 1*: The image depicts a book cover titled "When God Asks You to Do Something You Don't Want to Do," with a background of crashing ocean waves. The text is about jury members addressing trial observations. The image is irrelevant and unhelpful, meriting a score of 0-3.
  - *Example 2*: The image shows a close-up of banknotes (euro), arranged in a fan-like pattern. While the image is related to the concept of money, it does not directly enhance the comprehension of the text, which discusses the exchange rate of different currencies. Thus, it is not helpful for comprehension, earning a score of 0-3.
  - *Example 3*: The image is of a person looking at a blank computer screen, with no connection to the text discussing how the internet has influenced modern literature. The image does not provide any useful context or additional understanding of the text, so it warrants a score of 0-3.

- **Score 4-6**: The image provides some helpful context but may include extraneous or less relevant details.
  - *Example 1*: The image shows a man at a desk, shocked at his laptop, with colorful text and symbols in the background. The text discusses how hackers are caught. While the image is thematically related, it is not directly helpful in explaining the topic of cyber security, earning a score of 4-6.
  - *Example 2*: The image shows a car with a vented hood, which relates to the text discussing the aerodynamic and cooling benefits of such designs in mid-engine cars. However, the image doesn’t address specific details, like material considerations or the pressure zones discussed in the text, meaning it provides partial context but not a complete understanding. Therefore, it should be assigned a score of 4-6.
  - *Example 3*: The image of a group of college students at a campus event is referenced in the text discussing the benefits of a liberal arts education. While it offers a visual representation of student life, it doesn’t fully enhance the understanding of the specific academic benefits described, so it earns a score of 4-6.

- **Score 7-9**: The image is highly helpful in enhancing comprehension of the text.
  - *Example 1*: The image features a table comparing popular browsers and their operating systems, such as Edge, Safari, and Chrome. The text discusses why Internet Explorer (IE) is disliked. The image is highly relevant, providing a clear visual context that directly supports the reader’s understanding of the browser landscape, justifying a score of 7-9.
  - *Example 2*: The image is a close-up of a laptop keyboard, with highlighted keys. The text discusses the evolution of laptop designs, and the image directly supports the explanation, helping the reader understand the physical changes over time, which justifies a score of 7-9.
  - *Example 3*: The image shows the front page of a scientific journal, with headlines and an image illustrating the main topic. The text explains the content of the journal in detail, and the image directly helps readers better visualize the article’s focus, justifying a score of 7-9.

- **Score 10**: The image perfectly enhances and clarifies the text.
  - *Example 1*: The image is a detailed diagram of the human digestive system, showing organs like the stomach and intestines. The text explains the process of digestion, and the diagram complements this explanation perfectly, enhancing the reader's understanding of the content. The image and text work together seamlessly, earning a score of 10.
  - *Example 2*: The image shows a close-up of a public toilet seat, which is relevant to the context below that discusses the design differences between public and home toilet seats. The image directly complements the text, providing the exact visual context needed to understand the design differences. The score should be 10.
  - *Example 3*: The image is a map highlighting the trade routes discussed in the text about ancient civilizations. The map directly enhances the comprehension of the trade dynamics and locations mentioned in the text, offering a visual aid that clarifies the complex subject matter, thus earning a score of 10.

**Note**: The image needs to be helpful for the context above OR below it, not necessarily both. Be rigorous and objective in your assessment.

# Output Instructions
Your output should contain only 2 lines:
1. A brief explanation justifying the assigned score.
2. The score as an integer value.

Do not provide any additional information beyond the explanation and score.
"""


IMAGE_REFERENCE_PROMPT = """# Task Description
You are a multimodal document quality assessment assistant. You will receive an image and its accompanying textual context, formatted in markdown. Your task is to determine whether the image is explicitly referenced or explained within the surrounding text (both above and below the image).

# Context Above
```markdown
{context_above}
```

# Image
<IMAGE_PLACEHOLDER>

# Context Below
```markdown
{context_below}
```

# Scoring Criteria
Determine how well the image is referenced or explained in the surrounding text, assigning a score from 0 to 10:

- **Score 0**: The image is not mentioned or referenced in the text.
  - *Example 1*: The image is present, but there is no mention or acknowledgment of it in either the context above or below.
  - *Example 2*: The image shows a scenic mountain view, but the text is entirely about office workflows, with no reference to or explanation of the image.
  - *Example 3*: The image is of a fruit basket, but the text is about urban planning, without any reference to the image or its context.

- **Score 1-3**: The image is referenced implicitly, but the reference is inapparent, improper, or incorrect.
  - *Example 1*: The text discusses the Millennium Falcon’s hyperdrive system and kyber crystal energy field, which are depicted in the image. However, the image is not explicitly mentioned, and the connection is weak. The score is 1-3.
  - *Example 2*: The text mentions "the latest in gaming technology," and the image shows a futuristic gaming console. The image is not directly referenced, and the connection to the text is not clearly established. The score is 1-3.
  - *Example 3*: The text talks about space exploration and mentions rockets, but the image is a close-up of a satellite. The connection between the satellite and the discussion of space exploration is implied but not directly referenced. Thus, the score is 1-3.

- **Score 4-6**: The image is referenced implicitly or explicitly, but the reference is improper or partially relevant.
  - *Example 1*: The image shows a skier, and the text explicitly references it, but the discussion focuses on ski races, which is only loosely related to the skier's role. The reference is valid but not fully accurate, so the score is 4-6.
  - *Example 2*: The image shows a tree with roots exposed, and the text references “the importance of deep roots” in discussing personal growth. While the image supports the metaphor, the connection is not fully fleshed out, leading to a score of 4-6.
  - *Example 3*: The image is of a sunset, and the text discusses the environmental impact of light pollution. While the sunset relates to the topic of light, it does not directly contribute to the explanation of light pollution, resulting in a score of 4-6.

- **Score 7-9**: The image is explicitly referenced in a generally proper and correct manner.
  - *Example 1*: The image depicts a box of chips and is explicitly mentioned in the text at an appropriate point. While the reference is accurate, it might feel a little stiff, leading to a score of 7-9.
  - *Example 2*: The text discusses the structural components of a building, and the image shows a detailed diagram of the beams and supports. The reference is clear and mostly accurate, with a slight stiffness in phrasing, justifying a score of 7-9.
  - *Example 3*: The image depicts a city map, and the text discusses the layout of different districts. The map is explicitly referenced and properly aids the understanding of the text, but the reference could be smoother. The score should be 7-9.

- **Score 10**: The image is explicitly referenced with complete accuracy and proper placement.
  - *Example 1*: The image is explicitly mentioned and thoroughly explained in the text, with a discussion of NFL play calls perfectly aligned with the visual content. The placement and explanation of the image are spot-on, so it warrants a score of 10.
  - *Example 2*: The image shows a detailed diagram of the human circulatory system, and the text provides an in-depth explanation of how blood flows through the heart and veins. The image and text work in perfect harmony to enhance the reader’s understanding. The score should be 10.
  - *Example 3*: The image depicts a vintage car and is referenced explicitly in the context, which discusses the history and evolution of automobile designs. The reference is fully accurate, and the image provides essential clarity to the reader, justifying a perfect score of 10.

**Note**: The image only needs to be referenced or explained by its context above OR below, not necessarily both. Be rigorous and objective in your assessment.

# Output Instructions
Your output should consist of only two lines:
1. A brief explanation justifying the assigned score.
2. The score as an integer value.

Refrain from providing any additional information beyond the explanation and score.
"""


IMAGE_PLACEHOLDER = '<IMAGE_PLACEHOLDER>'


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


TEXT_METRICS = {
    'fluency': {
        'type': 'text_only',
        'prompt': FLUENCY_PROMPT,
        'scorer': None,
        'upper_bound': 10,
        'lower_bound': 0,
        'kwargs': ['text']
    },
    'response_relevancy': {
        'type': 'text_only',
        'prompt': RELEVANCE_PROMPT,
        'scorer': None,
        'upper_bound': 10,
        'lower_bound': 0,
        'kwargs': ['query', 'text']
    },
    'context_precision': {
        'type': 'text_only',
        'prompt': None,
        'scorer': LLMContextPrecisionWithoutReference() if LLMContextPrecisionWithoutReference is not None else None,
        'upper_bound': 1,
        'lower_bound': 0,
        'kwargs': ['user_input', 'response', 'retrieved_contexts']
    },
    'faithfulness': {
        'type': 'text_only',
        'prompt': None,
        'scorer': Faithfulness() if Faithfulness is not None else None,
        'upper_bound': 1,
        'lower_bound': 0,
        'kwargs': ['user_input', 'response', 'retrieved_contexts']
    }
}


IMAGE_PRECISION_METRICS = {
    'image_coherence': {
        'type': 'multi_modal',
        'prompt': IMAGE_COHERENCE_PROMPT,
        'scorer': None,
        'upper_bound': 10,
        'lower_bound': 0,
        'kwargs': ['context_above', 'context_below'],
        'num_images': 1
    },
    'image_helpfulness': {
        'type': 'multi_modal',
        'prompt': IMAGE_HELPFULNESS_PROMPT,
        'scorer': None,
        'upper_bound': 10,
        'lower_bound': 0,
        'kwargs': ['context_above', 'context_below'],
        'num_images': 1
    },
    'image_reference': {
        'type': 'multi_modal',
        'prompt': IMAGE_REFERENCE_PROMPT,
        'scorer': None,
        'upper_bound': 10,
        'lower_bound': 0,
        'kwargs': ['context_above', 'context_below'],
        'num_images': 1
    }
}


TEXT_WEIGHTS = {
    'fluency': 0.25,
    'context_precision': 0.25,
    'response_relevancy': 0.25,
    'faithfulness': 0.25
}


IMAGE_PRECISION_WEIGHTS = {
    'image_coherence': 0.25,
    'image_helpfulness': 0.25,
    'image_reference': 0.25,
}


@EVALUATOR.register_module
class FewShotEvaluator:
    def __init__(
        self,
        model_config: Dict = {},
        langchain_model_config: Dict = {},
        max_context_tokens: int = -1,
        multi_thread_config: dict = {},
        log_dir: str = None
    ) -> None:
        self.model = build_model(model_config)
        if not self.model.name.startswith('vlm'):
            logging.warning(f"the model for evaluator must be a VLM for multi-modal evaluation, but got {self.model.name}. Support text-only evaluation with this model.")
        self.langchain_model = build_model(langchain_model_config)
        
        self.max_context_tokens = max_context_tokens
        self.multi_thread_config = multi_thread_config
        
        assert log_dir is not None
        self.log_dir = os.path.join(log_dir, 'evaluate')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        openai_model = ChatOpenAI(
            model=self.langchain_model.model,
            api_key=self.langchain_model.api_key,
            base_url=self.langchain_model.base_url
        )
        openai_model_wrapper = LangchainLLMWrapper(langchain_llm=openai_model)
        for metric_name in TEXT_METRICS:
            metric = TEXT_METRICS[metric_name]
            if metric['scorer'] is not None and metric['scorer'].llm is None:
                metric['scorer'].llm = openai_model_wrapper
        
    def evaluate(
        self,
        query_id: Any,
        summarizer_id: Any,
        query: str,
        webpages: List[Dict],
        aux_images: List[Dict],
        raw_summary: str,
        output_images: List[Dict],
        metrics: List[str] = None
    ):
        # check if all requested metrics are valid
        all_metrics = list(TEXT_METRICS.keys()) + list(IMAGE_PRECISION_METRICS.keys())
        if metrics is not None:
            for item in metrics:
                if item not in all_metrics:
                    raise ValueError(f"metrics `{item}` is not defined.")
        else:
            metrics = all_metrics
        
        temp_manager = MultithreadManager(**self.multi_thread_config)
        
        samples = self.prepare_evaluate_sample(
            query_content=query,
            webpages=webpages,
            aux_images=aux_images,
            raw_summary=raw_summary,
            output_images=output_images
        )
        
        log_file = os.path.join(self.log_dir, summarizer_id, f"{query_id}.jsonl")
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # text metrics
        text_metrics = {key: TEXT_METRICS[key] for key in metrics if key in TEXT_METRICS}
        ragas_sample = SingleTurnSample(**samples['text_sample'])
        ragas_results = []
        
        if os.path.exists(log_file):
            with open(log_file, 'r+', encoding='utf-8') as f:
                records = [json.loads(line) for line in f.readlines()]
        else:
            records = []

        for name, metric in text_metrics.items():
            # ragas metrics should be run single-threaded
            if metric['scorer'] is not None:
                completed = False
                if os.path.exists(log_file):
                    # with open(log_file, 'r+', encoding='utf-8') as f:
                    #     records = [json.loads(line) for line in f.readlines()]
                    #     print(records)
                    for record in records:
                        if record['id'] == f"text-{name}" and record['success']:
                            ragas_results.append(record)
                            completed = True
                            break
                    
                    if completed:
                        continue
                
                nest_asyncio.apply()
                score = asyncio.run(metric['scorer'].single_turn_ascore(sample=ragas_sample))
                
                log = {
                    'id': f"text-{name}",
                    'success': True,
                    'result': {
                        'score': score,
                        'normed_score': max(metric['lower_bound'], min(metric['upper_bound'], score)) / metric['upper_bound']
                    },
                    'kwargs': {
                        'metric_name': name
                    }
                }
                ragas_results.append(log)
                with open(log_file, 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(log) + '\n')
            # other metrics are pushed into the multi-thread task queue
            else:
                prompt_kwargs = {}
                for kwarg in metric['kwargs']:
                    if kwarg in ['text']:
                        prompt_kwargs[kwarg] = samples['text_sample']['response']
                    elif kwarg in ['query']:
                        prompt_kwargs[kwarg] = samples['text_sample']['user_input']
                prompt = metric['prompt'].format(**prompt_kwargs)
                
                temp_manager.add_task(
                    self.get_evaluation_result,
                    log_file,
                    f"text-{name}",
                    prompt=prompt,
                    upper_bound=metric['upper_bound'],
                    lower_bound=metric['lower_bound'],
                    metric_name=name
                )
        
        # image precision metrics
        image_precision_metrics = {key: IMAGE_PRECISION_METRICS[key] for key in metrics if key in IMAGE_PRECISION_METRICS}
        for name, metric in image_precision_metrics.items():
            for i, sample in enumerate(samples['image_precision_samples']):
                prompt_kwargs = {}
                for kwarg in metric['kwargs']:
                    if kwarg in ['context_above', 'context_below']:
                        prompt_kwargs[kwarg] = sample[kwarg]
                    elif kwarg == 'image_caption':
                        prompt_kwargs[kwarg] = sample['image'][kwarg]
                        
                prompt = metric['prompt'].format(**prompt_kwargs)
                
                temp_manager.add_task(
                    self.get_evaluation_result,
                    log_file,
                    f"multi_modal-{name}-{i}",
                    prompt=prompt,
                    images=[sample['image']],
                    upper_bound=metric['upper_bound'],
                    lower_bound=metric['lower_bound'],
                    metric_name=name
                )
        
        results = temp_manager.execute_tasks()
        temp_manager.clear_tasks()
        results = results + ragas_results
        
        evaluate_result = {
            'text': {},
            'multi_modal': {}
        }

        for result in results:
            assert result['success'] is True and result['result'] is not None
            metric_name = result['kwargs']['metric_name']
            if metric_name in TEXT_METRICS.keys():
                evaluate_result['text'][metric_name] = result['result']
            else:
                if metric_name not in evaluate_result['multi_modal']:
                    evaluate_result['multi_modal'][metric_name] = [result['result']]
                else:
                    evaluate_result['multi_modal'][metric_name].append(result['result'])
        
        scores = {
            'text': {},
            'multi_modal': {}
        }
        
        # text scores
        for key in text_metrics:
            if key in evaluate_result['text']:
                scores['text'][key] = evaluate_result['text'][key]['normed_score']
            else:
                scores['text'][key] = 0.0
        
        # image precision
        for key in image_precision_metrics:
            if key in evaluate_result['multi_modal']:
                scores['multi_modal'][key] = sum([sample['normed_score'] for sample in evaluate_result['multi_modal'][key]]) / len(evaluate_result['multi_modal'][key])
            else:
                scores['multi_modal'][key] = 0.0
        
        logging.info(f"evaluation for query {query_id}, summarizer {summarizer_id} completed: {json.dumps(scores)}")

        return {
            'evaluate_result': evaluate_result,
            'evaluate_scores': scores
        }
    
    def extract_score_from_str(self, text: str, default: int = 0) -> int:
        scores = re.findall(r"\d+", text)
        if len(scores) == 0:
            return default

        score = int(scores[-1])
        return score
        
    def prepare_evaluate_sample(
        self,
        query_content: str,
        webpages: List[Dict],
        aux_images: List[Dict],
        raw_summary: str,
        output_images: List[Dict]
    ) -> List[Dict]:
        text_sample = {
            'user_input': query_content,
            'response': raw_summary.replace(IMAGE_PLACEHOLDER, ''),
            'retrieved_contexts': [webpage['content'] for webpage in webpages] + [image['detailed_image_caption'] for image in output_images],
            'reference': raw_summary.replace(IMAGE_PLACEHOLDER, '') # reference is not used but must be provided because of a bug in LLMContextPrecisionWithoutReference
        }
        
        cited_image_ids = [image['id'] for image in output_images]
        
        images = []
        for webpage in webpages:
            webpage_images = webpage['images']
            for image in webpage_images:
                image['id'] = len(images)
                if image['id'] in cited_image_ids:
                    image['cited'] = True
                else:
                    image['cited'] = False
                images.append(image)
        
        for image in aux_images:
            image['id'] = len(images)
            if image['id'] in cited_image_ids:
                image['cited'] = True
            else:
                image['cited'] = False
            images.append(image)
            
        summary_splits = raw_summary.split(IMAGE_PLACEHOLDER)
        if len(summary_splits) != len(output_images) + 1:
            logging.warning(f"there are {len(summary_splits)} splits with splitter `{IMAGE_PLACEHOLDER}`, but got {len(output_images)} output images, which is not a match. Ignore redundant image placeholders...")
            last_split = ' '.join(summary_splits[len(output_images):])
            summary_splits = summary_splits[:len(output_images)] + [last_split]
        
        # construct precision samples
        if self.max_context_tokens > 0:
            context_len = self.max_context_tokens // 2
            image_precision_sample_list = [
                {
                    'image': image,
                    'context_above': self.model.trim_text(text=''.join(summary_splits[:i + 1]), max_tokens=context_len, side='left'),
                    'context_below': self.model.trim_text(text=''.join(summary_splits[i + 1:]), max_tokens=context_len, side='right')
                } for i, image in enumerate(output_images)
            ]
        else:
            image_precision_sample_list = [
                {
                    'image': image,
                    'context_above': ''.join(summary_splits[:i + 1]),
                    'context_below': ''.join(summary_splits[i + 1:])
                } for i, image in enumerate(output_images)
            ]
        
        return {
            'text_sample': text_sample,
            'image_precision_samples': image_precision_sample_list
        }
        
    def prepare_annotation_sample(
        self,
        query_content: str,
        webpages: List[Dict],
        aux_images: List[Dict],
        raw_summary: str,
        output_images: List[Dict]
    ) -> List[Dict]:
        text_sample = {
            'user_input': query_content,
            'response': raw_summary.replace(IMAGE_PLACEHOLDER, ''),
            'retrieved_contexts': [webpage['content'] for webpage in webpages] + [image['detailed_image_caption'] for image in output_images],
            'reference': raw_summary.replace(IMAGE_PLACEHOLDER, '')
        }
        
        cited_image_ids = [image['id'] for image in output_images]
        
        images = []
        for webpage in webpages:
            webpage_images = webpage['images']
            for image in webpage_images:
                image['id'] = len(images)
                if image['id'] in cited_image_ids:
                    image['cited'] = True
                else:
                    image['cited'] = False
                images.append(image)
        
        for image in aux_images:
            image['id'] = len(images)
            if image['id'] in cited_image_ids:
                image['cited'] = True
            else:
                image['cited'] = False
            images.append(image)
            
        summary_splits = raw_summary.split(IMAGE_PLACEHOLDER)
        if len(summary_splits) != len(output_images) + 1:
            logging.warning(f"there are {len(summary_splits)} splits with splitter `{IMAGE_PLACEHOLDER}`, but got {len(output_images)} output images, which is not a match. Ignore redundant image placeholders...")
            last_split = ' '.join(summary_splits[len(output_images):])
            summary_splits = summary_splits[:len(output_images)] + [last_split]
        
        # construct precision samples
        if self.max_context_tokens > 0:
            context_len = self.max_context_tokens // 2
            image_precision_sample_list = [
                {
                    'image': image,
                    'context_above': self.model.trim_text(text=''.join(summary_splits[:i + 1]), max_tokens=context_len, side='left'),
                    'context_below': self.model.trim_text(text=''.join(summary_splits[i + 1:]), max_tokens=context_len, side='right')
                } for i, image in enumerate(output_images)
            ]
        else:
            image_precision_sample_list = [
                {
                    'image': image,
                    'context_above': ''.join(summary_splits[:i + 1]),
                    'context_below': ''.join(summary_splits[i + 1:])
                } for i, image in enumerate(output_images)
            ]
        
        return {
            'text_sample': text_sample,
            'image_precision_samples': image_precision_sample_list
        }
        
    def get_evaluation_result(self, prompt: str, images: List[Dict] = [], upper_bound: int = None, lower_bound: int = None):
        text_splits = prompt.split(IMAGE_PLACEHOLDER)
        assert len(text_splits) == len(images) + 1
        
        messages = [
            {
                'type': 'text',
                'text': text_splits[0]
            }
        ]
        for text_split, image in zip(text_splits[1:], images):
            messages.extend([
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': image['cached_image_url']
                    }
                },
                {
                    'type': 'text',
                    'text': text_split
                }
            ])
        result = self.model.chat(messages=messages)
        score = self.extract_score_from_str(text=result['response'])
        normed_score = max(lower_bound, min(upper_bound, score)) / upper_bound
        result.update({
            'score': score,
            'normed_score': normed_score
        })
        return result

    async def get_evaluation_result_ragas(self, scorer: Any, sample: Any):
        await scorer.single_turn_ascore(sample)
    