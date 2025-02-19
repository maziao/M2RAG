import re
from typing import Dict
from src.model import build_model
from src.question_filter import QUESTION_FILTER


IS_QUESTION_PROMPT_TEMPLATE = """# Task Description
You are a useful text classification assistant. You will be provided with a piece of text to be recognized. Your task is to classify the text with the guidelines.

# Guidelines
1. If the input text is not a question, the description is unclear, you are supposed to generate a single integer 0, which means it is an invalid question.
2. If there is no need to cite rich references to answer the question, i.e. the question can be fully addressed with a simple answer, you are supposed to generate a single integer 0, which means it is an invalid question.
3. If the question is not classified as invalid question according to the above two rules, then generate a single integer 1.

# Output Format
Your output should consist of two lines.
- The first line is your brief analysis of the input question content and why or why not it meets the requirements;
- The second line is an single integer 0 or 1.
Do not generate any other information other than the analysis and category indices.

# Input Text
{text}

# Output
"""


NEED_IMAGE_PROMPT_TEMPLATE = """# Task Description
You are a useful question classification assistant. Your task is to classify the input question with the guidelines.

# Guidelines
1. If inserting appropriate images in the answer to this question can provide a more comprehensive answer and make it easier for users to understand the content being discussed, then generate a single integer 1.
2. Otherwise, generate a single integer 0.

# Output Format
Your output should consist of two lines.
- The first line is your brief analysis of the input question content and why or why not it meets the requirements;
- The second line is an single integer 0 or 1.
Do not generate any other information other than the analysis and category indices.

# Input Question
{question}

# Output
"""


CLASSIFY_PROMPT_TEMPLATE = """# Task Description
You are a useful question classification assistant. Your task is to classify the input question with the guidelines.

# Category List
1. Society & Culture: This category encompasses questions about traditions, languages, customs, behaviors, and societal norms. Images can serve as a visual reference to cultural landmarks, traditional attire, or historical events that support the textual explanation.
Example:  What is the significance of the Day of the Dead celebration in Mexican culture?
2. Science & Mathematics: Questions in this category often deal with complex concepts that can be enhanced with diagrams, charts, or images. These visual aids can help in understanding scientific phenomena or mathematical theories.
Example: How does photosynthesis work in plants?
3. Health: Health-related questions often benefit from the use of images like anatomical diagrams, charts, or photos for better understanding of medical conditions, treatments, or fitness exercises.
Example: What are the symptoms and treatment options for carpal tunnel syndrome?
4. Education & Reference: This category includes academic topics and general knowledge where explanatory images, instructional graphics, and reference tables can provide clarity and enhance learning.
Example: What are the main parts of a plant cell and their functions?
5. Computers & Internet: Questions about technology, software, and internet usage can be more effectively answered with screenshots, infographics, and step-by-step visual guides.
Example: How do I set up a Wi-Fi network at home?
6. Sports: Sports-related queries can be explained better with visual demonstrations, play diagrams, athlete photos, and equipment images to illustrate techniques, rules, or historical moments.
Example: What are the basic rules of soccer?
7. Business & Finance: Financial and business concepts often involve data, charts, and visualizations that can make complex information more digestible and insightful.
Example: How does compound interest work and what impact can it have on investments?
8. Entertainment & Music: This category includes questions about movies, TV shows, music, and celebrities where images of album covers, scene snapshots, or charts of music theory can enrich the content.
Example: What was the cultural impact of the TV show 'Friends'?
9. Family & Relationships: Questions about family dynamics, parenting, and social relationships often benefit from illustrative images such as family diagrams, photos, and infographics about communication techniques or social scenarios.
Example: What are some effective conflict resolution strategies for couples?
10. Politics & Government: Political and governmental questions can be clarified with the help of charts, maps, and historical photographs that provide visual context to legislative processes, electoral maps, or political events.
Example: How does the electoral college system work in the United States?
11. Others

# Classification Guidelines
Select one or more categories from categories 1-10 that are most relevant to the content of the question as the category of the question. If the question does not belong to any of the categories 1-10, then classify it into category 11.

# Precautions
1. If the input text is classified into category 11, it cannot be assigned any other category.
2. Please first briefly analyze the content of the input text and find its association with the given category, and then output category indices based on the above analysis.

# Output Format
Your output should consist of two lines.
- The first line is your brief analysis of the input text content and its relationship with the given category;
- The second line consists of a list of integers separated by a space. Each integer represents the index of a category you assigned to the input question.
Do not generate any other information other than the analysis and category indices.

# Input Question
{question}

# Output
"""


@QUESTION_FILTER.register_module
class QuestionFilter:
    def __init__(self, model_config: Dict = {}) -> None:
        self.model = build_model(model_config)
    
    def _chat_single_round(self, prompt: str):
        """Get chat response from LLM

        Args:
            prompt (str): text input

        Return:
            str: text response from LLM
        Raises:
            NotImplementedError
        """
        return self.model.chat(
            messages=[
                {
                    'type': 'text',
                    'text': prompt
                }
            ]
        )
    
    def _get_index(self, text: str):
        indices = [int(index) for index in re.findall(r"\d+", text)]
        if len(indices) == 0:
            return 0
        if indices[0] not in [0, 1]:
            return 0
        return indices[0]
    
    def _get_indices(self, text: str):
        indices = [int(index) for index in re.findall(r"\d+", text)]
        if 0 in indices or len(indices) == 0:
            indices = [0]
        if 11 in indices:
            indices = [11]
        return indices
    
    def filter(self, question: str):
        is_question_prompt = IS_QUESTION_PROMPT_TEMPLATE.format(text=question)
        is_question_response = self._chat_single_round(prompt=is_question_prompt)
        is_question = self._get_index(text=is_question_response)
        
        if is_question == 1:
            need_image_prompt = NEED_IMAGE_PROMPT_TEMPLATE.format(question=question)
            need_image_response = self._chat_single_round(prompt=need_image_prompt)
            need_image = self._get_index(text=need_image_response)
            
            if need_image == 1:
                category_prompt = CLASSIFY_PROMPT_TEMPLATE.format(question=question)
                category_response = self._chat_single_round(prompt=category_prompt)
                category_indices = self._get_indices(text=category_response)
            else:
                category_response = None
                category_indices = None
        else:
            need_image_response = None
            need_image = None
            category_response = None
            category_indices = None
            
        return {
            "is_question_response": is_question_response,
            "is_question": is_question,
            "need_image_response": need_image_response,
            "need_image": need_image,
            "category_response": category_response,
            "category_indices": category_indices
        }
