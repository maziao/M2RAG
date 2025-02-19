from typing import Optional, List, Dict
from src.model import build_model
from src.image_handler.image_caption import IMAGE_CAPTION


VLM_IMAGE_CAPTION_PROMPT_TEMPLATE = """# Task Description
You are an assistant for describing the content of images. You will be provided with an image and you are supposed to generate a detailed description for it.

# Image
<IMAGE_PLACEHOLDER>

# Precautions
- Your description should be grounded to the image, you are not expected to guess if any part of the image is blurred or not clear.
- Descibe the image as detailed as possible, please note that your description should reflect the priority of the content in the image.
- Generate the caption in a single line, do not generate any other information other than the detailed image caption.

# Image Caption
"""


VLM_CONCISE_IMAGE_CAPTION_PROMPT_TEMPLATE = """# Task Description
You are an assistant for describing the content of images. You will be provided with an image and you are supposed to generate a concise description for it.

# Image
<IMAGE_PLACEHOLDER>

# Precautions
- Your description should be grounded to the image, you are not expected to guess if any part of the image is blurred or not clear.
- Descibe the image concisely, please note that your description should reflect the priority of the content in the image.
- Generate the caption in a single line, do not generate any other information other than the concise image caption.
- Note that the concise caption should be a phrase or contain no more than 1 sentence.

# Concise Image Caption
"""


VLM_IMAGE_CAPTION_WITH_QUERY_PROMPT_TEMPLATE = """# Task Description
You are an assistant for describing the content of images. You will be provided with an image with a relevant query. You are supposed to generate a detailed description for it.

# Query
{query_content}

# Image
<IMAGE_PLACEHOLDER>

# Precautions
- Your description should be grounded to the image, you are not expected to guess if any part of the image is blurred or not clear.
- Descibe the image as detailed as possible, please note that your description should reflect the priority of the content in the image.
- Generate the caption in a single line, do not generate any other information other than the detailed image caption.
- You can refer to the provided query for your generation if the content of the image is related to the query. If they are irrelevant, just ignore the query.

# Image Caption
"""


VLM_IMAGE_CAPTION_WITH_CONTEXT_PROMPT_TEMPLATE = """# Task Description
You are an assistant for describing the content of images. You will be provided with an image and its textual context. You are supposed to generate a detailed description for it.

# Context Above
{context_above}

# Image
<IMAGE_PLACEHOLDER>

# Context Below
{context_below}

# Precautions
- Your description should be grounded to the image, you are not expected to guess if any part of the image is blurred or not clear.
- Descibe the image as detailed as possible, please note that your description should reflect the priority of the content in the image.
- Generate the caption in a single line, do not generate any other information other than the detailed image caption.
- You can refer to the provided context for your generation if the context is relevant with the image content. If they are irrelevant, just ignore the context.

# Image Caption
"""


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"


@IMAGE_CAPTION.register_module
class ImageCaptionAgent:
    def __init__(
        self,
        model_config: Dict = {},
        use_context: bool = False,
        use_query: bool = False,
        max_context_tokens: Optional[int] = None,
        concise: bool = False
    ) -> None:
        self.model = build_model(model_config)
        assert self.model.name.startswith('vlm'), f"the model for image captioning must be a VLM, but got {self.model.name}"
        
        self.use_context = use_context
        self.use_query = use_query
        self.max_context_tokens = max_context_tokens
        self.concise = concise
    
    def _get_image_context(self, text: str):
        text_splits = text.split(IMAGE_PLACEHOLDER)
        assert len(text_splits) == 2
        context_above = text_splits[0]
        context_below = text_splits[1]
        
        if self.max_context_tokens is not None:
            context_above = self.model.trim_text(
                text=context_above,
                max_tokens=self.max_context_tokens // 2,
                side='left'
            )
            context_below = self.model.trim_text(
                text=context_below,
                max_tokens=self.max_context_tokens // 2,
                side='right'
            )
        return {
            'context_above': context_above,
            'context_below': context_below
        }
    
    def get_image_caption_single(
        self,
        image_url: str,
        ref_text: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ):
        if self.use_context:
            assert context is not None
            context = self._get_image_context(text=context)
            prompt = VLM_IMAGE_CAPTION_WITH_CONTEXT_PROMPT_TEMPLATE.format(
                context_above=context['context_above'],
                context_below=context['context_below']
            )
            splits = prompt.split(IMAGE_PLACEHOLDER)
            assert len(splits) == 2
            response = self.model.chat(
                messages=[
                    {
                        'type': 'text',
                        'text': splits[0]
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_url
                        }
                    },
                    {
                        'type': 'text',
                        'text': splits[1]
                    }
                ]
            )
        elif self.use_query:
            assert ref_text is not None
            prompt = VLM_IMAGE_CAPTION_WITH_QUERY_PROMPT_TEMPLATE.format(query_content=ref_text)
        elif self.concise:
            prompt = VLM_CONCISE_IMAGE_CAPTION_PROMPT_TEMPLATE
        else:
            prompt = VLM_IMAGE_CAPTION_PROMPT_TEMPLATE
        
        splits = prompt.split(IMAGE_PLACEHOLDER)
        assert len(splits) == 2
        response = self.model.chat(
            messages=[
                {
                    'type': 'text',
                    'text': splits[0]
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_url
                    }
                },
                {
                    'type': 'text',
                    'text': splits[1]
                }
            ]
        )

        return response
    
    def get_image_captions(
        self,
        image_urls: List[str],
        **kwargs
    ):
        raise NotImplementedError
