import re
import copy
from typing import Dict, List
from src.summarizer import SUMMARIZER
from src.summarizer.base import BaseSummarizer


IMAGE_PLACEHOLDER = "<IMAGE_PLACEHOLDER>"


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
class MultiModalSummarizer(BaseSummarizer):
    def __init__(
        self,
        name: str = None,
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
        output_detailed_caption: bool = False
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
        
        # max number of images (global)
        self.max_images = max_images        
        # max number of images (webpage)
        self.max_images_per_sample = max_images_per_sample
        # max number of auxiliary images
        self.max_aux_images = max_aux_images
        # max score of reserved images
        self.max_image_score = max_image_score
        # min score of reserved images
        self.min_image_score = min_image_score
        
        self.output_detailed_caption = output_detailed_caption
        
    def get_image_url_from_image(self, image: Dict) -> str:
        image_url = None
        if self.use_orig_image:
            image_url = image['image_url']
        else:
            image_url = image['cached_image_url']
        assert isinstance(image_url, str)
        return image_url
    
    def get_pseudo_image_url(self, image_id: int, cached_image_url: str) -> str:
        """Get pseudo image url with image index and cached image url (for extension).

        Args:
            image_id (int): index of the image in the input.
            cached_image_url (str): cached image url.

        Returns:
            str: images/{image_id}.{image_ext}
        """
        image_format = cached_image_url.split('.')[-1]
        return f"images/{image_id}.{image_format}"

    def replace_image_url_in_messages(self, messages: List[Dict], images: List[Dict]) -> List[Dict]:
        """Replace all image urls in the messages with original image url.

        Args:
            messages (List[Dict]): a list of messages
            images (List[Dict]): a list of images

        Returns:
            List[Dict]: the replaced messages
        """
        num_images_in_messages = sum([message['type'] in ['image_url', 'image_caption'] for message in messages])
        assert num_images_in_messages == len(images)
        
        counter = 0
        for message in messages:
            if message['type'] == 'image_url':
                message['image_url']['url'] = images[counter]['image_url']
                counter += 1
        return messages

    def handle_output_images(self, text: str, images: List[Dict] = []) -> Dict:
        candidate_image_placeholders = re.findall(r"\!\[(.*?)\]\((.*?)(\d+)(.*?)\)", text)
        
        output_images = []
        output_indices = []
        num_images = 0
        text_with_placeholders = copy.deepcopy(text)
        for placeholder in candidate_image_placeholders:
            placeholder_str = f"![{placeholder[0]}]({''.join(placeholder[1:])})"
            image_caption = placeholder[0].strip()
            image_index = int(placeholder[2])
            width = 60
            
            # ensure the selected image is valid, i.e. within the index range and does not have duplicates
            if image_index < len(images) and image_index not in output_indices:
                if isinstance(image_caption, str) and len(image_caption) > 0:
                    image_struct = IMAGE_STRUCT_WITH_CAPTION.format(image_url=images[image_index]['image_url'], width=width, image_caption=image_caption)
                else:
                    image_struct = IMAGE_STRUCT.format(image_url=images[image_index]['image_url'], width=width)
                
                if self.output_detailed_caption:
                    image_struct += f"\n[!] Image {image_index}{' - aux' if images[image_index]['is_aux_image'] else ''}; Original image caption: {images[image_index]['detailed_image_caption']}[!]\n"
                
                num_images += 1
                new_placeholder = IMAGE_PLACEHOLDER
                output_images.append({
                    'id': image_index,
                    'image_caption': image_caption,
                    'detailed_image_caption': images[image_index]['detailed_image_caption'],
                    'image_width': width,
                    'image_url': images[image_index]['image_url'],
                    'cached_image_url': images[image_index]['cached_image_url']
                })
                output_indices.append(image_index)
            else:
                image_struct = ''
                new_placeholder = ''
            text = text.replace(placeholder_str, image_struct)
            text_with_placeholders = text_with_placeholders.replace(placeholder_str, new_placeholder)
        
        return {
            'processed_response': text,
            'placeholder_response': text_with_placeholders,
            'output_images': output_images,
            'num_images': num_images
        }
