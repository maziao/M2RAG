import logging
from typing import List
try:
    from imagededup.methods import PHash
except Exception:
    PHash = None
    logging.warning(f"PHash from imagededup.methods is not currently imported, `PHashImageDeduplicateAgent` cannot be used.")
from src.image_handler.image_dedup import IMAGE_DEDUP
from src.image_handler.image_dedup.image_dedup import ImageDeduplicateAgent


@IMAGE_DEDUP.register_module
class PHashImageDeduplicateAgent(ImageDeduplicateAgent):
    def __init__(self) -> None:
        super().__init__()
        self.phasher = PHash()

    def get_duplicate_images(self, image_dir: str, **kwargs) -> List[str]:
        encodings = self.phasher.encode_images(image_dir=image_dir)
        duplicates = self.phasher.find_duplicates_to_remove(encoding_map=encodings)
        return duplicates
