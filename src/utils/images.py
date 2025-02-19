import base64
import requests
from io import BytesIO
from typing import Tuple
from PIL import Image, UnidentifiedImageError
try:
    import cairosvg
except Exception:
    print(f"[!] Warning: package `cairosvg` is not currently imported, .svg images will be ignored.")
    cairosvg = None


def convert_image_url_to_base64(image_url: str, size: Tuple[int, int] = None) -> str:
    session = requests.Session()
    response = session.get(url=image_url, timeout=3.0)
    image_file = BytesIO(response.content)
    
    if size is not None:
        image_file = resize_image(image_file=image_file, size=size)
        
    image_format = Image.open(image_file).format
    
    byte_data = image_file.getvalue()
    
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{image_format};base64,' + base64_str


def convert_image_path_to_base64(image_path: str, size: Tuple[int, int] = None) -> str:
    if image_path.startswith('file://'):
        image_path = image_path[7:]
        
    with open(image_path, "rb") as f:
        byte_data = f.read()
        image_file = BytesIO(byte_data)

    if size is not None:
        image_file = resize_image(image_file=image_file, size=size)
        
    image_format = Image.open(image_file).format
    
    byte_data = image_file.getvalue()
    
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return f'data:image/{image_format};base64,' + base64_str


def resize_image(image_file: BytesIO, size: Tuple[int, int]) -> BytesIO:
    pil_image = Image.open(image_file)
    image_format = pil_image.format
    if pil_image.width <= size[0] and pil_image.height <= size[1]:
        return image_file
    
    try:
        pil_image.thumbnail(size=size)
    except Exception:
        pil_image = pil_image.resize(size=size)

    resized_image_file = BytesIO()
    pil_image.save(resized_image_file, format=image_format)
    del pil_image
    return resized_image_file
    

def convert_image_url_to_pil_image(image_url: str):
    try:
        session = requests.Session()
        response = session.get(url=image_url, timeout=3.0)
    except Exception:
        return None
    
    try:
        img = Image.open(BytesIO(response.content))
        if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
            img.load()
            img = img.convert('RGB')
            tmp_img = BytesIO()
            img.save(tmp_img, "JPEG")
            img = Image.open(tmp_img)
            del tmp_img
    except UnidentifiedImageError:
        if cairosvg is not None:
            try:
                tmp_img = BytesIO()
                cairosvg.svg2png(file_obj=BytesIO(response.content), write_to=tmp_img)
                img = Image.open(tmp_img)
                img.load()
                img = img.convert('RGB')
                del tmp_img
                
                tmp_img = BytesIO()
                img.save(tmp_img, "JPEG")
                img = Image.open(tmp_img)
                del tmp_img
            except Exception:
                img = None
        else:
            img = None
        
    if img is not None:
        try:
            tmp_img = BytesIO()
            img.save(tmp_img, img.format)
            del tmp_img
        except Exception:
            img = None
    
    return img
