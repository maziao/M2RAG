import os
import json
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-file', type=str, required=True)
    parser.add_argument('--image-root', type=str, default=None)
    args = parser.parse_args()

    if args.image_root is None:
        image_root = os.environ.get('IMAGE_ROOT')
    else:
        image_root = args.image_root
    assert image_root is not None
    
    with open(args.raw_file, 'r+', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f.readlines()]

    new_dataset = []
    for sample in dataset:
        new_sample = {
            'id': sample['id'],
            'messages': [
                {
                    'role': 'user',
                    'content': sample['model_input']
                },
                {
                    'role': 'assistant',
                    'content': f"```markdown\n{sample['model_response']}\n```"
                }
            ]
        }
        if 'images' in sample:
            new_sample['images'] = [os.path.join(image_root, image) for image in sample['images']]
            for image in new_sample['images']:
                assert os.path.exists(image), image
        new_dataset.append(new_sample)

    output_dir = os.path.dirname(args.raw_file)
    output_file = os.path.join(output_dir, 'llama_factory.json')
    
    random.seed(3407)
    random.shuffle(new_dataset)

    with open(output_file, 'w+', encoding='utf-8') as f:
        json.dump(obj=new_dataset, fp=f, ensure_ascii=False, indent=4)
