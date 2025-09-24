import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel


def calc_text_features(prompts: list[str], text_cache: dict, batch_size: int = 20,
                       model_name: str = "stabilityai/stable-diffusion-2-1", device: str = 'cuda:0',
                       output_path: str = None) -> dict:
    """
    Calculate text features for a list of prompts
    :param prompts: list of prompts
    :param text_cache: cache for the text features
    :param batch_size: batch size for the model
    :param model_name: name of the model
    :param device: device to use
    :param output_path: path to save the annotation
    :return: dict of prompts and text features
    """
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    model = text_encoder.to(device)

    # add empty prompt
    prompts.append('')
    # remove duplicates
    prompts = list(set(prompts) - set(text_cache.keys()))
    print(f"Calculating text features for {len(prompts)} prompts")

    chunked_prompts = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
    with torch.no_grad(), torch.cuda.amp.autocast():
        for chunk in tqdm(chunked_prompts):
            texts = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, ).input_ids.to(device)

            text_features = model(texts, return_dict=True).last_hidden_state[:, 0:1]
            text_features = text_features.to(dtype=torch.float32, device='cpu')

            for prompt, text_feature in zip(chunk, text_features):
                text_cache[prompt] = text_feature

    if output_path is not None:
        torch.save(text_cache, output_path)

    return text_cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default="../../data/coin.pt")
    parser.add_argument("--text_cache", type=str, default='../../data/text_cache/stable-diffusion-2-1.pt')
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()

    annotation = torch.load(args.annotation)
    prompts = [anno['prompt'] for anno in annotation]

    if Path(args.text_cache).exists():
        text_cache = torch.load(args.text_cache)
    else:
        Path(args.text_cache).parent.mkdir(parents=True, exist_ok=True)
        text_cache = {}

    text_cache = calc_text_features(prompts, text_cache=text_cache, batch_size=args.batch_size,
                                    model_name=args.model_name, device=args.device, output_path=args.text_cache)
