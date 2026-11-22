import json
import os
import random

from torch.utils.data import Dataset


class CalibDataset(Dataset):
    """Calibration dataset for prompt-only generation collection."""

    def __init__(
        self,
        prompt_file="eval/geneval/prompts/evaluation_metadata_long.jsonl",
        num_samples_gen=4,
        seed=42,
    ):
        self.prompt_file = prompt_file
        self.num_samples_gen = num_samples_gen
        self.seed = seed
        self.prompts = self._load_prompts(prompt_file)

        if not self.prompts:
            raise ValueError(f"No valid prompts found in '{prompt_file}'.")

        if num_samples_gen > 0:
            if len(self.prompts) < num_samples_gen:
                raise ValueError(
                    f"Prompt file '{prompt_file}' contains {len(self.prompts)} prompts, "
                    f"but {num_samples_gen} were requested."
                )
            rng = random.Random(seed)
            self.prompts = rng.sample(self.prompts, num_samples_gen)

    def _load_prompts(self, prompt_file):
        if not os.path.exists(prompt_file):
            raise ValueError(f"Prompt file path '{prompt_file}' does not exist.")

        suffix = os.path.splitext(prompt_file)[1].lower()
        if suffix == ".json":
            return self._load_json_prompts(prompt_file)
        if suffix == ".jsonl":
            return self._load_jsonl_prompts(prompt_file)
        raise ValueError(
            f"Prompt file '{prompt_file}' must be a JSON or JSONL file. "
            "Expected a COCO-caption style JSON or GenEval-style metadata JSONL."
        )

    def _load_json_prompts(self, prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = []
        if isinstance(data, list):
            for item in data:
                item_prompts = self._extract_prompts_from_item(item)
                if item_prompts:
                    prompts.append(item_prompts[0])
        elif isinstance(data, dict):
            for key in ("data", "annotations", "images", "samples"):
                value = data.get(key)
                if isinstance(value, list):
                    for item in value:
                        item_prompts = self._extract_prompts_from_item(item)
                        if item_prompts:
                            prompts.append(item_prompts[0])
                    break
        return prompts

    def _load_jsonl_prompts(self, prompt_file):
        prompts = []
        with open(prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_prompts = self._extract_prompts_from_item(item)
                if item_prompts:
                    prompts.append(item_prompts[0])
        return prompts

    def _extract_prompts_from_item(self, item):
        if not isinstance(item, dict):
            return []

        prompts = []
        prompt = item.get("prompt")
        caption = item.get("caption")
        captions = item.get("captions")
        sentences = item.get("sentences")

        if isinstance(prompt, str) and prompt.strip():
            prompts.append(prompt.strip())
        elif isinstance(prompt, list):
            prompts.extend([text.strip() for text in prompt if isinstance(text, str) and text.strip()])

        if isinstance(caption, str) and caption.strip():
            prompts.append(caption.strip())
        elif isinstance(caption, list):
            prompts.extend([text.strip() for text in caption if isinstance(text, str) and text.strip()])

        if isinstance(captions, str) and captions.strip():
            prompts.append(captions.strip())
        elif isinstance(captions, list):
            prompts.extend([text.strip() for text in captions if isinstance(text, str) and text.strip()])

        # Karpathy Flickr8k/Flickr30k JSON stores the five captions for an
        # image as [{"raw": "...", ...}, ...]. Keep one prompt per image,
        # consistently with COCO records containing a caption list.
        if isinstance(sentences, list):
            prompts.extend(
                sentence["raw"].strip()
                for sentence in sentences
                if isinstance(sentence, dict)
                and isinstance(sentence.get("raw"), str)
                and sentence["raw"].strip()
            )

        return prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "mode": "Gen",
            "file_name": f"{idx:05d}.png",
        }
