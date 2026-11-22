import json
import argparse

import ImageReward as RM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--med_config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.input_json:
        if args.prompt is not None or args.image_path is not None:
            raise ValueError("Use either --input_json or --prompt/--image_path, not both.")
        with open(args.input_json, "r") as f:
            payload = json.load(f)
        pairs = payload["pairs"]
    else:
        if args.prompt is None or args.image_path is None:
            raise ValueError("Single-image mode requires both --prompt and --image_path.")
        pairs = [
            {
                "prompt": args.prompt,
                "image_path": args.image_path,
            }
        ]

    model = RM.load(
        name=args.ckpt,
        med_config=args.med_config,
        device=args.device,
    )

    scores = []
    for item in pairs:
        prompt = item["prompt"]
        image_path = item["image_path"]
        score = model.score(prompt, image_path)
        if isinstance(score, (list, tuple)):
            score = score[0]
        scores.append(float(score))

    result = {
        "mean": sum(scores) / len(scores) if scores else None,
        "scores": scores,
        "count": len(scores),
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f)
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
