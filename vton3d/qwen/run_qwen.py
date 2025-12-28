import argparse
from pathlib import Path
import gc

from PIL import Image
import torch
from diffusers import QwenImageEditPlusPipeline


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the Qwen Image Edit batch processor.
    """
    parser = argparse.ArgumentParser(description="Batch Qwen Image Edit pipeline.")

    parser.add_argument(
        "--model_path",
        type=str,
        default="ovedrive/Qwen-Image-Edit-2509-4bit",
        help="Path or HuggingFace model ID."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing person images."
    )
    parser.add_argument(
        "--clothing_image",
        type=str,
        required=True,
        help="Path to the clothing image that should replace the original clothing."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the edited images."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Replace the person's original clothing with the clothing image. "
            "Do not invent new features. Keep the person identical: pose, body, "
            "face, hair, and background must remain unchanged."
        ),
        help="Editing prompt."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="change pose",
        help="Negative prompt."
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="CFG scale value."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed."
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="File extensions treated as images."
    )

    return parser.parse_args()


def load_pipeline(model_path: str) -> QwenImageEditPlusPipeline:
    """
    Load the Qwen Image Edit pipeline with CPU offload enabled.
    """
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_model_cpu_offload()
    return pipe


def get_image_files(source_dir: Path, extensions: list[str]) -> list[Path]:
    """
    Return all image files in a directory matching the allowed extensions.
    """
    exts = {e.lower() for e in extensions}
    return [
        p for p in sorted(source_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]


def clear_gpu_cache():
    """
    Clear GPU and CPU memory caches after each image.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_qwen_from_config_dict(qwen_cfg: dict):
    """
    Run the Qwen clothing edit batch using a config dictionary (e.g. cfg['qwen']).
    """
    model_path = qwen_cfg.get("model_path", "ovedrive/Qwen-Image-Edit-2509-4bit")
    source_dir = Path(qwen_cfg["source_dir"])
    clothing_path = Path(qwen_cfg["clothing_image"])
    output_dir = Path(qwen_cfg["output_dir"])

    prompt = qwen_cfg.get(
        "prompt",
        (
            "Replace the person's original clothing with the clothing image. "
            "Do not invent new features. Keep the person identical: pose, body, "
            "face, hair, and background must remain unchanged."
        ),
    )
    negative_prompt = qwen_cfg.get("negative_prompt", "change pose")
    true_cfg_scale = float(qwen_cfg.get("true_cfg_scale", 4.0))
    num_inference_steps = int(qwen_cfg.get("num_inference_steps", 20))
    seed = int(qwen_cfg.get("seed", 0))
    extensions = qwen_cfg.get("extensions", [".jpg", ".jpeg", ".png"])

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not clothing_path.is_file():
        raise FileNotFoundError(f"Clothing image not found: {clothing_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    clothing_image = Image.open(clothing_path).convert("RGB")
    image_files = get_image_files(source_dir, extensions)

    if not image_files:
        raise RuntimeError(f"No images found in {source_dir}.")

    pipeline = load_pipeline(model_path)
    base_generator = torch.Generator(device="cpu").manual_seed(seed)

    for img_path in image_files:
        person_image = Image.open(img_path).convert("RGB")
        generator = torch.Generator(device="cpu").manual_seed(seed)

        inputs = {
            "image": [person_image, clothing_image],
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": true_cfg_scale,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)

        output_image = output.images[0]
        out_path = output_dir / f"{img_path.stem}.png"
        output_image.save(out_path)

        clear_gpu_cache()


def main():
    """
    CLI entry point for running Qwen clothing edit from the command line.
    """
    args = parse_args()
    qwen_cfg = {
        "model_path": args.model_path,
        "source_dir": args.source_dir,
        "clothing_image": args.clothing_image,
        "output_dir": args.output_dir,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.true_cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "extensions": args.extensions,
    }
    run_qwen_from_config_dict(qwen_cfg)


if __name__ == "__main__":
    main()
