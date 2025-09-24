import argparse
import os
from pathlib import Path

import torch


def get_ar_annotations(annotations: list[dict], ref_path: str) -> list[dict]:
    ref_path = Path(ref_path)
    if ref_path.exists():
        imgs = list(ref_path.glob("*.png"))
        for p in imgs:
            p.unlink()
    ref_path.mkdir(parents=True, exist_ok=True)

    all_steps = []
    image_idx = 0
    for anno in annotations:
        for step_idx, step in enumerate(anno['annotation']):
            step_info = step['subclip']
            # generation order
            step_info['order'] = step_idx
            step_info['clip_id'] = image_idx

            if step_idx > 0:
                # the previous step's last frame is the reference frame of this step
                step_info['ref_frame'] = ref_path / f"{image_idx - 1}.png"
            else:
                # remove ref_frame for the first step
                if 'ref_frame' in step_info:
                    del step_info['ref_frame']
            all_steps.append(step_info)
            image_idx += 1

    all_steps.sort(key=lambda x: x['order'])
    return all_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/val_coin_video_4o_rag.pt")
    parser.add_argument("--output", type=str, default="/tmp/val_coin_llm_plan_ar.pt")
    parser.add_argument("--ref_path", type=str, default="ref_images/ar")
    parser.add_argument("--data_path", type=str, default="pvg_data")
    parser.add_argument("--config", type=str, default="configs/test.yml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("-y", action="store_true", help="Don't ask for confirmation")
    args = parser.parse_args()

    annotations = torch.load(args.input)
    output = get_ar_annotations(annotations, args.ref_path)
    torch.save(output, args.output)

    run_cmd = (f"python main.py test "
               f"-c {args.config} "
               f"--trainer.logger.tags+=image:ar "
               f"--trainer.callbacks+=src.image.autoregress.callback.SaveLastFrame "
               f"--trainer.callbacks.save_path={args.ref_path} "
               f"--data.test_annotation_path=[{args.output}] "
               f"--data.num_workers=4 "
               f"--data.prefetch_factor=2 "
               f"--data.prompt_type=llm_plan "
               f"--data.use_ref_frame=true "
               f"--data.video_dir={args.data_path} "
               f"--ckpt={args.ckpt} ")
    print(run_cmd)
    if args.y or input("Do you want to run the autoregress pipeline? (y/n)") == "y":
        os.system(run_cmd)
