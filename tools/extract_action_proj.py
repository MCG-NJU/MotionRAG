import argparse
import torch
import os


def extract_action_proj_params(ckpt_path, output_path=None):
    """
    Extract parameters starting with 'action_proj' from a checkpoint file.

    Args:
        ckpt_path (str): Path to the checkpoint file
        output_path (str, optional): Path to save the extracted parameters
    """
    # Load the checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Get the state_dict (assuming it's directly in the checkpoint or under 'model' key)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint and 'state_dict' in checkpoint['model']:
        state_dict = checkpoint['model']['state_dict']
    else:
        state_dict = checkpoint  # Assume the checkpoint itself is the state_dict

    # Extract parameters starting with 'action_proj'
    action_proj_params = {}
    for key, value in state_dict.items():
        if key.startswith('action_proj_model.'):
            # Remove the 'action_proj' prefix
            new_key = key[len('action_proj_model.'):]
            action_proj_params[new_key] = value.clone()

    # Print statistics
    print(f"Total parameters in checkpoint: {len(state_dict)}")
    print(f"Extracted 'action_proj_model' parameters: {len(action_proj_params)}")

    # Save the extracted parameters
    torch.save(action_proj_params, output_path)
    print(f"Saved extracted parameters to {output_path}")

    return action_proj_params


def main():
    parser = argparse.ArgumentParser(description='Extract action_proj parameters from a checkpoint')
    parser.add_argument('ckpt_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--output', '-o', type=str, help='Path to save the extracted parameters', default=None)

    args = parser.parse_args()
    extract_action_proj_params(args.ckpt_path, args.output)


if __name__ == '__main__':
    main()