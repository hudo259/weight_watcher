import torch
import logging
from pathlib import Path

def print_checkpoint_keys_simple(checkpoint_path: str, 
                                 log_file_path: str,
                                 filter_pattern: str | None = None):
    """
    Simple version - just print keys with shapes.
    """
    logger = logging.getLogger("process_logger")
    logger.setLevel(logging.INFO)

    #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    #Save output to file, check if path exists and create if it doesn't
    save_path = Path(log_file_path)
    try:
        save_path.parents[0].mkdir(exist_ok=True, parents=True)

        file_handler = logging.FileHandler(str(save_path))
        file_handler.setLevel(logging.INFO)
        #file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(e)
        print("No logging to file")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    #console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"\nLoading: {checkpoint_path}")
    logger.info("=" * 80)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Detect state dict
    state_dict = {}
    for key in ['model_state', 'state_dict', 'model']:
        if key in ckpt:
            state_dict = ckpt[key]
            logger.info(f"State dict key: '{str(key)}'")
            break

    if len(state_dict) == 0:
        #skip non-tensor values in case ckpt itself contains mixed types 
        #e.g. epoch and best_metric floats:
        state_dict = {k: v for k, v in ckpt.items() 
                    if hasattr(v, 'shape')}
        logger.info("Checkpoint is directly a state dict")

    logger.info("=" * 80)
    
    # Get keys
    all_keys = sorted(state_dict.keys())
    
    if filter_pattern:
        keys = [k for k in all_keys if filter_pattern.lower() in k.lower()]
        logger.info(f"\nFiltered: {len(all_keys)} -> {len(keys)} keys (pattern: '{filter_pattern}')\n")
    else:
        keys = all_keys
        logger.info(f"\nTotal keys: {len(keys)}\n")
    
    # logger.info keys
    for i, key in enumerate(keys, 1):
        shape = tuple(state_dict[key].shape)
        dtype = state_dict[key].dtype
        logger.info(f"{i:4d}. {key:60s} {str(shape):20s} {dtype}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    total_params = sum(t.numel() for t in state_dict.values())
    logger.info(f"Total parameters: {len(keys)}")
    logger.info(f"Total elements: {total_params:,}")
    logger.info("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python print_checkpoint_keys.py <checkpoint path> <path to log file> [filter_pattern]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    log_file_path = sys.argv[2]
    filter_pattern = sys.argv[3] if len(sys.argv) > 3 else None
    
    print_checkpoint_keys_simple(checkpoint_path, log_file_path, filter_pattern)
