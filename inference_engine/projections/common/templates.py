import itertools
import re

#dict = {
#    # REQUIRED: Templates (what to decompose)
#    "templates": [
#        
#        # Simple pattern (single layer)
#        {
#            "pattern": "conv1.weight",
#            "type": "conv2d"
#        },
#        
#        # Pattern with indices (expands to multiple layers)
#        {
#            "pattern": "layer{i}.conv{j}.weight",
#            "indices": {
#                "i": [1, 2, 3, 4],      # list of specific values
#                "j": range(0, 3)        # Or Python range
#            },
#            "type": "conv2d"
#        },
#        
#        # Multiple indices (cartesian product)
#        {
#            "pattern": "encoder.layer.{i}.attention.{proj}.weight",
#            "indices": {
#                "i": range(0, 12),      # 12 layers
#                "proj": ["query", "key", "value"]  # 3 projections
#            },
#            "type": "attention_query"   # All get same type
#        },
#        
#        # Optional: Per-layer k (overrides global)
#        {
#            "pattern": "fc.weight",
#            "type": "linear",
#            "k": 10  # Use 10 components instead of global k=3
#        },
#
#        # Multihead attention per head decomposition
#        {
#            "pattern": "encoder.layer.{i}.attention.self.query.weight",
#            "indices": {"i": range(0, 12)},
#            "type": "attention_mha_head",
#            "num_heads": 12
#        }
#    ],
#    
#    # OPTIONAL: Exclusions (blacklist on top of whitelist)
#    "exclude": [
#        "*.bn*.weight",      # Batch norm weights
#        "*.bn*.bias",        # Batch norm biases
#        "*running_mean",     # Running statistics
#        "*running_var",
#        "*num_batches_tracked",
#        "*embedding*",       # Embeddings (often want to skip)
#        "*.bias"            # All biases (usually skip)
#    ],
#    
#    # OPTIONAL: Metadata (for organization)
#    "name": "ResNet50",
#    "description": "Standard ResNet50 decomposition",
#    "architecture": "resnet50",
#    "num_layers": 50,
#    "notes": "Decomposes all conv and fc layers, skips batch norm",
#    
#    # OPTIONAL: Processing options
#    "options": {
#        "default_k": 3,           # Default k if not specified per-layer
#        "device": "cuda",         # 'cuda' or 'cpu'
#        "state_dict_key": None,   # Auto-detect or specify
#        "verbose": True,
#        "chunk_size": 256         # For memory-efficient processing
#    }
#}

# Supported layer types:
LAYER_TYPES = {
    # Convolutional layers
    "conv1d": "Conv1d [out_ch, in_ch, k]",
    "conv2d": "Standard Conv2d [out_ch, in_ch, kH, kW]",
    "depthwise_conv": "Depthwise separable conv [ch, 1, kH, kW]",
    
    # Linear layers
    "linear": "Fully connected [out_features, in_features]",
    
    # Attention layers
    "attention_query": "Attention query projection",
    "attention_key": "Attention key projection",
    "attention_value": "Attention value projection",
    "attention_output": "Attention output projection",
    "attention_fused_qkv": "Fused Q/K/V [3*dim, dim]",
    "attention_mha_head": "Per-head MHA projection [num_heads, head_dim*in_dim]",
    "attention_mha_head_fused_qkv": "Per-head fused QKV [3*num_heads, head_dim*in_dim]",
    
    # Special
    "custom": "Custom decomposition (requires handler)"
}

def expand_pattern(pattern: str, indices: dict[str, range | list[int]]) -> list[str]:
    """
    Expand a pattern like "encoder.layer.{i}.attn.{j}" with index ranges.
    
    Args:
        pattern: Pattern string with placeholders like {i}, {j}, etc.
        indices: Dictionary mapping placeholder names to their possible values
    
    Returns:
        List of expanded keys
    
    Example:
        >>> pattern = "encoder.layer.{i}.attn.fc{j}.weight"
        >>> indices = {"i": range(0, 3), "j": [1, 2]}
        >>> expand_pattern(pattern, indices)
        [
            "encoder.layer.0.attn.fc1.weight",
            "encoder.layer.0.attn.fc2.weight",
            "encoder.layer.1.attn.fc1.weight",
            "encoder.layer.1.attn.fc2.weight",
            "encoder.layer.2.attn.fc1.weight",
            "encoder.layer.2.attn.fc2.weight",
        ]
    """
    
    # Find all placeholders in pattern
    placeholders = re.findall(r'\{(\w+)\}', pattern)
    
    if not placeholders:
        return [pattern]
    
    # Get all index combinations
    index_values = []
    for placeholder in placeholders:
        if placeholder in indices:
            values = indices[placeholder]
            # Convert range to list
            if isinstance(values, range):
                values = list(values)
            index_values.append(values)
        else:
            raise ValueError(f"Placeholder '{placeholder}' not found in indices")
    
    # Generate all combinations
    expanded = []
    for combination in itertools.product(*index_values):
        # Replace placeholders with actual values
        key = pattern
        for placeholder, value in zip(placeholders, combination):
            key = key.replace(f"{{{placeholder}}}", str(value))
        expanded.append(key)
    
    return expanded


def filter_by_exclusions(layer_mapping: dict[str, str], 
                         exclusions: list[str]) -> dict[str, str]:
    """
    Remove keys matching exclusion patterns.
    
    Args:
        layer_mapping: Dictionary of key -> type
        exclusions: List of exclusion patterns (supports wildcards)
    
    Returns:
        Filtered layer mapping
    
    Example:
        >>> layer_mapping = {"layer1.bn.weight": "norm", "layer1.conv.weight": "conv2d"}
        >>> exclusions = ["*.bn.*", "*embedding*"]
        >>> filter_by_exclusions(layer_mapping, exclusions)
        {"layer1.conv.weight": "conv2d"}
    """
    import fnmatch
    
    filtered = {}
    
    for key, layer_type in layer_mapping.items():
        # Check if key matches any exclusion pattern
        excluded = False
        for pattern in exclusions:
            if fnmatch.fnmatch(key, pattern):
                excluded = True
                break
        
        if not excluded:
            filtered[key] = layer_type
    
    return filtered

def expand_templates(config: dict) -> dict[str, str | dict]:
    """
    Expand template patterns with index ranges into explicit key mappings.
    
    Args:
        config: Configuration with templates
    
    Returns:
        Expanded mapping of keys to types
    
    Example:
        >>> config = {
        ...     "templates": [
        ...         {
        ...             "pattern": "layer{i}.conv{j}.weight",
        ...             "indices": {"i": range(0, 3), "j": [1, 2]},
        ...             "type": "conv2d"
        ...         },
        ...         {
        ...             "pattern": "encoder.layer.{i}.attention.query.weight",
        ...             "indices": {"i": range(0, 12)},
        ...             "type": "attention_mha_head",
        ...             "num_heads": 12
        ...         }
        ...     ]
        ... }
        >>> layer_mapping = expand_templates(config)
        >>> print(layer_mapping)
        {
            "layer0.conv1.weight": "conv2d",
            ...
            "encoder.layer.0.attention.query.weight": {"type": "attention_mha_head", "num_heads": 12},
            ...
        }
    """
    
    layer_mapping = {}
    
    for template in config.get("templates", []):
        pattern = template["pattern"]
        indices = template.get("indices", {})
        layer_type = template["type"]
        
        # Generate all combinations of indices
        if indices:
            expanded_keys = expand_pattern(pattern, indices)
        else:
            # No indices, just the pattern itself
            expanded_keys = [pattern]
        
        for key in expanded_keys:
            extra = {}

            if "num_heads" in template:
                extra["num_heads"] = template["num_heads"]
            if "k" in template:
                extra["k"] = template["k"]
                layer_mapping[key] = {
                    "type": layer_type, **extra} if extra else layer_type
            else:
                layer_mapping[key] = layer_type

    if "exclude" in config:
        layer_mapping = filter_by_exclusions(layer_mapping, config["exclude"])

    return layer_mapping
