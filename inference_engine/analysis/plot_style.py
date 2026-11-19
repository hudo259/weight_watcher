"""Labels, orders and colors shared by the paper figure scripts."""


MODEL_LABELS = {
    "convnextv2_base":  "ConvNeXt V2 Base",
    "cyclemlp_b2":      "CycleMLP B2",
    "mit_b2":           "MiT B2",
}

MODEL_ORDER = ["convnextv2_base", "cyclemlp_b2", "mit_b2"]


# Okabe-Ito colorblind-safe palette, paired by task:
# blues = classification, oranges = segmentation. Hatching adds a grayscale cue.
DATASET_COLORS = {
    "imagenet100":   "#0072B2",   # deep blue
    "isic2019":      "#56B4E9",   # sky blue
    "voc2012_sbd":   "#E69F00",   # amber
    "isic2018":      "#D55E00",   # vermillion
}
DATASET_HATCH = {
    "imagenet100":   "",
    "isic2019":      "///",
    "voc2012_sbd":   "",
    "isic2018":      "///",
}
DATASET_EDGE = {
    "imagenet100":   "#004E7C",
    "isic2019":      "#2E80A8",
    "voc2012_sbd":   "#A07000",
    "isic2018":      "#9A3E00",
}

DATASET_LABELS = {
    "imagenet100":   "ImageNet-100",
    "isic2019":      "ISIC2019",
    "voc2012_sbd":   "VOC2012",
    "isic2018":      "ISIC2018",
}


# Layer-type colors, distinct from the dataset palette above
LAYER_COLORS = {
    "conv2d":           "#009E73",   # green
    "depthwise_conv":   "#CC79A7",   # pink-purple
    "linear":           "#4477AA",   # slate blue
    "attention_output": "#EE7733",   # orange
    "attention_query":  "#AA3377",   # magenta-purple
}

# Short enough for narrow axes
LAYER_LABELS = {
    "conv2d":           "Conv2D",
    "depthwise_conv":   "DW Conv",
    "linear":           "Linear",
    "attention_output": "Attn-O",
    "attention_query":  "Attn-Q",
}

# Canonical order; each model shows only the types it has
LAYER_ORDER = [
    "conv2d", "depthwise_conv", "linear",
    "attention_output", "attention_query",
]
