#!/usr/bin/env python3
"""Convert trained AngleNet PyTorch model to Core ML .mlpackage."""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
ONNX_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "model.onnx")
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "WatchSync", "WatchSync", "WatchSecondHand.mlpackage"
)


class AngleNetExport(nn.Module):
    """Simplified model for export — avoids ops that confuse coremltools."""

    def __init__(self):
        super().__init__()
        from torchvision import models

        backbone = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def main():
    import coremltools as ct

    print("Loading model checkpoint...")
    model = AngleNetExport()
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Tracing model...")
    example_input = torch.randn(1, 3, 224, 224)

    # Use torch.jit.trace
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 224, 224),
                scale=1.0 / (255.0 * 0.226),
                bias=[
                    -0.485 / 0.229,
                    -0.456 / 0.224,
                    -0.406 / 0.225,
                ],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )

    mlmodel.author = "WatchSync"
    mlmodel.short_description = "Predicts second hand angle (sin, cos) from watch face image"
    mlmodel.input_description["image"] = "224x224 RGB watch face image"
    mlmodel.output_description["output"] = "2-element vector: [sin(angle), cos(angle)]"

    print(f"Saving to {OUTPUT_PATH}...")
    mlmodel.save(OUTPUT_PATH)
    print("Done! Core ML model saved.")


if __name__ == "__main__":
    main()
