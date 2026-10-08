# Vendored D-FINE-X inference

The model implementation, model YAML files, and license are copied from
`ascent/third_party/D-FINE`. The checkpoint is copied from ASCENT's
`pretrained_weights/dfine_x_obj2coco.pth` to LCN's
`weights/dfine_x_obj2coco.pth`.

Inference kernels and deploy conversion are unchanged. Package initializers
register only HGNetv2 and D-FINE inference components; training datasets,
criteria, optimizers, and FLOP profilers are not imported. Relative imports
keep registration isolated from other projects' generic `src` packages.
This is an inference-only vendor copy, not a training installation.

`vision_models/dfine_model.py` adapts ASCENT's `DFine.predict()` logic to LCN's
existing `set_classes()/detect()` interface. It uses RGB, direct 640×640 PIL
resize, `[0,1]` FP32 tensors, the original deploy postprocessor, and the
contiguous COCO-80 label order. No YOLO NMS or letterbox is added.

The ASCENT service filter is `score > 0.40`; LCN then accepts only the current
target category with `score >= 0.80`. Pixel XYXY boxes are clipped to image
bounds for compatibility with the existing SAM, screenshot, and Rerun paths.
No ASCENT source tree or model server is needed at runtime.
