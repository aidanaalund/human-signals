import onnx
from onnx import version_converter

model_path = "det.onnx"
model = onnx.load(model_path)
target_opset = 21

converted_model = version_converter.convert_version(model, target_opset)

onnx.save(converted_model, "det_v21.onnx")
print(f"Model successfully upgraded to Opset {target_opset}")