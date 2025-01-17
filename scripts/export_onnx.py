# /// script
# requires-python = "==3.9.7"
# dependencies = [
#     "funasr @ git+https://github.com/modelscope/FunASR.git@23c6d672881aedabd89e37397e04177f088eccf9",
#     "onnx",
#     "setuptools",
#     "torch",
#     "torchaudio",
# ]
# ///

"""
Run the following from the scripts folder: uv run export_onnx.py
"""


from funasr import AutoModel
import os

model = AutoModel(
    model="iic/emotion2vec_base",
    hub="ms"
)

res = model.export(type="onnx", quantize=False, opset_version=13, device='cpu', output_dir=".")
os.rename('emotion2vec', 'emotion2vec.onnx')
