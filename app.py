from predict import dispatch_with_config
import os
import io
import re
import base64

from PIL import Image
from flask import Flask, request, jsonify

# python3.6 predict.py --experiment mask_classifier --classes with_mask,without_mask --network mobilenet_v2 -lh 224 -lw 224 --resume ./checkpoints/lr_p001_with_mask_without_mask_224-mobilenet_v2-Epoch-211-Iteration-7400.pth <path_to_image>

model_config = {
    'classes': ['with_mask', 'without_mask'],
    'network': 'mobilenet_v2',
    'height': 224,
    'width': 224,
    'experiment': 'mask_classifier'
}
# from tf_model_helper import TFModel

app = Flask(__name__)

# Path to signature.json and model file
MODEL = 'lr_p001_with_mask_without_mask_224-mobilenet_v2-Epoch-211-Iteration-7400.pth'
MODEL_PATH = os.path.join('./model', MODEL)

print(MODEL_PATH)


@app.route('/ping', methods=["GET"])
def ping():
    print('Got a ping')
    response = {'message': 'pong'}
    return jsonify(response)


@app.route('/predict', methods=["POST"])
def predict_image():
    print('Incoming request!')
    req = request.get_json(force=True)
    image = _process_base64(req)
    model_config['model_path'] = MODEL_PATH
    predicted_class = dispatch_with_config(model_config, './incoming.png')
    response = {'message': 'success', 'prediction': predicted_class}
    return jsonify(response)


def _process_base64(json_data):
    image_data = json_data.get("image")
    image_data = re.sub(r"^data:image/.+;base64,", "", image_data)
    image_base64 = bytearray(image_data, "utf8")
    image = base64.decodebytes(image_base64)
    input_image = Image.open(io.BytesIO(image))
    input_image.save('incoming.png')
    return input_image


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
