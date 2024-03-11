import hopsworks
import h2o 
from flask import Flask, request, jsonify

project = hopsworks.login()
fs = project.get_feature_store()

app = Flask(__name__)

def load_model():
    h2o.init()
    mr = project.get_model_registry()
    retrieved_model = mr.get_model(name="earthquake_prediction_model", version=2)
    saved_model_dir = retrieved_model.download()
    mojo_file = saved_model_dir + "/best_model.mojo"
    return h2o.import_mojo(mojo_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.json
        test_data = h2o.H2OFrame(json_data)
        predictions = loaded_model.predict(test_data)
        predictions_list = predictions.as_data_frame().values.tolist()
        return jsonify(predictions=predictions_list)
    except Exception as e:
        return jsonify(error=str(e))
 

loaded_model = load_model()


  
