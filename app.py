from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import serial  # for COM port communication
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = joblib.load('fire_detection_model.pkl')

ser = serial.Serial('COM8', 9600, timeout=1)

sensor_data = {
    'temp': 0,
    'RH': 0
}

def read_from_com_port():
    global sensor_data
    while True:
        if ser.in_waiting > 0:
            # Read data from COM8 (assumes a format like "temp:30,RH:45")
            line = ser.readline().decode('utf-8').strip()
            if line:
                # Parse temperature and humidity
                try:
                    data_parts = line.split(',')
                    temp_value = float(data_parts[0].split(':')[1])
                    RH_value = float(data_parts[1].split(':')[1])
                    sensor_data = {'temp': temp_value, 'RH': RH_value}
                    print(f"Received from COM8: {sensor_data}")
                except (IndexError, ValueError) as e:
                    print(f"Error parsing data: {e}")

com_thread = threading.Thread(target=read_from_com_port)
com_thread.daemon = True
com_thread.start()

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Use the latest sensor data
    input_data = pd.DataFrame({
        'temp': [sensor_data['temp']],
        'RH': [sensor_data['RH']]
    })
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': prediction, 'temp': sensor_data['temp'], 'RH': sensor_data['RH']})

if __name__ == '__main__':
    app.run(debug=True)
