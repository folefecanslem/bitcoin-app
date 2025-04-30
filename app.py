import streamlit as st
import websocket
import threading
import json
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load your trained model and scaler
autoencoder = load_model("blockchain_autoencoder.h5")
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Threshold value (you can update this based on training)
anomaly_threshold = 0.02

# Shared state for Streamlit + WebSocket
st.set_page_config(page_title="Bitcoin Anomaly Detection", layout="wide")
st.title("🚨 Real-time Bitcoin Transaction Anomaly Detector")

latest_tx = st.empty()
status_display = st.empty()
mse_display = st.empty()

# === Feature Extraction Function ===
def extract_features(tx):
    input_val = sum(i['prev_out'].get('value', 0) for i in tx.get('inputs', []))
    output_val = sum(o.get('value', 0) for o in tx.get('out', []))
    num_inputs = len(tx.get('inputs', []))
    num_outputs = len(tx.get('out', []))
    fee = input_val - output_val if input_val > output_val else 0
    return [input_val, output_val, fee, num_inputs, num_outputs]

# === WebSocket Callbacks ===
def on_message(ws, message):
    tx_data = json.loads(message)
    tx = tx_data.get("x")
    if tx:
        features = extract_features(tx)
        scaled = scaler.transform([features])
        reconstructed = autoencoder.predict(scaled)
        mse = np.mean(np.square(scaled - reconstructed))
        label = "✅ Normal" if mse <= anomaly_threshold else "🚨 Anomalous"

        # Streamlit live updates
        latest_tx.markdown(f"**Transaction Hash**: `{tx['hash']}`")
        status_display.markdown(f"**Status**: {label}")
        mse_display.markdown(f"**Reconstruction MSE**: `{mse:.6f}`")

def on_open(ws):
    ws.send(json.dumps({"op": "unconfirmed_sub"}))

def on_error(ws, error):
    st.error(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    st.warning("WebSocket connection closed.")

# === Background Thread to Run WebSocket ===
def run_ws():
    ws = websocket.WebSocketApp("wss://ws.blockchain.info/inv",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

threading.Thread(target=run_ws, daemon=True).start()
