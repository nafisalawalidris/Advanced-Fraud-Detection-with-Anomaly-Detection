import requests

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# Define the payload with exactly 30 features (including Time)
payload = {
    "features": [
        0.0,  # Time
        -0.694242, -0.044075, 1.672773, 0.973366, -0.245117, 0.347068, 0.193679,  # V1-V7
        0.082637, 0.331128, -0.024923, 0.382854, -0.176911, 0.110507, 0.246585,  # V8-V14
        -0.392170, 0.330892, -0.063781, 0.244964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # V15-V28 (example values)
        149.62  # Amount
    ]
}

# Send the POST request to the FastAPI endpoint
try:
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    print("Response JSON:", response.json())
except requests.exceptions.HTTPError as http_err:
    print(f"HTTP error occurred: {http_err}")
    if response.status_code == 422:  # Unprocessable Entity
        print("Error details:", response.json())
except Exception as err:
    print(f"An error occurred: {err}")