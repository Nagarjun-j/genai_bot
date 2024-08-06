import requests

url = "https://q789819xlj.execute-api.us-west-2.amazonaws.com/dev/cvp/v1/vehicles/data/realTimeData"
payload = {
    "vin": "5NSVDK999JF937434",
    "interval": {
        "minute":1
    },
    "limit": 10
}

headers = {
    "x-api-key": "VsWlUhL16R4U0w4i7xJKS8cWSN2ET3Sea05TEC7f",
    "authorization": "allow",
    "Content-Type": "application/json"
}
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    # Print the response JSON
    print("Response:", response.json())
else:
    print(f"Failed to get response: {response.status_code}")