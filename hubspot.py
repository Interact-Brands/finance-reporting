import requests

# Replace 'YOUR_TOKEN' with your actual private app access token
access_token = 'pat-na1-61ac8528-b943-4cb7-a342-8690440d683c'
headers = {
    'Authorization': f'Bearer {access_token}',
    'Content-Type': 'application/json',
}

response = requests.get(
    # 'https://api.hubapi.com/crm/v3/objects/deals',
    'https://api.hubapi.com/crm/v3/objects/deals?after=14327237481',
    # 'https://api.hubapi.com/crm/v3/objects/partner-clients',


    headers=headers
)

# print(response.json())
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    print(data)
else:
    print(f'Error fetching data: {response.status_code}')
