import requests
import pandas as pd

def fetch_and_process_data(url):
    """
    Fetches data from a given URL using pagination and processes it into a pandas DataFrame.
    
    Parameters:
    - url: str, The initial URL to start fetching data from.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the fetched and processed data.
    """
    access_token = 'pat-na1-61ac8528-b943-4cb7-a342-8690440d683c'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
    }

    all_data = []
    while url:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data['results'])
            url = data.get('paging', {}).get('next', {}).get('link')
        else:
            print(f'Error fetching data: {response.status_code}')
            break

    # Extracting properties from each item in the results list
    properties_list = [item['properties'] for item in all_data]

    # Creating a DataFrame from the list of properties
    df = pd.DataFrame(properties_list)

    # Convert dates to date format without time
    df['createdate'] = pd.to_datetime(df['createdate']).dt.date
    df['hs_lastmodifieddate'] = pd.to_datetime(df['hs_lastmodifieddate']).dt.date
    df['closedate'] = df['closedate'].astype(str).str[:10]

    return df


def transform_closed_deals(df):
    """
    Transforms a DataFrame containing deal data by filtering closed deals,
    converting and extracting the date part of the closedate, calculating the time difference in days,
    and returning a subset of the DataFrame with specific columns.
    
    Parameters:
    - df: pd.DataFrame, The original DataFrame containing deal data.
    
    Returns:
    - pd.DataFrame: A transformed DataFrame containing only the createdate, closedate, and time_difference columns.
    """
    # Filter deals that have been closed
    closed_deals = df[df['closedate'].notna()]

    # Convert closedate to datetime, coercing errors to NaT
    closed_deals['closedate'] = pd.to_datetime(closed_deals['closedate'], errors='coerce')

    # Apply the transformation to extract the date part
    closed_deals['closedate'] = closed_deals['closedate'].dt.date


    # Calculate the time difference in days
    closed_deals['time_difference'] = (pd.to_datetime(closed_deals['closedate']) - 
                                        pd.to_datetime(closed_deals['createdate'])).dt.days

    closed_deals['closedate'] = pd.to_datetime(closed_deals['closedate'])

    # Return a subset of the DataFrame containing only the createdate, closedate, and time_difference columns
    return closed_deals

# Assuming df is the DataFrame returned by the fetch_and_process_data function
# transformed_df = transform_closed_deals(df)

# To display the transformed DataFrame
# print(transformed_df)


# Example usage
# initial_url = 'https://api.hubapi.com/crm/v3/objects/deals'
# df = fetch_and_process_data(initial_url)

# df = pd.read_csv('deals.csv')
# closed_deals = transform_closed_deals(df)

# print(closed_deals.head())
# Display the DataFrame
# print(closed_deals.head())


