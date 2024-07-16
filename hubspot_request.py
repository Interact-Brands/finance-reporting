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
    access_token = "KEY"
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

# Convert closedate to datetime, coercing errors to NaT
    # Apply the transformation to extract the date part
    closed_deals['hs_lastmodifieddate'] = pd.to_datetime(closed_deals['hs_lastmodifieddate'])

    # Calculate the time difference in days
    closed_deals['time_difference'] = (pd.to_datetime(closed_deals['closedate']) - 
                                        pd.to_datetime(closed_deals['createdate'])).dt.days

    closed_deals['closedate'] = pd.to_datetime(closed_deals['closedate'])

    # Return a subset of the DataFrame containing only the createdate, closedate, and time_difference columns
    return closed_deals

# Example usage
# initial_url = 'https://api.hubapi.com/crm/v3/objects/deals'
# df = fetch_and_process_data(initial_url)

# closed_deals = transform_closed_deals(df)

# # Save the combined dataframe back to the CSV file
# closed_deals.to_csv('deals.csv', index=False)


# print("Data has been updated successfully.")
# print(closed_deals.tail())

# # Ensure 'amount' column is numeric, convert non-numeric values to NaN
# closed_deals['amount'] = pd.to_numeric(closed_deals['amount'], errors='coerce')
# # Define the mapping for deal stages to categories
# deal_stage_mapping = {
#     'Proposals/Negotiation': ['decisionmakerboughtin', 'qualifiedtobuy'],
#     'Inbound/Discovery Call': ['appointmentscheduled'],
#     'Closed Lost': ['closedlost']
# }

# # Initialize sums for each category
# proposals_negotiation_sum = 0
# inbound_discovery_call_sum = 0
# closed_lost_sum = 0

# # Calculate the sums for each category
# proposals_negotiation_sum = closed_deals[closed_deals['dealstage'].isin(deal_stage_mapping['Proposals/Negotiation'])]['amount'].sum()
# inbound_discovery_call_sum = closed_deals[closed_deals['dealstage'].isin(deal_stage_mapping['Inbound/Discovery Call'])]['amount'].sum()
# closed_lost_sum = closed_deals[closed_deals['dealstage'].isin(deal_stage_mapping['Closed Lost'])]['amount'].sum()

# # Print the results
# print(f"Proposals/Negotiation Deals Amount Sum: {proposals_negotiation_sum}")
# print(f"Inbound/Discovery Call Deals Amount Sum: {inbound_discovery_call_sum}")
# print(f"Closed Lost Deals Amount Sum: {closed_lost_sum}")

# # Get the current month and the previous month
# current_month = closed_deals['hs_lastmodifieddate'].dt.to_period('M').max()
# previous_month = current_month - 1

# # Filter deals by current month and previous month
# current_month_deals = closed_deals[(closed_deals['hs_lastmodifieddate'].dt.to_period('M') == current_month) &
#                                    (closed_deals['dealstage'].isin(deal_stage_mapping['Proposals/Negotiation'] + deal_stage_mapping['Inbound/Discovery Call']))]
# previous_month_deals = closed_deals[(closed_deals['hs_lastmodifieddate'].dt.to_period('M') == previous_month) &
#                                     (closed_deals['dealstage'].isin(deal_stage_mapping['Proposals/Negotiation'] + deal_stage_mapping['Inbound/Discovery Call']))]

# # Count the number of pending deals for each month
# current_month_pending_deal_count = current_month_deals['hs_object_id'].count()
# previous_month_pending_deal_count = previous_month_deals['hs_object_id'].count()

# # Calculate the percentage change in pending deal count
# if previous_month_pending_deal_count != 0:
#     percentage_change = ((current_month_pending_deal_count - previous_month_pending_deal_count) / previous_month_pending_deal_count) * 100
# else:
#     percentage_change = float('inf')  # Handle division by zero case

# # Display the results
# print(f"Current Month Pending Deal Count: {current_month_pending_deal_count}")
# print(f"Previous Month Pending Deal Count: {previous_month_pending_deal_count}")
# print(f"Percentage Change in Pending Deals: {percentage_change:.2f}%")
