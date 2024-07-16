import pandas as pd
import datetime
# Read the CSV data into a DataFrame
data = pd.read_csv('rocketmoney.csv')

# Group by 'Institution Name' and 'Amount' and count the number of transactions per group
grouped = data.groupby(['Name', 'Amount']).size().reset_index(name='counts')

# # Filter for groups with more than one transaction (potential subscriptions)
subscriptions = grouped[grouped['counts'] > 2]

# Calculate the total amount of these subscriptions
total_subscription_amount = subscriptions['Amount'].sum()

# print(total_subscription_amount)

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')

# Get the last month
current_date = datetime.datetime.now()
if current_date.month == 1:
    last_month = 12
    last_year = current_date.year - 1
else:
    last_month = current_date.month - 1
    last_year = current_date.year

# Filter transactions to only include those from the last month
last_month_data = data[
    (data['Date'].dt.month == last_month) & 
    (data['Date'].dt.year == last_year)
]

# Group by 'Category' and sum the 'Amount' for the last month
category_spending_last_month = last_month_data.groupby('Category')['Amount'].sum().reset_index()

# Identify the category with the highest total spending for the last month
if not category_spending_last_month.empty:
    most_spent_category_last_month = category_spending_last_month.loc[category_spending_last_month['Amount'].idxmax()]


    # Print the most spent category and its summed amount for the last month
    most_spent_category_name_last_month = most_spent_category_last_month['Category']
    most_spent_category_amount_last_month = most_spent_category_last_month['Amount']
    print(f"Most Spent Category Last Month: {most_spent_category_name_last_month}")
    print(f"Total Amount Spent Last Month: ${most_spent_category_amount_last_month:.2f}")
else:
    print("No transactions found for the last month.")

data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')

# Get the current month and year
current_date = datetime.datetime.now()
current_month = current_date.month
current_year = current_date.year

# Identify the last month
if current_month == 1:
    last_month = 12
    last_year = current_year - 1
else:
    last_month = current_month - 1
    last_year = current_year

# Filter transactions to only include those from the current month and year
current_month_data = data[
    (data['Date'].dt.month == current_month) & 
    (data['Date'].dt.year == current_year)
]

# Filter transactions to only include those from the last month and year
last_month_data = data[
    (data['Date'].dt.month == last_month) & 
    (data['Date'].dt.year == last_year)
]

# Calculate the total amount spent this month
rockeymoney_total_current_month_amount = current_month_data['Amount'].sum()

# Calculate the total amount spent last month
total_last_month_amount = last_month_data['Amount'].sum()

# Calculate the percentage change in spending
if total_last_month_amount != 0:
    rocket_money_percentage_change = ((rockeymoney_total_current_month_amount - total_last_month_amount) / total_last_month_amount) * 100
else:
    rocket_money_percentage_change = float('inf')  # Handle the case where last month's spending is zero

# Display results

print(f"Total Amount Spent This Month: ${rockeymoney_total_current_month_amount:.2f}")
print(f"Total Amount Spent Last Month: ${total_last_month_amount:.2f}")
print(f"Percentage Change in Spending: {rocket_money_percentage_change:.2f}%")