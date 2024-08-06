import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from millify import millify
import datetime
import numpy as np
import os
import streamlit_option_menu
from streamlit_option_menu import option_menu
from hubspot_request import fetch_and_process_data, transform_closed_deals


from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go

from rocketmoney import rockeymoney_total_current_month_amount, rocket_money_percentage_change
from account_payable import ap_payroll_sum, ap_expensify_sum, ap_billcom_sum

from dotenv import load_dotenv
# Define a custom color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

# Fetch username and password from environment variables
USERNAME = os.getenv("STREAMLIT_USERNAME")
PASSWORD = os.getenv("STREAMLIT_PASSWORD")

USERNAME = st.secrets["STREAMLIT_USERNAME"]
PASSWORD = st.secrets["STREAMLIT_PASSWORD"]

# Login ------------------------------------------------------------------------------------------------------------------------------ 

# Define a function to check login credentials
def check_login(username, password):
    # Compare with environment variables
    return username == USERNAME and password == PASSWORD

# Initialize session state if not already done
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Display login page if not logged in
if not st.session_state.logged_in:
    st.set_page_config(page_title='Login', layout='centered', page_icon=':rocket:')
    
    st.title("Login to Finance Reports")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")
    
    if submit_button:
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password. Contact admins at elvin@interactbrands.com, christie@interactbrands.com for assistance.")

# Display the main page if logged in
if st.session_state.logged_in:
    st.set_page_config(page_title='Finances', layout='wide', page_icon=':rocket:')

# Login ------------------------------------------------------------------------------------------------------------------------------ 

    t1, t2 = st.columns((0.2,1)) 

    t1.image('images/interactbrands_logo.jpg', width = 120)
    t2.title("Finance Reports")


    initial_url = 'https://api.hubapi.com/crm/v3/objects/deals'
    # df = fetch_and_process_data(initial_url)

    df = pd.read_csv('deals.csv')
    closed_deals = transform_closed_deals(df)

    df = closed_deals
# Calculation of dashboard numbers-----------------------------------------------------------------------------------------------
    ###### Hubspot Deals
    deal_stage_mapping = {
        'Proposals/Negotiation': ['decisionmakerboughtin', 'qualifiedtobuy'],
        'Inbound/Discovery Call': ['appointmentscheduled'],
        'Closed Lost': ['closedlost']
    }

    # Initialize sums for each category
    proposals_negotiation_sum = 0
    inbound_discovery_call_sum = 0
    closed_lost_sum = 0

    # Calculate the sums for each category
    proposals_negotiation_sum = closed_deals[closed_deals['dealstage'].isin(deal_stage_mapping['Proposals/Negotiation'])]['amount'].sum()
    inbound_discovery_call_sum = closed_deals[closed_deals['dealstage'].isin(deal_stage_mapping['Inbound/Discovery Call'])]['amount'].sum()
    closed_lost_sum = closed_deals[closed_deals['dealstage'].isin(deal_stage_mapping['Closed Lost'])]['amount'].sum()

    # Calculate the number of deals for each stage
    proposals_negotiation_count = closed_deals[closed_deals['dealstage'].isin(deal_stage_mapping['Proposals/Negotiation'])].shape[0]
    inbound_discovery_call_count = closed_deals[closed_deals['dealstage'].isin(deal_stage_mapping['Inbound/Discovery Call'])].shape[0]

    # Total number of pending deals
    total_pending_deals_count = proposals_negotiation_count + inbound_discovery_call_count

    # Get the current month and the previous month
    current_month = closed_deals['hs_lastmodifieddate'].dt.to_period('M').max()
    previous_month = current_month - 1

    # Filter deals by current month and previous month
    current_month_deals = closed_deals[(closed_deals['hs_lastmodifieddate'].dt.to_period('M') == current_month) &
                                    (closed_deals['dealstage'].isin(deal_stage_mapping['Proposals/Negotiation'] + deal_stage_mapping['Inbound/Discovery Call']))]
    previous_month_deals = closed_deals[(closed_deals['hs_lastmodifieddate'].dt.to_period('M') == previous_month) &
                                        (closed_deals['dealstage'].isin(deal_stage_mapping['Proposals/Negotiation'] + deal_stage_mapping['Inbound/Discovery Call']))]

    # Count the number of pending deals for each month
    current_month_pending_deal_count = current_month_deals['hs_object_id'].count()
    previous_month_pending_deal_count = previous_month_deals['hs_object_id'].count()
    current_and_previous_month_pending_deal_count = current_month_pending_deal_count + previous_month_pending_deal_count
    
    # Calculate the percentage change in pending deal count
    if previous_month_pending_deal_count != 0:
        percentage_change = ((current_month_pending_deal_count - previous_month_pending_deal_count) / previous_month_pending_deal_count) * 100
    else:
        percentage_change = float('inf')  # Handle division by zero case
    ###### Hubspot Deals

    ###### RocketMoney
    # Read the CSV data into a DataFrame
    data = pd.read_csv('rocketmoney.csv')

    total_rm_ytd = data['Amount'].sum()

    # Group by 'Institution Name' and 'Amount' and count the number of transactions per group
    grouped = data.groupby(['Name', 'Amount']).size().reset_index(name='counts')

    # # Filter for groups with more than one transaction (potential subscriptions)
    subscriptions = grouped[grouped['counts'] > 2]

    # Calculate the total amount of these subscriptions
    total_subscription_amount = subscriptions['Amount'].sum()

    # Most Spending Category
        # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')

    # Get the last month
    current_date = datetime.now()
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
    
    most_spent_category_name_last_month = "None"
    most_spent_category_amount_last_month = 0.0
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

    ###### RocketMoney


# Calculation of dashboard numbers-----------------------------------------------------------------------------------------------

    with st.sidebar:
        selected = option_menu(
        menu_title = "Main Menu",
        options = ["Home1", "Home2", "Hubspot-Deals","Quickbooks - Balance", "Profit and Loss","Balance Sheet","Cash Flow", "RocketMoney","Bill COM","Issues"],
        icons = ["house","house","activity","activity", "activity","activity", "activity","bug", "phone", "phone"],
        menu_icon = "cast",
        default_index = 0,
        #orientation = "horizontal",
    )
    if selected == "Hubspot-Deals":
        st.header('Hubspot Finances')
        # Create a row layout
        c1, c2= st.columns(2)
        c3 = st.empty()  # Create an empty container
        c4 = st.empty()
        c5, c6= st.columns(2)

        with st.container():
            c1.write("YTD Sales")
            c2.write("YoY Sales")

        with st.container():
            c5.write("c5")
            c6.write("c6")

        with c1:
            today = datetime.now()
            last_month_start = today - timedelta(days=30)

            # Filter the DataFrame
            # Explicitly convert 'closedate' to datetime format
            df_last_month = df[df['closedate'] >= last_month_start]  # Direct comparison should now work

                    # Calculate the sum of the 'amount' column
            total_amount_last_month = df_last_month['amount'].sum()

            # Create the Plotly figure with the dynamic value
            fig = go.Figure()
            delta_value = 10000  # Example: set the delta value to $10,000

            indicator_trace = go.Indicator(
            mode="number+delta",
            value=total_amount_last_month,
            number={'prefix': "$"},
            delta={'position': "top", 'reference': delta_value},  # Use a meaningful reference value
            domain={'x': [0, 1], 'y': [0, 1]}
            )
            fig.add_trace(indicator_trace)

            fig.update_layout(
            # paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=color_palette[0],
            title_font_color=color_palette[0],
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis_gridwidth=1,
            yaxis_gridwidth=1,
            xaxis_zeroline=False,
            yaxis_zeroline=False
            )
            st.plotly_chart(fig)  # Use st.plotly_chart() to display the figure   

        with c2:
            # Filter closed won deals
            df_closed_deals = df[df['dealstage'] == 'closedwon']
            df_closed_deals['closedate'] = pd.to_datetime(df_closed_deals['closedate'])

            # Extract year and month from 'closedate' and convert to string
            df_closed_deals['year_month'] = df_closed_deals['closedate'].dt.to_period('M').apply(lambda x: x.strftime('%Y-%m'))

            deals_per_month = df_closed_deals.groupby('year_month').size().reset_index(name='count')

            # Create a gradient color scale for the line
            n_points = len(deals_per_month)
            colors = [f'rgba(0, 0, 255, {i / n_points})' for i in range(1, n_points + 1)]

            # Create the line chart with gradient color
            fig = go.Figure(data=[
            go.Scatter(
                    x=deals_per_month['year_month'],
                    y=deals_per_month['count'],
                    mode='lines+markers',
                    line=dict(color='rgba(0, 0, 255, 1)'),
                    marker=dict(
                    color=colors,
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey')
                    )
            )
            ])

            # Customize the chart
            fig.update_layout(
            title='Closed Won Deals Per Month',
            xaxis_title='Month',
            yaxis_title='Number of Closed Deals',
            legend=dict(
                    yanchor="top", y=0.95, xanchor="right", x=0.01,
                    bordercolor="#444", borderwidth=1,
                    bgcolor="white", font=dict(size=12)
            ),
            autosize=False,
            width=800,
            height=450,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white")
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig)


        with c3:
            # Assuming 'df' is your DataFrame
            relevant_stages_df = df[['closedate', 'dealstage']].dropna(subset=['dealstage']).replace({'dealstage': {'': None}}).dropna(subset=['dealstage'])
            relevant_stages_df['dealstage'] = relevant_stages_df['dealstage'].map({'qualifiedtobuy': 'Qualified to Buy', 'closedlost': 'Lost', 'closedwon': 'Won', 'decisionmakerboughtin': 'Decisionmaker Bought in', 'appointmentscheduled': 'Appointment'}).fillna('')

            relevant_stages_df['closedate'] = pd.to_datetime(relevant_stages_df['closedate'])
            relevant_stages_df['year_month'] = relevant_stages_df['closedate'].dt.to_period('M').apply(lambda x: x.strftime('%Y-%m'))

            deals_per_month_by_stage = relevant_stages_df.groupby(['year_month', 'dealstage']).size().reset_index(name='count')


            pivot_table = deals_per_month_by_stage.pivot(index='year_month', columns='dealstage', values='count').fillna(0)

            # Streamlit app to display the line chart
            st.title('Deals Per Month by Deal Stage')
            fig = px.line(pivot_table.reset_index(), x='year_month', y=pivot_table.columns,
                    labels={'count': 'Number of Deals'},
                    title='Deals Per Month by Deal Stage')
            fig.update_xaxes(title_text='Month')
            fig.update_yaxes(title_text='Number of Deals')

            st.plotly_chart(fig)
            

        with c4:
            df_closed_deals['closedate'] = pd.to_datetime(df_closed_deals['closedate'])

            # Extract year and month from 'closedate' and convert to string
            df_closed_deals['year_month'] = df_closed_deals['closedate'].dt.to_period('M').astype(str)
            monthly_deal_amounts = df_closed_deals.groupby('year_month')['amount'].sum().reset_index()

            # Prepare data for linear regression
            monthly_deal_amounts['date_num'] = pd.to_datetime(monthly_deal_amounts['year_month']).map(pd.Timestamp.toordinal)
            X = monthly_deal_amounts[['date_num']]
            y = monthly_deal_amounts['amount']

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Create future months
            last_date = pd.to_datetime(monthly_deal_amounts['year_month'].iloc[-1])
            future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
            future_months_str = future_months.to_period('M').astype(str)
            future_dates_num = future_months.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

            # Predict future values using the trained model
            future_amounts = model.predict(future_dates_num)

            # Manually construct the data for future months
            future_data = {
            'year_month': future_months_str,
            'amount': future_amounts
            }

            # Convert to DataFrame for easier manipulation
            future_df = pd.DataFrame(future_data)

            # Combine the past and future data
            all_data = pd.concat([monthly_deal_amounts[['year_month', 'amount']], future_df], ignore_index=True)

            # Create color scale
            colors = px.colors.sequential.Blues_r

            # Generate gradient color values
            n_colors = len(all_data)
            gradient_colors = [colors[int(i * (len(colors) - 1) / (n_colors - 1))] for i in range(n_colors)]

            # Create the base bar chart with gradient color scheme
            fig = go.Figure(data=[
            go.Bar(
                    x=all_data['year_month'],
                    y=all_data['amount'],
                    marker=dict(color=gradient_colors),
            )
            ])

            # Customize the chart
            fig.update_layout(
            title='Closed Won Deals Amounts Per Month with Trendline and Extended Forecast',
            xaxis_title='Month',
            yaxis_title='Amount',
            legend=dict(
                    yanchor="top", y=0.95, xanchor="right", x=0.01,
                    bordercolor="#444", borderwidth=1,
                    bgcolor="white", font=dict(size=12)
            ),
            autosize=False,
            width=800,
            height=450,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            bargap=0.15,  # gap between bars of adjacent bars
            bargroupgap=0.1  # gap between bars of the same group
            )

            # Add the trendline and forecast to the figure
            fig.add_trace(go.Scatter(
            x=all_data['year_month'], y=all_data['amount'], 
            mode='lines', name='Trendline & Forecast', line=dict(color='gray', dash='dashdot')
            ))

            # Display the chart in Streamlit
            st.plotly_chart(fig)
    
        with c5:
            
                    # Assuming df is your DataFrame with 'closedate' and 'dealstage' columns
            today = datetime.now()
            last_month_start = today - timedelta(days=30)

            # Filter the DataFrame for the last month
            df_last_month = df[df['closedate'] >= last_month_start]

            # Calculate the count of deals for each deal stage
            dealstage_counts = df_last_month['dealstage'].value_counts()

            # Calculate the total number of deals
            total_deals = dealstage_counts.sum()

            # Calculate the percentage of deals for each deal stage
            dealstage_percentages = (dealstage_counts / total_deals) * 100

            # Create a DataFrame for the pie chart
            pie_data = pd.DataFrame({
            'Deal Stage': dealstage_percentages.index,
            'Percentage': dealstage_percentages.values.round(2),
            'Count': dealstage_counts.values
            })

            # Plot the pie chart
            fig = px.pie(pie_data, values='Count', names='Deal Stage', 
                    title='Deal Stages for the Last Month',
                    hover_data=['Percentage'],
                    labels={'Deal Stage': 'Deal Stage'},
                    hole=0.5)

            # Update the layout
            fig.update_traces(textposition='inside', textinfo='percent+label')

            # Display the chart
            st.plotly_chart(fig)

    if selected == "Quickbooks - Balance":

        # Example data manipulation
        df = pd.DataFrame({
            'Account': ['Accounts Receivable', 'Accrued Expenses'],
            'Balance': [490463.55, -216086.85],
        })

        # Sample data
        accounts = [
            {
                'Name': 'Accounts Receivable',
                'CurrentBalance': 490463.55,
                'Classification': 'Asset',
                'AccountType': 'Accounts Receivable',
                'AcctNum': '12100',
                'CurrencyRef': {'value': 'USD', 'name': 'United States Dollar'}
            },
            {
                'Name': 'Accrued Expenses',
                'CurrentBalance': -216086.85,
                'Classification': 'Liability',
                'AccountType': 'Other Current Liability',
                'AcctNum': '20500',
                'CurrencyRef': {'value': 'USD', 'name': 'United States Dollar'}
            },
            {
                'Name': 'Accrued Interest N/P Wells Fargo',
                'CurrentBalance': 0,
                'Classification': 'Liability',
                'AccountType': 'Long Term Liability',
                'AcctNum': '25500',
                'CurrencyRef': {'value': 'USD', 'name': 'United States Dollar'}
            }
        ]

        # Displaying using st.metric
        st.title('Quickbooks Financial Overview')
        col1, col2, col3 = st.columns(3)
        curr_balance = 490463.55
        value_1 = f"${curr_balance:,.2f}"
        value_2 = f"${curr_balance:,.2f}"
        value_3 = f"${curr_balance:,.2f}"
        col1.metric("Accounts Receivable", value_1, "Asset")
        col1.metric("Expenses", "$462,365", "-Liability")
        col2.metric("Accounts Payable", "$145,429.81", "-Liability")
        col2.metric("Bill.com Out Clearing Payable", "$8331.73", "-Liability")
        col3.metric("Profit & Loss", "-$49,857", "Asset")
        col3.metric("Paid in Capital Surplus", "-$11,680", "Equity")
        
        for account in accounts:
            st.metric(label=account['Name'], value=f"${account['CurrentBalance']:,.2f}")
            # Sample data
        bank_accounts = [
            {'Name': 'CC - American Express (1001/3014)', 'Label': 'Credit Card', 'Amount': 74633.54},
            {'Name': 'CC - Chase INK', 'Label': 'Credit Card', 'Amount': 41595.56},
            {'Name': 'Checking Wells Fargo', 'Label': 'Checking', 'Amount': 162804.85},
        ]

        # Streamlit app title
        st.title("Bank Accounts Overview")

        # Function to display account info in a card-like format
        def display_account_card(account):
            st.markdown(
                f"""
                <div style="border:1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h3 style="color: #fff;">{account['Name']}</h3>
                    <p style="margin: 5px 0;"><strong>Label:</strong> {account['Label']}</p>
                    <p style="margin: 5px 0;"><strong>Current Amount:</strong> ${account['Amount']:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Display each account in a card
        for account in bank_accounts:
            display_account_card(account)
        # Using custom HTML/CSS for more control
        st.subheader('Detailed Financial Information')

        for account in accounts:
            st.write(f"""
            <div style="color: white; background: linear-gradient(135deg, #1e1e2e, #3a3a4a); padding: 10px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); margin-bottom: 10px;">
                <h4 style="color: #42caff; font-family: 'Arial', sans-serif;">{account['Name']}</h4>
                <p><strong>Account Type:</strong> {account['AccountType']}</p>
                <p><strong>Current Balance:</strong> <span style="color: #42caff;">${account['CurrentBalance']:,.2f}</span></p>
                <p><strong>Classification:</strong> {account['Classification']}</p>
                <p><strong>Account Number:</strong> {account['AcctNum']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Using custom HTML/CSS for more control and futuristic styling
        st.subheader('Custom Styled Financial Information')
        html_content = """
        <style>
            .financial-info {
                font-family: 'Arial', sans-serif;
                margin: 10px 0;
                padding: 20px;
                border: 1px solid #444;
                border-radius: 10px;
                background: linear-gradient(135deg, #1e1e2e, #3a3a4a);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                color: white;
            }
            .financial-info h4 {
                margin: 0 0 10px 0;
                font-size: 1.5em;
                color: #42caff;
            }
            .financial-info p {
                margin: 5px 0;
                color: #ddd;
            }
            .financial-info p span {
                color: #42caff;
            }
        </style>
        """

        st.markdown(html_content, unsafe_allow_html=True)

        for account in accounts:
            st.markdown(f"""
            <div class="financial-info">
                <h4>{account['Name']}</h4>
                <p><strong>Account Type:</strong> {account['AccountType']}</p>
                <p><strong>Current Balance:</strong> <span>${account['CurrentBalance']:,.2f}</span></p>
                <p><strong>Classification:</strong> {account['Classification']}</p>
                <p><strong>Account Number:</strong> {account['AcctNum']}</p>
            </div>
            """, unsafe_allow_html=True)
       
    if selected == "Profit and Loss":
        

        # Load the data
        df = pd.read_csv('profit_and_loss_report.csv')

        # Ensure no NaN values in Category column
        df['Category'].fillna('', inplace=True)

        # Data Preparation
        income_data = df[df['Category'] == 'Income']
        cogs_data = df[df['Category'] == 'Cost of Goods Sold']
        expenses_data = df[df['Category'] == 'Expenses']
        other_income_expenses_data = df[df['Category'] == 'Other Income/Expenses']

        # Calculate totals
        total_income = df[df['Category'] == 'Income']['Amount'].sum()
        total_cogs = df[df['Category'] == 'Cost of Goods Sold']['Amount'].sum()
        total_expenses = df[df['Category'] == 'Expenses']['Amount'].sum()
        total_other = df[df['Category'] == 'Other Income/Expenses']['Amount'].sum()

        gross_profit = total_income - total_cogs
        net_operating_income = gross_profit - total_expenses
        net_income = net_operating_income + total_other

        # Append totals to DataFrame
        totals = {
            'Category': ['Total Income', 'Total COGS', 'Gross Profit', 'Total Expenses', 'Net Operating Income', 'Total Other Income/Expenses', 'Net Income'],
            'Subcategory': ['', '', '', '', '', '', ''],
            'Amount': [total_income, total_cogs, gross_profit, total_expenses, net_operating_income, total_other, net_income]
        }
        totals_df = pd.DataFrame(totals)

        # Combine DataFrames
        final_df = pd.concat([df, totals_df], ignore_index=True)

        # Streamlit App
        st.title('Profit and Loss Report Visualization')

        # Pie Chart for Income and Expenses Distribution
        st.header('Income and Expenses Distribution')
        distribution_data = pd.DataFrame({
            'Category': ['Income', 'Cost of Goods Sold', 'Expenses', 'Other Income/Expenses'],
            'Amount': [total_income, total_cogs, total_expenses, total_other]
        })
        fig1 = px.pie(distribution_data, values='Amount', names='Category', title='Income and Expenses Distribution')
        st.plotly_chart(fig1)

        # Bar Chart for Detailed Income and Expense Categories
        st.header('Detailed Income and Expense Categories')
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=income_data['Subcategory'], y=income_data['Amount'], name='Income'))
        fig2.add_trace(go.Bar(x=expenses_data['Subcategory'], y=expenses_data['Amount'], name='Expenses'))
        fig2.update_layout(barmode='group', title='Detailed Income and Expense Categories', xaxis_title='Category', yaxis_title='Amount (USD)')
        st.plotly_chart(fig2)

        # Line Chart for Monthly Income and Expenses (assuming monthly data available)
        # Here, we assume 'Month' column exists for demonstration. Adjust accordingly if not present.
        # st.header('Monthly Income and Expenses')
        # monthly_data = df[df['Category'] == 'Monthly Data']  # Example placeholder
        # fig3 = px.line(monthly_data, x='Month', y='Amount', color='Category', title='Monthly Income and Expenses')
        # st.plotly_chart(fig3)

        # Stacked Bar Chart for COGS Breakdown
        st.header('COGS Breakdown')
        fig4 = go.Figure()
        for subcategory in cogs_data['Subcategory'].unique():
            sub_data = cogs_data[cogs_data['Subcategory'] == subcategory]
            fig4.add_trace(go.Bar(x=sub_data['Subcategory'], y=sub_data['Amount'], name=subcategory))
        fig4.update_layout(barmode='stack', title='COGS Breakdown', xaxis_title='Subcategory', yaxis_title='Amount (USD)')
        st.plotly_chart(fig4)

        # Summary Metrics for Key Financial Indicators
        st.header('Key Financial Indicators')
        st.metric(label="Total Income", value=f"${total_income:,.2f}")
        st.metric(label="Total COGS", value=f"${total_cogs:,.2f}")
        st.metric(label="Gross Profit", value=f"${gross_profit:,.2f}")
        st.metric(label="Total Expenses", value=f"${total_expenses:,.2f}")
        st.metric(label="Net Operating Income", value=f"${net_operating_income:,.2f}")
        st.metric(label="Net Income", value=f"${net_income:,.2f}")

    if selected == "RocketMoney":
        rm = pd.read_csv('rocketmoney.csv')
        df = rm.copy()
        print(df.columns)
        # Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

        # Summarize the spending by date
        daily_spending = df.groupby('Date')['Amount'].sum().reset_index()

        # Create a bar chart using Plotly
        fig = px.bar(daily_spending, x='Date', y='Amount', title='Daily Spendings')
        st.plotly_chart(fig)
        
        # Summarize the spending by category
        category_spending = df.groupby('Category')['Amount'].sum().reset_index()
        fig = px.pie(category_spending, values='Amount', names='Category', title='Spending by Category')
        st.plotly_chart(fig)

        # high spendings
        high_spendings = df.nlargest(15, 'Amount')
        fig = px.bar(high_spendings, x='Description', y='Amount', title='Top 10 High Spendings Transactions')
        st.plotly_chart(fig)

        # category spending
        category_time_spending = df.groupby(['Date', 'Category'])['Amount'].sum().unstack().fillna(0)
        fig = px.area(category_time_spending, x=category_time_spending.index, y=category_time_spending.columns, title='Category-wise Spendings Over Time')
        st.plotly_chart(fig)

        categories = df['Category'].unique()
        account_types = df['Account Type'].unique()

        # Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

        st.title('Interact-ive Spendings Breakdown Dashboard')

        # Dropdown for Category
        categories = df['Category'].unique()
        selected_category = st.selectbox('Select Category', categories)

        # Dropdown for Account Type
        account_types = df['Account Type'].unique()
        selected_account_type = st.selectbox('Select Account Type', account_types)

        # Date range slider
        date_range = st.slider(
            'Select Date Range',
            min_value=df['Date'].min().to_pydatetime(),
            max_value=df['Date'].max().to_pydatetime(),
            value=(df['Date'].min().to_pydatetime(), df['Date'].max().to_pydatetime())
        )

        # Filter the DataFrame based on selections
        filtered_df = df[
            (df['Category'] == selected_category) & 
            (df['Account Type'] == selected_account_type) & 
            (df['Date'] >= date_range[0]) & 
            (df['Date'] <= date_range[1])
        ]

        # Create a bar chart using Plotly
        fig = px.bar(filtered_df, x='Date', y='Amount', title='Spendings Breakdown')

        # Display the chart in Streamlit
        st.plotly_chart(fig)

    if selected == "Balance Sheet":
        st.subheader(f"**You Have selected {selected}**")
        balance = pd.read_csv('balance_sheet.csv')
        # st.write(balance)
        df = balance.copy()
            # Streamlit app
        df['Section'] = df['Section'].fillna('')

        st.title('Balance Sheet Dashboard')

        # Total Assets, Liabilities, and Equity
        total_assets = df[df['Account'] == 'Total ASSETS']['Amount'].values[0]
        total_liabilities = df[df['Account'] == 'Total Liabilities']['Amount'].values[0]
        total_equity = df[df['Account'] == 'Total Equity']['Amount'].values[0]

        # Displaying the key financial indicators
        st.header("Key Financial Indicators")
        st.metric("Total Assets", f"${total_assets:,.2f}")
        st.metric("Total Liabilities", f"${total_liabilities:,.2f}")
        st.metric("Total Equity", f"${total_equity:,.2f}")


            # Plotting Assets Breakdown
        st.header("Assets Breakdown")
        assets_df = df[df['Section'].str.contains("Asset")]
        fig_assets = px.pie(assets_df, names='Account', values='Amount', title='Assets Distribution')
        st.plotly_chart(fig_assets)

        # Plotting Liabilities Breakdown
        st.header("Liabilities Breakdown")
        liabilities_df = df[df['Section'].str.contains("Liabilities")]
        fig_liabilities = px.pie(liabilities_df, names='Account', values='Amount', title='Liabilities Distribution')
        st.plotly_chart(fig_liabilities)

    if selected == "Cash Flow":
        cash_flow = pd.read_csv('cashflow.csv')
        st.write(cash_flow)
        df = cash_flow.copy()

            # Check if DataFrame is empty
        if df.empty:
            st.error("The DataFrame is empty. Unable to proceed with analysis.")
        else:
            # Proceed with data analysis
            # Example: Display summary statistics by section
            if 'Section' in df.columns:
                section_totals = df.groupby('Section')['Amount'].sum().reset_index()
                st.subheader('Summary Statistics by Section')
                st.bar_chart(section_totals.set_index('Section'))
            else:
                st.error("Column 'Section' not found in the CSV file.")
            # Calculate and display summary statistics
                
    if selected == "Home1":
        # All Account Receivable
        account_receivable = 661851.50

        ar_current = 334721.00
        ar_30_days_overdue_sum = 193180.25
        ar_60_days_overdue_sum = 77226.75
        ar_90_days_overdue_sum = 0.0
        overall_overdue = ar_30_days_overdue_sum + ar_60_days_overdue_sum + ar_90_days_overdue_sum
        
        # Account Receivable Details
        account_receivable_details = f"""
        Overall Overdue: ${overall_overdue:,.2f} 
        - ${ar_current:,.2f} - current 
        - ${ar_30_days_overdue_sum:,.2f} - 30 days overdue
        - ${ar_60_days_overdue_sum:,.2f} - 60 days overdue
        - ${ar_90_days_overdue_sum:,.2f} - 90 days overdue
        """

        # All Account Payable
        account_payable = -202265.19
        
        ap_current = 76518.99
        ap_30_days_overdue_sum = 91466.45
        ap_60_days_overdue_sum = 16349.74
        ap_90_days_overdue_sum = 3820.0
        overall_overdue = ap_30_days_overdue_sum + ap_60_days_overdue_sum + ap_90_days_overdue_sum
        
        # Account Payable Details
        account_payable_details = f"""
        Overall Overdue: ${overall_overdue:,.2f} 
        - ${ap_current:,.2f} - current
        - ${ap_30_days_overdue_sum:,.2f} - 30 days overdue
        - ${ap_60_days_overdue_sum:,.2f} - 60 days overdue 
        - ${ap_90_days_overdue_sum:,.2f} - 90 days overdue 
        """

        # All Other Expenses and Forecast
        # Get the current month
        current_month = datetime.now().month

        # Calculate the scaling factor based on the current month
        scaling_factor = 12 / current_month

        other_expenses_sum = ap_payroll_sum + ap_expensify_sum
        # Calculate forecasted EOY values
        forecasted_payroll_eoy = ap_payroll_sum * scaling_factor
        forecasted_expensify_eoy = ap_expensify_sum * scaling_factor
        forecasted_other_expenses_sum = forecasted_payroll_eoy + forecasted_expensify_eoy

        # Calculate forecasted Rest of the Year values
        forecasted_payroll_roy = forecasted_payroll_eoy - ap_payroll_sum
        forecasted_expensify_roy = forecasted_expensify_eoy - ap_expensify_sum
        forecasted_other_expenses_sum_roy = forecasted_payroll_roy + forecasted_expensify_roy

        # Calculate forecasted rocketmoney EOY values
        forecasted_rocketmoney_eoy = total_rm_ytd * scaling_factor

        # Calculate forecasted Rest of the Year values
        forecasted_rocketmoney_roy = forecasted_rocketmoney_eoy - total_rm_ytd

        other_expenses_details = f"""
        - Payroll: ${ap_payroll_sum:,.2f}
        - Expensify: ${ap_expensify_sum:,.2f}
        """
        
        forecasted_expenses_details = f"""
        - Payroll: ${forecasted_payroll_roy:,.2f}
        - Expensify: ${forecasted_expensify_roy:,.2f}
        """

        # All Cash in/out the Door
        cash_in_the_door = account_receivable + proposals_negotiation_sum
        cash_out_the_door = account_payable + other_expenses_sum + forecasted_other_expenses_sum_roy
        # cash out the door will be AP + forecasted other expenses + freelance forecast

        pending_deals_details = f"""
        - $ {proposals_negotiation_sum:,} - Proposals/Negotiation
        - $ {inbound_discovery_call_sum:,} - Inbound/Discovery Call
        - {"50%":} - Account Close Ratio
        - {"45%":} - New Business Close Ratio
        """

        promising_deals_details = f"""
                - $ {proposals_negotiation_sum:,} - Promising Proposals
                - $ {inbound_discovery_call_sum:,} - Promosing Inbound
                - {"50%":} - Promising Account Close Ratio
                - {"45%":} - Promising New Business Close Ratio
                """
        rocketmoney_details = f"""
        - $ {total_rm_ytd:,.2f} - YTD Total
        - $ {total_subscription_amount:,} - Subscriptions
        - $ {most_spent_category_amount_last_month:,} - Most Spent Category: {most_spent_category_name_last_month}
        """

        rocketmoney_forecast_details = f"""
        - $ {total_rm_ytd:,.2f} - Forecast of RoY
        - $ {total_subscription_amount:,.2f} - Forecast of Subscriptions
        - $ {most_spent_category_amount_last_month:,.2f} - Forecast of Most Spent
        """

        sales_per_change = "10%"
        profit_per_change = "15%"
        col_home_left, col_home_right = st.columns(2)

        with col_home_left:
            metric_html = f"""
            <div style="color: green;">
                <span style="font-size: 20px; font-weight: bold;">Cash in the Door</span>
                <h2 style="font-size: 24px; font-weight: bold; color:green">${cash_in_the_door:,.2f}</h2>
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)

        with col_home_right:
            # Define the style for the metric
            metric_html = f"""
            <div style="color: red;">
                <span style="font-size: 20px; font-weight: bold;">Cash out the Door</span>
                <h2 style="font-size: 24px; font-weight: bold; color:red">${cash_out_the_door:,.2f}</h2>
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)

        deal_percentage_change = f"""{percentage_change:,.0f}% MoM"""

        # CSS for card-like border
        card_css = """
        <style>
        .card {
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: white;
        }
        </style>
        """

        # Inject CSS
        st.markdown(card_css, unsafe_allow_html=True)

        # Layout with cards
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            delta_value = "Asset"
            # st.metric(label="Account Receivable", value="$" + f"{account_receivable:,.2f}", delta=delta_value)
            metric_html = f"""
            <div style="color: green;">
                <span style="font-size: 20px; font-weight: bold;">Account Receivable</span>
                <h2 style="font-size: 24px; font-weight: bold; color:green">${account_receivable:,.2f}</h2>
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)

            # Optionally, you can still use st.metric for delta display
            st.metric(label="", value="", delta=delta_value)

            st.write(account_receivable_details)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            delta_value = "-Liability"
            # st.metric(label="Account Payable", value="$" + f"{:,.2f}", delta=delta_value)

            metric_html = f"""
            <div style="color: red;">
                <span style="font-size: 20px; font-weight: bold;">Account Payable</span>
                <h2 style="font-size: 24px; font-weight: bold; color:red">${account_payable:,.2f}</h2>
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)

            # Optionally, you can still use st.metric for delta display
            st.metric(label="", value="", delta=delta_value)

            st.write(account_payable_details)
            st.markdown('</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)

        with col3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    delta_value = "-Liability"
                    # st.metric(label="Other Expenses", value="$" + f"{other_expenses_sum:,.2f}", delta=delta_value)
                    metric_html = f"""
                    <div style="color: red;">
                        <span style="font-size: 20px; font-weight: bold;">YTD Expenses</span>
                        <h2 style="font-size: 24px; font-weight: bold; color:red">${other_expenses_sum:,.2f}</h2>
                    </div>
                    """
                    st.markdown(metric_html, unsafe_allow_html=True)
                    # Optionally, you can still use st.metric for delta display
                    st.metric(label="", value="", delta=delta_value)

                    st.write(other_expenses_details)
                    st.markdown('</div>', unsafe_allow_html=True)
        with col4:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    delta_value = "-"
                    # st.metric(label="Forecasted Rest of the Year Expenses", value="$" + f"{forecasted_other_expenses_sum_roy:,.2f}", delta=delta_value)
                    metric_html = f"""
                    <div style="color: red;">
                        <span style="font-size: 20px; font-weight: bold;">Forecasted Expenses</span>
                        <h2 style="font-size: 24px; font-weight: bold; color:red">${forecasted_other_expenses_sum_roy:,.2f}</h2>
                    </div>
                    """
                    st.markdown(metric_html, unsafe_allow_html=True)
                    # Optionally, you can still use st.metric for delta display
                    st.metric(label="", value="", delta=delta_value, delta_color="off")

                    st.write(forecasted_expenses_details)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)

        with col5:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            delta_value = deal_percentage_change

            metric_html = f"""
            <div style="color: green;">
                <span style="font-size: 20px; font-weight: bold;">Pending Deals</span>
                <h2 style="font-size: 24px; font-weight: bold; color:green">{current_month_pending_deal_count}</h2>
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])  # Adjust the ratio as needed
            with col1:
                st.metric(label="", value="", delta="Asset")
            with col2:
                st.metric(label="", value="", delta=delta_value)
            st.write(pending_deals_details)

        with col6:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            delta_value = deal_percentage_change

            metric_html = f"""
            <div style="color: green;">
                <span style="font-size: 20px; font-weight: bold;">Promising Deals</span>
                <h2 style="font-size: 24px; font-weight: bold; color:green">{current_month_pending_deal_count}</h2>
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])  # Adjust the ratio as needed
            with col1:
                st.metric(label="", value="", delta="Asset")
            with col2:
                st.metric(label="", value="", delta=delta_value)
            st.write(promising_deals_details)
            
        # Create columns
        col7, _ = st.columns([200, 1])  # Adjust column widths as needed

        # Populate col6
        with col7:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            # Define the projects with additional fields
            projects = [
                {"Name": "Del Monte", "Job code": "DM123", "Profitability": "25%"},
                {"Name": "Rally", "Job code": "RA456", "Profitability": "25%"},
                {"Name": "Gruns", "Job code": "GR789", "Profitability": "30%"},
                {"Name": "Nestle", "Job code": "NE101", "Profitability": "20%"}
                ]
            total_profitability = sum(int(project["Profitability"].strip('%')) for project in projects)
            average_profitability = total_profitability / len(projects)
            
            st.metric(label="Project Profitability", value=f"{average_profitability:.0f}%", delta="Overall Change")


            # Convert the projects list to a DataFrame
            projects_df = pd.DataFrame(projects)

            # Display the project details in an expander with a table
            with st.expander("Project Details"):
                st.table(projects_df.set_index([pd.Index(['']*len(projects_df))]))
            
            st.markdown('</div>', unsafe_allow_html=True)

        col8, col9 = st.columns(2)

        with col8:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(label="RocketMoney", value=rockeymoney_total_current_month_amount, delta=f"""{rocket_money_percentage_change}% MoM""")
            st.write(rocketmoney_details)
            st.markdown('</div>', unsafe_allow_html=True)

        with col9:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(label="Forecasted Rest of the Year RocketMoney", value=f"{forecasted_rocketmoney_roy:,.2f}", delta=f"""{rocket_money_percentage_change}% MoM""")
            st.write(rocketmoney_forecast_details)
            st.markdown('</div>', unsafe_allow_html=True)
 
        # Create columns
        col10, _ = st.columns([200, 1])  # Adjust column widths as needed
        with col10:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(label="Freelance Forecast", value="25%", delta="Overall Change")
            
            # Placeholder data for freelance forecast
            eoy_forecast = {
                "Total Revenue": "$90,000",
                "Total Projects": 36,
                "Average Revenue per Project": "$2,500",
                "Expected New Clients": 15,
                "Hours Worked": 1800
            }
            
            roy_forecast = {
                "Total Revenue": "$30,000",
                "Total Projects": 12,
                "Average Revenue per Project": "$2,500",
                "Expected New Clients": 5,
                "Hours Worked": 600
            }
            
            monthly_forecast = {
                "Total Revenue": "$7,500",
                "Total Projects": 3,
                "Average Revenue per Project": "$2,500",
                "Expected New Clients": 1-2,
                "Hours Worked": 150
            }
            
            with st.expander("Freelance Forecast Details"):
                st.write("### End of Year (EOY) Forecast")
                for key, value in eoy_forecast.items():
                    st.write(f"**{key}:** {value}")
                
                st.write("### Rest of Year (ROY) Forecast")
                for key, value in roy_forecast.items():
                    st.write(f"**{key}:** {value}")
                
                st.write("### Monthly Forecast")
                for key, value in monthly_forecast.items():
                    st.write(f"**{key}:** {value}")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Placeholder data for freelance projects
            projects = [
                {"name": "Del Monte", "hours_worked": 120, "rate": "$50/hr", "total_earnings": "$6000"},
                {"name": "Rally", "hours_worked": 80, "rate": "$60/hr", "total_earnings": "$4800"},
                {"name": "Gruns", "hours_worked": 100, "rate": "$55/hr", "total_earnings": "$5500"},
                {"name": "Nestle", "hours_worked": 90, "rate": "$65/hr", "total_earnings": "$5850"}
            ]
            
            with st.expander("Project Details"):
                for project in projects:
                    st.write(f"**{project['name']}**:")
                    st.write(f"Hours Worked: {project['hours_worked']}")
                    st.write(f"Rate: {project['rate']}")
                    st.write(f"Total Earnings: {project['total_earnings']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    # extra charts below-------------------------------------------------------------------------------------------------------------------
        c1, c2= st.columns(2)
        c3 = st.empty()  # Create an empty container
        c4 = st.empty()

        with st.container():
            c1.write("YTD Sales")
            c2.write("YoY Sales")

        with st.container():
            c3.write("YTD RocketMoney")

        with st.container():
            c4.write("c4")
            
        with c2:
            df_closed_deals = df[df['dealstage'] == 'closedwon']

            df_closed_deals['closedate'] = pd.to_datetime(df_closed_deals['closedate'])

            # Extract year and month from 'closedate' and convert to string
            df_closed_deals['year_month'] = df_closed_deals['closedate'].dt.to_period('M').astype(str)
            monthly_deal_amounts = df_closed_deals.groupby('year_month')['amount'].sum().reset_index()

                    # Separate and Filter Data by Year
            df_2023 = monthly_deal_amounts[monthly_deal_amounts['year_month'].str.startswith('2023')]
            df_2024 = monthly_deal_amounts[monthly_deal_amounts['year_month'].str.startswith('2024')]

            # Combine data while adding a "Year" column to differentiate
            df_combined = pd.concat([
                df_2023.assign(Year='2023'),
                df_2024.assign(Year='2024')
            ])

            # Create Plotly figure
            fig = go.Figure()

            # Add traces for each year with distinct colors
            fig.add_trace(go.Bar(
                x=df_combined[df_combined['Year'] == '2023']['year_month'],
                y=df_combined[df_combined['Year'] == '2023']['amount'],
                name='2023',
                marker_color='skyblue'
            ))

            fig.add_trace(go.Bar(
                x=df_combined[df_combined['Year'] == '2024']['year_month'],
                y=df_combined[df_combined['Year'] == '2024']['amount'],
                name='2024',
                marker_color='lightcoral'
            ))

            # Customize the chart layout
            fig.update_layout(
                title='Closed Won Deals Amounts Per Month (2023 vs. 2024)',
                xaxis_title='Month',
                yaxis_title='Amount',
                barmode='group', # Ensure bars are grouped side-by-side
                paper_bgcolor="black",
                plot_bgcolor="black",
                font=dict(color="white"),
                legend=dict(bgcolor='rgba(0,0,0,0)')  # Make the legend background transparent to blend with the plot
            )


            # Display the chart in Streamlit
            st.plotly_chart(fig)
        with c1:

            df_closed_deals = df[df['dealstage'] == 'closedwon']

            df_closed_deals['closedate'] = pd.to_datetime(df_closed_deals['closedate'])

            # Extract year and month from 'closedate' and convert to string
            df_closed_deals['year_month'] = df_closed_deals['closedate'].dt.to_period('M').astype(str)
            monthly_deal_amounts = df_closed_deals.groupby('year_month')['amount'].sum().reset_index()

            # Prepare data for linear regression
            monthly_deal_amounts['date_num'] = pd.to_datetime(monthly_deal_amounts['year_month']).map(pd.Timestamp.toordinal)
            X = monthly_deal_amounts[['date_num']]
            y = monthly_deal_amounts['amount']

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Create future months
            last_date = pd.to_datetime(monthly_deal_amounts['year_month'].iloc[-1])
            future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
            future_months_str = future_months.to_period('M').astype(str)
            future_dates_num = future_months.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

            # Predict future values using the trained model
            future_amounts = model.predict(future_dates_num)

            # Manually construct the data for future months
            future_data = {
            'year_month': future_months_str,
            'amount': future_amounts
            }

            # Convert to DataFrame for easier manipulation
            future_df = pd.DataFrame(future_data)

            # Combine the past and future data
            all_data = pd.concat([monthly_deal_amounts[['year_month', 'amount']], future_df], ignore_index=True)

            # Create color scale
            colors = px.colors.sequential.Blues_r

            # Generate gradient color values
            n_colors = len(all_data)
            gradient_colors = [colors[int(i * (len(colors) - 1) / (n_colors - 1))] for i in range(n_colors)]

            # Create the base bar chart with gradient color scheme
            fig = go.Figure(data=[
            go.Bar(
                    x=all_data['year_month'],
                    y=all_data['amount'],
                    marker=dict(color=gradient_colors),
            )
            ])

            # Customize the chart
            fig.update_layout(
            title='Closed Won Deals Amounts Per Month with Trendline',
            xaxis_title='Month',
            yaxis_title='Amount',
            legend=dict(
                    yanchor="top", y=0.95, xanchor="right", x=0.01,
                    bordercolor="#444", borderwidth=1,
                    bgcolor="white", font=dict(size=12)
            ),
            autosize=False,
            width=800,
            height=450,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            bargap=0.15,  # gap between bars of adjacent bars
            bargroupgap=0.1  # gap between bars of the same group
            )

            # Add the trendline and forecast to the figure
            fig.add_trace(go.Scatter(
            x=all_data['year_month'], y=all_data['amount'], 
            mode='lines', name='Trendline & Forecast', line=dict(color='gray', dash='dashdot')
            ))

            # Display the chart in Streamlit
            st.plotly_chart(fig)
        with c3:
            rm = pd.read_csv('rocketmoney.csv')
            df = rm.copy()
            print(df.columns)
            # Convert 'Date' column to datetime format
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

            # Summarize the spending by date
            daily_spending = df.groupby('Date')['Amount'].sum().reset_index()

            # Create a bar chart using Plotly
            fig = px.bar(daily_spending, x='Date', y='Amount', title='<a href="https://interact-finance.streamlit.app/" target="_blank">YTD Rocket Money</a>')
            st.plotly_chart(fig)


    # ---------------------------------------------------------------------------------------------------------------------------
    if selected == "Home2":

        # Define CSS styles once for all cards
        st.markdown(
            """
            <style>
            .card {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 15px;
                box-shadow: 2px 2px 5px #ddd;
                max-height: 400px;
                overflow-y: auto;
                text-align: center; /* Center align all text within the card */

            }
            .title {
                font-size: 20px;
                margin-bottom: 10px;
            }
            .item {
                margin-bottom: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Define data for each card
        card_data = [
            {
                "title": "Account Receivable",
                "items": {"Invoice 1": "1500", "Invoice 2": "850", "Invoice 3": "2200"},
            },
            {
                "title": "Account Payable",
                "items": {"Bill 1": "550", "Bill 2": "1200", "Bill 3": "380"},
            },
            {
                "title": "Pending Deals",
                "items": {
                    "Deal A": "Potential Value: $10,000",
                    "Deal B": "Potential Value: $5,500",
                    "Deal C": "Potential Value: $18,250",
                },
            },
            {
                "title": "RocketMoney",
                "items": {
                    "Savings Goal": "$500/month",
                    "Subscriptions": "Netflix, Spotify, Gym",
                    "Spending Categories": "Food, Rent, Entertainment",
                },
            },
        ]

            
        # Create two rows of columns
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

            # --- Card Rendering Functions ---
        with col1:
                # # Card 1 content (using the card_data[1])
                # card_content = f"""
                # <div class="card">
                #     <h2 class="title">{card_data[0]['title']}</h2>
                    
                #     {"100"}

                #     {"100"}
                # </div>
                # """
                # st.markdown(card_content, unsafe_allow_html=True)
            st.markdown(
                    """
                    <div class="card">
                        <h2 class="title">Account Receivable</h2>
                        <h3 class="title">100,000</h3>
                        <p class="item">Current: 1500</p>
                        <p class="item">30 Days Overdue: 850</p>
                        <p class="item">60 Days Overdue: 2200</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col2:
            # Card 2 content (using the card_data[1])
            st.markdown(
                    """
                    <div class="card">
                        <h2 class="title">Account Payable</h2>
                        <h3 class="title">100,000</h3>
                        <p class="item">Payroll: 1500</p>
                        <p class="item">Bill.com: 850</p>
                        <p class="item">Expensify: 2200</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col3:
            st.markdown(
                    """
                    <div class="card">
                        <h2 class="title">Pending Deals</h2>
                        <h3 class="title">100,000</h3>
                        <p class="item">Proposals/Negotiation: 1500</p>
                        <p class="item">Inbound/Discovery Call: 850</p>
                        <p class="item">Invoice 3: 2200</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col4:
            st.markdown(
                    """
                    <div class="card">
                        <h2 class="title">Rocket Money</h2>
                        <h3 class="title">100,000</h3>
                        <p class="item">Subscriptions: 1500</p>
                        <p class="item">Spending Categories: 850</p>
                        <p class="item">Invoice 3: 2200</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        c1, c2= st.columns(2)
        c3 = st.empty()  # Create an empty container
        c4 = st.empty()

        with st.container():
            c1.write("YTD Sales")
            c2.write("YoY Sales")

        with st.container():
            c3.write("c5")
            c4.write("c6")
        with c2:
            df_closed_deals = df[df['dealstage'] == 'closedwon']

            df_closed_deals['closedate'] = pd.to_datetime(df_closed_deals['closedate'])

            # Extract year and month from 'closedate' and convert to string
            df_closed_deals['year_month'] = df_closed_deals['closedate'].dt.to_period('M').astype(str)
            monthly_deal_amounts = df_closed_deals.groupby('year_month')['amount'].sum().reset_index()

                    # Separate and Filter Data by Year
            df_2023 = monthly_deal_amounts[monthly_deal_amounts['year_month'].str.startswith('2023')]
            df_2024 = monthly_deal_amounts[monthly_deal_amounts['year_month'].str.startswith('2024')]

            # Combine data while adding a "Year" column to differentiate
            df_combined = pd.concat([
                df_2023.assign(Year='2023'),
                df_2024.assign(Year='2024')
            ])

            # Create Plotly figure
            fig = go.Figure()

            # Add traces for each year with distinct colors
            fig.add_trace(go.Bar(
                x=df_combined[df_combined['Year'] == '2023']['year_month'],
                y=df_combined[df_combined['Year'] == '2023']['amount'],
                name='2023',
                marker_color='skyblue'
            ))

            fig.add_trace(go.Bar(
                x=df_combined[df_combined['Year'] == '2024']['year_month'],
                y=df_combined[df_combined['Year'] == '2024']['amount'],
                name='2024',
                marker_color='lightcoral'
            ))

            # Customize the chart layout
            fig.update_layout(
                title='Closed Won Deals Amounts Per Month (2023 vs. 2024)',
                xaxis_title='Month',
                yaxis_title='Amount',
                barmode='group', # Ensure bars are grouped side-by-side
                paper_bgcolor="black",
                plot_bgcolor="black",
                font=dict(color="white"),
                legend=dict(bgcolor='rgba(0,0,0,0)')  # Make the legend background transparent to blend with the plot
            )


            # Display the chart in Streamlit
            st.plotly_chart(fig)
        with c1:

            df_closed_deals = df[df['dealstage'] == 'closedwon']

            df_closed_deals['closedate'] = pd.to_datetime(df_closed_deals['closedate'])

            # Extract year and month from 'closedate' and convert to string
            df_closed_deals['year_month'] = df_closed_deals['closedate'].dt.to_period('M').astype(str)
            monthly_deal_amounts = df_closed_deals.groupby('year_month')['amount'].sum().reset_index()

            # Prepare data for linear regression
            monthly_deal_amounts['date_num'] = pd.to_datetime(monthly_deal_amounts['year_month']).map(pd.Timestamp.toordinal)
            X = monthly_deal_amounts[['date_num']]
            y = monthly_deal_amounts['amount']

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Create future months
            last_date = pd.to_datetime(monthly_deal_amounts['year_month'].iloc[-1])
            future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
            future_months_str = future_months.to_period('M').astype(str)
            future_dates_num = future_months.map(pd.Timestamp.toordinal).values.reshape(-1, 1)

            # Predict future values using the trained model
            future_amounts = model.predict(future_dates_num)

            # Manually construct the data for future months
            future_data = {
            'year_month': future_months_str,
            'amount': future_amounts
            }

            # Convert to DataFrame for easier manipulation
            future_df = pd.DataFrame(future_data)

            # Combine the past and future data
            all_data = pd.concat([monthly_deal_amounts[['year_month', 'amount']], future_df], ignore_index=True)

            # Create color scale
            colors = px.colors.sequential.Blues_r

            # Generate gradient color values
            n_colors = len(all_data)
            gradient_colors = [colors[int(i * (len(colors) - 1) / (n_colors - 1))] for i in range(n_colors)]

            # Create the base bar chart with gradient color scheme
            fig = go.Figure(data=[
            go.Bar(
                    x=all_data['year_month'],
                    y=all_data['amount'],
                    marker=dict(color=gradient_colors),
            )
            ])

            # Customize the chart
            fig.update_layout(
            title='Closed Won Deals Amounts Per Month with Trendline',
            xaxis_title='Month',
            yaxis_title='Amount',
            legend=dict(
                    yanchor="top", y=0.95, xanchor="right", x=0.01,
                    bordercolor="#444", borderwidth=1,
                    bgcolor="white", font=dict(size=12)
            ),
            autosize=False,
            width=800,
            height=450,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            bargap=0.15,  # gap between bars of adjacent bars
            bargroupgap=0.1  # gap between bars of the same group
            )

            # Add the trendline and forecast to the figure
            fig.add_trace(go.Scatter(
            x=all_data['year_month'], y=all_data['amount'], 
            mode='lines', name='Trendline & Forecast', line=dict(color='gray', dash='dashdot')
            ))

            # Display the chart in Streamlit
            st.plotly_chart(fig)
        with c3:
            rm = pd.read_csv('rocketmoney.csv')
            df = rm.copy()
            print(df.columns)
            # Convert 'Date' column to datetime format
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

            # Summarize the spending by date
            daily_spending = df.groupby('Date')['Amount'].sum().reset_index()

            # Create a bar chart using Plotly
            fig = px.bar(daily_spending, x='Date', y='Amount', title='YTD Rockey Money')
            st.plotly_chart(fig)