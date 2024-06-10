import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
import streamlit_option_menu
from streamlit_option_menu import option_menu
from hubspot_request import fetch_and_process_data, transform_closed_deals

from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go

# Define a custom color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

from datetime import datetime, timedelta

st.set_page_config(page_title='Finances',  layout='wide', page_icon=':rocket:')

t1, t2 = st.columns((0.2,1)) 

t1.image('images/interactbrands_logo.jpg', width = 120)
t2.title("Finance Reports of Interact")


initial_url = 'https://api.hubapi.com/crm/v3/objects/deals'
# df = fetch_and_process_data(initial_url)

df = pd.read_csv('dataframe.csv')


closed_deals = transform_closed_deals(df)

df = closed_deals

with st.sidebar:
    selected = option_menu(
    menu_title = "Main Menu",
    options = ["Hubspot-Deals","Quickbooks - Balance","RM","Bill COM","Issues", "Contacts", "Companies"],
    icons = ["house","activity","activity","activity","bug", "phone", "phone"],
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
        c1.write("Total deals closed in the last 30 days")
        c2.write("Closed Won deals Per Month")

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

        # fig.update_layout(paper_bgcolor="lightgray")
        st.plotly_chart(fig)  # Use st.plotly_chart() to display the figure
        # today = datetime.now()
        # last_month_start = today - timedelta(days=30)

        # # Filter the DataFrame
        # df_last_month = df[df['closedate'] >= last_month_start]

        # # Calculate the sum of the 'amount' column
        # total_amount_last_month = df_last_month['amount'].sum()

        # # Create the Plotly figure with the dynamic value
        # fig = go.Figure()

        # indicator_trace = go.Indicator(
        # mode="number+delta",
        # value=total_amount_last_month,
        # number={'prefix': "$"},
        # delta={'position': "top", 'reference': 320},
        # domain={'x': [0, 1], 'y': [0, 1]}
        # )
        # fig.add_trace(indicator_trace)

        # fig.update_layout(
        # paper_bgcolor="black",
        # plot_bgcolor="black",
        # font_color="white",
        # title_font_color="white",
        # xaxis_showgrid=False,
        # yaxis_showgrid=False,
        # xaxis_zeroline=False,
        # yaxis_zeroline=False
        # )

        # # Adjust the position of the title
        # fig.update_layout(title_x=0.5)

        # # Display the chart in Streamlit
        # st.plotly_chart(fig)
           
    with c2:
        # df_closed_deals = df[df['dealstage'] == 'closedwon']
        # df_closed_deals['closedate'] = pd.to_datetime(df_closed_deals['closedate'])
        # # Extract year and month from 'closedate' and convert to string
        # df_closed_deals['year_month'] = df_closed_deals['closedate'].dt.to_period('M').apply(lambda x: x.strftime('%Y-%m'))

        # deals_per_month = df_closed_deals.groupby('year_month').size().reset_index(name='count')

        # # Streamlit app to display the line chart
        # # st.title('Closed Won Deals Per Month')
        # fig = px.line(deals_per_month, x='year_month', y='count', title='',
        #         labels={'count': 'Number of Closed Deals'})
        # fig.update_xaxes(title_text='Month')
        # fig.update_yaxes(title_text='Number of Closed Deals')

        # st.plotly_chart(fig)
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
    pass
#     contacts = pd.read_csv('contacts.csv')
#     st.write(contacts)


if selected == "Contacts":
    st.subheader(f"**You Have selected {selected}**")
    contacts = pd.read_csv('contacts.csv')
    st.write(contacts)


if selected == "Companies":
    st.subheader(f"**You Have selected {selected}**")
    contacts = pd.read_csv('companies.csv')
    st.write(contacts)