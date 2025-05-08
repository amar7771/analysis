import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Configure the page
st.set_page_config(page_title='Telecom Churn Dashboard', layout='wide')

# Custom CSS to hide default Streamlit elements and adjust padding
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {
    padding-top: 0rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('📞 Telecom Customer Churn Dashboard')

@st.cache_data
def get_data():
    path = r"F:\telecom_customer_churn.csv"
    df = pd.read_csv(path)

    # Fill categorical missing values
    cols_to_fill = [
        'Multiple Lines', 'Internet Type', 'Online Security', 'Online Backup',
        'Offer',
        'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
        'Streaming Movies', 'Streaming Music', 'Unlimited Data'
    ]
    for col in cols_to_fill:
        df[col] = df[col].fillna(df[col].mode()[0])

    # KNN Imputation for numeric features
    cols = ['Avg Monthly Long Distance Charges', 'Avg Monthly GB Download']
    imputer = KNNImputer(n_neighbors=3)
    df[cols] = imputer.fit_transform(df[cols])

    return df

df = get_data()

# Create monthly revenue data (you can replace this with your actual monthly data)
monthly_revenue_data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Revenue': [12000, 15000, 17000, 13000, 18000, 20000, 19000, 22000, 21000, 23000, 25000, 24000]
}
monthly_revenue_df = pd.DataFrame(monthly_revenue_data)

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["Customer Overview", "Revenue Analysis", "Churn Analysis", "Offer Analysis"])

with tab1:
    st.header("Customer Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Distribution")
        gender_counts = df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig = px.bar(
            gender_counts,
            x="Gender",
            y="Count",
            color="Gender",
            title="Customer Gender Distribution",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Gender Insights"):
            st.write("""
            - The gender distribution is relatively balanced
            - No significant skew toward male or female customers
            - Slightly more male customers than female
            """)
    
    with col2:
        st.subheader("Marital Status")
        fig = px.pie(
            df,
            names="Married",
            title="Marital Status Distribution",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Marital Status Insights"):
            st.write("""
            - Shows the proportion of married vs unmarried customers
            - Can be correlated with other factors like number of dependents
            """)
    
    st.subheader("Age Distribution")
    fig = px.histogram(
        df,
        x="Age",
        nbins=20,
        title="Age Distribution of Customers",
        labels={"Age": "Customer Age"},
        template="plotly_white"
    )
    fig.update_layout(yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Age Distribution Insights"):
        st.write("""
        - Customer ages range from 20s to 70s
        - Highest concentration appears in mid-30s to 50s
        - Important for targeted marketing campaigns
        """)

with tab2:
    st.header("Revenue Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Contract Type")
        revenue_by_contract = df.groupby("Contract", as_index=False)["Total Revenue"].mean()
        fig = px.bar(
            revenue_by_contract,
            x="Contract",
            y="Total Revenue",
            color="Contract",
            title="Average Total Revenue by Contract Type",
            text_auto='.2s',
            labels={"Total Revenue": "Average Revenue (USD)"},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Contract Revenue Insights"):
            st.write("""
            - Two-year contracts generate the highest average revenue
            - Month-to-month contracts have the lowest average revenue
            - Shows the value of long-term customer retention
            """)
    
    with col2:
        st.subheader("Revenue vs Age")
        fig = px.scatter(
            df,
            x="Age",
            y="Total Revenue",
            color="Contract",
            title="Total Revenue by Customer Age",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Revenue vs Age Insights"):
            st.write("""
            - Shows how revenue varies by age group
            - Can identify which age groups are most valuable
            - Helps spot outliers or unusual patterns
            """)
    
    # New Monthly Revenue Plot
    st.subheader("Monthly Revenue Trend")
    fig = px.scatter(
        monthly_revenue_df,
        x='Month',
        y='Revenue',
        title='Monthly Total Revenue',
        labels={'Month': 'Month', 'Revenue': 'Total Revenue ($)'},
        color_discrete_sequence=['#1f77b4'],
        size='Revenue',
        hover_data=['Revenue']
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Total Revenue ($)',
        showlegend=False,
        template='plotly_white',
        xaxis={'tickangle': 45}
    )
    
    # Add trend line
    fig.add_traces(
        px.line(monthly_revenue_df, x='Month', y='Revenue').data[0]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Monthly Revenue Insights"):
        st.write("""
        - Shows revenue trends throughout the year
        - Helps identify seasonal patterns
        - Trend line highlights overall growth or decline
        """)
    
    st.subheader("Monthly vs Total Charges")
    fig = px.scatter(
        df,
        x="Monthly Charge",
        y="Total Charges",
        color="Contract",
        title="Monthly Charge vs Total Charges",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Charges Insights"):
        st.write("""
        - Relationship between monthly charges and total charges
        - Shows how contract length affects total revenue
        - Can identify customers with unusually high or low charges
        """)

with tab3:
    st.header("Churn Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Status Distribution")
        status_counts = df["Customer Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(
            status_counts,
            names="Status",
            values="Count",
            title="Customer Status Distribution",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Status Insights"):
            st.write("""
            - Shows proportion of churned vs stayed customers
            - Can include other statuses like 'Joined' if present
            - Baseline for churn rate analysis
            """)
    
    with col2:
        st.subheader("Churn Reasons")
        if 'Churn Category' in df.columns:
            churn_reasons = df[df['Churn Category'].notna()]['Churn Category'].value_counts().reset_index()
            churn_reasons.columns = ["Reason", "Count"]
            fig = px.bar(
                churn_reasons,
                x="Reason",
                y="Count",
                color="Reason",
                title="Primary Reasons for Churn",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Churn Reason Insights"):
                st.write("""
                - Shows top reasons customers leave
                - Helps identify areas for improvement
                - Competitor offers and dissatisfaction are common reasons
                """)
        else:
            st.warning("Churn category data not available in this dataset")
    
    st.subheader("Churn by Contract Type")
    if 'Customer Status' in df.columns and 'Contract' in df.columns:
        churn_by_contract = df.groupby(['Contract', 'Customer Status']).size().unstack().fillna(0)
        churn_by_contract['Churn Rate'] = (churn_by_contract['Churned'] / 
                                         (churn_by_contract['Churned'] + churn_by_contract['Stayed'])) * 100
        
        fig = px.bar(
            churn_by_contract,
            x=churn_by_contract.index,
            y="Churn Rate",
            title="Churn Rate by Contract Type",
            labels={"value": "Percentage", "variable": "Status"},
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Churn by Contract Insights"):
            st.write("""
            - Month-to-month contracts typically have highest churn
            - Long-term contracts usually have lower churn rates
            - Shows importance of contract type in retention
            """)
    else:
        st.warning("Required columns for churn analysis not available")

with tab4:
    st.header("Offer Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Most Frequently Purchased Offers")
        if 'Offer' in df.columns:
            offer_counts = df["Offer"].value_counts().reset_index()
            offer_counts.columns = ["Offer", "Count"]
            
            fig = px.bar(
                offer_counts,
                x="Offer",
                y="Count",
                color="Offer",
                title="Most Frequently Purchased Offers",
                labels={"Count": "Number of Customers"},
                template="plotly_white",
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title="Offer",
                yaxis_title="Number of Customers",
                xaxis_tickangle=45,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Offer Popularity Insights"):
                st.write("""
                - Shows which offers are most popular with customers
                - Helps identify successful marketing campaigns
                - Can reveal customer preferences
                """)
        else:
            st.warning("Offer data not available in this dataset")
    
    with col2:
        st.subheader("Offers by Revenue Generated")
        if 'Offer' in df.columns and 'Total Revenue' in df.columns:
            revenue_by_offer = df.groupby("Offer")["Total Revenue"].sum().reset_index()
            revenue_by_offer.columns = ["Offer", "Total Revenue"]
            revenue_by_offer = revenue_by_offer.sort_values("Total Revenue", ascending=False)
            
            fig = px.bar(
                revenue_by_offer,
                x="Offer",
                y="Total Revenue",
                color="Offer",
                title="Offers by Total Revenue Generated",
                labels={"Total Revenue": "Total Revenue ($)"},
                template="plotly_white",
                text_auto='.2s'
            )
            
            fig.update_layout(
                xaxis_title="Offer",
                yaxis_title="Total Revenue ($)",
                xaxis_tickangle=45,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("Offer Revenue Insights"):
                st.write("""
                - Shows which offers generate the most revenue
                - Popular offers may not always be the most profitable
                - Helps optimize marketing spend
                """)
        else:
            st.warning("Offer or revenue data not available in this dataset")

# Add a sidebar with filters
with st.sidebar:
    st.header("Filters")
    
    # Contract type filter
    contract_types = df['Contract'].unique() if 'Contract' in df.columns else []
    selected_contracts = st.multiselect(
        "Select Contract Types",
        options=contract_types,
        default=contract_types
    )
    
    # Age range filter
    min_age, max_age = (int(df['Age'].min()), int(df['Age'].max())) if 'Age' in df.columns else (0, 100)
    age_range = st.slider(
        "Select Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )
    
    # Offer filter (if available)
    if 'Offer' in df.columns:
        offers = df['Offer'].unique()
        selected_offers = st.multiselect(
            "Select Offers",
            options=offers,
            default=offers
        )
    else:
        selected_offers = []
    
    # Apply filters
    filtered_df = df.copy()
    if len(contract_types) > 0:
        filtered_df = filtered_df[filtered_df['Contract'].isin(selected_contracts)]
    if 'Age' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Age'] >= age_range[0]) &
            (filtered_df['Age'] <= age_range[1])
        ]
    if len(selected_offers) > 0 and 'Offer' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Offer'].isin(selected_offers)]
    
    st.header("Key Metrics")
    if 'Customer Status' in filtered_df.columns:
        churn_rate = (filtered_df['Customer Status'] == 'Churned').mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    if 'Total Revenue' in filtered_df.columns:
        avg_revenue = filtered_df['Total Revenue'].mean()
        st.metric("Average Revenue", f"${avg_revenue:,.2f}")
    
    st.header("About")
    st.write("""
    This dashboard analyzes telecom customer churn data.
    Use the filters to explore different customer segments.
    """)

# Update all visualizations with filtered data
if 'filtered_df' in locals():
    df = filtered_df