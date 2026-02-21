import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')

# --------------- healper functions ---------------
# daily orders dataframe
def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_approved_at').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "gross_sales"
    }, inplace=True)

    return daily_orders_df

# orders dataframe
def create_sum_order_items_df(df):
    sum_order_items_df = df.groupby('product_category_name')['order_item_id'].count().sort_values(ascending=False).reset_index()
    return sum_order_items_df

# by cities dataframe
def create_bycities_df(df):
    bycities_df = df.groupby('customer_city')['customer_id'].nunique().sort_values(ascending=False).head(10).reset_index()
    bycities_df.rename(columns={'customer_id': 'customer_count'}, inplace=True)
    return bycities_df

# rfm dataframe
def create_rfm_df(df):
    rfm_df = df
    reference_date = rfm_df['order_purchase_timestamp'].max()
    rfm_df = df.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
    'order_id': 'nunique',
    'price': 'sum'
    }).reset_index()

    rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    return rfm_df
# --------------- end healper functions ---------------

all_df = pd.read_csv('dashboard/merged_df.csv')

# Convert datetime
datetime_columns = ['order_purchase_timestamp','order_approved_at', 'order_delivered_customer_date']
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

all_df.sort_values(by='order_approved_at', inplace=True)
all_df.reset_index(drop=True, inplace=True)

# --------------- create filter ---------------

min_date = all_df["order_approved_at"].min()
max_date = all_df["order_approved_at"].max()

with st.sidebar:
    st.image("dashboard/logo.png")

    date_range = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

# Pastikan user memilih 2 tanggal
if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.warning("Silakan pilih tanggal mulai dan tanggal akhir.")
    st.stop()

start_date, end_date = date_range

# Filter dataframe
main_df = all_df[
    (all_df["order_approved_at"] >= pd.to_datetime(start_date)) &
    (all_df["order_approved_at"] <= pd.to_datetime(end_date))
]

# Generate dataframe turunan
daily_orders_df = create_daily_orders_df(main_df)
sum_order_items_df = create_sum_order_items_df(main_df)
bycities_df = create_bycities_df(main_df)
rfm_df = create_rfm_df(main_df)

# --------------- end create filter ---------------

# --------------- streamlit dashboard ---------------
st.header('E-Commerce Collection Dashboard :sparkles:')


# Daily Orders Section
st.subheader('Daily Orders')
if not daily_orders_df.empty:
    col1, col2 = st.columns(2)
    with col1:
        total_orders = daily_orders_df.order_count.sum()
        st.metric("Total orders", value=total_orders)
    with col2:
        total_sales = format_currency(daily_orders_df.gross_sales.sum(), "EUR ", locale='es_CO') 
        st.metric("Total Sales", value=total_sales)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        daily_orders_df["order_approved_at"],
        daily_orders_df["order_count"],
        marker='o', 
        linewidth=2,
        color="#90CAF9"
    )

    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=15)
    st.pyplot(fig)

else:
    st.warning("No data available for selected date range.")

# Best and Worst Performing Products Section
st.subheader("Best & Worst Performing Product") 
if not sum_order_items_df.empty:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    sns.barplot(x='order_item_id', y='product_category_name', data=sum_order_items_df.head(5), palette=colors, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Best Performing Products", loc="center", fontsize=50)
    ax[0].tick_params(axis='y', labelsize=30)

    sns.barplot(x='order_item_id', y='product_category_name', data=sum_order_items_df.sort_values(by='order_item_id', ascending=True).head(5), palette=colors, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Worst Performing Products", loc="center", fontsize=50)
    ax[1].tick_params(axis='y', labelsize=30)
    st.pyplot(fig)

else:
    st.warning("No data available for selected date range.")

# Customer Demographics Section
st.subheader("Customer Demographics")
if not bycities_df.empty:
    fig, ax = plt.subplots(figsize=(20, 10))
    n = len(bycities_df)
    colors = ["#90CAF9"] + ["#D3D3D3"] * (n - 1)
    sns.barplot(
        x="customer_count", 
        y="customer_city",
        data=bycities_df.sort_values(by="customer_count", ascending=False),
        palette=colors,
        ax=ax
    )
    ax.set_title("Number of Customer by cities", loc="center", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=15)
    st.pyplot(fig)

else:
    st.warning("No data available for selected date range.")

# RFM Analysis Section
st.subheader("RFM Distribution Analysis")

if not rfm_df.empty:

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- Recency ---
    axes[0].hist(rfm_df['recency'], bins=20, color="#90CAF9")
    axes[0].set_title("Recency Distribution", fontsize=30)
    axes[0].set_xlabel("Days Since Last Purchase", fontsize=20)
    axes[0].set_ylabel("Number of Customers", fontsize=20)
    axes[0].tick_params(axis='both', labelsize=20)

    # --- Frequency ---
    axes[1].hist(rfm_df['frequency'], bins=20, color="#90CAF9")
    axes[1].set_title("Frequency Distribution", fontsize=30)
    axes[1].set_xlabel("Number of Orders", fontsize=20)
    axes[1].tick_params(axis='both', labelsize=20)

    # --- Monetary ---
    axes[2].hist(np.log1p(rfm_df['monetary']), bins=20, color="#90CAF9")
    axes[2].set_title("Monetary Distribution (Log Scale)", fontsize=30)
    axes[2].set_xlabel("Log Total Spending", fontsize=20)
    axes[2].tick_params(axis='both', labelsize=20)

    # Tambahan grid supaya lebih profesional
    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.warning("No data available for selected date range.")

# --------------- end streamlit dashboard ---------------
