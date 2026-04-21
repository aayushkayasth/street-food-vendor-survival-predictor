import pandas as pd

def create_features(df):
    df = df.copy()

    df['revenue_per_customer'] = df['avg_daily_revenue_inr'] / (df['avg_daily_customers'] + 1)
    df['customers_per_hour'] = df['avg_daily_customers'] / (df['hours_open_per_day'] + 1)
    df['revenue_per_hour'] = df['avg_daily_revenue_inr'] / (df['hours_open_per_day'] + 1)

    df['monthly_revenue'] = df['avg_daily_revenue_inr'] * 30
    df['profit_estimate'] = df['monthly_revenue'] - df['monthly_stall_rent_inr']

    return df