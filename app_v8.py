import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import psutil
import os
from sklearn.metrics import mean_absolute_error
from prophet.plot import plot_components_plotly
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import math

# Constants
DATE_COL = "Date"
FILTER_COLUMNS = [
    'Campaign',
    'Campaign Title',
    'Campaign Category',
    'Campaign Stage',
    'Campaign Objective',
    'Campaign Platform',
    'Advertising Type',
    'Audience'
]

NON_AGGREGATABLE_METRICS = [
    # Unique or average counts
    "Reach",
    "Frequency",
    # Ratios
    "CTR",
    "CVR",
    # Unit‐cost metrics (do not sum these, but average or weight by volume)
    "CPM",
    "CPC",
    "CPV",
    "CPVC",
    "CPE",
    "CPCV",
]

performance_metrics = [
    'Impressions',
    'Reach',
    'Frequency',
    'Actions',
    'Post Engagements',
    'Link Clicks',
    'Video Completions',
    'ThruPlay Actions',
    'Landing Page Views',
    'Clicks',
    'Engagements',
    'Video Views',
    'Interactions',
    'Conversions',
    'CTR',
    'CVR'
]

cost_metrics = [
    "Cost",
    "CPM",
    "CPC",
    "CPV",
    "CPVC",
    "CPE",
    "CPCV"
]

# Page configuration
st.set_page_config(
    page_title="Media Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data():
    st.sidebar.header("Dataset Selection")

    dataset_map = {
        "Google (Cleaned)": "df_google_nonlog.csv",
        "Meta (Cleaned)": "df_meta_nonlog.csv",
        "Google (Log-transformed)": "df_google_log.csv",
        "Meta (Log-transformed)": "df_meta_log.csv",
        "Google (ROI Test)" : "df_google_log_ROI.csv",
        "Meta (ROI Test)" : "df_meta_log_ROI.csv"

    }
    selected_label = st.sidebar.selectbox("Choose a dataset", list(dataset_map.keys()))
    selected_file = dataset_map[selected_label]

    df = pd.read_csv(selected_file, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df, selected_label

def filter_data(df):
    st.sidebar.header("Filters")

    # Step 1: Store fixed fallback date range only once
    if "default_date_range" not in st.session_state:
        st.session_state.default_date_range = [
            df[DATE_COL].min().date(),
            df[DATE_COL].max().date()
        ]

    # Step 2: Initialize current filter state only once
    if "date_range" not in st.session_state:
        st.session_state.date_range = st.session_state.default_date_range

    # ✅ Step 3: Handle Reset BEFORE date_input is shown
    if st.sidebar.button("🔄 Reset Date Range"):
        st.session_state.date_range = st.session_state.default_date_range
        st.rerun()  # Trigger UI update so the widget reflects the reset

    # Step 4: Use a temporary widget (no key!)
    temp_date_range = st.sidebar.date_input(
        "Date Range",
        value=st.session_state.date_range,
        min_value=st.session_state.default_date_range[0],
        max_value=st.session_state.default_date_range[1],
    )

    # Step 5: Update state if user changes input
    if (
        isinstance(temp_date_range, (list, tuple)) and
        len(temp_date_range) == 2 and
        temp_date_range != st.session_state.date_range
    ):
        st.session_state.date_range = temp_date_range   

    # Step 6: Filter dataframe
    if len(st.session_state.date_range) == 2:
        filtered_df = df[
            (df[DATE_COL] >= pd.to_datetime(st.session_state.date_range[0])) &
            (df[DATE_COL] <= pd.to_datetime(st.session_state.date_range[1]))
        ]
    else:
        filtered_df = df  # fallback — don't filter

    # Year filter
    if 'Year' in df.columns:
        unique_year = filtered_df['Year'].dropna().unique()
        selected_year = st.sidebar.multiselect("Filter by Year", sorted(unique_year))
        if selected_year:
            filtered_df = filtered_df[filtered_df['Year'].isin(selected_year)]

    # Month filter
    if 'Month' in df.columns:
        unique_months = filtered_df['Month'].dropna().unique()
        selected_months = st.sidebar.multiselect("Filter by Month", sorted(unique_months))
        if selected_months:
            filtered_df = filtered_df[filtered_df['Month'].isin(selected_months)]

    # Week filter
    if 'Week' in df.columns:
        unique_weeks = filtered_df['Week'].dropna().unique()
        selected_weeks = st.sidebar.multiselect("Filter by Week", sorted(unique_weeks))
        if selected_weeks:
            filtered_df = filtered_df[filtered_df['Week'].isin(selected_weeks)]

    # Categorical filters
    applied_filters = []
    for col in FILTER_COLUMNS:
        if col in filtered_df.columns:
            options = sorted(filtered_df[col].dropna().unique())
            default = options if len(options) <= 10 else None
            selected = st.sidebar.multiselect(f"Filter by {col}", options, default=default)
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]
                applied_filters.append(col)
    return filtered_df, applied_filters

def plot_combined_metrics_line_chart(df, metric1, metric2, col_filter):
    """Line chart comparing two metrics on the same plot"""

    color_col = "Campaign" if "Campaign" in col_filter else None

    # Aggregate both metrics
    agg_cols = [DATE_COL]
    if color_col:
        agg_cols.append(color_col)

    plot_df = df.groupby(agg_cols)[[metric1, metric2]].sum().reset_index()

    # Melt the dataframe into long format
    plot_df_melted = plot_df.melt(id_vars=agg_cols, value_vars=[metric1, metric2],
                                  var_name="Metric", value_name="Value")

    # Plot
    fig = px.line(
        plot_df_melted,
        x=DATE_COL,
        y="Value",
        color="Metric" if not color_col else "Metric + Campaign",
        line_group=color_col if color_col else None,
        facet_col=color_col if color_col else None,
        title=f"Daily {metric1} vs {metric2}",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        height=600,
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="lightgrey",
            rangeslider=dict(visible=True),
            tickformat="%b %d\n%Y"
        ),
        yaxis=dict(
            title="Metric Value",
            showgrid=True,
            gridcolor="lightgrey",
            rangemode="tozero"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_metric_line_chart(df, metric,col_filter):
    """Enhanced line chart with smart aggregation and visualization"""

    # Determine color column dynamically
    color_col = "Campaign" if "Campaign" in col_filter else None

    # Group accordingly
    if color_col:
        plot_df = df.groupby([DATE_COL, color_col])[metric].sum().reset_index()
    else:
        plot_df = df.groupby(DATE_COL)[metric].sum().reset_index()

    title = f"Daily {metric} Totals"
        
    # Create the line chart
    fig = px.line(
        plot_df,
        x=DATE_COL,
        y=metric,
        color=color_col,
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Enhance layout
    fig.update_layout(
        height=600,
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="lightgrey",
            rangeslider=dict(visible=True),
            tickformat="%b %d\n%Y"
        ),
        yaxis=dict(
            title=metric,
            showgrid=True,
            gridcolor="lightgrey",
            rangemode="tozero"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Add range selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_metrics_subplots(df,metric1, metric2, col_filter):
    color_col = "Campaign" if "Campaign" in col_filter else None

    # Group accordingly
    if color_col:
        plot_df = df.groupby([DATE_COL, color_col])[[metric1,metric2]].sum().reset_index()
    else:
        plot_df = df.groupby(DATE_COL)[[metric1,metric2]].sum().reset_index()

    title = f"{metric1} and {metric2} Over Time"

    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=(metric1, metric2))

    fig.add_trace(go.Scatter(
        x=plot_df[DATE_COL], y=plot_df[metric1],
        mode='lines', name=metric1
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df[DATE_COL], y=plot_df[metric2],
        mode='lines', name=metric2
    ), row=2, col=1)

    fig.update_layout(
        title=title,
        height=700,
        template="plotly_white",
        hovermode="x unified",
        xaxis2=dict(rangeslider=dict(visible=True)),
        legend=dict(orientation="h", y=1.05, x=1, xanchor='right'),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
    
def plot_two_metrics(df, metric1, metric2, col_filter):
    color_col = "Campaign" if "Campaign" in col_filter else None

    # Group accordingly
    if color_col:
        plot_df = df.groupby([DATE_COL, color_col])[[metric1,metric2]].sum().reset_index()
    else:
        plot_df = df.groupby(DATE_COL)[[metric1,metric2]].sum().reset_index()

    title = f"{metric1 +' and ' + metric2} Over Time"

    fig = go.Figure()

    # Add first metric
    fig.add_trace(go.Scatter(
        x=plot_df[DATE_COL],
        y=plot_df[metric1],
        mode='lines',
        name=metric1
    ))

    # Add second metric
    fig.add_trace(go.Scatter(
        x=plot_df[DATE_COL],
        y=plot_df[metric2],
        mode='lines',
        name=metric2
    ))

    # Layout customization
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)),
        height=600,
        legend=dict(orientation="h", y=1.02, x=1, xanchor='right'),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_dr(df):
    def diminishing_returns(x, a, b):
        return a * (1 - np.exp(-b * x))
    
    roi_by_stage = df.groupby(['Campaign Stage', 'Cost'], as_index=False)['Stage ROI'].mean()
    roi_by_objective = df.groupby(['Campaign Objective', 'Cost'], as_index=False)['Stage ROI'].mean()

    stages = roi_by_stage['Campaign Stage'].unique()
    objectives = roi_by_objective['Campaign Objective'].unique()

    # ----- Stage Plot (unchanged) -----
    fig1 = go.Figure()
    for stage in stages:
        stage_df = roi_by_stage[roi_by_stage['Campaign Stage'] == stage].copy()
        x = stage_df['Cost'].values
        y = stage_df['Stage ROI'].values

        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            continue

        try:
            popt, _ = curve_fit(diminishing_returns, x, y, bounds=(0, [np.inf, 1]))
            a, b = popt
            stage_df['Fitted ROI'] = diminishing_returns(x, *popt)
            stage_df = stage_df.sort_values(by='Cost')

            fig1.add_trace(go.Scatter(x=stage_df['Cost'], y=stage_df['Fitted ROI'],
                                      mode='lines', name=f'{stage} Curve', line=dict(width=3)))

            saturation_x = -np.log(0.05) / b
            saturation_y = diminishing_returns(saturation_x, *popt)
            fig1.add_trace(go.Scatter(x=[saturation_x], y=[saturation_y],
                                      mode='markers+text', name=f'{stage} Saturation',
                                      marker=dict(size=10),
                                      text=[f"{stage} 95% Saturation"],
                                      textposition='top center'))
        except Exception as e:
            st.write(f"Skipping stage {stage} due to fitting error: {e}")

    fig1.update_layout(title='Diminishing Returns Curves by Campaign Stage with Saturation Points',
                       xaxis_title='Cost', yaxis_title='Stage ROI', template='plotly_white', height=500)
    st.plotly_chart(fig1, use_container_width=True)

    # ----- Objective Plot as Subplots -----
    n_cols = 3  
    n_rows = math.ceil(len(objectives) / n_cols)
    fig2 = make_subplots(rows=n_rows, cols=n_cols,
                         subplot_titles=[str(obj) for obj in objectives],
                         shared_xaxes=False, shared_yaxes=False,
                         vertical_spacing=0.15, horizontal_spacing=0.07)

    for idx, objective in enumerate(objectives):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        objective_df = roi_by_objective[roi_by_objective['Campaign Objective'] == objective].copy()
        x = objective_df['Cost'].values
        y = objective_df['Stage ROI'].values

        if len(objective_df['Cost'].unique()) < 3 or np.std(x) == 0 or np.std(y) == 0:
            st.write(f"Skipping {objective}: not enough variation or data")
            continue

        try:
            popt, _ = curve_fit(diminishing_returns, x, y, bounds=(0, [np.inf, 1]))
            a, b = popt
            fitted_y = diminishing_returns(x, *popt)
            objective_df['Fitted ROI'] = fitted_y
            objective_df = objective_df.sort_values(by='Cost')

            # Fitted curve
            fig2.add_trace(go.Scatter(x=objective_df['Cost'], y=objective_df['Fitted ROI'],
                                      mode='lines', name=f'{objective} Curve',
                                      line=dict(width=3), showlegend=False),
                           row=row, col=col)

            # Saturation point
            saturation_x = -np.log(0.05) / b
            saturation_y = diminishing_returns(saturation_x, *popt)
            fig2.add_trace(go.Scatter(x=[saturation_x], y=[saturation_y],
                                      mode='markers+text',
                                      name=f'{objective} Saturation',
                                      marker=dict(size=10),
                                      text=[f"95% Saturation"],
                                      textposition='top center',
                                      showlegend=False),
                           row=row, col=col)

        except Exception as e:
            st.write(f"Skipping objective {objective} due to fitting error: {e}")

    fig2.update_layout(height=350 * n_rows,
                       title_text="Diminishing Returns by Campaign Objective",
                       template="plotly_white",
                       margin=dict(t=80))

    st.plotly_chart(fig2, use_container_width=True)

def plot_roi(df):
    roi_by_stage = df.groupby('Campaign Stage', as_index=False)['Stage ROI'].mean()
    roi_stage_time = df.groupby([DATE_COL, 'Campaign Stage'], as_index=False)['Stage ROI'].mean()
    roi_by_objective = df.groupby('Campaign Objective', as_index=False)['Stage ROI'].mean()
    roi_objective_time = df.groupby([DATE_COL, 'Campaign Objective'], as_index=False)['Stage ROI'].mean()

    def diminishing_returns(x, a, b):
        return a * (1 - np.exp(-b * x))
    
    # 1. Stage
    fig1 = px.bar(
        roi_by_stage,
        x='Campaign Stage',
        y='Stage ROI',
        title='Average Stage ROI by Campaign Stage',
        labels={'Stage ROI': 'Stage ROI (per $ spent)'},
        color='Campaign Stage',
        text_auto='.2f'
    )
    fig1.update_layout(yaxis_title='Stage ROI', xaxis_title='Campaign Stage')
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Stage Over Time
    fig2 = px.line(
        roi_stage_time,
        x='Date',
        y='Stage ROI',
        color='Campaign Stage',
        title='Stage ROI Over Time by Campaign Stage',
        markers=True,
        labels={'Stage ROI': 'ROI per $'}
    )
    fig2.update_layout(xaxis_title='Date', yaxis_title='Stage ROI')
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Objective
    fig3 = px.bar(
        roi_by_objective,
        x='Campaign Objective',
        y='Stage ROI',
        title='Average Stage ROI by Campaign Objective',
        labels={'Stage ROI': 'Stage ROI (per $ spent)'},
        color='Campaign Objective',
        text_auto='.2f'
    )
    fig3.update_layout(yaxis_title='Stage ROI', xaxis_title='Campaign Objective')
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Objective Over Time
    fig4 = px.line(
        roi_objective_time,
        x='Date',
        y='Stage ROI',
        color='Campaign Objective',
        title='Stage ROI Over Time by Campaign Objective',
        markers=True,
        labels={'Stage ROI': 'ROI per $'}
    )
    fig4.update_layout(xaxis_title='Date', yaxis_title='Stage ROI')
    st.plotly_chart(fig4, use_container_width=True)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def plot_forecast(df, metric, days=120):
    """Prophet forecasting with enhanced error handling and seasonality components"""
    if metric.lower() in NON_AGGREGATABLE_METRICS:
        st.warning(f"Forecasting not supported for {metric}")
        return
    
    # Prepare time series data
    df['Date'] = pd.to_datetime(df['Date'])

    ts = (
        df
        .groupby('Date')[[metric] + cost_metrics]
        .sum()
        .reset_index()
    )

    if len(ts) < 365:
        st.warning("Insufficient data for forecasting (min 365 days required)")
        return

    test_days   = 30    # for back‐testing  
    train = ts.iloc[:-test_days].copy()
    test  = ts.iloc[-test_days:].copy()

    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
    for reg in cost_metrics:
        m.add_regressor(reg)
    m.fit(train.rename(columns={'Date':'ds', metric:'y'}))

    future = m.make_future_dataframe(periods=test_days)
    # bring in regressor values for both train+test
    future = future.merge(ts.rename(columns={'Date':'ds'})[['ds']+cost_metrics],
                        on='ds', how='left')
    fc = m.predict(future)

    # extract just the test‐period forecasts
    pred_prophet = (
        fc.set_index('ds')['yhat']
        .loc[test['Date']]
        .values
    )

    #	here: cost_metrics + simple calendar features
    def make_features(df_):
        X = df_[cost_metrics].copy()
        X['dow']   = df_['Date'].dt.dayofweek
        X['month'] = df_['Date'].dt.month
        return X

    X_train = make_features(train)
    y_train = train[metric]
    X_test  = make_features(test)

    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)

    results = []
    for name, yhat in [('Prophet', pred_prophet), ('XGBoost', pred_xgb)]:
        y_true = test[metric].values
        results.append({
            'Model': 	name,
            'MAE':   	mean_absolute_error(y_true, yhat),
            'MAPE (%)':  mape(y_true, yhat),
            'SMAPE (%)': smape(y_true, yhat),
            'R2':    	r2_score(y_true, yhat)
        })
    metrics_df = pd.DataFrame(results)
    # print(metrics_df.to_string(index=False))

    # --- after you’ve built `metrics_df` ---
    # identify which model had the lower MAPE (%) on the test split
    best = metrics_df.sort_values("MAPE (%)").iloc[0]["Model"]  
    best_row  = metrics_df.loc[metrics_df['MAPE (%)'].idxmin()]

    # extract model name and its MAPE (%)
    best_model = best_row['Model']
    best_MAPE   = best_row['MAPE (%)']

    st.write(f"Best back‑test model: {best_model} with MAPE (%) = {best_MAPE:.3f}")

    # define your forecast horizon
    future_days = days
    last_date   = ts['Date'].max()

    # build a future frame long enough for both
    future_all = m.make_future_dataframe(periods=future_days)
    future_all = future_all.merge(
        ts.rename(columns={'Date':'ds'})[['ds']+cost_metrics],
        on='ds', how='left'
    )
    future_all[cost_metrics] = future_all[cost_metrics].ffill().fillna(0)

    # Prophet forecast for all future dates
    fc_all = m.predict(future_all)
    fc_60 = fc_all[fc_all['ds'] > last_date].copy()

    # XGBoost forecast for the future (assumes regressors are carried forward)
    future_dates = pd.date_range(last_date + pd.Timedelta(1,'D'), periods=future_days)
    # grab the last known cost_metrics row and repeat it
    last_regs = ts.set_index('Date')[cost_metrics].loc[last_date]
    future_regs = pd.DataFrame([last_regs.values]*future_days,
                            index=future_dates, columns=cost_metrics)
    df_feat_60 = pd.DataFrame({"Date": future_dates})
    df_feat_60 = pd.concat([df_feat_60, future_regs.reset_index(drop=True)], axis=1)
    X_feat_60 = make_features(df_feat_60)

    pred_xgb_60 = xgb.predict(X_feat_60)

    # now branch to whichever was best
    if best == "Prophet":
        # use Prophet’s yhat, lower, upper
        forecast_df = fc_60.rename(columns={
            "yhat":       "Forecast",
            "yhat_lower": "Lower CI",
            "yhat_upper": "Upper CI"
        })[['ds','Forecast','Lower CI','Upper CI']]
    else:
        # use XGBoost — no CI
        forecast_df = pd.DataFrame({
            'ds':       future_dates,
            'Forecast': pred_xgb_60
        })
 
    # final Plotly chart
    fig = go.Figure([
        go.Scatter(x=ts['Date'], y=ts[metric], mode='lines', name='Historical'),
        go.Scatter(x=forecast_df['ds'], y=forecast_df['Forecast'],
                mode='lines', name=f'{future_days}‑Day {best} Forecast')
    ])

    if best == "Prophet":
        fig.add_trace(go.Scatter(
            x=list(forecast_df['ds']) + list(forecast_df['ds'][::-1]),
            y=list(forecast_df['Upper CI']) + list(forecast_df['Lower CI'][::-1]),
            fill='toself', fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'), name='95% CI'
        ))

    fig.update_layout(
        title=f"{metric}: {future_days}‑Day Forecast by {best}",
        xaxis_title='Date', yaxis_title=metric
    )

    st.plotly_chart(fig, use_container_width=True)

    # Seasonality components section
    if best_model == "Prophet":
        st.subheader("Seasonality Analysis")

        # build a year’s worth of future dates
        future_year = m.make_future_dataframe(periods=365, include_history=False)
        # merge in all regressors you added at training
        future_year = future_year.merge(
            ts.rename(columns={'Date':'ds'})[['ds'] + cost_metrics],
            on='ds', how='left'
        )
        # fill down so no missing regressor values
        future_year[cost_metrics] = future_year[cost_metrics].ffill().fillna(0)

        # get the full seasonal forecast
        forecast_year = m.predict(future_year)

        # WEEKLY
        weekly = (
        forecast_year[['ds','weekly']]
        .assign(day=lambda d: d['ds'].dt.day_name())
        .groupby('day')['weekly']
        .mean()
        .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        .reset_index()
        )
        fig_w = px.line(weekly, x='day', y='weekly',
                        title="Weekly Seasonality Pattern",
                        labels={'weekly':'Seasonal Impact','day':'Day of Week'},
                        template="plotly_white")
        st.plotly_chart(fig_w, use_container_width=True)

        # YEARLY
        monthly = (
        forecast_year[['ds','yearly']]
        .assign(month=lambda d: d['ds'].dt.month_name())
        .groupby('month')['yearly']
        .mean()
        .reindex([
            'January','February','March','April','May','June',
            'July','August','September','October','November','December'
        ])
        .reset_index()
        )
        fig_y = px.line(monthly, x='month', y='yearly',
                        title="Yearly Seasonality Pattern",
                        labels={'yearly':'Seasonal Impact','month':'Month'},
                        template="plotly_white")
        st.plotly_chart(fig_y, use_container_width=True)

    else:
        st.subheader("Approximate Seasonality from XGBoost")

        # 1. Prepare history with day & month
        hist = ts.copy()
        hist['dow']   = hist['Date'].dt.dayofweek
        hist['month'] = hist['Date'].dt.month

        # 2. Generate XGBoost predictions on history
        hist_feat = make_features(hist)
        hist['xgb_pred'] = xgb.predict(hist_feat)

        # 3. Weekly seasonality (by day of week)
        day_map = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
            4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        weekly = (
            hist
            .groupby('dow')['xgb_pred']
            .mean()
            .reindex(range(7))
            .reset_index()
        )
        weekly['day'] = weekly['dow'].map(day_map)

        fig_weekly = px.line(
            weekly,
            x='day', y='xgb_pred',
            title="Weekly Seasonality (XGBoost)",
            labels={'xgb_pred': 'Predicted Value', 'day': 'Day of Week'},
            template="plotly_white"
        )
        st.plotly_chart(fig_weekly, use_container_width=True)

        # 4. Yearly seasonality (by month)
        month_map = {
            1:'January', 2:'February', 3:'March',   4:'April',
            5:'May',     6:'June',     7:'July',    8:'August',
            9:'September',10:'October',11:'November',12:'December'
        }
        monthly = (
            hist
            .groupby('month')['xgb_pred']
            .mean()
            .reindex(range(1,13))
            .reset_index()
        )
        monthly['month_name'] = monthly['month'].map(month_map)

        fig_yearly = px.line(
            monthly,
            x='month_name', y='xgb_pred',
            title="Yearly Seasonality (XGBoost)",
            labels={'xgb_pred': 'Predicted Value', 'month_name': 'Month'},
            template="plotly_white"
        )
        st.plotly_chart(fig_yearly, use_container_width=True)


def main():
    # Load data
    df, selected_label = load_data()
    
    # Apply filters
    filtered_df, applied_filters = filter_data(df)
    
    # Main layout
    st.title("📊 Media Performance Analysis")
    
    # Data summary
    # First row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Date Range", f"{filtered_df[DATE_COL].min().strftime('%b-%Y')} "
                f"to {filtered_df[DATE_COL].max().strftime('%b-%Y')}")
    with col2:
        st.metric("Unique Campaigns", filtered_df["Campaign"].nunique())
        
    
    # Second row
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Total Records", len(filtered_df))
    with col4:
        st.metric("Total Cost", round(filtered_df["Cost"].sum(), 2))

    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Year']

    # Filter performance and cost metrics to only include those in the numeric columns
    available_performance_metrics = sorted([col for col in performance_metrics if col in numeric_cols])
    available_cost_metrics = sorted([col for col in cost_metrics if col in numeric_cols])

    # Streamlit metric selection UI
    metric1 = st.selectbox("Select Cost Metric", available_cost_metrics)
    metric2 = st.selectbox("Select Performance Metric", available_performance_metrics)

    # Chart selection
    chart_type = st.radio("Visulisation Option", ["Trend","Forecast Performance", "ROI", "Diminishing Curve"], horizontal=True)
    
    if chart_type == "Trend":
        plot_combined_metrics_line_chart(filtered_df, metric1, metric2, applied_filters)
        plot_metric_line_chart(filtered_df, metric1,applied_filters)
        plot_metric_line_chart(filtered_df, metric2,applied_filters)

    elif chart_type == "ROI" and "ROI" in selected_label:
        st.write("Reset Data Range to unfiltered for full")
        plot_roi(filtered_df)

    elif chart_type == "Diminishing Curve" and "ROI" in selected_label:
        plot_dr(filtered_df)
    
    else:
        days = st.slider("Forecast Performance Metric Horizon (days)", 30, 120, 180)
        plot_forecast(filtered_df, metric2, days)
    
    # Raw data preview
    with st.expander("View Filtered Data"):
        st.dataframe(filtered_df.sort_values(DATE_COL, ascending=False), 
                    height=300)

if __name__ == "__main__":
    main()

def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=1)
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / (1024 * 1024)  # in MB
    return cpu_percent, ram_usage

cpu, ram = get_resource_usage()
st.write(f"**CPU Usage**: {cpu:.2f}%")
st.write(f"**RAM Usage**: {ram:.2f} MB")