import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import ModelConfig, AppConfig
from src.data.data_loader import StockDataLoader
from src.models.lstm_model import StockPredictor
from src.utils.metrics import calculate_metrics
from src.utils.logger import setup_logger
from src.models.model_storage import ModelStorage
from src.utils.company_lookup import get_company_aliases

logger = setup_logger("streamlit_app")
# Initialize configurations
model_config = ModelConfig()
app_config = AppConfig()

# Page configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Application title and description
st.title("Stock Price Prediction App")
st.markdown("""
This app uses historical stock data to predict future stock prices using custom trained LSTM model (trained locally on each run, for updated predictions)
Select a company name to get started!
""")

def get_company_options() -> list:
    companies = get_company_aliases()
    # Create list of tuples (display_name, ticker)
    options = []
    for company in companies:
        # Add full name
        options.append((f"{company['name']} ({company['ticker']})", company['ticker']))
        # Add aliases
        # options.extend([(alias, company['ticker']) for alias in company['aliases']])
    return sorted(options, key=lambda x: x[0])

def display_metrics(data, company_info):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    with col2:
        st.metric("Market Cap", f"${company_info.get('marketCap', 0)/1e9:.2f}B")
    with col3:
        daily_return = ((data['Close'].iloc[-1] - data['Close'].iloc[-2])/data['Close'].iloc[-2]) * 100
        st.metric("Daily Return", f"{daily_return:.2f}%")

def display_prediction_metrics(actual, predicted):
    st.subheader("Model Performance Metrics")
    try:
        metrics = calculate_metrics(actual, predicted)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("RMSE", f"{metrics.get('RMSE', 0)}")
        with col2:
            st.metric("MAE", f"{metrics.get('MAE', 0)}")
        with col3:
            st.metric("R² Score", f"{metrics.get('R2', 0)}")
        with col4:
            st.metric("MSE", f"{metrics.get('MSE', 0)}")
        with col5:
            st.metric("Direction Accuracy", f"{metrics.get('Direction_Accuracy', 0):.1f}%")
            st.caption("Accuracy of predicting price movement direction")
    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error("Could not calculate performance metrics")

def display_plots(data, predictions):
    # Date range selector
    st.subheader("Historical Data Range")
    date_range = st.slider(
        "Select date range",
        min_value=data.index[0].date(),
        max_value=data.index[-1].date(),
        value=(data.index[-365].date(), data.index[-1].date())
    )
    
    # Filter data based on date range
    mask = (data.index.date >= date_range[0]) & (data.index.date <= date_range[1])
    filtered_data = data[mask]
    
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(predictions), freq='D')
    ticker_name = getattr(data, 'name', 'Stock')
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Stock Price', 'Volume')
    )
    
    # Updated colors and line styles
    fig.add_trace(
        go.Scatter(x=filtered_data.index, y=filtered_data['Close'],
                  name="Historical Price",
                  line=dict(color='#2E86C1', width=2)),  # Bright blue
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=future_dates, y=predictions.flatten(),
                  name="Predicted Price",
                  line=dict(color='#28B463', width=2)),  # Bright green
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=filtered_data.index, y=filtered_data['Volume'],
               name="Volume",
               marker_color='#85C1E9'),  # Light blue
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"Stock Price Analysis for {ticker_name}",
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    st.plotly_chart(fig, use_container_width=True)

def add_download_buttons(data, predictions):
    col1, col2 = st.columns(2)
    
    # Get ticker name with fallback
    ticker_name = getattr(data, 'name', 'Stock')
    
    with col1:
        csv = data.to_csv()
        st.download_button(
            label="Download Historical Data",
            data=csv,
            file_name=f'{ticker_name}_historical_data.csv',
            mime='text/csv',
        )
    
    with col2:
        predictions_df = pd.DataFrame({
            'Date': pd.date_range(start=data.index[-1], periods=len(predictions), freq='D'),
            'Predicted_Price': predictions.flatten()
        })
        csv_predictions = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv_predictions,
            file_name=f'{ticker_name}_predictions.csv',
            mime='text/csv',
        )

# Main app logic
def main():
    # Replace text input with dropdown
    options = get_company_options()
    selected_option = st.selectbox(
        "Select Company",
        options=options,
        format_func=lambda x: x[0]  # Show display name in dropdown
    )
    
    if selected_option:
        ticker = selected_option[1]  # Get ticker from selected option
        try:
            data_loader = StockDataLoader(model_config)
            predictor = StockPredictor(model_config)
            model_storage = ModelStorage(app_config)
            
            # Try to load existing model
            loaded_model = model_storage.load_latest_model(predictor.model, ticker)
            if loaded_model:
                predictor.model = loaded_model
                st.success(f"Loaded existing model for {ticker}")
            
            with st.spinner('Fetching data...'):
                data, company_info = data_loader.fetch_stock_data(ticker)
            
            if data is None or data.empty:
                st.error(f"No data available for {ticker}. Please check the company symbol.")
                return
                
            if len(data) < model_config.SEQUENCE_LENGTH:
                st.error(f"Not enough historical data for {ticker}. Need at least {model_config.SEQUENCE_LENGTH} days.")
                return
                
            display_metrics(data, company_info)
            
            with st.spinner('Preparing data...'):
                train_loader, scaler = data_loader.prepare_data(data)
            
            with st.spinner('Training model...'):
                try:
                    losses = predictor.train(train_loader)
                    if not losses:
                        st.error("Model training failed to produce results.")
                        return
                except RuntimeError as e:
                    st.error(f"Error during model training: {str(e)}")
                    logger.error(f"Training error: {str(e)}")
                    return
                except Exception as e:
                    st.error("An unexpected error occurred during training. Please try again.")
                    logger.error(f"Unexpected training error: {str(e)}")
                    return
            
            # Continue with predictions only if training was successful
            scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))
            predictions = predictor.predict(scaled_data, scaler)
            
            # Display plots
            display_plots(data, predictions)
            
            # Save model after successful training
            model_storage.save_model(predictor.model, ticker)
            
            # Add download buttons
            add_download_buttons(data, predictions)
            
            # Calculate and display metrics
            display_prediction_metrics(data['Close'].values[-len(predictions):], predictions)
                
        except Exception as e:
            logger.error(f"Error in main app execution: {str(e)}")
            st.error("An error occurred while processing your request. Please try again.")
            

if __name__ == "__main__":
    main()