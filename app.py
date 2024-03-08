

# Import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_excel('Crude Oil Prices Daily.xlsx')
df = df.dropna()

# Preprocess data
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Scaled_Price'] = scaler.fit_transform(df[['Closing Value']])
    return df, scaler

df, scaler = preprocess_data(df)

# Create sequences for LSTM
sequence_length = 30  # Adjust as needed
sequences = []
prices = df['Scaled_Price'].to_numpy()

for i in range(len(prices) - sequence_length):
    sequences.append(prices[i:i+sequence_length+1])

# Convert to NumPy array
sequences = np.array(sequences)

# Split data into features and target
X, y = sequences[:, :-1], sequences[:, -1]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model()

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Model Loss on Test Data: {loss}")

# Save the model (optional)
model.save("crude_oil_price_lstm_model")

# Streamlit app
def main():
    st.title("Oil Price Prediction")

    # User input for date selection
    selected_month = st.selectbox("Select a month:", range(1, 13))
    selected_year = st.number_input("Select a year:", min_value=df.index.year.min(), max_value=df.index.year.max())

    # User input for prediction
    if st.button("Predict"):
        # Convert user input to datetime
        selected_date = pd.to_datetime(f"{selected_year}-{selected_month}-01")
    
        # Get the previous 30 days of data, if available
        input_data = df.loc[:selected_date]['Scaled_Price'].tail(30)
    
        # Check if there are enough data points
        if len(input_data) >= 30:
            # Reshape input data for LSTM model
            input_data = np.reshape(input_data.values, (1, 30, 1))
        
            # Load the LSTM model
            lstm_model = load_model("crude_oil_price_lstm_model")
        
            # Predict for the selected date
            prediction = lstm_model.predict(input_data)
            prediction = scaler.inverse_transform(prediction)
        
            # Display the prediction
            st.write(f"Prediction for {selected_date}: ${prediction[0, 0]:.2f}")
        else:
            st.warning("Insufficient data for prediction. Please select an earlier date.")

if __name__ == "__main__":
    main()
