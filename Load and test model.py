import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model("trained_model.h7")

# Load the new data from CSV
new_data = pd.read_csv("/Users/samuelzobev/Downloads/End of year project/AAPL.csv")

# Assuming the new data has the same columns as the original training data
columns = list(new_data)[1:6]

# Convert the new data to float data type
new_data_for_prediction = new_data[columns].astype(float)

# Load the scaler used during training and transform the new data
scaler = StandardScaler()
scaler = scaler.fit(new_data_for_prediction)
scaled_new_data_for_prediction = scaler.transform(new_data_for_prediction)

# Assuming 'past' and 'future' values are the same as used during training
past = 30
future = 1

# Prepare the new data for prediction in the correct shape
X_new = []
for i in range(past, len(scaled_new_data_for_prediction) - future + 1):
    X_new.append(scaled_new_data_for_prediction[i - past:i, 0:scaled_new_data_for_prediction.shape[1]])
X_new = np.array(X_new)

# Make predictions using the trained model
predictions = model.predict(X_new)
# Reshape the predictions array to remove any dimensions with a length of 1
predictions = np.squeeze(predictions)

# Reshape the scale_ attribute to have the same shape as the predictions array
scale_reshaped = scaler.scale_.reshape(1, -1)

# Transpose the scale_reshaped array to match the shape of the predictions array
scale_transposed = scale_reshaped.T

# Transpose the predictions array to match the shape of the scale_transposed array
predictions_transposed = predictions.T

# Repeat the scale_transposed array to match the shape of the predictions_transposed array
scale_transposed = np.repeat(scale_transposed, predictions_transposed.shape[1], axis=1)
 
# Inverse transform the predictions using the scaler
actual_predictions = scaler.inverse_transform(predictions_transposed * scale_transposed)

# Assuming the 'Date' column is present in the new_data DataFrame
dates = new_data['Date'].iloc[past + future - 1:]

# Create a new DataFrame to store the predicted stock prices along with dates
predicted_data = pd.DataFrame({'Date': dates, 'Predicted_Stock_Price': actual_predictions[:, 0]})

# Display the predicted data
print(predicted_data)

predicted_data.to_csv("predicted_data.csv", index=False)


