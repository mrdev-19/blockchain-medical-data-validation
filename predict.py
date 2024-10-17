import pandas as pd
import numpy as np
import tensorflow as tf

def pred(dev):
    # Load xtest dataset
    xtest_df = pd.read_csv('xtest_dataset.csv')
    xtest_array = xtest_df.to_numpy()

    # Load pre-trained model
    loaded_model = tf.keras.models.load_model('dev.h5')
    print(f'Model loaded successfully.')

    # Perform predictions
    predictions = loaded_model.predict(xtest_array)

    # Convert probabilities to class labels using a threshold (e.g., 0.5)
    predicted_labels = (predictions > 0.5).astype(int)

    # Append predictions column to the original dataframe
    xtest_df['Predicted_Label'] = predicted_labels

    # Save the updated dataframe to a CSV file
    xtest_df.to_csv('predictions_and_original.csv', index=False)
    print('Predictions appended and saved to predictions_and_original.csv.')
    return xtest_df