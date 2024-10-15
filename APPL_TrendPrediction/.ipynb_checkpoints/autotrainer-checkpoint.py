import os
import time
import csv
import argparse
import numpy as np
import yfinance as yf
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import save_model
from keras.callbacks import EarlyStopping
import joblib
from tqdm import tqdm


# Verzeichnisse erstellen
os.makedirs('./models', exist_ok=True)
os.makedirs('./results', exist_ok=True)
os.makedirs('./scaler', exist_ok=True)

# Hyperparameter für die zufällige Suche
hyperparameters = {
    'windowsize': [1, 2, 5, 10, 15, 20, 30, 60],
    'batchsize': [1, 5, 14, 30, 64, 128, 256, 512],
    'epochs': [30],
    'units': [1, 2, 5, 6, 8, 12, 16, 32, 64, 128],
    'lstm_layers': [1, 2, 3],
    'future_steps': [2, 3, 4, 5, 10, 30, 60]
}

# Daten von Yahoo Finance laden
symbol = "AAPL"  # Apple-Aktien
start_date = "2000-01-01"
end_date = "2024-10-10"

val_size = 0.2  # Anteil der Validierungsdaten
csv_filename = f'./results/hyperparameter_search_results_{start_date}_{end_date}.csv'

# Prüfen, ob Modell mit den Parametern bereits existiert
def model_exists(window_size, future_steps, batch_size, epochs, units, lstm_layers):
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1:6] == [str(window_size), str(future_steps), str(batch_size), str(epochs), str(units), str(lstm_layers)]:
                    return True
    return False

# Ergebnisse in eine CSV-Datei schreiben
def write_to_csv(model_name, window_size, future_steps, batch_size, epochs, units, lstm_layers, mse, mae, r2):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Model Name', 'Window Size', 'Future Steps', 'Batch Size', 'Epochs', 'Units', 'LSTM Layers', 'MSE', 'MAE', 'R2'])
        writer.writerow([model_name, window_size, future_steps, batch_size, epochs, units, lstm_layers, mse, mae, r2])

# Sliding-Window-Daten erstellen
def create_sliding_window(data, target, window_size, future_steps):
    X, y = [], []
    for i in range(len(data) - window_size - future_steps):
        X.append(data[i:i + window_size])
        y.append(target[i + window_size:i + window_size + future_steps])
    return np.array(X), np.array(y)

# Zufällige Hyperparameter-Suche
def random_search(n_iter=10, scaled_features=None, scaled_target=None, target_scaler=None):
    results = []
    
    for iteration in tqdm(range(n_iter), desc="Hyperparameter-Suche"):
        # Zufällige Auswahl von Hyperparametern
        window_size = np.random.choice(hyperparameters['windowsize'])
        batch_size = np.random.choice(hyperparameters['batchsize'])
        epochs = np.random.choice(hyperparameters['epochs'])
        units = int(np.random.choice(hyperparameters['units']))
        lstm_layers = np.random.choice(hyperparameters['lstm_layers'])
        future_steps = np.random.choice(hyperparameters['future_steps'])

        tqdm.write(f"Iteration {iteration + 1}: window_size={window_size}, future_steps={future_steps}, batch_size={batch_size}, epochs={epochs}, units={units}, lstm_layers={lstm_layers}")

        # Prüfen, ob Modell bereits existiert
        if model_exists(window_size, future_steps, batch_size, epochs, units, lstm_layers):
            tqdm.write(f"Modell mit den Parametern existiert bereits, überspringe...")
            continue

        # Daten vorbereiten
        X, y = create_sliding_window(scaled_features, scaled_target, window_size=window_size, future_steps=future_steps)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)

        # Modell erstellen
        model = Sequential()

        # LSTM-Schichten hinzufügen
        model.add(LSTM(units=units, return_sequences=(lstm_layers > 1), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))

        for layer in range(1, lstm_layers):
            model.add(LSTM(units=units, return_sequences=(layer < lstm_layers - 1)))
            model.add(Dropout(0.2))

        # Dense-Schicht hinzufügen
        model.add(Dense(units=future_steps))

        # Modell kompilieren
        model.compile(optimizer='adam', loss='mean_squared_error')
        earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Modell trainieren
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0, callbacks=[earlystop])

        # Vorhersagen für den gesamten Datensatz machen
        X_windowed = create_sliding_window(scaled_features, scaled_target, window_size=window_size, future_steps=future_steps)[0]
        y_pred = model.predict(X_windowed)

        # Vorhersagen zurückskalieren
        y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, future_steps))

        # Echte und vorhergesagte Werte für die letzten future_steps
        tmax = len(scaled_target)
        start_idx = tmax - future_steps
        y_true_final_span = target_scaler.inverse_transform(scaled_target[start_idx:].reshape(-1, 1)).flatten()
        y_pred_final_span = y_pred_rescaled[-1, :]

        # Metriken berechnen
        mse = mean_squared_error(y_true_final_span, y_pred_final_span)
        mae = mean_absolute_error(y_true_final_span, y_pred_final_span)
        r2 = r2_score(y_true_final_span, y_pred_final_span)

        # Modell speichern
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_name = f"model_{timestamp}_ws{window_size}_fs{future_steps}_bs{batch_size}_epochs{epochs}_units{units}_layers{lstm_layers}.keras"
        save_model(model, f'./models/{model_name}')

        # Ergebnisse speichern
        write_to_csv(model_name, window_size, future_steps, batch_size, epochs, units, lstm_layers, mse, mae, r2)

        results.append((window_size, future_steps, batch_size, epochs, units, lstm_layers, mse, mae, r2, model_name))

    # Beste Hyperparameter ausgeben
    results = sorted(results, key=lambda x: x[5])  # Sortiere nach MSE
    best_params = results[0]
    print("\nBeste Hyperparameter-Kombination:")
    print(f"window_size={best_params[0]}, future_steps={best_params[1]}, batch_size={best_params[2]}, epochs={best_params[3]}, units={best_params[4]}, lstm_layers={best_params[5]}")

    return results

if __name__ == '__main__':
    print("Skript gestartet.")
    # Argumentparser für die Anzahl der Iterationen
    parser = argparse.ArgumentParser(description='LSTM Hyperparameter-Suche')
    parser.add_argument('--iterations', type=int, default=10, help='Anzahl der Iterationen für die Hyperparameter-Suche')
    args = parser.parse_args()


    df = yf.download(symbol, start=start_date, end=end_date)
    features = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    target = df['Close']

    # Daten skalieren
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1))

    # Skalierer speichern
    joblib.dump(feature_scaler, './scaler/feature_scaler.pkl')
    joblib.dump(target_scaler, './scaler/target_scaler.pkl')

    # Hyperparameter-Suche starten
    random_search(n_iter=args.iterations, scaled_features=scaled_features, scaled_target=scaled_target, target_scaler=target_scaler)
