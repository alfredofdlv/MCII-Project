import torch
import torch.nn as nn
import torch.optim as optim

class ComplexLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout_rate=0.5):
        super(ComplexLSTMModel, self).__init__()
        
        # Capas LSTM bidireccionales
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=False)
        
        # Capa LSTM unidireccional final
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim3, batch_first=True)
        
        # Capas densas
        self.fc1 = nn.Linear(hidden_dim3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        # Normalización por lotes (puedes probar eliminarla si es necesario)
        self.batchnorm = nn.BatchNorm1d(hidden_dim3)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        # Usamos la última salida de la secuencia
        x = self.batchnorm(x[:, -1, :])  # Normalización por lotes
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout para evitar overfitting
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ComplexGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout_rate=0.1):
        super(ComplexGRUModel, self).__init__()

        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=False)
        self.gru3 = nn.GRU(hidden_dim2, hidden_dim3, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm = nn.BatchNorm1d(hidden_dim3)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)

        x = self.batchnorm(x[:, -1, :])  # Usamos la última salida de la secuencia
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout para evitar overfitting
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x