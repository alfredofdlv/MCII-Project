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
import torch
import torch.nn as nn

class AudioLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=10, n_layers=2, dropout_rate=0.3):
        super(AudioLSTM, self).__init__()
        
        # Definir la LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_rate)
        
        # Capas densas
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)  # Reducimos las dimensiones a la mitad
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)  # Capa de salida
        
        # Definir Dropout y ReLU
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        # Pasamos la entrada a través de la LSTM
        lstm_out, hidden = self.lstm1(x, hidden)
        
        # Aplicamos el Dropout
        out = self.dropout(lstm_out)
        
        # Usamos solo la última salida de la secuencia
        out = out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Pasamos por las capas densas
        out = self.fc1(out)
        out = self.relu(out)  # ReLU en la primera capa densa
        out = self.fc2(out)   # Capa de salida

        return out

    def init_hidden(self, batch_size):
        # Inicializar el estado oculto
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        return hidden


class ComplexGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout_rate):
        super(ComplexGRUModel, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.gru3 = nn.GRU(hidden_dim2, hidden_dim3, batch_first=True)
        
        # BatchNorm aplicada a la salida completa de la última capa GRU
        self.batchnorm = nn.BatchNorm1d(hidden_dim3)
        
        self.fc1 = nn.Linear(hidden_dim3, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)
        
        # BatchNorm en la secuencia completa y selecciona el último paso
        x = self.batchnorm(x.transpose(1, 2)).transpose(1, 2)  # (batch, features, sequence)
        x = x[:, -1, :]  # Usa el último paso después de la BatchNorm
        
        # Cambiamos a tanh para la activación
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate):
        super(ImprovedGRUModel, self).__init__()
        
        # Primera capa GRU
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)  # Regularización después de la primera GRU
        
        # Segunda capa GRU
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)  # Regularización después de la segunda GRU
        
        # Global pooling para capturar toda la secuencia
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)  # Alternativa: AdaptiveAvgPool1d
        #self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.activation = nn.Tanh()
        # Capa densa final
        self.fc1 = nn.Linear(hidden_dim2, output_dim)
        

    def forward(self, x):
        # Primera capa GRU
        x, _ = self.gru1(x)  # (batch, seq_len, hidden_dim1)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm en la dimensión de características
        x = self.dropout1(x)
        
        # Segunda capa GRU
        x, _ = self.gru2(x)  # (batch, seq_len, hidden_dim2)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout2(x)
        
        # Global pooling para reducir la dimensión temporalSSSS
        # Pooling
        x = x.transpose(1, 2)  # (batch, hidden_dim2, seq_len)
        
        x_max = self.global_max_pooling(x).squeeze(2)
        #x_avg = self.global_avg_pooling(x).squeeze(2)
        x = x_max 
        
        # Capa completamente conectada
        x = self.activation(x)
        x = self.fc1(x)  # Salida sin activación (para usar con CrossEntropyLoss)
        return x
    

class AttGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate, num_heads=4):
        super(AttGRUModel, self).__init__()

        # Primera capa GRU
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)  # Regularización después de la primera GRU

        # Segunda capa GRU
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)  # Regularización después de la segunda GRU

        # Capa de atención multi-cabeza
        self.attention = nn.MultiheadAttention(hidden_dim2, num_heads=num_heads, batch_first=True)

        # Global pooling para capturar toda la secuencia
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)  # Puedes alternar entre MaxPool o AvgPool

        # Capa de activación
        self.activation = nn.Tanh()

        # Capa densa final
        self.fc1 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Primera capa GRU
        x, _ = self.gru1(x)  # (batch_size, seq_len, hidden_dim1)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm en la dimensión de características
        x = self.dropout1(x)

        # Segunda capa GRU
        x, _ = self.gru2(x)  # (batch_size, seq_len, hidden_dim2)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)  # BatchNorm después de la segunda GRU
        x = self.dropout2(x)

        # Capa de atención multi-cabeza
        # La entrada de la atención debe ser (seq_len, batch_size, hidden_dim2)
        x_attn, _ = self.attention(x, x, x)  # Aplicamos la atención sobre sí misma

        # Global pooling para reducir la dimensión temporal
        x = x_attn.transpose(1, 2)  # (batch_size, hidden_dim2, seq_len)
        x_max = self.global_max_pooling(x).squeeze(2)  # Aplicamos Max Pooling y eliminamos la dimensión extra

        # Capa completamente conectada
        x = self.activation(x_max)
        x = self.fc1(x)  # Salida sin activación (para usar con CrossEntropyLoss)

        return x