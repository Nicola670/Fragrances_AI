import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simulazione di dati: rappresentazione numerica degli oli essenziali
# Ogni olio e rappresentato come un vettore unico
lista = ["lavanda", "menta", "arancia", "rosmarino", "eucalipto", "limone", 
    "pompelmo", "sandalo", "vaniglia", "patchouli", "ylang-ylang", 
    "cedro", "bergamotto", "chiodi_di_garofano", "gelsomino", "rosa", 
    "neroli", "camomilla", "timo", "geranio","caramello","cioccolato","nocciola","mandorla","marzapane","miele",
    "musk","castoreum","civetta","vaniglia","cuoio","amber","fava_tonka","cacao", "benzoino", "incenso", "mirra",
"labdano"]

oli_essenziali = {fragranza: [1 if i == j else 0 for j in range(len(lista))] for i, fragranza in enumerate(lista)}
#print(oli_essenziali)
# Dataset di abbinamenti (buoni: 1, cattivi: 0)
dataset = [
  ([oli_essenziali["lavanda"], oli_essenziali["menta"]], 1),    # Lavanda e Menta: buona
    ([oli_essenziali["vaniglia"], oli_essenziali["patchouli"]], 1), # Vaniglia e Patchouli: buona
    ([oli_essenziali["arancia"], oli_essenziali["bergamotto"]], 1), # Arancia e Bergamotto: buona
    ([oli_essenziali["rosa"], oli_essenziali["gelsomino"]], 1),    # Rosa e Gelsomino: buona
    ([oli_essenziali["cedro"], oli_essenziali["sandalo"]], 1),     # Cedro e Sandalo: buona
    ([oli_essenziali["miele"], oli_essenziali["vaniglia"]], 1),    # Miele e Vaniglia: buona
    ([oli_essenziali["ylang-ylang"], oli_essenziali["neroli"]], 1), # Ylang-ylang e Neroli: buona
    ([oli_essenziali["cacao"], oli_essenziali["nocciola"]], 1),    # Cacao e Nocciola: buona
    ([oli_essenziali["incenso"], oli_essenziali["mirra"]], 1),     # Incenso e Mirra: buona
    ([oli_essenziali["musk"], oli_essenziali["amber"]], 1),         # Musk e Amber: buona
    ([oli_essenziali["lavanda"], oli_essenziali["caramello"]], 0),    # Lavanda e Caramello: cattiva
    ([oli_essenziali["menta"], oli_essenziali["vaniglia"]], 0),       # Menta e Vaniglia: cattiva
    ([oli_essenziali["eucalipto"], oli_essenziali["musk"]], 0),       # Eucalipto e Musk: cattiva
    ([oli_essenziali["rosmarino"], oli_essenziali["cioccolato"]], 0), # Rosmarino e Cioccolato: cattiva
    ([oli_essenziali["timo"], oli_essenziali["cacao"]], 0),           # Timo e Cacao: cattiva
    ([oli_essenziali["patchouli"], oli_essenziali["miele"]], 0),      # Patchouli e Miele: cattiva
    ([oli_essenziali["benzoino"], oli_essenziali["camomilla"]], 0),   # Benzoino e Camomilla: cattiva
    ([oli_essenziali["mandorla"], oli_essenziali["incenso"]], 0),     # Mandorla e Incenso: cattiva
    ([oli_essenziali["geranio"], oli_essenziali["castoreum"]], 0),    # Geranio e Castoreum: cattiva
    ([oli_essenziali["fava_tonka"], oli_essenziali["chiodi_di_garofano"]], 0), # Fava Tonka e Chiodi di Garofano: cattiva
    ([oli_essenziali["pompelmo"], oli_essenziali["limone"]], 1),        # Pompelmo e Limone: buona
    ([oli_essenziali["pompelmo"], oli_essenziali["bergamotto"]], 1),    # Pompelmo e Bergamotto: buona
    ([oli_essenziali["pompelmo"], oli_essenziali["menta"]], 1),         # Pompelmo e Menta: buona
    ([oli_essenziali["pompelmo"], oli_essenziali["lavanda"]], 1),       # Pompelmo e Lavanda: buona
    ([oli_essenziali["pompelmo"], oli_essenziali["rosmarino"]], 1),     # Pompelmo e Rosmarino: buona
    ([oli_essenziali["pompelmo"], oli_essenziali["arancia"]], 1),       # Pompelmo e Arancia: buona
    ([oli_essenziali["pompelmo"], oli_essenziali["geranio"]], 1),       # Pompelmo e Geranio: buona
    ([oli_essenziali["pompelmo"], oli_essenziali["ylang-ylang"]], 1),   # Pompelmo e Ylang-ylang: buona
    ([oli_essenziali["pompelmo"], oli_essenziali["patchouli"]], 0),     # Pompelmo e Patchouli: cattiva
    ([oli_essenziali["pompelmo"], oli_essenziali["miele"]], 0),         # Pompelmo e Miele: cattiva
    ([oli_essenziali["pompelmo"], oli_essenziali["cedro"]], 0),         # Pompelmo e Cedro: cattiva
    ([oli_essenziali["pompelmo"], oli_essenziali["cacao"]], 0),         # Pompelmo e Cacao: cattiva
    ([oli_essenziali["pompelmo"], oli_essenziali["incenso"]], 0),       # Pompelmo e Incenso: cattiva
    ([oli_essenziali["pompelmo"], oli_essenziali["fava_tonka"]], 0),    # Pompelmo e Fava Tonka: cattiva
    ([oli_essenziali["pompelmo"], oli_essenziali["castoreum"]], 0),     # Pompelmo e Castoreum: cattiva
    ([oli_essenziali["pompelmo"], oli_essenziali["vaniglia"]], 0),      # Pompelmo e Vaniglia: cattiva
]


# Preparare input e output
X = []
y = []
for (abbinamento, label) in dataset:
    # Concatenare i vettori dei due oli per rappresentare l'abbinamento
    X.append(np.concatenate(abbinamento))
    y.append(label)

#un tensore p una struttura dati (variabile, lista matrice...) che e ottimizzata per 
#l'esecuzione dei calcoli, che possono essere eseguiti sia da cpu che gpu
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Definire la rete neurale
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 8)  # Strato nascosto
        self.fc2 = nn.Linear(8, 1)          # Strato di output
        self.sigmoid = nn.Sigmoid()         # Funzione di attivazione

    def forward(self, x):
        x = torch.relu(self.fc1(x))         # Relu per lo strato nascosto
        x = self.sigmoid(self.fc2(x))      # Sigmoid per output (valore tra 0 e 1)
        return x

# Inizializzare la rete
input_size = X.shape[1]  # Dimensione degli input (es. 10 elementi, 5 oli x 2)
model = SimpleNN(input_size)

# Configurazione dell'ottimizzatore e della funzione di perdita
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Addestramento
epochs = 1000 #termine per indicare ciclo di addestramento
for epoch in range(epochs):
    """I gradienti servono a calcolare l'errore (perdita) e a regolare i parametri del modello
    Un gradiente misura il tasso di variazione di una funzione in relazione alle sue variabili di input. 
    E l'equivalente multidimensionale della derivata in una singola dimensione."""

    optimizer.zero_grad()            # Azzera i gradienti
    outputs = model(X)               # Forward pass
    loss = criterion(outputs, y)    # Calcola la perdita
    loss.backward()                  # Backward pass (calcolo gradienti)
    optimizer.step()                 # Aggiornamento pesi
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Testare il modello
def valuta_abbinamento(olio1, olio2):
    input_test = torch.tensor([np.concatenate([oli_essenziali[olio1], oli_essenziali[olio2]])], dtype=torch.float32)
    predizione = model(input_test).item()
    return "Buono" if predizione >= 0.5 else "Cattivo"

l = []
for i in range(2):
    a = input("Inserisci le fragranze da inserire:  ")
    l.append(a)
result = valuta_abbinamento(l[0], l[1])
print(result)

"""
print("\\nTest modello:")
goods = []
for i in lista:
    
    for k in lista:
        risultato = valuta_abbinamento(i, k)
        print(i,k , ":", risultato)
        
        if risultato == "Buono" and i != k and (i, k) not in goods:
            goods.append((i, k))
            print(i, k, ": ",risultato)

for i in goods:
    print(i[0], i[1])"""