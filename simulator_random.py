import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

inhibition = np.array([-8, -13, -18])
excitation = np.array([8, 13])
connection_strengths = {
    "ISe": np.hstack([inhibition, excitation]),
    "ISi": np.hstack([inhibition, excitation]),
    "IFB": inhibition,
    "ISB": inhibition,
    "A": np.array([18]),
    "RS": inhibition,
    "RSB": excitation,
    "RFB": excitation,
}

## Generate Regular neurons with presets
datos_RS = pd.read_csv("RS_values.csv")
datos_RSB = pd.read_csv("RSB_values.csv")
datos_RFB = pd.read_csv("RFB_values.csv")

# Convertir columnas a numérico (forzando errores a NaN (Not A Number) si hay valores no convertibles)
datos_RS["freq(ms)"] = pd.to_numeric(datos_RS["freq(ms)"], errors="coerce")

datos_RSB["freq_inter"] = pd.to_numeric(datos_RSB["freq_inter"], errors="coerce")
datos_RSB["freq_intra"] = pd.to_numeric(datos_RSB["freq_intra"], errors="coerce")

datos_RFB["freq_inter"] = pd.to_numeric(datos_RFB["freq_inter"], errors="coerce")
datos_RFB["freq_intra"] = pd.to_numeric(datos_RFB["freq_intra"], errors="coerce")

# Aplicar filtros después de la conversión
aceptables_RS = datos_RS[(datos_RS["freq(ms)"] > 5) & (datos_RS["freq(ms)"] < 6)].index.to_list()
aceptables_RSB = datos_RSB[(datos_RSB["freq_inter"] > 0.4) & (datos_RSB["freq_inter"] < 0.5) & (datos_RSB["freq_intra"] < 20)].index.to_list()
aceptables_RFB = datos_RFB[(datos_RFB["freq_inter"] > 2) & (datos_RFB["freq_inter"] < 9) & (datos_RFB["freq_intra"] > 100)].index.to_list()

names_irreg = ["ISe", "ISi", "ISB", "IFB", "A"]
names_reg = ["RS", "RSB", "RFB"]
nombres_neu = names_irreg + names_reg  # Contiene los tipos de neuronas
names_all = names_irreg + names_reg

def random_activation(nsamples, p=0.05):
    return np.random.uniform(0, 1, nsamples) < p

def periodic_activation(nsamples, p1=0.01, p2=0.05, period=10):
    # return np.random.uniform(0, 1, nsamples) < p
    random_start = np.random.randint(period)
    signal = np.zeros(nsamples, dtype=bool)
    signal[random_start::period] = True
    signal[np.random.uniform(0, 1, nsamples) < p1] = 0
    signal[np.random.uniform(0, 1, nsamples) < p2] = 1
    return signal

def periodic_burst_activation(nsamples, p1 = 0.75, p2 = 0.1):
    signal = np.empty(nsamples,dtype=bool)
    for i in range(nsamples):
        if i >= 1 and signal[i-1]:
            signal[i] = np.random.uniform(0,1) < p1
        else:
            signal[i] = np.random.uniform(0,1) < p2
    return signal

def random_burst_activation(nsamples, p1 = 0.75, p2 = 0.1):
    signal = np.empty(nsamples,dtype=bool)
    for i in range(nsamples):
        if i >= 1 and signal[i-1]:
            signal[i] = np.random.uniform(0,1) < p1
        else:
            signal[i] = np.random.uniform(0,1) < p2
    return signal


def run_simulator_random(adj_matrix=None, conn_matrix=None, simulation_time=100, neuron_amounts=(0, 0, 0, 0, 0, 12, 6, 2), record_volt=False):
    """
    Izhikevich model
    """

    assert not record_volt, "The random simulator does not support volt outputs"

    resolution = 1e-3
    nneurons = sum(neuron_amounts)
    nsamples = int(simulation_time / resolution)
    p = 0.05/nneurons
    period = 250

    cursor = 0
    activations = np.empty((nsamples, sum(neuron_amounts)), dtype=bool)
    for (neuron_type, neuron_amount) in zip(names_all, neuron_amounts):
        if neuron_amount <= 0:
            continue

        match neuron_type:
            case "ISe":
                activations[:, cursor:cursor+neuron_amount] = np.asarray([random_activation(nsamples, p) for _ in range(neuron_amount)]).T
            case "ISi":
                activations[:, cursor:cursor+neuron_amount] = np.asarray([random_activation(nsamples, p) for _ in range(neuron_amount)]).T
            case "IFB":
                activations[:, cursor:cursor+neuron_amount] = np.asarray([random_burst_activation(nsamples, 0.5, p*5) for _ in range(neuron_amount)]).T
            case "ISB":
                activations[:, cursor:cursor+neuron_amount] = np.asarray([random_burst_activation(nsamples, 0.5, p*5) for _ in range(neuron_amount)]).T
            case "A":
                activations[:, cursor:cursor+neuron_amount] = np.asarray([random_activation(nsamples, p) for _ in range(neuron_amount)]).T
            case "RS":
                activations[:, cursor:cursor+neuron_amount] = np.asarray([periodic_activation(nsamples, p/50, p/5, period=period) for _ in range(neuron_amount)]).T
            case "RSB":
                activations[:, cursor:cursor+neuron_amount] = np.asarray([periodic_burst_activation(nsamples, 0.99, p/5) for _ in range(neuron_amount)]).T
            case "RFB":
                activations[:, cursor:cursor+neuron_amount] = np.asarray([periodic_burst_activation(nsamples, 0.99, p/5) for _ in range(neuron_amount)]).T
        
        cursor += neuron_amount

    sim_vector = np.argwhere(activations.T)
    sim = [[] for i in range(nneurons)]
    for (neuron_idx, sample_idx) in sim_vector:
        sim[neuron_idx].append(float(sample_idx)*resolution)
    
    return sim
