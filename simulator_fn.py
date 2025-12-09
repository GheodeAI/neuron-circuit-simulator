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


def run_simulator(adj_matrix, conn_matrix=None, simulation_time=100, neuron_amounts=(0, 0, 0, 0, 0, 12, 6, 2), record_volt=False):
    """
    Izhikevich model
    """

    assert len(neuron_amounts) == 8
    assert adj_matrix.ndim == 2
    assert adj_matrix.shape[0] == adj_matrix.shape[1] == 8

    ISe, ISi, ISB, IFB, A, RS, RSB, RFB = neuron_amounts

    f = np.random.choice(aceptables_RS, size=RS, replace=True).tolist() if aceptables_RS else []
    g = np.random.choice(aceptables_RSB, size=RSB, replace=True).tolist() if aceptables_RSB else []
    h = np.random.choice(aceptables_RFB, size=RFB, replace=True).tolist() if aceptables_RFB else []

    ## Simulation parameters
    a = np.hstack(
        (
            [0.02] * (ISe + ISi),
            [0.14] * ISB,
            [0.1] * IFB,
            [0.02] * A,
            datos_RS.loc[f, "a"],
            datos_RSB.loc[g, "a"],
            datos_RFB.loc[h, "a"],
        ),
        dtype=np.float64,
    )

    # np.random.uniform(a, b, n) genera n valores aleatorios en el rango [a, b] con distribución uniforme.
    # si ISB=5 se podría generar la lista [0.2631, 0.2638, 0.2635, 0.2632, 0.2639]
    b = np.hstack(
        (
            [0.2] * (ISe + ISi),
            np.random.uniform(0.263, 0.264, ISB),
            np.random.uniform(0.249, 0.251, IFB),
            [0.2] * A,
            datos_RS.loc[f, "b"],
            datos_RSB.loc[g, "b"],
            datos_RFB.loc[h, "b"],
        ),
        dtype=np.float64,
    )

    c = np.hstack(
        (
            [-65] * (ISe + ISi + ISB + IFB + A + RS),
            datos_RSB.loc[g, "c"],
            datos_RFB.loc[h, "c"],
        ),
        dtype=np.float64,
    )

    d = np.hstack(([8] * (ISe + ISi), np.random.uniform(-8, -8, ISB).tolist(), np.random.uniform(-8, -7.95, IFB).tolist(), [8] * A), dtype=np.float64)

    periodo = np.hstack([d, datos_RS.loc[f, "period"], datos_RSB.loc[g, "period"], datos_RFB.loc[h, "period"]])

    num_irre = [ISe, ISi, ISB, IFB, A]
    num_reg = [RS, RSB, RFB]

    cantidad_neu = num_irre + num_reg  # Cuántas neuronas hay de cada tipo
    size = sum(cantidad_neu)  # Número total de neuronas

    # Creación de la matriz (circuito) de conexiones, con el tamaño (size) igual al número de neuronas
    circuito = np.zeros((size, size))  # Matriz de conexiones 26x26 llena de ceros, lo que representa que
    # inicialmente ninguna neurona está conectada con otra

    tipos = np.repeat(nombres_neu, cantidad_neu)

    # Si t (un elemento de tipos) está en ["IFB", "ISB", "RSB", "RFB"]
    # enumerate(tipos) genera pares (i, t), donde i es el índice y t es el tipo de neurona en tipos
    # la salida es burst = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25], porque imprime los índices,
    # y al no haber ni ISe ni ISi, empieza imprimiendo el índice 10
    burst = [i for i, t in enumerate(tipos) if t in ["IFB", "ISB", "RSB", "RFB"]]

    datos_conexiones = pd.DataFrame({"nombre": nombres_neu, "numero_neurons_tipo": np.cumsum(cantidad_neu)})

    circuito_df = pd.DataFrame(circuito, columns=tipos, index=tipos)

    # Identificación de posiciones
    # Sacan en una lista los índices de las columnas correspondientes a neuronas irregulares --> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    pos_irreg = np.asarray([i for i, col in enumerate(circuito_df.columns) if col in names_irreg], dtype=int)

    # Sacan en una lista los índices de las columnas correspondientes a neuronas regulares --> [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    pos_reg = np.asarray([i for i, col in enumerate(circuito_df.columns) if col in names_reg], dtype=int)

    for i_idx, (row, row_name) in enumerate(zip(adj_matrix, names_all)):
        for j_idx, (n_con, col_name) in enumerate(zip(row, names_all)):
            idx_to_change_row = np.where(tipos == row_name)[0]
            idx_to_change_col = np.where(tipos == col_name)[0]

            volt_strength = adj_matrix[i_idx, j_idx]
            neurons_to_connect = int(conn_matrix[i_idx, j_idx])

            if len(idx_to_change_row) == 0 or len(idx_to_change_col) == 0 or volt_strength == 0 or neurons_to_connect == 0:
                continue

            # Generate a fixed number of connections with the specified strength
            total_neurons = len(idx_to_change_row)*len(idx_to_change_col)

            conn_values = np.zeros(total_neurons)
            idx_to_add = np.random.permutation(total_neurons)[:min(neurons_to_connect, total_neurons)]

            conn_values[idx_to_add] = volt_strength
            conn_values = conn_values.reshape((len(idx_to_change_row), len(idx_to_change_col)))

            circuito_df.iloc[idx_to_change_row, idx_to_change_col] = conn_values

    # # Fill diagonal
    # circuito_df = circuito_df * (1 - np.eye(len(circuito_df)))

    ######### Parámetros controlables del circuito############
    max_delay = 5  # Establece un valor máximo para el retraso que se puede generar
    min_delay = 1  # Establece un valor mínimo para el retraso
    tiempo = simulation_time
    tiempo_ms = tiempo * 1000

    ############  parámetros y variables internos###############
    delays = np.random.choice(range(min_delay, max_delay + 1), size, replace=True)
    contador = np.zeros((size, 2))

    volt = c.copy()
    reg = np.full(size, -13.0)
    inputs = np.zeros(size)

    lim = sum(num_irre)

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    punto_medio = reg - (b / a)

    # tclave = np.arccos((-16 - punto_medio) * a / b) / a
    tclave = np.arccos((-16 - punto_medio) * a / b) / a
    t = np.zeros(size)

    grupo_tag = pd.factorize(circuito_df.columns)[0] + 1
    grupos_nume = max(grupo_tag)
    nombres_grupos = list(pd.factorize(circuito_df.columns)[1])
    list_aferentes = np.where(circuito_df.columns == "A")[0]

    # almacenar la actividad neuronal
    activations = [[] for _ in range(size)]

    volt = np.array(volt, dtype=np.float64)
    if record_volt:
        volt_full = np.empty((size, tiempo_ms))

    idx = 0

    avg_time_diffeq = 0
    avg_time_disp = 0
    avg_time_correct = 0
    circuito = circuito_df.to_numpy()

    for i in range(tiempo_ms):
        time_t = i + 1
        time_t_ms = time_t / 1000
        t = t + 1

        # inputs += np.random.normal(0, 1, inputs.shape)

        volt += 0.5 * ((0.04 * volt + 5) * volt + 140 - reg + inputs)
        # reg[pos_irreg] += 0.5 * a[pos_irreg] * (b[pos_irreg] * volt[pos_irreg] - reg[pos_irreg])
        # reg[pos_reg] -= 0.5 * np.sin(t[pos_reg] * a[pos_reg]) * b[pos_reg]

        volt += 0.5 * ((0.04 * volt + 5) * volt + 140 - reg + inputs)
        # reg[pos_irreg] += 0.5 * a[pos_irreg] * (b[pos_irreg] * volt[pos_irreg] - reg[pos_irreg])
        # reg[pos_reg] -= 0.5 * np.sin(t[pos_reg] * a[pos_reg]) * b[pos_reg]
        reg[pos_irreg] += a[pos_irreg] * (b[pos_irreg] * volt[pos_irreg] - reg[pos_irreg])
        reg[pos_reg] -= np.sin(t[pos_reg] * a[pos_reg]) * b[pos_reg]

        inputs.fill(0)

        if record_volt:
            volt_full[:, i] = np.fmin(volt, 30)

        # Si volt supera los 30 se considera un disparo
        disp = np.where(volt > 30)[0]
        if disp.size > 0:
            # DR = np.intersect1d(pos_reg, disp)
            DI = np.intersect1d(pos_irreg, disp)
            DA = np.intersect1d(disp, list_aferentes)

            contador[disp, 0] += delays[disp]
            contador[disp, 1] += 1

            volt[disp] = c[disp]
            reg[DI] += d[DI]
            reg[DA] = -13

            for k in disp:
                activations[k].append(time_t_ms)

        contador2 = np.ceil(contador[:, 0] / delays)
        contador[:, 0] -= contador[:, 1]
        contador[:, 1] = np.ceil(contador[:, 0] / delays)
        y = np.where(contador[:, 1] < contador2)[0]

        if y.size > 0:
            inputs = circuito[y, :].sum(axis=0) if y.size > 1 else circuito[y, :].flatten()
            receptor = np.intersect1d(np.where(inputs != 0)[0], pos_reg)

            # Cálculo seguro aunque receptor esté vacío
            t_sel = t[receptor]
            tclave_sel = tclave[receptor]
            periodo_sel = periodo[receptor]
            inputs_sel = inputs[receptor]
            delta = (tclave_sel - (t_sel % (periodo_sel / 2))) * (inputs_sel / 20)
            t[receptor] = np.round(t_sel + delta)
            reg[receptor] = punto_medio[receptor] + b[receptor] * np.cos(a[receptor] * t[receptor]) / a[receptor]

    
    if record_volt:
        result = activations, volt_full
    else:
        result = activations

    return result

def generate_random_neuron_matrix(activation_p = None):
    adjmat = np.empty((3, 3))
    for idx_i, name in enumerate(["RS", "RSB", "RFB"]):
        for idx_j, _ in enumerate(["RS", "RSB", "RFB"]):
            if activation_p is None:
                adjmat[idx_i, idx_j] = np.random.choice(np.concatenate([[0], connection_strengths[name]]))
            elif np.random.uniform(0,1) < activation_p:
                adjmat[idx_i, idx_j] = np.random.choice(connection_strengths[name])
            else:
                adjmat[idx_i, idx_j] = 0
    conn_matrix = np.random.randint(1, 6, size=(3, 3))

    return np.array([adjmat, conn_matrix]).flatten()

def calculate_frequency_left(activations, bin_size=0.2):
    """
    Calculates the event frequency at each point by calculating the frequency of a 
    bin of the specified size centered at each activation.
    """
    
    collapsed_sim = np.asarray(sorted(list(set(sum(activations, start=[])))))
    sim_len = len(collapsed_sim)
    freq = np.empty(len(collapsed_sim))

    for idx, act_time in enumerate(collapsed_sim):
        ## Count number of activations in bin 
        # Get start and end of the bin (half way each size)
        bin_start_idx = np.searchsorted(collapsed_sim, act_time - bin_size, side='left')

        bin_start = collapsed_sim[bin_start_idx]
        
        # Number of activations in the bin = size of the bin
        n_activations_in_bin = idx - bin_start_idx + 1

        # Use frequency formula
        if act_time - bin_start > bin_size/2:
            freq[idx] = (n_activations_in_bin-1) / (act_time - bin_start)
        else:
            freq[idx] = n_activations_in_bin / bin_size
        
    return freq

def calculate_frequency_centered(activations, bin_size=0.2):
    """
    Calculates the event frequency at each point by calculating the frequency of a 
    bin of the specified size centered at each activation.
    """
    
    collapsed_sim = np.asarray(sorted(list(set(sum(sim, start=[])))))
    sim_len = len(collapsed_sim)
    freq = np.empty(len(collapsed_sim))

    for idx, act_time in enumerate(collapsed_sim):
        ## Count number of activations in bin 
        # Get start and end of the bin (half way each size)
        bin_start_idx = np.searchsorted(collapsed_sim, act_time - bin_size/2, side='left')
        bin_end_idx = np.searchsorted(collapsed_sim, act_time + bin_size/2, side='right') - 1

        bin_start = collapsed_sim[bin_start_idx]
        bin_end = collapsed_sim[bin_end_idx]
        
        # Number of activations in the bin = size of the bin
        n_activations_in_bin = bin_end_idx - bin_start_idx + 1

        # Use frequency formula
        if bin_end - bin_start > bin_size/2:
            freq[idx] = (n_activations_in_bin-1) / (bin_end - bin_start)
        else:
            freq[idx] = n_activations_in_bin / bin_size
        
    return freq_list

def get_metrics(sim, simulation_time, burst_thesh=0.2):
    hmean_freq = 0
    mean_n_activations = 0

    mean_n_naive_activations = 0
    n_non_empty_activtions = 0
    n_low_activtions = 0
    cv_acc = 0
    cv_count = 0
    for activations in sim:
        n_naive_activations = len(activations)
        mean_n_naive_activations += n_naive_activations

        timing_diffs = np.diff(activations)
        n_activations = np.count_nonzero(timing_diffs > burst_thesh)
        mean_n_activations += n_activations

        n_non_empty_activtions += int(n_activations > 0)
        n_low_activtions += int(n_activations < 3)

        if n_naive_activations > 0:
            cv_acc += np.nanstd(activations)/np.nanmean(activations)
            cv_count += 1

    cv = cv_acc/cv_count
        
    if n_non_empty_activtions != 0:
        mean_n_naive_activations /= n_non_empty_activtions
        if mean_n_naive_activations != 0: 
            hmean_naive_freq = simulation_time / mean_n_naive_activations

        mean_n_activations /= n_non_empty_activtions
        if mean_n_activations != 0: 
            hmean_freq = simulation_time / mean_n_activations


    return {
        "Freq": float(hmean_freq),
        "Freq_naive": float(hmean_naive_freq),
        "N_dead_neurons": int(n_low_activtions),
        "CV": float(cv),
    }


if __name__ == "__main__":
    adj_matrix = generate_random_adjmat(0.5)
    adj_matrix[:-3, :] = 0
    adj_matrix[:, :-3] = 0

    # adj_matrix = np.zeros((8,8))
    # adj_matrix[5, 6] = -13
    # adj_matrix[7, 6] = -8
    # adj_matrix[6, 5] = 8
    # adj_matrix[6, 2] = 8
    # adj_matrix[2, 0] = -13
    # adj_matrix[2, 1] = -13
    # adj_matrix[4, 0] = 18
    # adj_matrix[4, 1] = 18
    # adj_matrix[4, 3] = 8
    # adj_matrix[3, 4] = 8

    simulation_time=100
    neuron_amounts=(0, 0, 0, 0, 0, 12, 6, 2)

    conn_matrix = np.random.randint(1, 6, size=adj_matrix.shape)
    # conn_matrix = np.random.randint(1, 2, size=adj_matrix.shape)
    # conn_matrix = np.eye(adj_matrix.shape[0])
    # conn_matrix = np.zeros(adj_matrix.shape)
    # conn_matrix[-3:, -3:] = np.array([[10,10,10],[0,0,0],[0,0,0]])

    print(adj_matrix)
    print(conn_matrix)

    t0 = time.time()
    sim, volts = run_simulator(adj_matrix, conn_matrix, simulation_time=simulation_time, neuron_amounts=neuron_amounts, record_volt=True)
    t1 = time.time()
    print(t1 - t0)

    t0 = time.time()
    metrics = get_metrics(sim, simulation_time=simulation_time)
    t1 = time.time()
    print(t1 - t0)

    print(metrics)


    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax_flat = ax.flatten()
    idx_list = np.random.permutation(volts.shape[0])[:6]

    ax[0, 0].plot(volts[0, -1000:])
    ax[0, 1].plot(volts[1, -1000:])
    ax[0, 2].plot(volts[14, -1000:])
    ax[1, 0].plot(volts[15, -1000:])
    ax[1, 1].plot(volts[18, -1000:])
    ax[1, 2].plot(volts[19, -1000:])

    ax[0, 0].set(ylim=(None, 32))
    ax[0, 1].set(ylim=(None, 32))
    ax[0, 2].set(ylim=(None, 32))
    ax[1, 0].set(ylim=(None, 32))
    ax[1, 1].set(ylim=(None, 32))
    ax[1, 2].set(ylim=(None, 32))

    plt.show()
