import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def run_simulator_simple(adj_matrix, simulation_time = 100, neuron_amounts=(0, 0, 0, 0, 0, 12, 6, 2)):
    assert len(neuron_amounts) == 8
    assert adj_matrix.ndim == 2 
    assert adj_matrix.shape[0] == adj_matrix.shape[1] == 8

    ISe, ISi, ISB, IFB, A, RS, RSB, RFB = neuron_amounts

    ## Generate Regular neurons with presets
    datos_RFB = pd.read_csv("RFB_values.csv")
    datos_RS = pd.read_csv("RS_values.csv")
    datos_RSB = pd.read_csv("RSB_values.csv")

    # Convertir columnas a numérico (forzando errores a NaN (Not A Number) si hay valores no convertibles)
    datos_RFB["freq_inter"] = pd.to_numeric(datos_RFB["freq_inter"], errors='coerce')
    datos_RFB["freq_intra"] = pd.to_numeric(datos_RFB["freq_intra"], errors='coerce')
    datos_RS["freq(ms)"] = pd.to_numeric(datos_RS["freq(ms)"], errors='coerce')
    datos_RSB["freq_inter"] = pd.to_numeric(datos_RSB["freq_inter"], errors='coerce')
    datos_RSB["freq_intra"] = pd.to_numeric(datos_RSB["freq_intra"], errors='coerce')

    # Aplicar filtros después de la conversión
    aceptables_RFB = datos_RFB[(datos_RFB["freq_inter"] > 2) & (datos_RFB["freq_inter"] < 9) & (datos_RFB["freq_intra"] > 100)].index.to_list()
    aceptables_RS = datos_RS[(datos_RS["freq(ms)"] > 5) & (datos_RS["freq(ms)"] < 6)].index.to_list()
    aceptables_RSB = datos_RSB[(datos_RSB["freq_inter"] > 0.4) & (datos_RSB["freq_inter"] < 0.5) & (datos_RSB["freq_intra"] < 20)].index.to_list()

    h = np.random.choice(aceptables_RFB, size=RFB, replace=True).tolist() if aceptables_RFB else [] #En R, si aceptables_RFB está vacío aquí lanzaba un error
    f = np.random.choice(aceptables_RS, size=RS, replace=True).tolist() if aceptables_RS else []
    g = np.random.choice(aceptables_RSB, size=RSB, replace=True).tolist() if aceptables_RSB else []

    ## Simulation parameters
    a = np.hstack((
        [0.02] * (ISe + ISi),
        [0.14] * ISB,
        [0.1] * IFB,
        [0.02] * A,
        datos_RS.loc[f, "a"],
        datos_RSB.loc[g, "a"],
        datos_RFB.loc[h, "a"],
    ), dtype=np.float64)

    #np.random.uniform(a, b, n) genera n valores aleatorios en el rango [a, b] con distribución uniforme.
        #si ISB=5 se podría generar la lista [0.2631, 0.2638, 0.2635, 0.2632, 0.2639]
    b = np.hstack((
        [0.2] * (ISe + ISi),
        np.random.uniform(0.263, 0.264, ISB),
        np.random.uniform(0.249, 0.251, IFB),
        [0.2] * A,
        datos_RS.loc[f, "b"],
        datos_RSB.loc[g, "b"],
        datos_RFB.loc[h, "b"],
    ), dtype=np.float64)

    c = np.hstack((
        [-65] * (ISe + ISi + ISB + IFB + A + RS),
        datos_RSB.loc[g, "c"],
        datos_RFB.loc[h, "c"],
    ), dtype=np.float64)

    d = np.hstack((
        [8] * (ISe + ISi),
        np.random.uniform(-8, -8, ISB).tolist(),
        np.random.uniform(-8, -7.95, IFB).tolist(),
        [8] * A
    ), dtype=np.float64)
    # print(a)
    # print(b)
    # print(c)
    # print(d)

    # periodo = d + datos_RS.loc[f, "period"].tolist() + datos_RSB.loc[g, "period"].tolist() + datos_RFB.loc[h, "period"].tolist()
    periodo = np.hstack([d, datos_RS.loc[f, "period"], datos_RSB.loc[g, "period"], datos_RFB.loc[h, "period"]])
    # print(periodo)
    # periodo = np.array(periodo)

    num_irre = [ISe, ISi, ISB, IFB, A]
    num_reg = [RS, RSB, RFB]
    names_irreg = ["ISe", "ISi", "ISB", "IFB", "A"]
    names_reg = ["RS", "RSB", "RFB"]
    names_all = names_irreg + names_reg

    cantidad_neu = num_irre + num_reg # Cuántas neuronas hay de cada tipo
    size = sum(cantidad_neu)  # Número total de neuronas

    # Creación de la matriz (circuito) de conexiones, con el tamaño (size) igual al número de neuronas
    nombres_neu = names_irreg + names_reg # Contiene los tipos de neuronas
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
    pos_irreg = [i for i, col in enumerate(circuito_df.columns) if col in names_irreg]

    # Sacan en una lista los índices de las columnas correspondientes a neuronas regulares --> [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    pos_reg = [i for i, col in enumerate(circuito_df.columns) if col in names_reg]

    # circuito[which(tipos=="RSB"),which(tipos=="RS")] <- conexiones(RSB,RS,13,4)
    # circuito[which(tipos=="RS"),which(tipos=="RFB")] <- conexiones(RS,RFB,-13,4)
    # circuito[which(tipos=="RS"),which(tipos=="RSB")] <- conexiones(RS,RSB,-13,4)
    # circuito[which(tipos=="RFB"),which(tipos=="RSB")] <- conexiones(RFB,RSB,13,4)
    # circuito[which(tipos=="RSB"),which(tipos=="A")] <- conexiones(RSB,A,6,4)
    # circuito[which(tipos=="A"),which(tipos=="A")] <- conexiones(A,A,13,4)
    # circuito[which(tipos=="A"),which(tipos=="RS")] <- conexiones(A,RS,18,4)
    # circuito[which(tipos=="A"),which(tipos=="RFB")] <- conexiones(A,RFB,13,4)
    # circuito[which(tipos=="RS"),which(tipos=="A")] <- conexiones(RS,A,13,4)
    # circuito[which(tipos=="RS"),which(tipos=="ISe")] <- conexiones(RS,ISe,13,4)
    # circuito[which(tipos=="RFB"),which(tipos=="ISB")] <- conexiones(RFB,ISB,13,4)
    # circuito[which(tipos=="ISB"),which(tipos=="IFB")] <- conexiones(ISB,IFB,13,4)
    # circuito[which(tipos=="ISe"),which(tipos=="IFB")] <- conexiones(ISe,IFB,13,4)
    
    for i_idx, (row, row_name) in enumerate(zip(adj_matrix, names_all)):
        for j_idx, (n_con, col_name) in enumerate(zip(row, names_all)):
            idx_to_change_row = np.where(tipos == row_name)[0]
            idx_to_change_col = np.where(tipos == col_name)[0]
            possible_values = connection_strengths[row_name]
            
            # Randomized values in range
            # values_to_generate = np.random.choice(possible_values, len(idx_to_change_row)*len(idx_to_change_col))
            # if len(idx_to_change_col) != 0 and len(idx_to_change_row) != 0:
            #     print(circuito_df.iloc[idx_to_change_row, idx_to_change_col])
            #     print(values_to_generate)
            #     circuito_df.iloc[idx_to_change_row, idx_to_change_col] = np.reshape(values_to_generate, (len(idx_to_change_row), len(idx_to_change_col)))
            #     print(circuito_df.iloc[idx_to_change_row, idx_to_change_col])

            # Single random value
            # values_to_generate = np.random.choice(possible_values)
            # circuito_df.iloc[idx_to_change_row, idx_to_change_col] = values_to_generate

            # Random masked by adjacency matrix
            # values_to_generate = adj_matrix[i_idx, j_idx]*np.random.choice(possible_values)
            # circuito_df.iloc[idx_to_change_row, idx_to_change_col] = values_to_generate
            
            # Copy from by the adjacenty matrix
            circuito_df.iloc[idx_to_change_row, idx_to_change_col] = adj_matrix[i_idx, j_idx]

    # Fill diagonal
    circuito_df = circuito_df * (1-np.eye(len(circuito_df)))



    ######### Parámetros controlables del circuito############
    max_delay = 5 # Establece un valor máximo para el retraso que se puede generar
    min_delay = 1 # Establece un valor mínimo para el retraso
    tiempo = simulation_time

    ############  parámetros y variables internos###############
    delays = np.random.choice(range(min_delay, max_delay + 1), size, replace=True)
    contador = np.zeros((size, 2))

    volt = c.copy()
    reg = np.full(size, -13.)
    inputs = np.zeros(size)

    lim = sum(num_irre)

    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    punto_medio = reg - (b / a)
    # punto_medio = reg
    # punto_medio = reg - (b * np.cos(a * 0) / a - b * np.cos(a * np.pi / a) / a) / 2
    # print(punto_medio-punto_medio2)

    tclave = np.arccos((-16 - punto_medio) * a / b) / a
    t = np.zeros(size)

    # grupos_nume <- round(size/grupos_tama+0.4) #(es R)
    grupo_tag = pd.factorize(circuito_df.columns)[0] + 1
    #print(grupo_tag)   # en R imprime esto -->  3 3 3 3 3 4 4 4 4 4 2 2 2 2 2 1 5 5 5 5 5 5 5 5 5 5 !! (los nº van en función del orden alfabético)
    grupos_nume = max(grupo_tag)
    nombres_grupos = list(pd.factorize(circuito_df.columns)[1])
    #print(nombres_grupos) #EN R LO IMPRIME EN OTRO ORDEN!!!!!!!! (en R se imprimen por orden alfabético)
    # grupo_tag[which(tipos=="A")] <- grupos_nume #(es R)
    list_aferentes = np.where(circuito_df.columns == "A")[0]
    #print(list_aferentes) #EN R IMPRIME 16 Y AQUÍ 15, PORQUE AQUÍ SE EMPIEZA A CONTAR DESDE 0

    # almacenar la actividad neuronal
    sim = [[] for _ in range(size)]
    sim_con = []
    sim_con_2 = []
    grupo = [[] for _ in range(grupos_nume)]

    volt = np.array(volt, dtype=np.float64)
    volt_full = np.empty((size, tiempo*1000))

    idx = 0

    for i in range(1000*tiempo):
        i = i + 1
        t = t + 1

        volt[pos_irreg] += 0.5 * ((0.04 * volt[pos_irreg] + 5) * volt[pos_irreg] + 140 - reg[pos_irreg] + inputs[pos_irreg])
        volt[pos_irreg] += 0.5 * ((0.04 * volt[pos_irreg] + 5) * volt[pos_irreg] + 140 - reg[pos_irreg] + inputs[pos_irreg])
        reg[pos_irreg] += a[pos_irreg] * (b[pos_irreg] * volt[pos_irreg] - reg[pos_irreg])

        volt[pos_reg] += 0.5 * ((0.04 * volt[pos_reg] + 5) * volt[pos_reg] + 140 - reg[pos_reg] + inputs[pos_reg])
        volt[pos_reg] += 0.5 * ((0.04 * volt[pos_reg] + 5) * volt[pos_reg] + 140 - reg[pos_reg] + inputs[pos_reg])
        reg[pos_reg] -= np.sin(t[pos_reg] * a[pos_reg]) * b[pos_reg]

        inputs.fill(0)

        # Si volt supera los 30 se considera un disparo
        disp = np.where(volt > 30)[0]
        if disp.size > 0:
            disp = disp.astype(int)
            # DR = np.intersect1d(pos_reg, disp).astype(int)
            DI = np.intersect1d(pos_irreg, disp).astype(int)
            DA = np.intersect1d(disp, list_aferentes).astype(int)

            contador[disp, 0] += delays[disp]
            contador[disp, 1] += 1

            volt[disp] = c[disp]
            reg[DI] += d[DI]
            reg[DA] = -13

            for k in disp:
                #CREAR VARIABLE j+1
                #POSIBLE ERRATA EN j + i / 1000
                sim[k].append(i / 1000)
                grupo[grupo_tag[k]-1].append(i / 1000)
                if k not in burst:
                    sim_con_2.append(i / 1000)
                if k not in list_aferentes:
                    sim_con.append(i/ 1000)

        contador2 = np.ceil(contador[:, 0] / delays)
        contador[:, 0] -= contador[:, 1]
        contador[:, 1] = np.ceil(contador[:, 0] / delays)
        y = np.where(contador[:, 1] < contador2)[0]

        if y.size > 0:
            inputs = circuito[y, :].sum(axis=0) if y.size > 1 else circuito[y, :].reshape(-1)
            receptor = np.intersect1d(np.where(inputs != 0)[0], pos_reg)

            # Cálculo seguro aunque receptor esté vacío
            t_sel = t[receptor]
            tclave_sel = tclave[receptor]
            periodo_sel = periodo[receptor]
            inputs_sel = inputs[receptor]
            delta = (tclave_sel - (t_sel % (periodo_sel / 2))) * (inputs_sel / 20)
            t[receptor] = np.round(t_sel + delta)
            reg[receptor] = punto_medio[receptor] + b[receptor] * np.cos(a[receptor] * t[receptor]) / a[receptor]
        
        volt_full[:, idx] = volt
        idx += 1
    
    # Convertir la simulación final en un DataFrame
    maximo = max(len(s) for s in sim)
    final = pd.DataFrame({f"U{str(i).zfill(2)}": sim[i] + [np.nan] * (maximo - len(sim[i])) for i in range(size)})
    final.columns = tipos

    return sim, final, volt_full

def generate_random_adjmat():
    adjmat = np.empty((8,8))
    for idx_i, name in enumerate(["ISe", "ISi", "IFB", "ISB", "A", "RS", "RSB", "RFB"]):
        for idx_j, _ in enumerate(["ISe", "ISi", "IFB", "ISB", "A", "RS", "RSB", "RFB"]):
            options = np.hstack([0, connection_strengths[name]])
            adjmat[idx_i, idx_j] = np.random.choice(options)
    
    return adjmat

if __name__ == "__main__":
    adj_matrix = generate_random_adjmat()
    print(adj_matrix)

    sim, final, volts = run_simulator(adj_matrix)
    # print(sim)
    # print(final)
    # plt.plot(volts[0, :1000])
    # plt.show()