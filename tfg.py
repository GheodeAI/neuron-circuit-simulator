import pandas as pd
import numpy as np

# Carga y selección de neuronas aceptables. Se leen 3 archivos csv que contienen datos sobre distintos tipos de neuronas.
# De cada archivo se filtran las neuronas que son aceptables según los parámetros (freq_inter, freq_intra) que serán las que luego
# se usarán en la simulación.
# Carga de datos desde archivos CSV
datos_RFB = pd.read_csv("RFB_valors.csv2", sep=';')
# Convertir columnas a numérico (forzando errores a NaN (Not A Number) si hay valores no convertibles)
datos_RFB["freq_inter"] = pd.to_numeric(datos_RFB["freq_inter"], errors='coerce')
datos_RFB["freq_intra.ms."] = pd.to_numeric(datos_RFB["freq_intra.ms."], errors='coerce')
# Aplicar filtros después de la conversión
aceptables_RFB = datos_RFB[(datos_RFB["freq_inter"] > 2) & (datos_RFB["freq_inter"] < 9) & (datos_RFB["freq_intra.ms."] > 100)].index.to_list()
#aceptables_RFB = aceptables_RFB[1:] + [aceptables_RFB[-1] + 1]

datos_RS = pd.read_csv("RS_valors.csv2", sep=';')
datos_RS["freq(ms)"] = pd.to_numeric(datos_RS["freq(ms)"], errors='coerce')
aceptables_RS = datos_RS[(datos_RS["freq(ms)"] > 5) & (datos_RS["freq(ms)"] < 6)].index.to_list()
#aceptables_RS = aceptables_RS[1:] + [aceptables_RS[-1] + 1]

print("RFB indices:", aceptables_RFB)
print(len(aceptables_RFB))
print("RS indices:", aceptables_RS)
print(len(aceptables_RS))

datos_RSB = pd.read_csv("RSB_valors.csv2", sep=';')
datos_RSB["freq_inter"] = pd.to_numeric(datos_RSB["freq_inter"], errors='coerce')
datos_RSB["freq_intra.ms."] = pd.to_numeric(datos_RSB["freq_intra.ms."], errors='coerce')
aceptables_RSB = datos_RSB[(datos_RSB["freq_inter"] > 0.4) & (datos_RSB["freq_inter"] < 0.5) & (datos_RSB["freq_intra.ms."] < 20)].index.to_list()
aceptables_RSB = [x + 1 for x in aceptables_RSB]


##Caracterización neuronas##
# Number of neurons of each class / Asignación del número de neuronas de cada tipo
# ISe --> Irregular Spike ------
# ISi --> Irregular Spike ------
# ISB --> Irregular slow burst (irregulares en ráfaga lentos)
# IFB --> #Irregular fast burst (irregulares en ráfaga rápidos)
# A --> ???
# RS --> Regular spike (regulares simples)
# RSB --> Regular slow burst (regulares en ráfaga lentos)
# RFB --> Regular fast burst (regulares en ráfaga rápidos)
ISe, ISi, ISB, IFB, A, RS, RSB, RFB = 5, 5, 0, 5, 1, 0, 10, 0

# Selección de parámetros de cada neurona
# Three constants are needed to each class
# Regular neurons depend on a Simple harmonious Movement function, so they need angular velocity and Amplitud.
# The constant "a" of the regular neurons depends on the period.
h = np.random.choice(aceptables_RFB, size=RFB, replace=True).tolist() if aceptables_RFB else [] #En R, si aceptables_RFB está vacío aquí lanzaba un error
f = np.random.choice(aceptables_RS, size=RS, replace=True).tolist() if aceptables_RS else []
g = np.random.choice(aceptables_RSB, size=RSB, replace=True).tolist() if aceptables_RSB else []


#Construyendo unas listas
#Ej: crea una lista de 0.02 el número de ISe+ISi veces --> 5+5=10 --> [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
#datos_RS.loc[f, "a"].tolist()  -->  .loc[f, "a"] selecciona las filas f de la columna "a" y lo convierte a lista.
    #si f = [1, 3] y "a" en esas filas es [0.5, 0.6], entonces la lista = [0.5, 0.6]

a = ([0.02] * (ISe + ISi) + [0.14] * ISB + [0.1] * IFB + [0.02] * A +
     datos_RS.loc[f, "a"].tolist() + datos_RSB.loc[g, "a"].tolist() + datos_RFB.loc[h, "a"].tolist())

#np.random.uniform(a, b, n) genera n valores aleatorios en el rango [a, b] con distribución uniforme.
    #si ISB=5 se podría generar la lista [0.2631, 0.2638, 0.2635, 0.2632, 0.2639]
b = ([0.2] * (ISe + ISi) + np.random.uniform(0.263, 0.264, ISB).tolist() +
     np.random.uniform(0.249, 0.251, IFB).tolist() + [0.2] * A +
     datos_RS.loc[f, "b"].tolist() + datos_RSB.loc[g, "b"].tolist() + datos_RFB.loc[h, "b"].tolist())

c = ([-65] * (ISe + ISi + ISB + IFB + A) + [-65] * RS + datos_RSB.loc[g, "c"].tolist() + datos_RFB.loc[h, "c"].tolist())

d = ([8] * (ISe + ISi) + np.random.uniform(-8, -8, ISB).tolist() + np.random.uniform(-8, -7.95, IFB).tolist() + [8] * A)

periodo = d + datos_RS.loc[f, "period"].tolist() + datos_RSB.loc[g, "period"].tolist() + datos_RFB.loc[h, "period"].tolist()
periodo = np.array(periodo)

#print(a)
#print(b)
#print(c)
#print(d)
#print(periodo)


######## Diseño circuito ########
num_irre = [ISe, ISi, ISB, IFB, A]
num_reg = [RS, RSB, RFB]
names_irreg = ["ISe", "ISi", "ISB", "IFB", "A"]
names_reg = ["RS", "RSB", "RFB"]

cantidad_neu = num_irre + num_reg # Cuántas neuronas hay de cada tipo
size = sum(cantidad_neu)  # Número total de neuronas


# Creación de la matriz (circuito) de conexiones, con el tamaño (size) igual al número de neuronas
nombres_neu = names_irreg + names_reg # Contiene los tipos de neuronas
circuito = np.zeros((size, size))  # Matriz de conexiones 26x26 llena de ceros, lo que representa que
                                    # inicialmente ninguna neurona está conectada con otra

'''
tipos contiene:
['ISe', 'ISe', 'ISe', 'ISe', 'ISe',
 'ISi', 'ISi', 'ISi', 'ISi', 'ISi',
 'ISB', 'ISB', 'IFB', 'IFB', 'IFB', 'IFB', 'IFB',
 'A',
 'RSB', 'RSB', 'RSB', 'RSB', 'RSB', 'RSB', 'RSB', 'RSB', 'RSB', 'RSB']
 '''
tipos = np.repeat(nombres_neu, cantidad_neu)

# Si t (un elemento de tipos) está en ["IFB", "ISB", "RSB", "RFB"]
# enumerate(tipos) genera pares (i, t), donde i es el índice y t es el tipo de neurona en tipos
# la salida es burst = [10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25], porque imprime los índices,
    # y al no haber ni ISe ni ISi, empieza imprimiendo el índice 10
burst = [i for i, t in enumerate(tipos) if t in ["IFB", "ISB", "RSB", "RFB"]]
#burst = burst[1:] + [burst[-1] + 1] # ????????

# np.cumsum calcula la suma acumulativa de cantidad_neu ([5, 5, 0, 5, 1, 0, 10, 0]), por lo tanto --> [5, 10, 12, 17, 18, 18, 28, 28]
# Pandas combina nombres_neu y np.cumsum(cantidad_neu) en un DataFrame, cuya salida es:
'''
  nombre  numero_neurons_tipo
0    ISe                    5
1    ISi                   10
2    ISB                   10
3    IFB                   15
4      A                   16
5     RS                   16
6    RSB                   26
7    RFB                   26
'''
datos_conexiones = pd.DataFrame({"nombre": nombres_neu, "numero_neurons_tipo": np.cumsum(cantidad_neu)})

# Creación de etiquetas para la matriz circuito. Se crea un DataFrame en el que cada fila y cada columna representa
    # una neurona de tipos. El valor en cada celda representa la conexión entre dos tipos de neuronas, inicialmente 0
# Salida:
'''
ISe  ISe  ISe  ISe  ISe  ISi  ISi  ...  RSB  RSB  RSB  RSB  RSB  RSB  RSB
ISe  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISe  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISe  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISe  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISe  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISi  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISi  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISi  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISi  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
ISi  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
IFB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
IFB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
IFB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
IFB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
IFB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
A    0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
RSB  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0
[26 rows x 26 columns]
'''
circuito_df = pd.DataFrame(circuito, columns=tipos, index=tipos)

# Identificación de posiciones
# Sacan en una lista los índices de las columnas correspondientes a neuronas irregulares --> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
pos_irreg = [i for i, col in enumerate(circuito_df.columns) if col in names_irreg]
#pos_irreg = pos_irreg[1:] + [pos_irreg[-1] + 1] # ????????
# Sacan en una lista los índices de las columnas correspondientes a neuronas regulares --> [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
pos_reg = [i for i, col in enumerate(circuito_df.columns) if col in names_reg]
#pos_reg = pos_reg[1:] + [pos_reg[-1] + 1]  # ????????
#print(pos_irreg)
#print(pos_reg)

############################??????????????????????????????????????????????????????????????????????????????
# generador aleatorio de   #
#         conexiones       #
############################

# the function "conexiones" create a connection between two types of neurons

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

#############################
#        SIMULACIÓN         #
#############################

# library("plot.matrix") #???? (es R)

######### Parámetros controlables del circuito############
max_delay = 5 # Establece un valor máximo para el retraso que se puede generar
min_delay = 1 # Establece un valor mínimo para el retraso
tiempo = 100

############  parámetros y variables internos###############
delays = np.random.choice(range(min_delay, max_delay + 1), size, replace=True)
contador = np.zeros((size, 2))

volt = c.copy()
reg = np.full(size, -13)
inputs = np.zeros(size)


lim = sum(num_irre)

a = np.array(a)
punto_medio = reg - (b * np.cos(a * 0) / a - b * np.cos(a * np.pi / a) / a) / 2
tclave = np.arccos((-16 - punto_medio) * a / b) / a
t = np.zeros(size)

# grupos_nume <- round(size/grupos_tama+0.4) #???? (es R)
grupo_tag = pd.factorize(circuito_df.columns)[0] + 1
#print(grupo_tag)   # en R imprime esto -->  3 3 3 3 3 4 4 4 4 4 2 2 2 2 2 1 5 5 5 5 5 5 5 5 5 5 !! (los nº van en función del orden alfabético)
grupos_nume = max(grupo_tag)
nombres_grupos = list(pd.factorize(circuito_df.columns)[1])
#print(nombres_grupos) #EN R LO IMPRIME EN OTRO ORDEN!!!!!!!! (en R se imprimen por orden alfabético)
# grupo_tag[which(tipos=="A")] <- grupos_nume #???? (es R)
list_aferentes = np.where(circuito_df.columns == "A")[0]
#print(list_aferentes) #EN R IMPRIME 16 Y AQUÍ 15, PORQUE AQUÍ SE EMPIEZA A CONTAR DESDE 0


###################sim de disp####################

# almacenar la actividad neuronal
sim = [[] for _ in range(size)]
sim_con = []
sim_con_2 = []
grupo = [[] for _ in range(grupos_nume)]


#volt = np.pad(volt, (0, 26 - len(volt)), constant_values=-65)
#print(volt)
volt = np.array(volt, dtype=np.float64)
#print(volt)
a = np.array(a, dtype=np.float64)
b = np.array(b, dtype=np.float64)
reg = np.array(reg, dtype=np.float64)


for j in range(tiempo):
    grupocorto = [[] for _ in range(grupos_nume)]
    sim_short_con = []
    sim_short_con_2 = []
    sim_short = [[] for _ in range(size)]

    for i in range(1000):
        t = t + 1

        # POR QUÉ HAY DOS LÍNEAS IGUALES AQUÍ AL PRINCIPIO?
        volt[pos_irreg] += 0.5 * ((0.04 * volt[pos_irreg] + 5) * volt[pos_irreg] + 140 - reg[pos_irreg] + inputs[pos_irreg])
        volt[pos_irreg] += 0.5 * ((0.04 * volt[pos_irreg] + 5) * volt[pos_irreg] + 140 - reg[pos_irreg] + inputs[pos_irreg])
        reg[pos_irreg] += a[pos_irreg] * (b[pos_irreg] * volt[pos_irreg] - reg[pos_irreg])
        #print(volt)
        #print(volt[pos_irreg])
        #print(reg[pos_irreg])

        ####################################A PARTIR DE AQUÍ DAN ERRORES############################################
        # POR QUÉ HAY DOS LÍNEAS IGUALES AQUÍ AL PRINCIPIO?
        volt[pos_reg] += 0.5 * ((0.04 * volt[pos_reg] + 5) * volt[pos_reg] + 140 - reg[pos_reg] + inputs[pos_reg])
        volt[pos_reg] += 0.5 * ((0.04 * volt[pos_reg] + 5) * volt[pos_reg] + 140 - reg[pos_reg] + inputs[pos_reg])
        #volt[pos_reg] = volt[pos_reg] + 0.5 * ((0.04 * volt[pos_reg] + 5) * volt[pos_reg] + 140 - reg[pos_reg] + inputs[pos_reg])
        #volt[pos_reg] = 0.5 * ((0.04 * volt[pos_reg] + 5) * volt[pos_reg] + 140 - reg[pos_reg] + inputs[pos_reg])
        #print(volt[pos_reg])
        reg[pos_reg] -= np.sin(t[pos_reg] * a[pos_reg]) * b[pos_reg] #ESTÁ BIEN (?)
        #print(volt)
        #reg[pos_reg] = reg[pos_reg] - np.sin(t[pos_reg] * a[pos_reg]) * b[pos_reg]
        #print(reg[pos_reg])

        inputs.fill(0)

        # Si volt supera los 30 se considera un disparo
        #print(volt)
        disp = np.where(volt > 30)[0]
        #print(disp)
        if disp.size > 0:
            disp = disp.astype(int)
            #print(volt)
            #print(disp)
            DR = np.intersect1d(pos_reg, disp)
            #print(DR)
            DI = np.intersect1d(pos_irreg, disp)
            #print(DI)
            DA = np.intersect1d(disp, list_aferentes)
            #print(DA)

            contador[disp, 0] += delays[disp]
            #print(contador[disp, 0])
            contador[disp, 1] += 1
            #print(contador[disp, 1])

            c = np.array(c) ###############hacer conversión justo cuando defino c
            volt[disp] = c[disp]
            #print(volt[disp])
            d = np.array(d) ###############hacer conversión justo cuando defino d
            reg[DI] += d[DI]
            #print(reg[DI])
            reg[DA] = -13
            #print(reg[DA])

            for k in disp:
                sim_short[k].append(j + i / 1000)###################!!!!!#####################en R el primer número es 1,9xx, 0,9xx
                #print(sim_short[k])
                grupocorto[grupo_tag[k] - 1].append(j + i / 1000)###################!!!!!#####################en R el primer número es 1,9xx, 0,9xx
                #print(grupocorto[grupo_tag[k] - 1])
                if k not in burst:
                    sim_short_con_2.append(j + i / 1000)
                    #print(sim_short_con_2)
                if k not in list_aferentes:
                    sim_short_con.append(j + i / 1000)
                    #print(sim_short_con)

        contador2 = np.ceil(contador[:, 0] / delays)
        #print(contador2)
        contador[:, 0] -= contador[:, 1]
        #print(contador[:, 0])
        contador[:, 1] = np.ceil(contador[:, 0] / delays)
        y = np.where(contador[:, 1] < contador2)[0]
        #print(y)

        '''
        if y.size > 0:
            inputs = circuito[y, :].sum(axis=0) if y.size > 1 else circuito[y, :]
            #print(inputs)
            receptor = np.intersect1d(np.where(inputs != 0)[0], pos_reg)
            #print(receptor)
            t[receptor] = np.round(t[receptor] + ((tclave[receptor] - t[receptor] % (periodo[receptor] / 2)) * (inputs[receptor] / 20)))
            print(t[receptor])
            reg[receptor] = punto_medio[receptor] + b[receptor] * np.cos(a[receptor] * t[receptor]) / a[receptor]
        '''
        if y.size > 0:
            inputs = circuito[y, :].sum(axis=0) if y.size > 1 else circuito[y, :].reshape(-1)
            # print(inputs)
            receptor = np.intersect1d(np.where(inputs != 0)[0], pos_reg)
            # print(receptor)
            # Cálculo seguro aunque receptor esté vacío
            t_sel = t[receptor]
            tclave_sel = tclave[receptor]
            periodo_sel = periodo[receptor]
            inputs_sel = inputs[receptor]
            delta = (tclave_sel - (t_sel % (periodo_sel / 2))) * (inputs_sel / 20)
            t[receptor] = np.round(t_sel + delta)
            #print(t[receptor])
            reg[receptor] = punto_medio[receptor] + b[receptor] * np.cos(a[receptor] * t[receptor]) / a[receptor]
            #print(reg[receptor])

    sim_con.extend(sim_short_con) #!!!!!!!!!!!!!!!!!!! EN R SIEMPRE EMPIEZA EN 0 Y LUEGO CONTINUA CON 1,9XX
    #print(sim_con)
    sim_con_2.extend(sim_short_con_2)
    #print(sim_con_2)
    for i in range(size):
        sim[i].extend(sim_short[i]) #!!!!!!!!!!!!!!!!!!! EN R SIEMPRE LOS NÚMEROS SON UNO MÁS, OTRA VEZ EL PROBLEMA DEL 1,9XX
        #print(sim[i])
    for i in range(grupos_nume):
        grupo[i].extend(grupocorto[i]) #!!!!!!!!!!!!!!!!!!! EN R SIEMPRE LOS NÚMEROS SON UNO MÁS, OTRA VEZ EL PROBLEMA DEL 1,9XX
        #print(grupo[i])

# Convertir la simulación final en un DataFrame
maximo = max(len(s) for s in sim) #!!!!! PARECE QUE SIEMPRE SACA ALREDEDOR DE 320/330, EN CAMBIO EN R SACA ALREDEDOR DE 240 !!!!!!
#print(maximo)
final = pd.DataFrame({f"U{str(i).zfill(2)}": sim[i] + [np.nan] * (maximo - len(sim[i])) for i in range(size)})
#print(final) #!!!!!!!!!!!!!!!!!!!!!!!!!!! EN R ES T0DO NA, AQUí NO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
final.columns = tipos

'''
# Paso 1: agregar sim_con a grupo (si aplica, este paso depende del contexto completo)
grupo.append(sim_con)  # <-- si tienes este paso, mantenlo
# Paso 2: calcular el máximo largo de las listas en `sim`
maximo = max(len(s) for s in sim)
# Paso 3: crear un DataFrame de NaNs con maximo filas y size columnas
final = pd.DataFrame(np.nan, index=range(maximo), columns=range(size))
# Paso 4: rellenar columna por columna con los valores de sim[i]
for i in range(size):
    final.iloc[:len(sim[i]), i] = sim[i]
# Paso 5: generar nombres como U01, U02, ..., Uxx
colnames_temp = [f"U{str(i + 1).zfill(2)}" for i in range(size)]
final.columns = colnames_temp
# Paso 6: reemplazar nombres de columnas por `tipos`
final.columns = tipos
# Mostrar resultado final
print(final)
'''