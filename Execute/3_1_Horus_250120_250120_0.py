### Subseccion: Este es el formal (ocupo solo enero 2025 por ahora) pero aquí se está consruyendo el modelo de optimización
# 240108: inclusion de modo_tactico, y opción de ejecución con periodo mensual
# Opción de particionar DATE en Semana o Mes (aún no, primero integar las restricciones propias del escenario)
### Seccion:  Librerías
import sys
import pandas as pd
import numpy as np
import mip
import os
import datetime as dt

import warnings
warnings.filterwarnings("ignore")
import time
t_0 = time.time()
t_inicio_0 = time.time()
#sys.exit('En df_R3c, en rango 1, tengo A > 0...eso no puede ser...A = 0, B >= 0')
### Seccion:  Parámetros
escenario = 'Base MX' #'Base MX operativo'
modo_tactico = True

refrescar_lectura = True
mode = 'PC'
# paises = ['MX'] #['UY', 'CL', 'MX', 'CO', 'BR', 'AR'] # Paises a considerar ['CL']
# modo_proyeccion = 'LIBRE' #['PRESUPUESTO', 'LIBRE'] #['LIBRE', 'PRESUPUESTO']
ajuste_metricas = True # Ajuste a versión de presupuesto (True)

grano = 'F-LT'
lista_duplicidades = ['F-LT', 'F', 'LT', 'TOTALES']  #, 'F-SF'
escribir_modelo = True, 9999
activar_holguras = False
campo_last_touch = 'TIPO_MEDIO'
version_simplificada = False # Activar para buscar factibilidad. Elimina las restricciones: 8 - Todas  las restricciones propias del escenario
# Además, permite un solo rango de inversión
### Seccion:  Funciones Transversales
carpeta_modulos = '../../Modulos/'  #/home/data/python/performance_automation/Modulos
if mode == 'VM': # Esto tiene que ir en funcion generar dag
    carpeta_modulos = '/home/data/python/performance_automation/Modulos/'

sys.path.insert(0, carpeta_modulos)
import funciones_transversales as tf
client = tf.generar_cliente()

dir_histogramas = '../Data/histogramas/'
carpeta_input = '../Inputs/'
dir_data = '../Data/'
actualizar = False

df_dias_eventos = tf.generar_calendario_eventos_new(actualizar = actualizar, carpeta_input = carpeta_input)
df_dias_eventos.head()
df_times = pd.DataFrame()
t0 = time.time()

def medir_tiempo(t0, texto, df_times, output = True):
    
    if output:
        print(f'\n\n Medición de tiempo en {texto} \n\n\n\n')
    t = (time.time() - t0) / 60
    df = pd.DataFrame({'texto': [texto], 'tiempo': [t]})
    df_times = pd.concat([df_times, df])
    t0 = time.time()
    return df_times, t0

### Seccion:  0. Escenario
df_escenarios = tf.lectura_segura_Gsheets('Horus Escenarios', 'Escenarios', encabezado = 1, refrescar_lectura = refrescar_lectura)
df_escenarios = df_escenarios[df_escenarios['ESCENARIO'] == escenario].reset_index(drop = True)
print(df_escenarios)

df_cols = tf.lectura_segura_Gsheets('Horus Escenarios', 'cols')
df_escenarios.columns = df_cols.columns

df_escenarios
### Subseccion: Medios
df_medios = tf.lectura_segura_Gsheets('clacomizacion_corp_v0', 'DICCIONARIO_MEDIOS_CORP', refrescar_lectura = refrescar_lectura)
#df_medios = df_medios[['TIPO_MEDIO', 'NATURALEZA_MEDIO']].drop_duplicates().reset_index(drop = True)
df_medios['NATURALEZA_MEDIO'] = np.where(df_medios['NATURALEZA_MEDIO'] == 'Pago', 'PAGO', 'ORGANICO')
df_medios = df_medios.drop(columns = 'LAST_TOUCH_CHANNEL')
df_medios = df_medios.rename(columns = {'LAST_TOUCH_CHANNEL_CORP': 'LAST_TOUCH_CHANNEL'})
df_medios = df_medios.drop_duplicates().reset_index(drop = True)
df_medios.head()
df_medios = df_medios.sort_values(campo_last_touch).reset_index(drop = True)
df_medios_org, df_medios_pago = df_medios[df_medios['NATURALEZA_MEDIO'] == 'ORGANICO'].reset_index(drop = True), df_medios[df_medios['NATURALEZA_MEDIO'] == 'PAGO'].reset_index(drop = True)

lista_org, lista_pago = list(df_medios_org[campo_last_touch].unique()), list(df_medios_pago[campo_last_touch].unique())
print(lista_org), print(len(lista_org))
print(lista_pago), print(len(lista_pago))
df_configuracion = df_escenarios[df_escenarios['TIPO'] == 'C'].reset_index(drop = True)
df_configuracion

k = 1
# Temporalidad por defecto: Año actual + k Próximos años (Se puede cambiar)
dias_proyeccion = [dt.datetime.today().replace(year = dt.datetime.today().year, month = 1, day = 1).date(), dt.datetime.today().replace(year = dt.datetime.today().year + k, month = 12, day = 31).date()] # declarar arriba, al inicio del código
#dias_proyeccion = [pd.to_datetime(dias_proyeccion[0]), pd.to_datetime(dias_proyeccion[1])]

paises = ['MX', 'CL', 'UY', 'CO', 'BR', 'AR', 'PE'] # Por defecto
familias_ids = ['*'] # Todo
last_touch_lista = ['*'] #lista_pago[:1] #+ lista_org[:1] #['SEM - Shopping', 'Typed', 'Afliliados', 'Internal', 'SEM - Non Brand', 'SEM - Brand'] # # Todo
lista_fuentes, lista_canales = ['SODIMAC', 'ES', 'SIS'], ['WEB', 'APP']

for c in df_configuracion.columns:
    if 'Base + Config' in c:
        name = '_'.join(c.split('_')[1:])
        value = df_configuracion[c][0]
        if name == 'MODO_PROYECCION': # Modo proyección
            modo_proyeccion = value
            if modo_proyeccion not in ['PRESUPUESTO', 'LIBRE']:
                sys.exit(f'MODO_PROYECCION {modo_proyeccion} no es correcto')
        
        if (name == 'PAIS') and (value != ''):
            paises = value.split(',')
                        
        if (name == 'DATE') and (value != ''):
            dias_proyeccion_new = pd.to_datetime(value.split('>')[0]).date(), pd.to_datetime(value.split('>')[1]).date()
            dias_proyeccion[0] = max(dias_proyeccion[0], dias_proyeccion_new[0])
            dias_proyeccion[1] = min(dias_proyeccion[1], dias_proyeccion_new[1])
        
        if (name == 'PERIODO') and (value != ''):
            periodo_0, periodo_1 = value.split('>')
            dia0 = dt.datetime(int(periodo_0.split('-')[0]), int(periodo_0.split('-')[1]), 1).date()
            dia1 = dt.datetime(int(periodo_1.split('-')[0]), int(periodo_1.split('-')[1]), 1).date()
            dia1 = (dt.datetime(dia1.year, dia1.month, 1) + pd.DateOffset(months = 1) - pd.DateOffset(days = 1)).date()
            dias_proyeccion[0] = max(dias_proyeccion[0], dia0)
            dias_proyeccion[1] = min(dias_proyeccion[1], dia1)
        
        if (name == 'AÑO') and (value != ''):
            year_0, year_1 = value.split('>')
            dia0 = dt.datetime(int(year_0), 1, 1).date()
            dia1 = dt.datetime(int(year_1), 12, 31).date()
            dias_proyeccion[0] = max(dias_proyeccion[0], dia0)
            dias_proyeccion[1] = min(dias_proyeccion[1], dia1)
        
        if (name == 'FAMILIA') and (value != ''):
            familias_lista = value.split(',')
            familias_ids = [int(f) for f in familias_lista]
        
        if (name == campo_last_touch) and (value != ''):
            last_touch_lista = value.split(',')
        
        if (name == 'FUENTE') and (value != ''):
            lista_fuentes = value.split(',')
        
        if (name == 'CANAL') and (value != ''):
            lista_canales = value
        
        if (name == 'CANAL_BASE') and (value != ''):
            sys.exit('CANAL_BASE en configuración debe ir vacío')
            

dias_proyeccion = [pd.to_datetime(dias_proyeccion[0]), pd.to_datetime(dias_proyeccion[1])]
dias_proyeccion
### Seccion:  1. Lectura de Parámetros
"""
ias_proyeccion = [dt.datetime.today().replace(year = dt.datetime.today().year, month = 1, day = 1).date(), dt.datetime.today().replace(year = dt.datetime.today().year + 1, month = 12, day = 31).date()] # declarar arriba, al inicio del código
dias_proyeccion = [pd.to_datetime(dias_proyeccion[0]), pd.to_datetime(dias_proyeccion[1])]
dias_proyeccion # Desde el primer día del año actual, hasta el último día del próximo año (por defecto)

print('Small Batch [un mes](eliminar)')
print('Small Batch [un día](eliminar)')
dias_proyeccion = [dt.datetime(2025, 1, 1).date(), dt.datetime(2025, 2, 28).date()] # declarar arriba, al inicio del código
dias_proyeccion = [pd.to_datetime(dias_proyeccion[0]), pd.to_datetime(dias_proyeccion[1])]
dias_proyeccion
"""
### Seccion: # 1.1 Conjuntos
diccionario_dimensiones = {'TOTALES': '', 'F': 'FAMILIA', 'LT': campo_last_touch, 'SF': 'SUBFAMILIA', 'F1': 'F1', 'G1': 'G1'} 
campos_grano = []
for d in grano.split('-'):
    campos_grano.append(diccionario_dimensiones[d])
campos_grano # campos asociados al grano
df_P2 = tf.parquet_act(f'{dir_data}Parametros/P2')
df_P2['DATE'] = pd.to_datetime(df_P2['DATE'])
df_P2 = df_P2[df_P2['PAIS'].isin(paises)]
df_alpha_grano_pago = df_P2[campos_grano].drop_duplicates()
df_alpha_grano_pago = df_alpha_grano_pago.sort_values(campos_grano).reset_index(drop = True)
df_alpha_grano_pago = df_alpha_grano_pago[df_alpha_grano_pago[campos_grano].notna().all(axis = 1)].reset_index(drop = True) # alpha_tongo (alpha_grano)
df_alpha_grano_pago # alpha_tongo (alpha_grano)
lista_familias = df_alpha_grano_pago['FAMILIA'].unique()

familia_sm = ['']
for f in lista_familias:
    
    if '*' in familias_ids:
        familia_sm = lista_familias[:]
        break
    
    for s in familias_ids:
        formatted_number = str(s).zfill(2)
        if formatted_number in f:
            familia_sm.append(f)
            continue

familia_sm
lista_LT_pago = df_alpha_grano_pago[campo_last_touch].unique()

if '*' in last_touch_lista:
    last_touch_lista_pago = lista_LT_pago
else:
    last_touch_lista_pago = []
    for lt in lista_LT_pago:
        if lt in last_touch_lista:
            last_touch_lista_pago.append(lt)

last_touch_lista_pago
df_alpha_grano_pago = df_alpha_grano_pago[df_alpha_grano_pago['FAMILIA'].isin(familia_sm)].reset_index(drop = True) # Reducción de df_alpha_grano_pago
df_alpha_grano_pago = df_alpha_grano_pago[df_alpha_grano_pago['TIPO_MEDIO'].isin(last_touch_lista_pago)].reset_index(drop = True)
tm_seleccion = list(df_alpha_grano_pago['TIPO_MEDIO'].unique())
df_P3 = tf.parquet_act(f'{dir_data}Parametros/P3')
df_P3['DATE'] = pd.to_datetime(df_P3['DATE'])
df_P3 = df_P3[df_P3['PAIS'].isin(paises)]
df_alpha_grano_organico = df_P3[campos_grano].drop_duplicates()
df_alpha_grano_organico = df_alpha_grano_organico.sort_values(campos_grano).reset_index(drop = True)
df_alpha_grano_organico = df_alpha_grano_organico[df_alpha_grano_organico[campos_grano].notna().all(axis = 1)].reset_index(drop = True) # alpha_tongo (alpha_grano)
df_alpha_grano_organico
df_alpha_grano_organico.FAMILIA.unique()
lista_LT_organico = df_alpha_grano_organico[campo_last_touch].unique()

if '*' in last_touch_lista:
    last_touch_lista_organico = lista_LT_organico
else:
    last_touch_lista_organico = []
    for lt in lista_LT_organico:
        if lt in last_touch_lista:
            last_touch_lista_organico.append(lt)

last_touch_lista_organico
df_alpha_grano_organico = df_alpha_grano_organico[df_alpha_grano_organico['FAMILIA'].isin(familia_sm)].reset_index(drop = True)
df_alpha_grano_organico = df_alpha_grano_organico[df_alpha_grano_organico['TIPO_MEDIO'].isin(last_touch_lista_organico)].reset_index(drop = True)
tm_seleccion += list(df_alpha_grano_organico['TIPO_MEDIO'].unique())
print(tm_seleccion)
df_beta = df_P2[['PAIS', 'DATE', 'CANAL', 'FUENTE']].drop_duplicates()

# Filtros

df_beta = df_beta[df_beta['PAIS'].isin(paises)]
df_beta = df_beta[(df_beta['DATE'] >= dias_proyeccion[0]) & (df_beta['DATE'] <= dias_proyeccion[1])]
df_beta = df_beta[df_beta['FUENTE'].isin(lista_fuentes)]
df_beta = df_beta[df_beta['CANAL'].isin(lista_canales)]
df_beta = df_beta.sort_values(['PAIS', 'DATE', 'CANAL', 'FUENTE']).reset_index(drop = True)
df_beta # beta
df_times, t0 = medir_tiempo(t0, '1.1 Conjuntos', df_times)
### Seccion: # 1.2. Data Histórica
def list_to_gcp(lista):
    s = '('
    for i in lista:
        s += f'"{i}",'
    s = s[:-1] + ')'
    return s
def request_df_duplicidades(tf, client, tabla_data, duplicacion, paises, dia_desde, dia_hasta, diccionario_dimensiones, dia_0, df_factores_PE, co_sin_fcom = True):
    
    if dia_desde >= str(dt.datetime.today().date()): # >= hoy
        return pd.DataFrame()
    # CO sin FCOM
    str_co = ''
    if co_sin_fcom:
        str_co = 'and not (PAIS = "CO" and FUENTE != "SODIMAC")'
    
    # otras dimensiones (que dependen de la duplicacion)
    
    str_dimensiones, str_numeros = 'PAIS, DATE, CANAL, FUENTE, ', '1, 2, 3, 4, '
    for i, elemento in enumerate(duplicacion.split('-')):
        if elemento == 'TOTALES':
            break
        str_dimensiones += diccionario_dimensiones[elemento] + ', '
        str_numeros += f'{i + 5}, '
    
    str_dimensiones, str_numeros = str_dimensiones[:-2], str_numeros[:-2]
    
    """
    ejecutar_PE = False
    if 'PE' in paises:
        ejecutar_PE = True
    """
        
    #paises = list(set(paises) - {'PE'}) # paises sin PE, mientras que está parchada APP SODIMAC PE

    # paises          
    paises = list_to_gcp(paises)      
    
    # query   
    # dia_0 se ocupa por defecto, en caso de que no exista df_representantes y quiera actualizarse por completo  
    """  
    if ejecutar_PE:
        
        query = f'SELECT {str_dimensiones}, SUM(VENTA_COLOCADA) AS VENTA_COLOCADA, SUM(ORDENES) AS ORDENES, SUM(VISITAS) AS VISITAS FROM `{tabla_data}_{duplicacion}` where PAIS in {paises} {str_co} and FUENTE_DATOS = "REAL" and DATE >= "{dia_0}" and DATE >= "{dia_desde}" and DATE <= "{dia_hasta}" group by {str_numeros} order by 1, 2' # main request para entrenamiento
        print(query)
        
        dia_hasta = pd.to_datetime(str(dia_hasta)[:10]).date()
        dia_desde_definitivo = max(pd.to_datetime(str(dia_0)[:10]), pd.to_datetime(str(dia_desde)[:10])).date()
        print('desde, hasta', dia_desde_definitivo, dia_hasta)
        
        if ((dt.datetime(2024, 5, 8).date() >= dia_desde_definitivo) and (dt.datetime(2024, 5, 8).date() <= dia_hasta)) or ((dt.datetime(2024, 6, 8).date() >= dia_desde_definitivo) and (dt.datetime(2024, 6, 8).date() <= dia_hasta)):
            
        #sys.exit('desarrollar request para PE')
    """
                                                                                                                                                                                                                                                                    
    query = f'SELECT {str_dimensiones}, SUM(VENTA_COLOCADA) AS VENTA_COLOCADA, SUM(ORDENES) AS ORDENES, SUM(VISITAS) AS VISITAS, SUM(INVERSION) AS INVERSION FROM `{tabla_data}_{duplicacion}` where PAIS in {paises} {str_co} and FUENTE_DATOS = "REAL" and DATE >= "{dia_0}" and DATE >= "{dia_desde}" and DATE <= "{dia_hasta}" group by {str_numeros} order by 1, 2' # main request para entrenamiento
    
    # request y df
    df =  tf.request_GCP_vnew(
            nombre_tabla = "",
            specific_query = query,
            client = client,
            output = True, permitir_fallos = False)
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Corrección con factores PE cuando corresponde
    df = df.merge(df_factores_PE, on = ['PAIS', 'DATE', 'CANAL', 'FUENTE'], how = 'left')
    df['FACTOR'] = df['FACTOR'].fillna(1)
    df['VENTA_COLOCADA'] = df['VENTA_COLOCADA'] * df['FACTOR']
    df = df.drop(columns = ['FACTOR'])
    return df

dir_data_back = f'{dir_data}Back/'
print(dir_data_back) # Donde almacenar

df_representantes_base = pd.DataFrame(columns = ['DUPLICACION', 'PAIS', 'DATE'])
if 'Horus_data.pkl' in os.listdir(dir_data_back):
    print('SI')
    df_representantes_base = tf.parquet_act(f'{dir_data_back}Horus_data')
df_representantes_base
df_representantes = pd.DataFrame({'REPRESENTANTE': lista_duplicidades})
df_representantes['DIA_DESDE'], df_representantes['DIA_HASTA'] = str(dias_proyeccion[0])[:10], str(dias_proyeccion[1])[:10]

df_paises = pd.DataFrame({'PAIS': paises})
df_paises['AUX'] = 'aux'

co_sin_fcom = True
paises_no_andes = ['BR', 'MX', 'AR', 'UY']
paises_andes = ['CL', 'PE']
if co_sin_fcom:
    paises_no_andes.append('CO')

df_representantes['AUX'] = 'aux'
df_representantes = df_representantes.merge(df_paises, on = 'AUX')
df_representantes = df_representantes.drop(columns = ['AUX'])
df_representantes = df_representantes[['PAIS', 'REPRESENTANTE', 'DIA_DESDE', 'DIA_HASTA']]
df_representantes = df_representantes[~((df_representantes['PAIS'].isin(paises_no_andes)) & (df_representantes['REPRESENTANTE'].isin(['F1', 'G1'])))].reset_index(drop = True)
df_representantes
actualizar_data_historica = False # Parámetro
tabla_data = 'sod-corp-plp-beta.ETL_main_2023.Nexus_Main_230829_0_v10'
df_factores_PE = tf.lectura_segura_Gsheets('Factor_correccion_app_PE', 'Factores', refrescar_lectura = refrescar_lectura)
df_factores_PE['DATE'] = pd.to_datetime(df_factores_PE['DATE'])
df_factores_PE['FACTOR'] = df_factores_PE['FACTOR'].astype(float)
df_factores_PE
df_representantes
set(df_representantes_base.columns)
alguna_ejecucion = False

for i in range(len(df_representantes)):
    representante, dia_desde, dia_hasta, pais = df_representantes.loc[i, 'REPRESENTANTE'], df_representantes.loc[i, 'DIA_DESDE'], df_representantes.loc[i, 'DIA_HASTA'], df_representantes.loc[i, 'PAIS']
    print(representante, dia_desde, dia_hasta, pais)
    
    df_dates = pd.date_range(start = dia_desde, end = dia_hasta, freq = 'D')
    df_dates = pd.DataFrame({'DATE': df_dates})
    df_dates['AUX'] = 'aux'
    
    df_pais = pd.DataFrame({'PAIS': [pais]})
    df_pais['AUX'] = 'aux'
    
    df_req = df_dates.merge(df_pais, on = 'AUX')
    df_req = df_req.drop(columns = ['AUX'])
    df_req['DUPLICACION'] = representante

    if not actualizar_data_historica:
        df_existente = df_representantes_base[['DUPLICACION', 'PAIS', 'DATE']].drop_duplicates().reset_index(drop = True)
        df_existente['EXISTE'] = True
        
        df_req = df_req.merge(df_existente, on = ['DUPLICACION', 'PAIS', 'DATE'], how = 'left')
        df_req['EXISTE'] = df_req['EXISTE'].fillna(False)
    else:
        df_req['EXISTE'] = False
        
    df_req = df_req[~df_req['EXISTE']]

    if len(df_req) == 0: # Están todos los datos actualizados
        continue
        
    min_date, max_date, paises_ejecucion = str(df_req['DATE'].min())[:10], str(df_req['DATE'].max())[:10], df_req['PAIS'].unique()
    
    df_ejecucion = request_df_duplicidades(tf, client, tabla_data, representante, paises_ejecucion, min_date, max_date, diccionario_dimensiones, dia_desde, df_factores_PE)
    
    if len(df_ejecucion) > 0:
        alguna_ejecucion = True
    df_ejecucion['DUPLICACION'] = representante
    #print(df_ejecucion)
    df_representantes_base = pd.concat([df_representantes_base, df_ejecucion])
    df_representantes_base['DATE'] = pd.to_datetime(df_representantes_base['DATE'])
    df_representantes_base = df_representantes_base.drop_duplicates(subset = list(set(df_representantes_base.columns) - {'VENTA_COLOCADA', 'ORDENES', 'VISITAS'}), keep = 'last').reset_index(drop = True) # Se mantienen con prioridad los nuevos datos

if alguna_ejecucion:
    df_representantes_base = df_representantes_base.fillna("")
    tf.parquet_act(f'{dir_data_back}Horus_data', variable = df_representantes_base, mode = 'save') # Respaldo de la nueva data
df_times, t0 = medir_tiempo(t0, '1.2 Data Hist', df_times)
### Seccion: # 1.3. Parámetros
# Lectura general de parámetros
for archivo in os.listdir(f'{dir_data}Parametros/'):
    break # Por ahora, esto es solo revisión
    print(archivo)
    df = tf.parquet_act(f'{dir_data}Parametros/{archivo[:-4]}')
    print(df.head(2))
# Las metas están disponibles en los archivos
dir_metas = '../Data/metas/'

for archivo in os.listdir(dir_metas):
    print(archivo)
    df = tf.parquet_act(f'{dir_metas}{archivo[:-4]}')
    print(df.head())
campos_beta = ['PAIS', 'DATE', 'CANAL', 'FUENTE']
dic_parametros = {}
campos_beta_def = campos_beta
if modo_tactico:
    campos_beta_def = campos_beta + ['PERIODO']
    campos_beta_def.remove('DATE')


df_beta_def = df_beta.copy()
if modo_tactico:
    df_beta_def['PERIODO'] = df_beta_def['DATE'].astype(str).str[:7]
    #df_beta_def['PERIODO'] = np.where(df_beta_def['PERIODO'] == str(ultimo_dia_real)[:7], np.where(df_beta_def['DATE'] <= ultimo_dia_real, df_beta_def['PERIODO'] + '_REAL', df_beta_def['PERIODO'] + '_PREDICT'), df_beta_def['PERIODO'])
    df_beta_def = df_beta_def[campos_beta_def].drop_duplicates().reset_index(drop = True)

df_beta_def
### Seccion: ## 1.3.1 Parámetros TC & TP [1-5]
### Subseccion: Raw parámetros 1-5
paises
df_real = df_representantes_base[df_representantes_base['DUPLICACION'] == grano]
df_real['TC'], df_real['TP'] = df_real['ORDENES'] / df_real['VISITAS'], df_real['VENTA_COLOCADA'] / df_real['ORDENES']
df_real = df_real[campos_beta + campos_grano + ['VISITAS', 'TC', 'TP']]
df_real[['VISITAS', 'TC', 'TP']] = df_real[['VISITAS', 'TC', 'TP']].fillna(0)
df_real

df_real_visitas = df_real[campos_beta + campos_grano + ['VISITAS']]
df_real_visitas['METRICA'] = 'VISITAS'
df_real_visitas = df_real_visitas.rename(columns = {'VISITAS': 'VALOR_REAL'})

df_real_tc = df_real[campos_beta + campos_grano + ['TC']]
df_real_tc['METRICA'] = 'TC'
df_real_tc = df_real_tc.rename(columns = {'TC': 'VALOR_REAL'})

df_real_tp = df_real[campos_beta + campos_grano + ['TP']]
df_real_tp['METRICA'] = 'TP'
df_real_tp = df_real_tp.rename(columns = {'TP': 'VALOR_REAL'})

df_real = pd.concat([df_real_visitas, df_real_tc, df_real_tp])
df_real
#ultimo_dia_real = df_real.DATE.max()

def tactico(df, ajustar_ratios, modo_tactico):
    
    if not modo_tactico:
        df['y'] = df['y'].fillna(0)
        return df
    
    if ajustar_ratios[0]:
        existen_visitas = True
        
        columns_dims = set(df.columns)
        columns_dims = columns_dims - set([ajustar_ratios[1]]) - set(['y'])
        print(columns_dims)

        df = df.pivot_table(index = list(columns_dims), columns = ajustar_ratios[1], values = ajustar_ratios[2]).reset_index()
        if 'VISITAS' not in df.columns:
            df['VISITAS'] = 1
            existen_visitas = False
        
        df['ORDENES'] = df['TC'] * df['VISITAS']
        df['VENTA_COLOCADA'] = df['TP'] * df['ORDENES']
        df = df[list(columns_dims) + ['VISITAS', 'ORDENES', 'VENTA_COLOCADA']]
        df[['VISITAS', 'ORDENES', 'VENTA_COLOCADA']] = df[['VISITAS', 'ORDENES', 'VENTA_COLOCADA']].fillna(0)
        df['PERIODO'] = df['DATE'].astype(str).str[:7]
        
        #periodo_actual = str(ultimo_dia_real)[:7]
        #df['PERIODO'] = np.where(df['PERIODO'] == periodo_actual, np.where(df['DATE'] <= ultimo_dia_real, df['PERIODO'] + '_REAL', df['PERIODO'] + '_PREDICT'), df['PERIODO'])
        
        columns_dims_periodo = list(columns_dims) + ['PERIODO']
        columns_dims_periodo.remove('DATE')
        
        df = df[columns_dims_periodo + ['VISITAS', 'ORDENES', 'VENTA_COLOCADA']].groupby(columns_dims_periodo, as_index = False).sum()
        
        df['TC'] = df['ORDENES'] / df['VISITAS']
        df['TP'] = df['VENTA_COLOCADA'] / df['ORDENES']
        df = df.drop(columns = ['ORDENES', 'VENTA_COLOCADA'])
        df = df.melt(id_vars = columns_dims_periodo, var_name = 'METRICA', value_name = 'y')
        
        if not existen_visitas:
            df = df[df['METRICA'] != 'VISITAS'].reset_index(drop = True)
        df['y'] = df['y'].fillna(0)

    return df
dia_desde, dia_hasta
dic_presupuesto = {True: '_ajustado', False: ''}

# Orgánicos
dfP3 = tf.parquet_act(f'{dir_data}Parametros/P3{dic_presupuesto[ajuste_metricas]}')
dfP3 = dfP3[dfP3['PAIS'].isin(paises)].reset_index(drop = True)

dfP3 = dfP3[(dfP3['DATE'] >= dias_proyeccion[0]) & (dfP3['DATE'] <= dias_proyeccion[1])].reset_index(drop = True)
dfP3 = dfP3[dfP3['MODO_PROYECCION'] == modo_proyeccion].reset_index(drop = True) # Presupuesto o libre, según ajustar métricas
dfP3 = dfP3.merge(df_real, how = 'left', on = campos_beta + campos_grano + ['METRICA'])

dfP3['y2'] = dfP3['real'].fillna(dfP3['predict'])
dfP3['y'] = dfP3['VALOR_REAL'].fillna(dfP3['y2'])
dfP3 = dfP3[campos_beta + campos_grano + ['METRICA', 'y']]
dfP3 = tactico(dfP3, [True, 'METRICA', 'y'], modo_tactico)
print(dfP3)
#sys.exit('Crear función f: tactico, que permite mensualizar los registros diarios')

dic_parametros['TCO'] = dfP3[dfP3['METRICA'] == 'TC'][campos_beta_def + campos_grano + ['y']] # P1
dic_parametros['TPO'] = dfP3[dfP3['METRICA'] == 'TP'][campos_beta_def + campos_grano + ['y']] # P3
dic_parametros['VO'] = dfP3[dfP3['METRICA'] == 'VISITAS'][campos_beta_def + campos_grano + ['y']] # P5
# Pagos

dfP2 = tf.parquet_act(f'{dir_data}Parametros/P2')
dfP2 = dfP2[dfP2['PAIS'].isin(paises)].reset_index(drop = True)
dfP2 = dfP2[(dfP2['DATE'] >= dias_proyeccion[0]) & (dfP2['DATE'] <= dias_proyeccion[1])].reset_index(drop = True)
dfP2 = dfP2.merge(df_real, how = 'left', on = campos_beta + campos_grano + ['METRICA'])
dfP2['y2'] = dfP2['REAL'].fillna(dfP2['PREDICCION'])
dfP2['y'] = dfP2['VALOR_REAL'].fillna(dfP2['y2'])
dfP2 = dfP2[campos_beta + campos_grano + ['METRICA', 'y']]
dfP2 = tactico(dfP2, [True, 'METRICA', 'y'], modo_tactico)
dic_parametros['TCP'] = dfP2[dfP2['METRICA'] == 'TC'][campos_beta_def + campos_grano + ['y']] # P2
dic_parametros['TPP'] = dfP2[dfP2['METRICA'] == 'TP'][campos_beta_def + campos_grano + ['y']] # P4
### Subseccion: Deben existir para los conjuntos alpha grano y beta
def list_to_gcp(lista):
    s = '('
    for i in lista:
        s += f'"{i}",'
    s = s[:-1] + ')'
    return s


str_paises = list_to_gcp(paises)

tabla_data_all = tabla_data[:-2] + '9'
tabla_data_all

query = f'select distinct DUPLICACION, PAIS, CANAL, FUENTE, FAMILIA, SUBFAMILIA, TIPO_MEDIO, NATURALEZA_MEDIO from {tabla_data_all} where FUENTE_DATOS = "REAL" and pais in {str_paises}'

df_alpha_beta_duplicacion = tf.request_GCP_vnew(
                            nombre_tabla = "",
                            specific_query = query,
                            client = client,
                            output = False, permitir_fallos = False)

df_alpha_beta_duplicacion['AUX'] = 'aux'
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion.merge(df_dates, on = 'AUX')
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion.drop(columns = ['AUX'])
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion[df_alpha_beta_duplicacion['DUPLICACION'].isin(lista_duplicidades)].reset_index(drop = True)
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion.sort_values(['DUPLICACION', 'PAIS', 'DATE', 'CANAL', 'FUENTE', 'FAMILIA', 'SUBFAMILIA', 'TIPO_MEDIO', 'NATURALEZA_MEDIO']).reset_index(drop = True)
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion[['DUPLICACION', 'PAIS', 'DATE', 'CANAL', 'FUENTE', 'FAMILIA', 'SUBFAMILIA', 'TIPO_MEDIO', 'NATURALEZA_MEDIO']]

print(df_alpha_beta_duplicacion.DUPLICACION.unique())
if modo_tactico:
    df_alpha_beta_duplicacion['PERIODO'] = df_alpha_beta_duplicacion['DATE'].astype(str).str[:7]
    df_alpha_beta_duplicacion = df_alpha_beta_duplicacion.drop(columns = ['DATE']).drop_duplicates().reset_index(drop = True)
    
# Limpieza
print(df_alpha_beta_duplicacion.DUPLICACION.unique())
# Si duplicacion = F y FAMILIA = "", excluir
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion[~((df_alpha_beta_duplicacion['DUPLICACION'] == 'F') & (df_alpha_beta_duplicacion['FAMILIA'] == ''))].reset_index(drop = True)
# Si duplicacion = F-LT y (familia = "" o tipo_medio = "") excluir
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion[~((df_alpha_beta_duplicacion['DUPLICACION'] == 'F-LT') & ((df_alpha_beta_duplicacion['FAMILIA'] == '') | (df_alpha_beta_duplicacion['TIPO_MEDIO'] == '')))].reset_index(drop = True)
# Si duplicacion = F-SF y (familia = "" o subfamilia = "") excluir
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion[~((df_alpha_beta_duplicacion['DUPLICACION'] == 'F-SF') & ((df_alpha_beta_duplicacion['FAMILIA'] == '') | (df_alpha_beta_duplicacion['SUBFAMILIA'] == '')))].reset_index(drop = True)
# Si duplicacion = LT y tipo_medio = "" excluir
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion[~((df_alpha_beta_duplicacion['DUPLICACION'] == 'LT') & (df_alpha_beta_duplicacion['TIPO_MEDIO'] == ''))].reset_index(drop = True)
print(df_alpha_beta_duplicacion.DUPLICACION.unique())
df_alpha_beta_duplicacion = df_alpha_beta_duplicacion[(df_alpha_beta_duplicacion['FAMILIA'].isin(list(familia_sm) + [''])) & (df_alpha_beta_duplicacion['TIPO_MEDIO'].isin(list(tm_seleccion) + ['']))].reset_index(drop = True)
df_alpha_beta_duplicacion = df_beta_def.merge(df_alpha_beta_duplicacion, on = campos_beta_def, how = 'left')
df_alpha_beta_duplicacion
print(df_alpha_beta_duplicacion.DUPLICACION.unique())
df_alpha_beta_duplicacion_grano = df_alpha_beta_duplicacion[df_alpha_beta_duplicacion['DUPLICACION'] == grano].reset_index(drop = True)
df_alpha_beta_duplicacion_grano = df_alpha_beta_duplicacion_grano[campos_beta_def + campos_grano]
df_alpha_beta_duplicacion_grano['IN'] = True
df_alpha_beta_duplicacion_grano
"""
df_alpha_beta_duplicacion_grano_def = df_alpha_beta_duplicacion_grano.copy()
if modo_tactico:
    df_alpha_beta_duplicacion_grano_def['PERIODO'] = df_alpha_beta_duplicacion_grano_def['DATE'].astype(str).str[:7]
    #df_alpha_beta_duplicacion_grano_def['PERIODO'] = np.where(df_alpha_beta_duplicacion_grano_def['PERIODO'] == str(ultimo_dia_real)[:7], np.where(df_alpha_beta_duplicacion_grano_def['DATE'] <= ultimo_dia_real, df_alpha_beta_duplicacion_grano_def['PERIODO'] + '_REAL', df_alpha_beta_duplicacion_grano_def['PERIODO'] + '_PREDICT'), df_alpha_beta_duplicacion_grano_def['PERIODO'])
    df_alpha_beta_duplicacion_grano_def = df_alpha_beta_duplicacion_grano_def.drop(columns = ['DATE']).drop_duplicates().reset_index(drop = True)

df_alpha_beta_duplicacion_grano_def
"""
df_beta_def['AUX'] = 'aux'
df_alpha_grano_organico['AUX'] = 'aux'
df_alpha_grano_pago['AUX'] = 'aux'

df_beta_alpha_grano_organico = df_beta_def.merge(df_alpha_grano_organico, on = 'AUX')
df_beta_alpha_grano_pago = df_beta_def.merge(df_alpha_grano_pago, on = 'AUX')

for df in [df_beta_def, df_alpha_grano_organico, df_alpha_grano_pago, df_beta_alpha_grano_organico, df_beta_alpha_grano_pago]:
    df.drop(columns = ['AUX'], inplace = True) # Con inplace para que funcione, ya que df es variable iterativa
df_beta_alpha_grano_organico.head()
# Solo los casos que existen en df_alpha_beta_duplicacion_grano
df_beta_alpha_grano_organico = df_beta_alpha_grano_organico.merge(df_alpha_beta_duplicacion_grano, on = campos_beta_def + campos_grano, how = 'left')
df_beta_alpha_grano_pago = df_beta_alpha_grano_pago.merge(df_alpha_beta_duplicacion_grano, on = campos_beta_def + campos_grano, how = 'left')

df_beta_alpha_grano_organico['IN'] = df_beta_alpha_grano_organico['IN'].fillna(False)
df_beta_alpha_grano_pago['IN'] = df_beta_alpha_grano_pago['IN'].fillna(False)

df_beta_alpha_grano_organico = df_beta_alpha_grano_organico[df_beta_alpha_grano_organico['IN']].reset_index(drop = True)
df_beta_alpha_grano_pago = df_beta_alpha_grano_pago[df_beta_alpha_grano_pago['IN']].reset_index(drop = True)

df_beta_alpha_grano_organico = df_beta_alpha_grano_organico.drop(columns = ['IN'])
df_beta_alpha_grano_pago = df_beta_alpha_grano_pago.drop(columns = ['IN'])
df_beta_alpha_grano_organico
for m in ['TCO', 'TPO', 'VO', 'TCP', 'TPP']:
    df = dic_parametros[m]
    if m[-1:] == 'O':
        df = df_beta_alpha_grano_organico.merge(df, on = campos_beta_def + campos_grano, how = 'left')
    else:
        df = df_beta_alpha_grano_pago.merge(df, on = campos_beta_def + campos_grano, how = 'left')
    df['y'] = df['y'].fillna(0)
    dic_parametros[m] = df
    print(m,  m[-1:])
    print(df.head())

df_times, t0 = medir_tiempo(t0, '1.3.1 Pars 1-5', df_times)
### Seccion: ## 1.3.3 Relaciones duplicidad [8]
dias_proyeccion
df = tf.parquet_act(f'{dir_data}Parametros/P1')
df = df[(df['DATE'] >= dias_proyeccion[0]) & (df['DATE'] <= dias_proyeccion[1])].reset_index(drop = True)
df = df[df['PAIS'].isin(paises)].reset_index(drop = True)
df_arcos = df[['AGREGADO', 'DESAGREGADO']].drop_duplicates().reset_index(drop = True)
df_arcos = df_arcos[(df_arcos['AGREGADO'].isin(lista_duplicidades)) & (df_arcos['DESAGREGADO'].isin(lista_duplicidades))]

# Para evitar ciclos en el grafo (doble-dependencias)
df_arcos =  df_arcos[df_arcos['AGREGADO'] != ""]
df_arcos = df_arcos[~((df_arcos['AGREGADO'] == 'LT') & (df_arcos['DESAGREGADO'] == 'F-LT'))].reset_index(drop = True)
df = df[~((df['AGREGADO'] == 'LT') & (df['DESAGREGADO'] == 'F-LT'))]
print(df_arcos)
df
if modo_tactico:
    df['PERIODO'] = df['DATE'].astype(str).str[:7]

df = df[['AGREGADO', 'DESAGREGADO'] + campos_beta_def + campos_grano + ['VENTA_COLOCADA_AGREGADO', 'VENTA_COLOCADA_DESAGREGADO', 'ORDENES_AGREGADO', 'ORDENES_DESAGREGADO', 'VISITAS_AGREGADO', 'VISITAS_DESAGREGADO']].groupby(['AGREGADO', 'DESAGREGADO'] + campos_beta_def + campos_grano, as_index = False).sum()

for m in ['VENTA_COLOCADA', 'ORDENES', 'VISITAS']:
    df[f'f_{m}'] = df[f'{m}_AGREGADO'] / df[f'{m}_DESAGREGADO']

df
### Subseccion: grafo
import networkx as nx
import matplotlib.pyplot as plt

# Crear un grafo dirigido
G = nx.DiGraph()

# Agregar los nodos y las aristas al grafo
for agregado, desagregado in zip(df_arcos["AGREGADO"], df_arcos["DESAGREGADO"]):
    G.add_edge(desagregado, agregado)

# Dibujar el grafo
plt.figure(figsize = (8, 6))
pos = nx.spring_layout(G)  # Posiciones de los nodos para un diseño agradable
nx.draw_networkx_nodes(G, pos, node_size = 500, node_color = "lightblue")
nx.draw_networkx_edges(G, pos, arrowstyle = "->", arrowsize = 20, edge_color = "gray")
nx.draw_networkx_labels(G, pos, font_size = 10, font_color = "black")

# Mostrar el grafo
plt.title("Grafo: AGREGADO y DESAGREGADO", fontsize = 14)
plt.axis("off")
plt.show()
dic_parametros['RELACIONES DUPLICIDAD'] = df
df_times, t0 = medir_tiempo(t0, '1.3.3 Pars 8', df_times)
### Seccion: ## 1.3.4 Metas [9]
paises
df_relaciones_canal_fuente = pd.DataFrame({'CANAL_BASE': ['WEB', 'APP', 'FCOM', 'FCOM', 'FCOM', 'FCOM'],
                              'CANAL': ['WEB', 'APP', 'WEB', 'APP', 'WEB', 'APP'],
                              'FUENTE': ['SODIMAC', 'SODIMAC', 'SIS', 'SIS', 'ES', 'ES']})
df_relaciones_canal_fuente
### Subseccion: 0_df_resumen_metas_general
df = tf.parquet_act(f'{dir_metas}0_df_resumen_metas_general')

if not modo_tactico:
    df_pais_t = df_beta_def[['DATE', 'PAIS']].drop_duplicates().reset_index(drop = True)
    df_pais_t['PERIODO'] = df_pais_t['DATE'].astype(str).str[:7]
    
else:
    df_pais_t = df_beta_def[['PERIODO', 'PAIS']].drop_duplicates().reset_index(drop = True)
    
df_pais_t['IN'] = True
df = df.merge(df_pais_t, on = ['PERIODO', 'PAIS'], how = 'left')

df['IN'] = df['IN'].fillna(False)
df = df[df['IN']].reset_index(drop = True)
df = df.drop(columns = ['IN'])

df['RATIO_NETA_COLOCADA'] = df['VENTA_NETA'] / df['VENTA_COLOCADA']
df['TP'] = df['VENTA_COLOCADA'] / df['ORDENES']
df['TC'] = df['ORDENES'] / df['VISITAS']
df
dic_parametros['METAS'] = {}
# Ratio VN / VC por df_beta

df_vnvc = df[['PAIS', 'CANAL', 'PERIODO', 'RATIO_NETA_COLOCADA']]

#year0, yearF = int(df['PERIODO'].min()[:4]), int(df['PERIODO'].max()[:4])
# dataframe con todas las fechas en estos años

#df_dates = pd.date_range(start = f'{year0}-01-01', end = f'{yearF}-12-31', freq = 'D')
#df_dates = pd.DataFrame({'DATE': df_dates})
#df_dates['PERIODO'] = df_dates['DATE'].astype(str).str[:7]
#df_dates

df_fcom = df_vnvc[df_vnvc['CANAL'] == 'FCOM'][['PAIS', 'PERIODO', 'RATIO_NETA_COLOCADA']]
df_sodimac = df_vnvc[df_vnvc['CANAL'] != 'FCOM']

# fcom
if len(df_fcom) > 0:
    df_fcom['AUX'] = 'aux'
    df_fuente_canal_fcom = pd.DataFrame({'CANAL': ['APP', 'APP', 'WEB', 'WEB'], 'FUENTE': ['ES', 'SIS', 'ES', 'SIS']})
    df_fuente_canal_fcom['AUX'] = 'aux'
    df_fcom = df_fcom.merge(df_fuente_canal_fcom, on = 'AUX')
    df_fcom = df_fcom.drop(columns = ['AUX'])
    #df_fcom = df_dates.merge(df_fcom, on = ['PERIODO'], how = 'left')
    df_fcom = df_fcom[['PAIS', 'PERIODO', 'CANAL', 'FUENTE', 'RATIO_NETA_COLOCADA']]

# sodimac (azul)
df_sodimac['FUENTE'] = 'SODIMAC'
#df_sodimac = df_dates.merge(df_sodimac, on = ['PERIODO'], how = 'left')
df_sodimac = df_sodimac[['PAIS', 'PERIODO', 'CANAL', 'FUENTE', 'RATIO_NETA_COLOCADA']]

# concat
df_ratio_neta_colocada = pd.concat([df_fcom, df_sodimac])[['PAIS', 'PERIODO', 'CANAL', 'FUENTE', 'RATIO_NETA_COLOCADA']].reset_index(drop = True)
df_ratio_neta_colocada.columns.name = None
dic_parametros['METAS']['RATIO NETA COLOCADA'] = df_ratio_neta_colocada
df_ratio_neta_colocada.head()
# ordenes & visitas
for metrica in ['VENTA_NETA', 'VENTA_COLOCADA', 'ORDENES', 'VISITAS', 'TP', 'TC']:
    df_metrica = df[['PAIS', 'CANAL', 'PERIODO', metrica]].rename(columns = {'CANAL': 'CANAL_BASE'})
    
    metrica0 = metrica
    if metrica == 'VENTA_NETA':
        metrica0 = 'VENTA_NETA_BASE'
    dic_parametros['METAS'][metrica0] = df_metrica
    print(df_metrica.head())
### Subseccion: 1_df_resumen_metas_otros
df = tf.parquet_act(f'{dir_metas}1_df_resumen_metas_otros')
df = df.merge(df_pais_t, on = ['PERIODO', 'PAIS'], how = 'left')
df['IN'] = df['IN'].fillna(False)
df = df[df['IN']].reset_index(drop = True)
df = df.drop(columns = ['IN'])
df
# Impuesto

df_impuesto = df[['PAIS', 'PERIODO', 'IMPUESTO']].reset_index(drop = True)
df_impuesto.columns.name = None
dic_parametros['METAS']['IMPUESTO'] = df_impuesto
df_impuesto.head()
# Inversion
df_inversion = df[['PAIS', 'PERIODO', 'INVERSION']].reset_index(drop = True)
df_inversion.columns.name = None
dic_parametros['METAS']['INVERSION'] = df_inversion
df_inversion.head()
# % Tráfico Orgánico
df_trafico_organico = df[['PAIS', 'PERIODO', 'PORCENTAJE_TRAFICO_ORGANICO']].reset_index(drop = True)
df_trafico_organico['PORCENTAJE_TRAFICO_ORGANICO'] *= (1 / 100)
df_trafico_organico.columns.name = None
dic_parametros['METAS']['PORCENTAJE_TRAFICO_ORGANICO'] = df_trafico_organico
df_trafico_organico.head()
### Subseccion: 2_df_resumen_metas_detalle
df = tf.parquet_act(f'{dir_metas}2_df_resumen_metas_detalle')
df = df.merge(df_pais_t, on = ['PERIODO', 'PAIS'], how = 'left')
df['IN'] = df['IN'].fillna(False)
df = df[df['IN']].reset_index(drop = True)
df_metas_venta = df[['PAIS', 'FAMILIA', 'SUBFAMILIA', 'PERIODO', 'VENTA_NETA']].reset_index(drop = True)
dic_parametros['METAS']['VENTA_NETA'] = df_metas_venta
df_metas_venta
dic_parametros['METAS'].keys()
# Solo revisión
for m in dic_parametros['METAS'].keys():
    print('m', m)
    df = dic_parametros['METAS'][m]
    df = df[df['PERIODO'] >= "2025-01"]
    print(df.head())
df_times, t0 = medir_tiempo(t0, '1.3.4 Pars 9', df_times)
### Seccion: ## 1.3.5. Matriz de relación tráfico IN - tráfico Total [10]
### Subseccion: Data visitas IN
dir_data = '../Data/'
dir_data_horus = '../Data/Horus/'
df_data_visitas = pd.DataFrame()
if 'df_data_visitas.pkl' in os.listdir(dir_data_horus):
    df_data_visitas = tf.parquet_act(f'{dir_data_horus}df_data_visitas')
df_data_visitas

if len(df_data_visitas) > 0:
    df_data_visitas_existentes = df_data_visitas[['DATE', 'PAIS']].drop_duplicates().reset_index(drop = True)
    df_data_visitas_existentes['DATE'] = pd.to_datetime(df_data_visitas_existentes['DATE'])
    df_data_visitas_existentes['EXISTE'] = True

df_data_visitas_existentes
sys.path.insert(0, '')
from dias_comparacion import dias_comparacion 
query_max_date = f'select max(DATE) as DATE from `{tabla_data}_{grano}` where FUENTE_DATOS = "REAL"'

df_max_date = tf.request_GCP_vnew(
    nombre_tabla = "",
    specific_query = query_max_date,
    client = client,
    output = False, permitir_fallos = False)

df_max_date['DATE'] = pd.to_datetime(df_max_date['DATE'])

dia_hasta = df_max_date['DATE'][0].date()

dia_desde = (dia_hasta + dt.timedelta(days = 1)).replace(year = dia_hasta.year - 1)#.strftime('%Y-%m-%d')
dia_desde, dia_hasta
dias_proyeccion
df_dias_predecir = dias_comparacion(paises, [dias_proyeccion[0].date(), dias_proyeccion[1].date()], df_dias_eventos, [dia_desde, dia_hasta])
df_dias_predecir = df_dias_predecir[(df_dias_predecir['DATE'] >= dias_proyeccion[0]) & (df_dias_predecir['DATE'] <= dias_proyeccion[1])].reset_index(drop = True)
df_dias_predecir # Asigna a cada día de la proyección, uno que ya exista (histórico)
dia_desde, dia_hasta = df_dias_predecir['DATE_COMPARACION'].min(), df_dias_predecir['DATE_COMPARACION'].max()
dia_hasta = dt.datetime.now().date() - dt.timedelta(days = 1)
dia_desde, dia_hasta
df_dates = pd.date_range(start = dia_desde, end = dia_hasta, freq = 'D')
df_dates = pd.DataFrame({'DATE': df_dates})
df_dates.head()
if len(df_data_visitas) > 0:
    df_data_visitas['DATE'] = pd.to_datetime(df_data_visitas['DATE'])

for pais in paises:
    
    df_dates_pais = df_dates.copy()
    df_dates_pais['PAIS'] = pais
    if len(df_data_visitas) > 0:
        df_dates_pais = df_dates_pais.merge(df_data_visitas_existentes, how = 'left', on = ['DATE', 'PAIS'])
        df_dates_pais['EXISTE'] = df_dates_pais['EXISTE'].fillna(False)
        df_dates_pais = df_dates_pais[df_dates_pais['EXISTE'] != True]
    
    if len(df_dates_pais) == 0: # Si no hay nada, el país está completamente actualizado
        continue
    
    min_date, max_date = str(df_dates_pais['DATE'].min())[:10], str(df_dates_pais['DATE'].max())[:10]
    
    query_detalle = f'PAIS = "{pais}" and DATE >= "{min_date}" and DATE <= "{max_date}"'
    
    # VISITAS IN
    query = f'SELECT * FROM `sod-corp-plp-beta.ETL_main_2023.VisitasIN_241122_0_v2` where {query_detalle} and duplicacion = "{grano}"'

    # request y df
    df_visitas_in =  tf.request_GCP_vnew(
            nombre_tabla = "",
            specific_query = query,
            client = client,
            output = True, permitir_fallos = False)

    df_visitas_in['DATE'] = pd.to_datetime(df_visitas_in['DATE'])
    #df_visitas_in.to_csv(f'{dir_data_horus}1df_visitas_in.csv', index = False, sep = ';', decimal = ',')
    #df_visitas = df_visitas.merge(df_medios, on = 'LAST_TOUCH_CHANNEL', how = 'left')
    #df_visitas = df_visitas[campos_beta + campos_grano + ['VISITAS_IN']].groupby(campos_beta + campos_grano, as_index = False).sum()
    
    df_visitas_in = df_visitas_in.merge(df_medios, on = 'LAST_TOUCH_CHANNEL', how = 'left')
    df_visitas_in = df_visitas_in[campos_beta + campos_grano + ['VISITAS_IN']].groupby(campos_beta + campos_grano, as_index = False).sum()    
    
    # VISITAS
    query = f'SELECT * FROM `sod-corp-plp-beta.ETL_main_2023.Nexus_Main_230829_0_v3` where {query_detalle} and duplicacion = "{grano}"'

    # request y df
    df_visitas =  tf.request_GCP_vnew(
            nombre_tabla = "",
            specific_query = query,
            client = client,
            output = True, permitir_fallos = False)

    df_visitas['DATE'] = pd.to_datetime(df_visitas['DATE'])
    #df_visitas.to_csv(f'{dir_data_horus}2df_visitas.csv', index = False, sep = ';', decimal = ',')
    
    df_visitas = df_visitas.merge(df_medios, on = 'LAST_TOUCH_CHANNEL', how = 'left')
    df_visitas = df_visitas[campos_beta + campos_grano + ['VISITAS']].groupby(campos_beta + campos_grano, as_index = False).sum()
    #df_visitas.to_csv(f'{dir_data_horus}2b_df_visitas.csv', index = False, sep = ';', decimal = ',')
    
    # RATIO
    df_ratio = df_visitas_in.merge(df_visitas, on = campos_beta + campos_grano, how = 'outer')
    df_ratio = df_ratio.fillna(0)
    #df_ratio.to_csv(f'{dir_data_horus}df_ratio.csv', index = False, sep = ';', decimal = ',')
    df_ratio['RATIO'] = np.where(df_ratio['VISITAS'] > 0, df_ratio['VISITAS_IN'] / df_ratio['VISITAS'], 1)
    df_ratio = df_ratio[campos_beta + campos_grano + ['RATIO']]
    df_ratio

    # VISITAS ALL
    query = f'SELECT * FROM `sod-corp-plp-beta.ETL_main_2023.Nexus_Main_230829_0_v10_{grano}` where {query_detalle} and duplicacion = "{grano}" and FUENTE_DATOS = "REAL"'

    # request y df
    df_visitas_all =  tf.request_GCP_vnew(
            nombre_tabla = "",
            specific_query = query,
            client = client,
            output = True, permitir_fallos = False)

    df_visitas_all['DATE'] = pd.to_datetime(df_visitas_all['DATE'])

    df_visitas_all = df_visitas_all[campos_beta + campos_grano + ['VISITAS']].groupby(campos_beta + campos_grano, as_index = False).sum()

    df_visitas_all_final = df_visitas_all.merge(df_ratio, on = campos_beta + campos_grano, how = 'left')
    #df_visitas_all_final.to_csv(f'{dir_data_horus}df_visitas_all_final.csv', index = False, sep = ';', decimal = ',')
    df_visitas_all_final['RATIO'] = df_visitas_all_final['RATIO'].fillna(1)
    df_visitas_all_final['VISITAS_IN'] = df_visitas_all_final['VISITAS'] * df_visitas_all_final['RATIO']
    df_visitas_all_final = df_visitas_all_final.drop(columns = ['RATIO'])
    
    df_visitas_all_final['DATE'] = pd.to_datetime(df_visitas_all_final['DATE'])
    df_data_visitas = pd.concat([df_visitas_all_final, df_data_visitas]).reset_index(drop = True) # Prioridad al nuevo registro
    
    df_data_visitas = df_data_visitas.drop_duplicates(subset = campos_beta + campos_grano, keep = 'first').reset_index(drop = True)

tf.parquet_act(f'{dir_data_horus}df_data_visitas', df_data_visitas, 'save')
    
# https://chatgpt.com/c/673e3b77-464c-8000-a9d7-5bb500da438d
dia_desde, dia_hasta
### Subseccion: Matriz entropía
df_data_visitas = tf.parquet_act(f'{dir_data_horus}df_data_visitas') # Se lee y se filtra df_data_visitas # este para inversion 250109
df_data_visitas['DATE'] = pd.to_datetime(df_data_visitas['DATE'])
df_data_visitas = df_data_visitas[df_data_visitas['PAIS'].isin(paises)]
df_data_visitas = df_data_visitas[(df_data_visitas['DATE'] >= str(dia_desde)) & (df_data_visitas['DATE'] <= str(dia_hasta))].reset_index(drop = True)
df_data_visitas
periodo_temporal = 'DATE'
if modo_tactico:
    periodo_temporal = 'PERIODO'
    
df_beta_alpha_grano_pago_visitas = df_beta_alpha_grano_pago.drop(columns = periodo_temporal).drop_duplicates().reset_index(drop = True)
df_beta_alpha_grano_pago_visitas['AUX'] = 'aux'

df_dates_comp = df_dias_predecir[['DATE_COMPARACION']].drop_duplicates().reset_index(drop = True).rename(columns = {'DATE_COMPARACION': 'DATE'})
df_dates_comp['AUX'] = 'aux'

df_beta_alpha_grano_pago_visitas = df_beta_alpha_grano_pago_visitas.merge(df_dates_comp, on = 'AUX')
df_beta_alpha_grano_pago_visitas = df_beta_alpha_grano_pago_visitas.drop(columns = ['AUX'])

df_beta_alpha_grano_pago_visitas['IN'] = True

df_data_visitas['DATE'], df_beta_alpha_grano_pago_visitas['DATE'] = pd.to_datetime(df_data_visitas['DATE']), pd.to_datetime(df_beta_alpha_grano_pago_visitas['DATE'])
df_data_visitas = df_data_visitas.merge(df_beta_alpha_grano_pago_visitas, on = campos_beta + campos_grano, how = 'left')
df_data_visitas['IN'] = df_data_visitas['IN'].fillna(False)
df_data_visitas = df_data_visitas[df_data_visitas['IN']].reset_index(drop = True)
df_data_visitas = df_data_visitas.drop(columns = ['IN'])
df_data_visitas
#dfa = df_data_visitas[df_data_visitas['DATE'] == "2024-02-02"]
#dfa[dfa.TIPO_MEDIO == 'Reference Domain'].head(50)
iterador = campos_beta + [campo_last_touch]
lista_clacom = list(set(campos_grano) - {campo_last_touch})

print(iterador)
print(lista_clacom)
import tqdm
import numpy as np
from scipy.optimize import minimize
# Función de entropía a maximizar
def entropy(M):
    M = M.reshape(n, n)
    return -np.sum(M[M > 0] * np.log(M[M > 0]))

# Restricción M * a = b
def matrix_vector_constraint(M):
    M = M.reshape(n, n)
    return np.dot(M, a) - b

# Restricción de diagonal fija = 1
def diagonal_constraint(M):
    M = M.reshape(n, n)
    return [M[i, i] - 1 for i in range(n)]

def matriz_max_entropia(df_data_visitas_i, lista_clacom):
    # Configuración para n x n
    global n, a, b
    n = len(df_data_visitas_i['VISITAS_IN'])  # Tamaño de la matriz
    a = np.array(df_data_visitas_i['VISITAS_IN'])
    b = np.array(df_data_visitas_i['VISITAS'])
    
    #a = np.array(a, dtype=float)
    #b = np.array(b, dtype=float)
    
    #print(type(a), a.dtype, type(b), b.dtype)

    # Restricciones
    constraints = [
        {"type": "eq", "fun": diagonal_constraint},  # Diagonal fija
        {"type": "eq", "fun": matrix_vector_constraint},  # M * a = b
    ]

    # Límites: Mij >= 0
    bounds = [(0, None)] * (n * n)

    # Inicialización de la matriz (puede ser identidad)
    M_initial = np.eye(n).flatten()

    # Optimización
    result = minimize(
        entropy,
        M_initial,
        bounds = bounds,
        constraints = constraints,
        method = 'SLSQP'
    )

    # Resultado
    M_optimized = result.x.reshape(n, n)
    
    # Ajusta el resultado a un dataframe Familia 1, familia 2, resultado M
    
    df_clacom = df_data_visitas_i[lista_clacom]
    for j in ['INICIAL', 'FINAL']:
        df_clacom_j = df_clacom.copy()
        for c in lista_clacom:
            df_clacom_j = df_clacom_j.rename(columns = {c: f'{c}_{j}'})
        df_clacom_j['AUX'] = 'aux'
        if j == 'INICIAL':
            df_clacom_all = df_clacom_j.copy()
        else:
            df_clacom_all = df_clacom_all.merge(df_clacom_j, on = 'AUX', how = 'left')
    df_clacom_all = df_clacom_all.drop(columns = ['AUX'])
    df_clacom_all['M'] = M_optimized.flatten()
    
    # comprueba las restricciones
    df_clacom_all_igual = df_clacom_all[(df_clacom_all['FAMILIA_INICIAL'] == df_clacom_all['FAMILIA_FINAL']) & (df_clacom_all['M'] != 1)]
    if len(df_clacom_all_igual) > 0:
        print('ERROR')
        print(df_clacom_all_igual)
        sys.exit('Restricciones 1...diagonal != 1')
    
    if df_clacom_all['M'].min() < 0:
        print('ERROR')
        print(df_clacom_all[df_clacom_all['M'] < 0])
        sys.exit('Restricciones 2...Mij < 0')

    return df_clacom_all
### Subseccion: Cálculo de entropía para casos no existentes
df_iterador = df_data_visitas[iterador].drop_duplicates().reset_index(drop = True) # Lo que necesito
df_iterador
df_matriz_entropia = pd.DataFrame()
if 'df_matriz_entropia.pkl' in os.listdir(dir_data_horus):
    df_matriz_entropia = tf.parquet_act(f'{dir_data_horus}df_matriz_entropia')

if len(df_matriz_entropia) > 0:
    df_matriz_entropia_existente = df_matriz_entropia[iterador].drop_duplicates().reset_index(drop = True)
    df_matriz_entropia_existente['EXISTE'] = True

df_matriz_entropia
if len(df_matriz_entropia) > 0: # Solo si existe algo
    df_iterador = df_iterador.merge(df_matriz_entropia_existente, on = iterador, how = 'left')
    df_iterador['EXISTE'] = df_iterador['EXISTE'].fillna(False)
    df_iterador = df_iterador[~df_iterador['EXISTE']].reset_index(drop = True) # Se procesará solo lo que no existe
    df_iterador = df_iterador.drop(columns = ['EXISTE']) 
df_iterador
paso = 25
subset = campos_beta + [campo_last_touch] + [f'{j}_INICIAL' for j in lista_clacom] + [f'{j}_FINAL' for j in lista_clacom]

if len(df_matriz_entropia) > 0:
    df_matriz_entropia['DATE'] = pd.to_datetime(df_matriz_entropia['DATE'])

for i in tqdm.tqdm(range(len(df_iterador))):
    
    #print('ELIMINAR')
    df_i = df_iterador.iloc[i: i + 1]
    df_i['INCLUIR'] = True
    df_data_visitas_i = df_data_visitas.merge(df_i, on = iterador, how = 'left')
    df_data_visitas_i['INCLUIR'] = df_data_visitas_i['INCLUIR'].fillna(False)
    df_data_visitas_i = df_data_visitas_i[df_data_visitas_i['INCLUIR']].reset_index(drop = True)
    df_data_visitas_i['VISITAS_IN'] = np.minimum(df_data_visitas_i['VISITAS_IN'], df_data_visitas_i['VISITAS']) # Casos inconsistentes donde visitas in puede ser > visitas
    df_clacom_all = matriz_max_entropia(df_data_visitas_i, lista_clacom)
    
    df_clacom_all['INCLUIR'] = True
    df_clacom_all = df_clacom_all.merge(df_i, on = 'INCLUIR', how = 'left')
    
    df_clacom_all = df_clacom_all[df_clacom_all['M'] > 0]
    df_clacom_all = df_clacom_all[campos_beta + [campo_last_touch] + [f'{j}_INICIAL' for j in lista_clacom] + [f'{j}_FINAL' for j in lista_clacom] + ['M']]
    df_clacom_all['DATE'] = pd.to_datetime(df_clacom_all['DATE'])
    
    df_matriz_entropia = pd.concat([df_clacom_all, df_matriz_entropia]).reset_index(drop = True) # Prioridad al nuevo registro
    df_matriz_entropia = df_matriz_entropia.drop_duplicates(subset = subset, keep = 'first').reset_index(drop = True) # Todo actualizado. guardar según paso
    
    if ((i + 1) % paso == 0) or (i == len(df_iterador) - 1):
        print('Guardar')
        tf.parquet_act(f'{dir_data_horus}df_matriz_entropia', df_matriz_entropia, 'save') # Se guarda el resultado

### Subseccion: Parámetro matriz entropía
df_matriz_entropia = tf.parquet_act(f'{dir_data_horus}df_matriz_entropia')
df_matriz_entropia

df_dates['IN'] = True
df_matriz_entropia['DATE'] = pd.to_datetime(df_matriz_entropia['DATE'])
df_matriz_entropia = df_matriz_entropia.merge(df_dates, on = 'DATE', how = 'left')
df_matriz_entropia['IN'] = df_matriz_entropia['IN'].fillna(False)
df_matriz_entropia = df_matriz_entropia[df_matriz_entropia['IN']].reset_index(drop = True)
df_matriz_entropia = df_matriz_entropia.drop(columns = ['IN'])
df_matriz_entropia = df_matriz_entropia[df_matriz_entropia['PAIS'].isin(paises)]

df_matriz_entropia = df_matriz_entropia.rename(columns = {'DATE': 'DATE_COMPARACION'})
df_dias_predecir = df_dias_predecir[['DATE', 'PAIS', 'DATE_COMPARACION']]
df_dias_predecir['DATE_COMPARACION'] = pd.to_datetime(df_dias_predecir['DATE_COMPARACION'])
df_dias_predecir['DATE'] = pd.to_datetime(df_dias_predecir['DATE'])

df_matriz_entropia = df_matriz_entropia.merge(df_dias_predecir, on = ['PAIS', 'DATE_COMPARACION'], how = 'left')
df_matriz_entropia = df_matriz_entropia[df_matriz_entropia['DATE'].notna()].reset_index(drop = True) # No interesan los dates fuera del horizonte
df_matriz_entropia
if modo_tactico:
    df_matriz_entropia = df_matriz_entropia.rename(columns = {'FAMILIA_INICIAL': 'FAMILIA'})
    df_matriz_entropia = df_matriz_entropia.merge(df_data_visitas.rename(columns = {'DATE': 'DATE_COMPARACION'}), how = 'left', on = ['PAIS', 'DATE_COMPARACION', 'CANAL', 'FUENTE'] + campos_grano)
    df_matriz_entropia['M_VISITAS_IN'] = df_matriz_entropia['M'] * df_matriz_entropia['VISITAS']
    df_matriz_entropia['PERIODO'] = df_matriz_entropia['DATE'].astype(str).str[:7]
    df_matriz_entropia = df_matriz_entropia.rename(columns = {'FAMILIA': 'FAMILIA_INICIAL'})
    df_matriz_entropia = df_matriz_entropia[campos_beta_def + [campo_last_touch] + ['FAMILIA_INICIAL', 'FAMILIA_FINAL', 'M_VISITAS_IN', 'VISITAS_IN']].groupby(campos_beta_def + [campo_last_touch] + ['FAMILIA_INICIAL', 'FAMILIA_FINAL'], as_index = False).sum()
    df_matriz_entropia['M'] = df_matriz_entropia['M_VISITAS_IN'] / df_matriz_entropia['VISITAS_IN']
    df_matriz_entropia['M'] = np.where(df_matriz_entropia['M'] > 1, 1, df_matriz_entropia['M'])
    df_matriz_entropia = df_matriz_entropia.drop(columns = ['M_VISITAS_IN', 'VISITAS_IN'])
df_matriz_entropia
dic_parametros['RELACION_VISITAS'] = df_matriz_entropia
df_matriz_entropia
df_times, t0 = medir_tiempo(t0, '1.3.5 Pars 10', df_times)
### Seccion: ## 1.3.2 Parámetros Visitas In Pago e Inversión [6-7]
df_beta_in = df_beta.copy()
df_beta_in['IN'] = True
df_data_visitas = tf.parquet_act(f'{dir_data_horus}df_data_visitas') # (Histórica) Se lee y se filtra df_data_visitas # este para inversion 250109 (Histórica)
df_data_visitas['DATE'] = pd.to_datetime(df_data_visitas['DATE'])
#df_data_visitas = df_data_visitas[df_data_visitas['PAIS'].isin(paises)]
#df_data_visitas = df_data_visitas[(df_data_visitas['DATE'] >= str(dia_desde)) & (df_data_visitas['DATE'] <= str(dia_hasta))].reset_index(drop = True)
df_data_visitas = df_data_visitas.merge(df_beta_in, on = campos_beta, how = 'left')
df_data_visitas['IN'] = df_data_visitas['IN'].fillna(False)
df_data_visitas = df_data_visitas[df_data_visitas['IN']].reset_index(drop = True)
df_data_visitas = df_data_visitas.drop(columns = ['IN'])
df_data_visitas
tf.parquet_act(f'{dir_data}Parametros/P4').DATE.min()
df_beta_in
df = tf.parquet_act(f'{dir_data}Parametros/P4')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.merge(df_beta_in, on = campos_beta, how = 'left')
df['IN'] = df['IN'].fillna(False)
df = df[df['IN']]
df = df.drop(columns = ['IN'])
if len(df_data_visitas) > 0:  # Solo si hay data histórica.Si el horizonte es futuro, entonces df_data_visitas = vacío
    df = df[df['DATE'] > df_data_visitas.DATE.max()].reset_index(drop = True) # Solo para días que no existan en los reales
df
#sys.exit("Eliminar acá abajo df = df[df['RANGO'] <= 3].reset_index(drop = True)..small batch")
#df = df[df['RANGO'] <= 2].reset_index(drop = True)
limite = 20
if version_simplificada:
    limite = 2

df_rangos = pd.DataFrame({'RANGO': [i for i in range(1, limite + 1)]})
df_rangos['AUX'] = 'aux'
df_rangos
# Depende del caso la union del real con el proyectado

if modo_tactico:
    # SI es modo tactico, se agrupa por periodo y se le suma la data real a la misma base
    df['PERIODO'] = df['DATE'].astype(str).str[:7]
    df_data_visitas['PERIODO'] = df_data_visitas['DATE'].astype(str).str[:7]
    df_data_visitas = df_data_visitas[campos_beta_def + campos_grano + ['VISITAS_IN']].groupby(campos_beta_def + campos_grano, as_index = False).sum()
    df_representantes_base['PERIODO'] = df_representantes_base['DATE'].astype(str).str[:7]
    df_inversion = df_representantes_base[campos_beta_def + campos_grano + ['INVERSION']].groupby(campos_beta_def + campos_grano, as_index = False).sum()
    
    df = df[campos_beta_def + campos_grano + ['RANGO', 'I_0', 'I_F'] + ['INVERSION', 'VISITAS_PAGO', 'VP0', 'VPF', 'A', 'B']].groupby(campos_beta_def + campos_grano + ['RANGO', 'I_0', 'I_F'], as_index = False).sum()

    df_real = df_data_visitas.merge(df_inversion, on = campos_beta_def + campos_grano, how = 'left')
    df_real = df_real.rename(columns = {'VISITAS_IN': 'VISITAS_0', 'INVERSION': 'INVERSION_0'})
    
    df = df.merge(df_real, on = campos_beta_def + campos_grano, how = 'left')
    df[['INVERSION_0', 'VISITAS_0']] = df[['INVERSION_0', 'VISITAS_0']].fillna(0)

    df['I_0'] += df['INVERSION_0']
    df['I_F'] += df['INVERSION_0']
    df['INVERSION'] += df['INVERSION_0']
    df['VISITAS_PAGO'] += df['VISITAS_0']
    df['VP0'] += df['VISITAS_0']
    df['VPF'] += df['VISITAS_0']
    df['B'] += (df['VISITAS_0'] - df['A'] * df['INVERSION_0'])

    df['K0'] = abs(df['A'] * df['I_0'] + df['B'] - df['VP0'])
    df['KF'] = abs(df['A'] * df['I_F'] + df['B'] - df['VPF'])
    
    df_valid = df[(df['K0'] >= 0.0001) | (df['KF'] >= 0.0001)]
    if len(df_valid) > 0:
        print('ERROR')
        print(df_valid)
        sys.exit('Error en la validación de los datos')
    
    df = df.drop(columns = ['K0', 'KF'])

else:
    df_inversion = df_representantes_base[campos_beta + campos_grano + ['INVERSION']].groupby(campos_beta + campos_grano, as_index = False).sum()
    df_data_visitas = df_data_visitas[campos_beta + campos_grano + ['VISITAS_IN']].groupby(campos_beta + campos_grano, as_index = False).sum()
    df_real = df_data_visitas.merge(df_inversion, on = campos_beta + campos_grano, how = 'left').rename(columns = {'VISITAS_IN': 'VISITAS_PAGO'})
    df_real['I_0'], df_real['I_F'] = df_real['INVERSION'], df_real['INVERSION']
    df_real['VP0'], df_real['VPF'] = df_real['VISITAS_PAGO'], df_real['VISITAS_PAGO']
    df_real['A'] = 0
    df_real['B'] = df_real['VISITAS_PAGO']
    df_real['AUX'] = 'aux'
    df_real = df_real.merge(df_rangos, on = 'AUX')
    df_real = df_real.drop(columns = ['AUX'])
    df_real # No importa el rango, ya está predefinida la inversión y las visitas (reales)
    df = pd.concat([df, df_real]).reset_index(drop = True)
df
df = df[campos_beta_def + campos_grano + ['RANGO', 'I_0', 'I_F', 'VP0', 'VPF', 'A', 'B']]
#df = df[(df['DATE'] >= dias_proyeccion[0]) & (df['DATE'] <= dias_proyeccion[1])].reset_index(drop = True)
#if modo_tactico:
#    df['PERIODO'] = df['DATE'].astype(str).str[:7]
#df = df[campos_beta_def + campos_grano + ['RANGO', 'I_0', 'I_F', 'A', 'B']].groupby(campos_beta_def + campos_grano + ['RANGO', 'I_0', 'I_F'], as_index = False).sum()
#df['VP0'], df['VPF'] = df['I_0'] * df['A'] + df['B'], df['I_F'] * df['A'] + df['B']

df_rango = df[['RANGO']].drop_duplicates().reset_index(drop = True)
df_rango['AUX'] = 'aux'
df_beta_alpha_grano_pago['AUX'] = 'aux'
df_beta_alpha_grano_pago_rango = df_beta_alpha_grano_pago.merge(df_rango, on = 'AUX')
df_beta_alpha_grano_pago_rango = df_beta_alpha_grano_pago_rango.drop(columns = ['AUX'])
df_beta_alpha_grano_pago = df_beta_alpha_grano_pago.drop(columns = ['AUX'])
df = df_beta_alpha_grano_pago_rango.merge(df, on = campos_beta_def + campos_grano + ['RANGO'], how = 'left')
df = df.reset_index(drop = True)  
dic_parametros['VISITAS_PAGO'] = df
df

df_times, t0 = medir_tiempo(t0, '1.3.2 Pars 6-7', df_times)
df_beta_alpha_grano_pago_rango
#sys.exit('Revisar tiempos')
#sys.exit('Atención abajo en restricciones > declarar días historicos cuando sea requerido (como en inversion y visitas in), y no cuando ya esté integraddo en el parámetro')
# como el caso de sumas organicas y pagas
dic_parametros.keys()
### Seccion:  Modelo de Optimización
model = mip.Model("Horus", sense = mip.MAXIMIZE) # Se crea el modelo de optimización (maximización)

# , log_level=2 muestra info en el model.optimize() ??
### Seccion: # Variables
df_beta_alpha_grano_organico
df_beta_alpha_grano_organico['NAME'] = ''
df_beta_alpha_grano_pago['NAME'] = ''

for c in campos_beta_def + campos_grano:
    df_beta_alpha_grano_organico['NAME'] += (df_beta_alpha_grano_organico[c].astype(str) + '_')
    df_beta_alpha_grano_pago['NAME'] += (df_beta_alpha_grano_pago[c].astype(str) + '_')

df_beta_alpha_grano_organico['NAME'] = df_beta_alpha_grano_organico['NAME'].str[:-1]
df_beta_alpha_grano_pago['NAME'] = df_beta_alpha_grano_pago['NAME'].str[:-1]
df_alpha_beta_duplicacion['NAME'] = ''
for c in ['DUPLICACION'] + campos_beta_def + ['FAMILIA', 'SUBFAMILIA'] + [campo_last_touch, 'NATURALEZA_MEDIO']:
    df_alpha_beta_duplicacion['NAME'] += (df_alpha_beta_duplicacion[c].astype(str) + '_')
df_alpha_beta_duplicacion['NAME'] = df_alpha_beta_duplicacion['NAME'].str[:-1]
df_alpha_beta_duplicacion
dic_variables = {}
df_times, t0 = medir_tiempo(t0, 'M Vars Ini', df_times)
### Seccion: ## Venta Colocada [1-3]
df_alpha_beta_duplicacion
df_alpha_beta_duplicacion.DUPLICACION.unique()
# SO (Sales Orgánicas)

# Crear variables para cada fila y almacenarlas en una nueva columna
df_SO = df_beta_alpha_grano_organico.copy()
df_SO['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'SO_{name}') for name in df_beta_alpha_grano_organico['NAME'].values]
print(df_SO)

# SP (Sales Pagas)
df_SP = df_beta_alpha_grano_pago.copy()
df_SP['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'SP_{name}') for name in df_beta_alpha_grano_pago['NAME'].values]
print(df_SP)

# ST (Sales Totales)
df_ST = df_alpha_beta_duplicacion.copy()
df_ST['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'ST_{name}') for name in df_alpha_beta_duplicacion['NAME'].values]
print(df_ST)

# Venta neta
df_ST_net = df_alpha_beta_duplicacion.copy()
df_ST_net['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'ST_net_{name}') for name in df_alpha_beta_duplicacion['NAME'].values]


dic_variables['Venta'] = {'SO': df_SO, 'SP': df_SP, 'ST': df_ST, 'ST_net': df_ST_net}
df_times, t0 = medir_tiempo(t0, 'M Vars VC', df_times)
### Seccion: ## Órdenes Colocadas [4-6]
# OO (Ordenes Orgánicas)
df_OO = df_beta_alpha_grano_organico.copy()
df_OO['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'OO_{name}') for name in df_beta_alpha_grano_organico['NAME'].values]
print(df_OO)

# OP (Ordenes Pagas)
df_OP = df_beta_alpha_grano_pago.copy()
df_OP['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'OP_{name}') for name in df_beta_alpha_grano_pago['NAME'].values]
print(df_OP)

# OT (Ordenes Totales)
df_OT = df_alpha_beta_duplicacion.copy()
df_OT['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'OT_{name}') for name in df_alpha_beta_duplicacion['NAME'].values]
print(df_OT)

dic_variables['Ordenes'] = {'OO': df_OO, 'OP': df_OP, 'OT': df_OT}
df_times, t0 = medir_tiempo(t0, 'M Vars OC', df_times)
### Seccion: ## Visitas [7-9]
# VO (Visitas Orgánicas)
df_VO = df_beta_alpha_grano_organico.copy()
df_VO['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'VO_{name}') for name in df_beta_alpha_grano_organico['NAME'].values]
print(df_VO)

# VP (Visitas Pagas)
df_VP = df_beta_alpha_grano_pago.copy()
df_VP['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'VP_{name}') for name in df_beta_alpha_grano_pago['NAME'].values]
print(df_VP)

# VP (Visitas Pagas IN)
df_VP_in = df_beta_alpha_grano_pago.copy()
df_VP_in['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'VP_in_{name}') for name in df_beta_alpha_grano_pago['NAME'].values]
print(df_VP_in)

# VT (Visitas Totales)
df_VT = df_alpha_beta_duplicacion.copy()
df_VT['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'VT_{name}') for name in df_alpha_beta_duplicacion['NAME'].values]
print(df_VT)

dic_variables['Visitas'] = {'VO': df_VO, 'VP': df_VP, 'VP_in': df_VP_in, 'VT': df_VT}

df_times, t0 = medir_tiempo(t0, 'M Vars Vis', df_times)
### Seccion: ## Inversión [10-11]
# Inversion (para grano)
df_Inv = df_beta_alpha_grano_pago.copy()
df_Inv['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'Inv_{name}') for name in df_beta_alpha_grano_pago['NAME'].values]
print(df_Inv)

# Inv para todos los niveles de duplicación
df_Inv_all = df_alpha_beta_duplicacion[df_alpha_beta_duplicacion['NATURALEZA_MEDIO'].isin(['Pago', '']).reset_index(drop = True)]
df_Inv_all['X'] = [model.add_var(var_type = 'C', lb = 0, name = f'Inv_all_{name}') for name in df_Inv_all['NAME'].values]
print(df_Inv_all)
df_y = dic_parametros['VISITAS_PAGO']
df_y = df_y[campos_beta_def + campos_grano + ['RANGO']].drop_duplicates().reset_index(drop = True)

df_y['NAME'] = ''
for c in campos_beta_def + campos_grano + ['RANGO']:
    df_y['NAME'] += (df_y[c].astype(str) + '_')
df_y['NAME'] = df_y['NAME'].str[:-1]
df_y['X'] = [model.add_var(var_type = 'B', name = f'Y_{name}') for name in df_y['NAME'].values] # Binario del rango seleccionado
df_y

df_Inv_rango = df_y[campos_beta_def + campos_grano + ['RANGO', 'NAME']]
df_Inv_rango['X'] = [model.add_var(var_type = 'C', name = f'Inv_rango_{name}') for name in df_Inv_rango['NAME'].values]

dic_variables['Inversion'] = {'Inv': df_Inv, 'Inv_all': df_Inv_all, 'Y': df_y, 'Inv_rango': df_Inv_rango}
dic_variables['Inversion']['Y']
df_times, t0 = medir_tiempo(t0, 'M Vars Inv', df_times)
### Seccion: # Función Objetivo
escribir_modelo
def probar_factibilidad(model):
    model.max_gap = 0.01
    status = model.optimize(max_seconds = 300, relax = True, msg=True) # Primero prueba factibilidad

    print('Relaxed')
    print(status)
    print(model.objective_value)
    return None

def escribir_modelo_opt(activar, archivo, modo, linea):
    if activar:
        with open(archivo, modo) as writefile:
            writefile.write(linea)
    return None
# Tasas de cambio

query = "SELECT * FROM `sod-corp-plp-beta.ETL_performance_2023.exrate` WHERE DATE = (select max(date) from `sod-corp-plp-beta.ETL_performance_2023.exrate`)"

df_exrate = tf.request_GCP_vnew(
                            nombre_tabla = "",
                            specific_query = query,
                            client = client,
                            output = False, permitir_fallos = False)

df_exrate = df_exrate[['PAIS', 'EXRATE']]

# Función objetivo

# Venta Neta
df_venta_neta = dic_variables['Venta']['ST_net']
df_venta_neta_total = df_venta_neta[df_venta_neta['DUPLICACION'] == 'TOTALES'].reset_index(drop = True)
df_venta_neta_total = df_venta_neta_total.merge(df_exrate, on = 'PAIS', how = 'left')
df_venta_neta_total['EXRATE'] = round(df_venta_neta_total['EXRATE'], 3)
df_venta_neta_total['VENTA_NETA_USD'] = df_venta_neta_total['X'] / df_venta_neta_total['EXRATE']
VN_USD = sum(df_venta_neta_total['VENTA_NETA_USD'])


df_inversion = dic_variables['Inversion']['Inv']
df_inversion = df_inversion.merge(df_exrate, on = 'PAIS', how = 'left')
df_inversion['EXRATE'] = round(df_inversion['EXRATE'], 3)
df_inversion['INVERSION_USD'] = df_inversion['X'] / df_inversion['EXRATE']
INV_USD = sum(df_inversion['INVERSION_USD'])

Z = VN_USD - INV_USD

model.objective = mip.maximize(Z)

escribir_modelo_opt(activar = escribir_modelo, archivo = f'Modelo de optimizacion.txt', modo = 'w', linea = f'Función Objetivo \n')
escribir_modelo_opt(activar = escribir_modelo, archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'----------------  \n\n\n\n\n\n\n\n')
escribir_modelo_opt(activar = escribir_modelo, archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'Z: {Z}  \n\n')
df_times, t0 = medir_tiempo(t0, 'M FO', df_times)
### Seccion: # Restricciones
### Seccion: ## 1. Grano orgánico
### Subseccion: El grano orgánico, viene de los parámetros
dic_parametros.keys()
escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'Restricciones\n\n\n')
# a. Visitas Orgánicas

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR1a\n\n')

df_R1a = dic_variables['Visitas']['VO'].merge(dic_parametros['VO'], on = campos_beta_def + campos_grano, how = 'left')
df_R1a = df_R1a.rename(columns = {'y': 'VO'})
df_R1a = df_R1a.fillna(0)
df_R1a['VO'] = round(df_R1a['VO'], 3)

if activar_holguras:
    df_R1a['H1a_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H1a_p_{name}') for name in df_R1a['NAME'].values]
    df_R1a['H1a_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H1a_n_{name}') for name in df_R1a['NAME'].values]
    #H1a_p = model.add_var(var_type = 'C', lb = 0, name = f'H1a_p') # Holgura positiva
    #H1a_n = model.add_var(var_type = 'C', lb = 0, name = f'H1a_p') # Holgura negativa

for i in tqdm.tqdm(range(len(df_R1a))):
    
    #if version_simplificada:
    #    break
    
    name = df_R1a["NAME"][i]

    if activar_holguras:
        #R1a = (df_R1a['X'][i] == df_R1a['VO'][i] + H1a_p - H1a_n)#, f"R1_{df_R1['NAME'][i]}"
        R1a = (df_R1a['X'][i] == round(df_R1a['VO'][i], 3) + df_R1a['H1a_p'][i] - df_R1a['H1a_n'][i])
    else:
        R1a = (df_R1a['X'][i] == round(df_R1a['VO'][i], 3))
        
    model += R1a
    
    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R1a({name}): {R1a}  \n')

# b. Ordenes Orgánicas

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR1b\n\n')

df_R1b = dic_variables['Ordenes']['OO'].merge(dic_parametros['VO'], on = campos_beta_def + campos_grano, how = 'left')
df_R1b = df_R1b.rename(columns = {'y': 'VO'})
df_R1b = df_R1b.merge(dic_parametros['TCO'], on = campos_beta_def + campos_grano, how = 'left')
df_R1b = df_R1b.rename(columns = {'y': 'TCO'})
df_R1b = df_R1b.fillna(0)
df_R1b['OO'] = df_R1b['VO'] * df_R1b['TCO']
df_R1b['OO'] = round(df_R1b['OO'], 3)

if activar_holguras:

    print('Si')
    df_R1b['H1b_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H1b_p_{name}') for name in df_R1b['NAME'].values]
    df_R1b['H1b_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H1b_n_{name}') for name in df_R1b['NAME'].values]
    #H1b_p = model.add_var(var_type = 'C', lb = 0, name = f'H1b_p') # Holgura positiva
    #H1b_n = model.add_var(var_type = 'C', lb = 0, name = f'H1b_n') # Holgura negativa
    
for i in tqdm.tqdm(range(len(df_R1b))):

    #if version_simplificada:
    #    break
    
    name = df_R1b["NAME"][i]

    if activar_holguras:
        #R1b = (df_R1b['X'][i] == df_R1b['OO'][i] + H1b_p + H1b_n)#, f"R1_{df_R1['NAME'][i]}"
        R1b = (df_R1b['X'][i] == df_R1b['OO'][i] + df_R1b['H1b_p'][i] - df_R1b['H1b_n'][i])
    else: 
        R1b = (df_R1b['X'][i] == df_R1b['OO'][i])
        
    model += R1b
    
    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R1b({name}): {R1b}  \n')

# c. Ventas Orgánicas
escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR1c\n\n')

df_R1c = dic_variables['Venta']['SO'].merge(dic_parametros['VO'], on = campos_beta_def + campos_grano, how = 'left')
df_R1c = df_R1c.rename(columns = {'y': 'VO'})
df_R1c = df_R1c.merge(dic_parametros['TCO'], on = campos_beta_def + campos_grano, how = 'left')
df_R1c = df_R1c.rename(columns = {'y': 'TCO'})
df_R1c = df_R1c.merge(dic_parametros['TPO'], on = campos_beta_def + campos_grano, how = 'left')
df_R1c = df_R1c.rename(columns = {'y': 'TPO'})
df_R1c = df_R1c.fillna(0)
df_R1c['SO'] = df_R1c['VO'] * df_R1c['TCO'] * df_R1c['TPO']
df_R1c['SO'] = round(df_R1c['SO'], 3)

if activar_holguras:
    #H1c_p = model.add_var(var_type = 'C', lb = 0, name = f'H1c_p') # Holgura positiva
    #H1c_n = model.add_var(var_type = 'C', lb = 0, name = f'H1c_n') # Holgura negativa
    df_R1c['H1c_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H1c_p_{name}') for name in df_R1c['NAME'].values]
    df_R1c['H1c_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H1c_n_{name}') for name in df_R1c['NAME'].values]
    
    
for i in tqdm.tqdm(range(len(df_R1c))):
    
    name = df_R1c["NAME"][i]
    
    if activar_holguras:
        #R1c = (df_R1c['X'][i] == df_R1c['SO'][i] + H1c_p - H1c_n)
        R1c = (df_R1c['X'][i] == df_R1c['SO'][i] + df_R1c['H1c_p'][i] - df_R1c['H1c_n'][i])
    else:
        R1c = (df_R1c['X'][i] == df_R1c['SO'][i])
    model += R1c
    
    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R1c({name}): {R1c}  \n')

df_times, t0 = medir_tiempo(t0, 'M R1', df_times)
### Seccion: ## 2. Grano pago
### Subseccion: No se incluyen visitas en este caso y, a diferencia de orgánico, no se generan las variables como producto de parámetros
dic_parametros.keys()
# a. Ordenes Pago

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR2a\n\n')

df_R2a = dic_variables['Ordenes']['OP'].rename(columns = {'X': 'X_OP'})
df_R2a = df_R2a.merge(dic_variables['Visitas']['VP'], on = campos_beta_def + campos_grano + ['NAME'], how = 'left').rename(columns = {'X': 'X_VP'})
df_R2a = df_R2a.merge(dic_parametros['TCP'], on = campos_beta_def + campos_grano, how = 'left').rename(columns = {'y': 'TCP'})
df_R2a['TCP'] = df_R2a['TCP'].fillna(0)
df_R2a['TCP'] = round(df_R2a['TCP'], 3)

if activar_holguras:
    df_R2a['H2a_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H2a_p_{name}') for name in df_R2a['NAME'].values]
    df_R2a['H2a_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H2a_n_{name}') for name in df_R2a['NAME'].values]
                       
    #H2a_p = model.add_var(var_type = 'C', lb = 0, name = f'H2a_p') # Holgura positiva
    #H2a_n = model.add_var(var_type = 'C', lb = 0, name = f'H2a_n') # Holgura negativa
    
for i in tqdm.tqdm(range(len(df_R2a))):
    
    name = df_R2a["NAME"][i]
    
    if activar_holguras:
        #R2a = (df_R2a['X_OP'][i] == df_R2a['X_VP'][i] * df_R2a['TCP'][i] + H2a_p - H2a_n)
        R2a = (df_R2a['X_OP'][i] == df_R2a['X_VP'][i] * df_R2a['TCP'][i] + df_R2a['H2a_p'][i] - df_R2a['H2a_n'][i])
    else:
        R2a = (df_R2a['X_OP'][i] == df_R2a['X_VP'][i] * df_R2a['TCP'][i])
    model += R2a

    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R2a({name}): {R2a}  \n')

# b. Venta Pago

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR2b\n\n')

df_R2b = dic_variables['Venta']['SP'].rename(columns = {'X': 'X_SP'})
df_R2b = df_R2b.merge(dic_variables['Ordenes']['OP'], on = campos_beta_def + campos_grano + ['NAME'], how = 'left').rename(columns = {'X': 'X_OP'})
df_R2b = df_R2b.merge(dic_parametros['TPP'], on = campos_beta_def + campos_grano, how = 'left').rename(columns = {'y': 'TPP'})
df_R2b['TPP'] = df_R2b['TPP'].fillna(0)
df_R2b['TPP'] = round(df_R2b['TPP'], 3)

if activar_holguras:
    #H2b_p = model.add_var(var_type = 'C', lb = 0, name = f'H2b_p') # Holgura positiva
    #H2b_n = model.add_var(var_type = 'C', lb = 0, name = f'H2b_n') # Holgura negativa
    df_R2b['H2b_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H2b_p_{name}') for name in df_R2b['NAME'].values]
    df_R2b['H2b_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H2b_n_{name}') for name in df_R2b['NAME'].values]

for i in tqdm.tqdm(range(len(df_R2b))):
    
    name = df_R2b["NAME"][i]
    
    if activar_holguras:
        #R2b = (df_R2b['X_SP'][i] == df_R2b['X_OP'][i] * df_R2b['TPP'][i] + H2b_p - H2b_n)
        R2b = (df_R2b['X_SP'][i] == df_R2b['X_OP'][i] * df_R2b['TPP'][i] + df_R2b['H2b_p'][i] - df_R2b['H2b_n'][i])
    else:
        R2b = (df_R2b['X_SP'][i] == df_R2b['X_OP'][i] * df_R2b['TPP'][i])
    model += R2b

    if i < escribir_modelo[1]:
        #print(str(R2b))
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R2b({name}): {R2b}  \n')

df_R2b
df_times, t0 = medir_tiempo(t0, 'M R2', df_times)
### Seccion: ## 3. Inversión & Tráfico Pago IN
df = dic_parametros['VISITAS_PAGO']
df = df.sort_values(by = campos_beta_def + campos_grano + ['RANGO']).reset_index(drop = True)
df
df['MIN_VIS_IN'] = df['A'] * df['I_0'] + df['B']
df['MAX_VIS_IN'] = df['A'] * df['I_F'] + df['B']

# Vis = A * I + b
escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR3a: Para cada combinación, la suma de y en los rangos tiene que ser = 1...esto es, y solo puede existir en uno y solo un rango\n\n')

lista_rangos = list(df_rango.RANGO.unique())

# Variable Y
df_y = dic_variables['Inversion']['Y']

# Para cada combinación, la suma de y en los rangos tiene que ser = 1...esto es, y solo puede existir en uno y solo un rango
# Pivotear el DataFrame
df_y = df_y.pivot_table(
    index = campos_beta_def + campos_grano,  # Mantener estas columnas
    columns = 'RANGO',  # RANGO como columnas
    values = 'X',  # Usar X como valores
    aggfunc = 'first'  # Tomar el primer valor en caso de duplicados
).reset_index()

df_y

df_y['NAME'] = ''
for c in campos_beta_def + campos_grano:
    df_y['NAME'] += (df_y[c].astype(str) + '_')
df_y['NAME'] = df_y['NAME'].str[:-1]

if activar_holguras:
    #df_R3b['H3b1'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3b1{name}') for name in df_R3b['NAME'].values]
    df_y['H3a_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3a_p_{name}') for name in df_y['NAME'].values]
    df_y['H3a_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3a_n_{name}') for name in df_y['NAME'].values]

# Esta restricción, no tiene holguras

for i in tqdm.tqdm(range(len(df_y))):
    
    name = df_y["NAME"][i]
    #R3a = ((df_R2b['X_SP'][i] == df_R2b['X_OP'][i] * df_R2b['TPP'][i]))
    s = 0
    for c in lista_rangos:
        s += df_y[c][i]
        
    if activar_holguras:
        R3a = (s == 1 + df_y['H3a_p'][i] - df_y['H3a_n'][i])
    else:
        R3a = s == 1
    model += R3a

    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3a({name}): {R3a}  \n')

df_times, t0 = medir_tiempo(t0, 'M R3a', df_times)
# R3b: El rango seleccionado depende de la inversión
# Variable Inv
df_Inv = dic_variables['Inversion']['Inv_rango']
df_Inv = df_Inv[campos_beta_def + campos_grano + ['RANGO', 'X']].rename(columns = {'X': 'Inv_rango'})
print(df_Inv.head())

df_y_inv = dic_variables['Inversion']['Y']
df_y_inv = df_y_inv[campos_beta_def + campos_grano + ['RANGO', 'X']].rename(columns = {'X': 'X_Y'})
print(df_y_inv.head())
df_R3b = df_y_inv.merge(df_Inv, on = campos_beta_def + campos_grano + ['RANGO'], how = 'left')
df_R3b = df_R3b.merge(df[campos_beta_def + campos_grano + ['RANGO'] + ['I_0', 'I_F']], on = campos_beta_def + campos_grano + ['RANGO'], how = 'left')
df_R3b

df_R3b['NAME'] = ''
for c in campos_beta_def + campos_grano + ['RANGO']:
    df_R3b['NAME'] += (df_R3b[c].astype(str) + '_')
df_R3b['NAME'] = df_R3b['NAME'].str[:-1]

df_R3b = df_R3b.fillna(0)
# 3b1 y 3b2: I >= min * y & I <= max *y

if activar_holguras:
    #H3b1 = model.add_var(var_type = 'C', lb = 0, name = f'H3b1') # Holgura positiva
    #H3b2 = model.add_var(var_type = 'C', lb = 0, name = f'H3b2') # Holgura negativa
    
    df_R3b['H3b1'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3b1_{name}') for name in df_R3b['NAME'].values] # Holgura positiva
    df_R3b['H3b2'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3b2_{name}') for name in df_R3b['NAME'].values] # Holgura negativa

#df_R3b['RHS1'] = df_R3b['I_0'] * df_R3b['X_Y']
#df_R3b['RHS2'] = df_R3b['I_F'] * df_R3b['X_Y']

df_R3b['I_0'], df_R3b['I_F'] = round(df_R3b['I_0'], 3), round(df_R3b['I_F'], 3)

df_R3b['RHS1'] = df_R3b['I_0'] * df_R3b['X_Y']
if activar_holguras:
    df_R3b['RHS1'] -= df_R3b['H3b1']

df_R3b['RHS2'] = df_R3b['I_F'] * df_R3b['X_Y']
if activar_holguras:
    df_R3b['RHS2'] += df_R3b['H3b2']


escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR3b\n\n')

for i in tqdm.tqdm(range(len(df_R3b))):
    name = df_R3b["NAME"][i]
    
    #if activar_holguras:
    #    R3b1 = (df_R3b['Inv_rango'][i] >= df_R3b['RHS1'][i] - H3b1)   
    #    R3b2 = (df_R3b['Inv_rango'][i] <= df_R3b['RHS2'][i] + H3b2)
    #else:
    #    R3b1 = (df_R3b['Inv_rango'][i] >= df_R3b['RHS1'][i])
    #    R3b2 = (df_R3b['Inv_rango'][i] <= df_R3b['RHS2'][i])
 
    R3b1 = (df_R3b['Inv_rango'][i] >= df_R3b['RHS1'][i])
    R3b2 = (df_R3b['Inv_rango'][i] <= df_R3b['RHS2'][i])
    model += R3b1
    model += R3b2    
    
    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3b1({name}): {R3b1}  \n')
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3b2({name}): {R3b2}  \n')


"""escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR3b1\n\n')

for i in tqdm.tqdm(range(len(df_R3b))):
    
    name = df_R3b["NAME"][i]
    if activar_holguras:
        R3b1 = (df_R3b['Inv_rango'][i] >= df_R3b['I_0'][i] * df_R3b['X_Y'][i] - df_R3b['H3b1'][i])
    else:
        R3b1 = (df_R3b['Inv_rango'][i] >= df_R3b['I_0'][i] * df_R3b['X_Y'][i])
    model += R3b1

    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3b1({name}): {R3b1}  \n')
    
# 3b2: I <= max * y

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR3b2\n\n')

for i in tqdm.tqdm(range(len(df_R3b))):
    
    name = df_R3b["NAME"][i]
    if activar_holguras:
        R3b2 = (df_R3b['Inv_rango'][i] <= df_R3b['I_F'][i] * df_R3b['X_Y'][i] + df_R3b['H3b2'][i])
    else:
        R3b2 = (df_R3b['Inv_rango'][i] <= df_R3b['I_F'][i] * df_R3b['X_Y'][i])
    model += R3b2

    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3b2({name}): {R3b2}  \n')"""
df_times, t0 = medir_tiempo(t0, 'M R3b 1-2', df_times)
# 3b3

df_Inv = dic_variables['Inversion']['Inv_rango']

# Eliminar este if / else: Dejar solo lo que está en el if
if 'X' in df_Inv.columns:
    df_Inv = df_Inv[campos_beta_def + campos_grano + ['RANGO', 'X']].rename(columns = {'X': 'Inv_rango'})
else:
    df_Inv = df_Inv[campos_beta_def + campos_grano + ['RANGO', 'Inv_rango']]
    
df_Inv_total = dic_variables['Inversion']['Inv']
df_Inv_total = df_Inv_total[campos_beta_def + campos_grano + ['X']].rename(columns = {'X': 'Inv_total'})

df_R3b3 = df_Inv.merge(df_Inv_total, how = 'left', on = campos_beta_def + campos_grano)
#df_R3b3 = df_R3b3[campos_beta + campos_grano + ['Inv_total', 'Inv_rango']].groupby(campos_beta + campos_grano + ['Inv_total'], as_index = False).sum().reset_index(drop = True)

#df_names = dic_variables['Inversion']['Inv'][campos_beta + campos_grano + ['NAME']]
#df_R3b3 = df_R3b3.merge(df_names, how = 'left', on = campos_beta + campos_grano)

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR3b3\n\n')


df_inv_total = df_R3b3[['Inv_total']] # 250116
df_inv_total['Inv_total_NAME'] = df_inv_total['Inv_total'].astype(str)
df_inv_total = df_inv_total.drop_duplicates().reset_index(drop = True)

# Group by con name (en los campos del group by tiene no puede haver una var, por eso se hace la reversa después): Con esta solución, Tiempos bajan de 22 mins a 2 seg (99.85%)
df_R3b3['Inv_total_NAME'] = df_R3b3['Inv_total'].astype(str)
df_R3b3_agr = df_R3b3[campos_beta_def + campos_grano + ['Inv_rango', 'Inv_total_NAME']].groupby(campos_beta_def + campos_grano + ['Inv_total_NAME'], as_index = False).sum().reset_index(drop = True)
df_R3b3_agr = df_R3b3_agr.merge(df_inv_total, on = 'Inv_total_NAME', how = 'left')
df_R3b3_agr

if activar_holguras:
    df_R3b3_agr['H3b3_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3b3_p_{name}') for name in df_R3b3_agr['Inv_total_NAME'].values]
    df_R3b3_agr['H3b3_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3b3_n_{name}') for name in df_R3b3_agr['Inv_total_NAME'].values]

for i in tqdm.tqdm(range(len(df_R3b3_agr))):
    
    if activar_holguras:
        R3b3 = (df_R3b3_agr['Inv_total'][i] == df_R3b3_agr['Inv_rango'][i] + df_R3b3_agr['H3b3_p'][i] - df_R3b3_agr['H3b3_n'][i])
    else:
        R3b3 = (df_R3b3_agr['Inv_total'][i] == df_R3b3_agr['Inv_rango'][i])
    model += R3b3
    
    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3b3({name}): {R3b3}  \n')
df_times, t0 = medir_tiempo(t0, 'M R3b3', df_times)
df_beta_alpha_grano_pago
#df = df.drop_duplicates(subset = list(set(df.columns) - {'RANGO'})).reset_index(drop = True) # Casos con la inversión fija (A = 0)
#df
# 3c
# A * I + B
#Límites

# df_L: VP_min - max (h) {a(h) + b(h) * I_max}

df = df.fillna(0)
df_max_I = df[campos_beta_def + campos_grano + ['RANGO', 'I_F']].groupby(campos_beta_def + campos_grano, as_index = False).max().rename(columns = {'I_F': 'I_MAX'})
df_VP0_min = df[campos_beta_def + campos_grano + ['RANGO', 'VP0']].groupby(campos_beta_def + campos_grano, as_index = False).min().rename(columns = {'VP0': 'VP0_MIN'})
df_L = df[campos_beta_def + campos_grano + ['RANGO', 'A', 'B']].merge(df_max_I[campos_beta_def + campos_grano + ['I_MAX']], on = campos_beta_def + campos_grano, how = 'left')
df_L = df_L.merge(df_VP0_min[campos_beta_def + campos_grano + ['VP0_MIN']], on = campos_beta_def + campos_grano, how = 'left')
df_L['L'] = df_L['VP0_MIN'] - (df_L['A'] * df_L['I_MAX'] + df_L['B'])
df_L = df_L[campos_beta_def + campos_grano + ['L']].groupby(campos_beta_def + campos_grano, as_index = False).min()
df_L
## AQUI!!!
df_VPF_MAX = df[campos_beta_def + campos_grano + ['VPF']].groupby(campos_beta_def + campos_grano, as_index = False).max().rename(columns = {'VPF': 'VPF_MAX'})
df_B_MIN = df[campos_beta_def + campos_grano + ['B']].groupby(campos_beta_def + campos_grano, as_index = False).min().rename(columns = {'B': 'B_MIN'})

df_U = df_VPF_MAX.merge(df_B_MIN, on = campos_beta_def + campos_grano, how = 'left')
df_U['U'] = df_U['VPF_MAX'] - df_U['B_MIN']
df_U = df_U.drop(columns = ['VPF_MAX', 'B_MIN'])
df_U
df_R3c = df_R3b[campos_beta_def + campos_grano + ['NAME'] + ['RANGO', 'X_Y']]

df_Inv_total = dic_variables['Inversion']['Inv']
df_Inv_total = df_Inv_total[campos_beta_def + campos_grano + ['X']].rename(columns = {'X': 'X_Inv'})
df_R3c = df_R3c.merge(df_Inv_total, on = campos_beta_def + campos_grano, how = 'left')
df_R3c = df_R3c.merge(df_L, on = campos_beta_def + campos_grano, how = 'left')
df_R3c = df_R3c.merge(df_U, on = campos_beta_def + campos_grano, how = 'left')
df_R3c = df_R3c.merge(df[campos_beta_def + campos_grano + ['RANGO', 'A', 'B']], on = campos_beta_def + campos_grano + ['RANGO'], how = 'left')
df_R3c = df_R3c.fillna(0)
df_R3c

df_VP_in = dic_variables['Visitas']['VP_in'][campos_beta_def + campos_grano + ['X']]
df_VP_in = df_VP_in.rename(columns = {'X': 'X_VP_in'})
df_R3c = df_R3c.merge(df_VP_in, on = campos_beta_def + campos_grano, how = 'left')
df_R3c
#df_R3c1: VP - [A + B*I] <= U * (1-y)
df_R3c['A'], df_R3c['B'], df_R3c['U'], df_R3c['L'] = round(df_R3c['A'], 3), round(df_R3c['B'], 3), round(df_R3c['U'], 3), round(df_R3c['L'], 3)

df_R3c['LHS'] = df_R3c['X_VP_in'] - (df_R3c['A'] * df_R3c['X_Inv'] + df_R3c['B'])
df_R3c['RHSU'] = df_R3c['U'] * (1 - df_R3c['X_Y'])
df_R3c['RHSL'] = df_R3c['L'] * (1 - df_R3c['X_Y'])


if activar_holguras:
    #H3c1 = model.add_var(var_type = 'C', lb = 0, name = f'H3c1') # Holgura positiva
    #H3c2 = model.add_var(var_type = 'C', lb = 0, name = f'H3c2') # Holgura negativa
    df_R3c['H3c1'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3c1_{name}') for name in df_R3c['NAME'].values] # Holgura positiva
    df_R3c['H3c2'] = [model.add_var(var_type = 'C', lb = 0, name = f'H3c2_{name}') for name in df_R3c['NAME'].values] # Holgura negativa
    df_R3c['RHSU'] += df_R3c['H3c1']
    df_R3c['RHSL'] -= df_R3c['H3c2']

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR3c\n\n')

for i in tqdm.tqdm(range(len(df_R3c))):
    
    name = df_R3c["NAME"][i]
    
    R3c1 = (df_R3c['LHS'][i] <= df_R3c['RHSU'][i])
    R3c2 = (df_R3c['LHS'][i] >= df_R3c['RHSL'][i])
    
    """
    if activar_holguras:
        #R3c1 = (df_R3c['X_VP_in'][i] - (df_R3c['A'][i] * df_R3c['X_Inv'][i] + df_R3c['B'][i]) <= df_R3c['U'][i] * (1 - df_R3c['X_Y'][i])) + H3c1
        #R3c2 = (df_R3c['X_VP_in'][i] - (df_R3c['A'][i] * df_R3c['X_Inv'][i] + df_R3c['B'][i]) >= df_R3c['L'][i] * (1 - df_R3c['X_Y'][i])) - H3c2
        R3c1 = (df_R3c['X_VP_in'][i] - (df_R3c['A'][i] * df_R3c['X_Inv'][i] + df_R3c['B'][i]) <= df_R3c['U'][i] * (1 - df_R3c['X_Y'][i])) + df_R3c['H3c1'][i]
        R3c2 = (df_R3c['X_VP_in'][i] - (df_R3c['A'][i] * df_R3c['X_Inv'][i] + df_R3c['B'][i]) >= df_R3c['L'][i] * (1 - df_R3c['X_Y'][i])) - df_R3c['H3c2'][i]
    else:
        R3c1 = (df_R3c['X_VP_in'][i] - (df_R3c['A'][i] * df_R3c['X_Inv'][i] + df_R3c['B'][i]) <= df_R3c['U'][i] * (1 - df_R3c['X_Y'][i]))
        R3c2 = (df_R3c['X_VP_in'][i] - (df_R3c['A'][i] * df_R3c['X_Inv'][i] + df_R3c['B'][i]) >= df_R3c['L'][i] * (1 - df_R3c['X_Y'][i]))
    """
    model += R3c1
    model += R3c2

    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3c1({name}): {R3c1}  \n')
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3c2({name}): {R3c2}  \n')
    
"""  
#df_R3c2: VP - [A + B*I] >= L * (1-y)

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR3c2\n\n')

for i in tqdm.tqdm(range(len(df_R3c))):
    
    name = df_R3c["NAME"][i]
    if activar_holguras:
        R3c2 = (df_R3c['X_VP_in'][i] - (df_R3c['A'][i] * df_R3c['X_Inv'][i] + df_R3c['B'][i]) >= df_R3c['L'][i] * (1 - df_R3c['X_Y'][i])) - H3c2
    else:
        R3c2 = (df_R3c['X_VP_in'][i] - (df_R3c['A'][i] * df_R3c['X_Inv'][i] + df_R3c['B'][i]) >= df_R3c['L'][i] * (1 - df_R3c['X_Y'][i]))
    model += R3c2

    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R3c2({name}): {R3c2}  \n')
"""
df_times, t0 = medir_tiempo(t0, 'M R3c', df_times)
### Seccion: ## 4. Suma de variables aditivas
simbolos = {'Venta': 'S', 'Ordenes': 'O', 'Visitas': 'V', 'Inversion': 'I'}
### Seccion: ### 4.1. Relación de Inv con Inv all
# Desarrollo actual (reemplazar bloque de arriba)
df_inv_grano_0 = dic_variables['Inversion']['Inv']
df_inv_grano_0 = df_inv_grano_0.rename(columns = {'X': 'X_Inv_0'})
df_inv_all = dic_variables['Inversion']['Inv_all'].rename(columns = {'X': 'X_Inv_1'})
df_inv_all
for duplicacion in df_inv_all['DUPLICACION'].unique():
    
    #if duplicacion != "F-LT":
    #    continue
    print(duplicacion)
    df_inv_all_duplicacion = df_inv_all[df_inv_all['DUPLICACION'] == duplicacion].reset_index(drop = True)

    campos_dim = campos_beta_def[:]

    for d in duplicacion.split('-'):
        dim_name = diccionario_dimensiones[d]
        if dim_name != "":
            campos_dim += [dim_name]
    
    campos_metricas = ['X_Inv_1']
    df_inv_all_duplicacion = df_inv_all_duplicacion[campos_dim + campos_metricas].groupby(campos_dim, as_index = False).sum().reset_index(drop = True)
    
    # # Lo de arriba es solo reducir campos..no debería "agruparse" realmente...el len no debería bajar...a diferencia de abajo, en la agrupación de df_inv_grano_0
    
    campos_metricas = ['X_Inv_0']
    df_inv_grano_0_agr = df_inv_grano_0[['NAME'] + campos_dim + campos_metricas].groupby(campos_dim, as_index = False).sum().reset_index(drop = True)
    
    df_R = df_inv_grano_0_agr.merge(df_inv_all_duplicacion, on = campos_dim, how = 'left')
    print(df_R.head())

    escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR40: Relación de inversión grano con inversión x {duplicacion} \n\n')
    
    #if activar_holguras:
    #    df_R[f'H4a0_p_{duplicacion}'] = [model.add_var(var_type = 'C', lb = 0, name = f'H4a0_p{name}') for name in df_R['NAME'].values]
    #    df_R[f'H4a0_n_{duplicacion}'] = [model.add_var(var_type = 'C', lb = 0, name = f'H4a0_n{name}') for name in df_R['NAME'].values]
        
    for i in tqdm.tqdm(range(len(df_R))):
        name = df_R["NAME"][i]
        #if activar_holguras:
        #    R4a0 = (df_R[f'X_Inv_0'][i] == df_R[f'X_Inv_1'][i] + df_R['H4a0_p'][i] - df_R['H4a0_n'][i])
        #else:
        R4a0 = (df_R[f'X_Inv_0'][i] == df_R[f'X_Inv_1'][i])
            
        #R4a0 = (df_R[f'X_Inv_0'][i] == df_R[f'X_Inv_1'][i])
        model += R4a0

        if i < escribir_modelo[1]:
            escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R4a0({name}): {R4a0}  \n')
        
    # sys.exit('Seguir revisión')
### Seccion: ### 4.2. Otras métricas
for m in simbolos:
    if m == 'Inversion':
        continue
    print(m)
    m_s = simbolos[m]
    df_o = dic_variables[m][f'{m_s}O'].rename(columns = {'X': f'X_{m_s}O'}) # grano
    #print(df_o.head())
    df_p = dic_variables[m][f'{m_s}P'].rename(columns = {'X': f'X_{m_s}P'}) # grano
    #print(df_p.head())
    df_t = dic_variables[m][f'{m_s}T'].rename(columns = {'X': f'X_{m_s}T'}) # todos
    #print(df_t.head())
    
    # Relación de grano con total
    
    escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR4 {m}\n\n')
    
    # df_R4a: Orgánico
    df_R4a = df_o.merge(df_t[campos_beta_def + campos_grano + [f'X_{m_s}T']], on = campos_beta_def + campos_grano, how = 'left')
    
    for i in tqdm.tqdm(range(len(df_R4a))):
        R4a = (df_R4a[f'X_{m_s}O'][i] == df_R4a[f'X_{m_s}T'][i])
        model += R4a

        if i < escribir_modelo[1]:
            escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R4a({df_R4a["NAME"][i]}): {R4a}  \n')
    
    # df_R4b: Pago
    df_R4b = df_p.merge(df_t[campos_beta_def + campos_grano + [f'X_{m_s}T']], on = campos_beta_def + campos_grano, how = 'left')
    
    for i in tqdm.tqdm(range(len(df_R4b))):
        R4b = (df_R4b[f'X_{m_s}P'][i] == df_R4b[f'X_{m_s}T'][i])
        model += R4b

        if i < escribir_modelo[1]:
            escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R4b({df_R4b["NAME"][i]}): {R4b}  \n')
    
df_times, t0 = medir_tiempo(t0, 'M R4', df_times)
### Seccion: ## 5. Relación de niveles de duplicidad
lista_duplicidades
df_relaciones_duplicidad = dic_parametros['RELACIONES DUPLICIDAD']
df_relaciones_duplicidad = df_relaciones_duplicidad[(df_relaciones_duplicidad['FAMILIA'].isin(list(familia_sm) + [''])) & (df_relaciones_duplicidad['TIPO_MEDIO'].isin(list(tm_seleccion) + ['']))]
df_relaciones_duplicidad = df_relaciones_duplicidad[(df_relaciones_duplicidad['AGREGADO'].isin(lista_duplicidades)) & (df_relaciones_duplicidad['DESAGREGADO'].isin(lista_duplicidades))].reset_index(drop = True)
df_relaciones_duplicidad
dic_relaciones = {'Venta': 'VENTA_COLOCADA', 'Ordenes': 'ORDENES', 'Visitas': 'VISITAS', 'Inversion': 'INVERSION'}
diccionario_dimensiones
"""
df_relaciones_duplicidad['NAME'] = df_relaciones_duplicidad.index.astype(str)

H5_p, H5_n = {}, {}

for m in simbolos:
    if not activar_holguras:
        break
    
    df_relaciones_duplicidad[f'H5_p_{m}'] = [model.add_var(var_type = 'C', lb = 0, name = f'H5_p_{m}_{name}') for name in df_relaciones_duplicidad['NAME'].values] # Holgura positiva
    df_relaciones_duplicidad[f'H5_n_{m}'] = [model.add_var(var_type = 'C', lb = 0, name = f'H5_n_{m}_{name}') for name in df_relaciones_duplicidad['NAME'].values] # Holgura negativa
    
    #H5_p[m] = model.add_var(var_type = 'C', lb = 0, name = f'H5_p_{m}')
    #H5_n[m] = model.add_var(var_type = 'C', lb = 0, name = f'H5_n_{m}')
"""                           

df_beta_alpha_all = pd.concat([df_beta_alpha_grano_organico, df_beta_alpha_grano_pago], axis = 0).reset_index(drop = True)
df_beta_alpha_all = df_beta_alpha_all[campos_beta_def + ['FAMILIA']].drop_duplicates().reset_index(drop = True)
df_beta_alpha_all

df_beta_alpha_all_totales = df_beta_alpha_all[campos_beta_def].drop_duplicates().reset_index(drop = True)
print(simbolos)
print(dic_relaciones)
df_relaciones_duplicidad['f_INVERSION'] = 1 # new 241226
dic_variables.keys()
df_relaciones_duplicidad

df_relaciones_duplicidad
H = 0
# Suma de holguras
conjunto_holguras_contraste = set()

df_relaciones_duplicidad = df_beta_def.merge(df_relaciones_duplicidad, how = 'left', on = list(df_beta_def.columns)).reset_index(drop = True)

for m in simbolos:
    
    m_s = simbolos[m]
    metrica_name = dic_relaciones[m]
    
    if m == 'Inversion': # new 241226
        df_m = dic_variables[m]['Inv_all'].rename(columns = {'X': f'X_{m_s}T'}) # todos
    else:
        df_m = dic_variables[m][f'{m_s}T'].rename(columns = {'X': f'X_{m_s}T'}) # todos
    
    for i in range(len(df_arcos)):
        
        agr, desagr = df_arcos['AGREGADO'][i], df_arcos['DESAGREGADO'][i]
        
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'\nR5 {m} {agr} | {desagr}\n\n')
        
        campos_extra = []
        for c in agr.split('-'):
            if c == 'TOTALES':
                continue
            campos_extra.append(diccionario_dimensiones[c])
            
        print(m, m_s, metrica_name)
        print(agr, desagr)

        # Caso específico agr &  desagr
        df_relaciones_duplicidad_i = df_relaciones_duplicidad[(df_relaciones_duplicidad['AGREGADO'] == agr) & (df_relaciones_duplicidad['DESAGREGADO'] == desagr)].reset_index(drop = True)

        """ Se agregan tuplas (de campos_beta_def) que podrían no existir. Son necesarias para el cumplimiento correcto de las restricciones"""
        if agr == 'TOTALES':
            df_relaciones_duplicidad_i_tuplas = df_relaciones_duplicidad_i[campos_beta_def].drop_duplicates().reset_index(drop = True)
            df_relaciones_duplicidad_i_tuplas['IN'] = True
            
            #print('A')
            #print(df_relaciones_duplicidad_i_tuplas)
            df_beta_alpha_all_totales_aux = df_beta_alpha_all_totales.copy()
            df_beta_alpha_all_totales_aux = df_beta_alpha_all_totales_aux.merge(df_relaciones_duplicidad_i_tuplas, on = campos_beta_def, how = 'left')
            
            #print('B')
            #print(df_beta_alpha_all_totales_aux)
            
            df_beta_alpha_all_totales_aux['IN'] = df_beta_alpha_all_totales_aux['IN'].fillna(False)
            df_beta_alpha_all_totales_aux = df_beta_alpha_all_totales_aux[~df_beta_alpha_all_totales_aux['IN']].reset_index(drop = True)
            df_beta_alpha_all_totales_aux = df_beta_alpha_all_totales_aux.drop(columns = ['IN'])
            df_relaciones_duplicidad_i = pd.concat([df_relaciones_duplicidad_i, df_beta_alpha_all_totales_aux], axis = 0).reset_index(drop = True)
            
        else:
            df_relaciones_duplicidad_i_tuplas = df_relaciones_duplicidad_i[campos_beta_def + ['FAMILIA']].drop_duplicates().reset_index(drop = True)
            df_relaciones_duplicidad_i_tuplas['IN'] = True
            df_beta_alpha_all_aux = df_beta_alpha_all.copy()
            df_beta_alpha_all_aux = df_beta_alpha_all_aux.merge(df_relaciones_duplicidad_i_tuplas, on = campos_beta_def + ['FAMILIA'], how = 'left')
            df_beta_alpha_all_aux['IN'] = df_beta_alpha_all_aux['IN'].fillna(False)
            df_beta_alpha_all_aux = df_beta_alpha_all_aux[~df_beta_alpha_all_aux['IN']].reset_index(drop = True)
            df_beta_alpha_all_aux = df_beta_alpha_all_aux.drop(columns = ['IN'])

            df_relaciones_duplicidad_i = pd.concat([df_relaciones_duplicidad_i, df_beta_alpha_all_aux], axis = 0).reset_index(drop = True)

        # Fillna con el valor que esá arriba
        df_relaciones_duplicidad_i[['AGREGADO', 'DESAGREGADO'] + campos_grano] = df_relaciones_duplicidad_i[['AGREGADO', 'DESAGREGADO'] + campos_grano].fillna(method = 'ffill')
        df_relaciones_duplicidad_i = df_relaciones_duplicidad_i.fillna(0)
        
        """ Se itera en los distintos casos"""
        
        if 'DATE' in campos_beta_def:
            df_relaciones_duplicidad_i['DATE'], df_m['DATE'] = pd.to_datetime(df_relaciones_duplicidad_i['DATE']), pd.to_datetime(df_m['DATE'])
        
        df_agr = df_m[['DUPLICACION'] + campos_beta_def + campos_grano + [f'X_{m_s}T']].rename(columns = {'DUPLICACION': 'AGREGADO', f'X_{m_s}T': f'X_{m_s}T_AGR'})
        df_agr = df_agr[df_agr['AGREGADO'] == agr]
        
        df_desagr = df_m[['DUPLICACION'] + campos_beta_def + campos_grano + [f'X_{m_s}T']].rename(columns = {'DUPLICACION': 'DESAGREGADO', f'X_{m_s}T': f'X_{m_s}T_DESAGR'})
        df_desagr = df_desagr[df_desagr['DESAGREGADO'] == desagr]

        if list(df_relaciones_duplicidad_i['FAMILIA'].unique()) == ['']:
            df_desagr['FAMILIA'] =  ''

        if list(df_relaciones_duplicidad_i[campo_last_touch].unique()) == ['']:
            df_desagr[campo_last_touch] =  ''

        #print('A')
        #print(df_relaciones_duplicidad_i)
        #print(df_agr)
        
        df_relaciones_duplicidad_i = df_relaciones_duplicidad_i.merge(df_agr, on = ['AGREGADO'] + campos_beta_def + campos_grano, how = 'left')
        df_relaciones_duplicidad_i = df_relaciones_duplicidad_i.merge(df_desagr, on = ['DESAGREGADO'] + campos_beta_def + campos_grano, how = 'left')

        #print('A0') # cambio 250116 (drop duplicates se hace abajo, post declarar name como str)
        df_relaciones_duplicidad_i_dict_nombres = df_relaciones_duplicidad_i[[f'X_{m_s}T_AGR']]#.drop_duplicates()
        #print(df_relaciones_duplicidad_i_dict_nombres)
        df_relaciones_duplicidad_i_dict_nombres[f'X_{m_s}T_AGR_NAME'] = df_relaciones_duplicidad_i_dict_nombres[f'X_{m_s}T_AGR'].astype(str)
        df_relaciones_duplicidad_i_dict_nombres = df_relaciones_duplicidad_i_dict_nombres.drop_duplicates().reset_index(drop = True)
        #print(df_relaciones_duplicidad_i_dict_nombres)
    
        #print('A2')  
        #print(df_relaciones_duplicidad_i)
        
        df_relaciones_duplicidad_i[f'X_{m_s}T_AGR_NAME'] = df_relaciones_duplicidad_i[f'X_{m_s}T_AGR'].astype(str)
        #print('A2.5')  
        #print(df_relaciones_duplicidad_i)
        
        df_relaciones_duplicidad_i = df_relaciones_duplicidad_i[['AGREGADO', 'DESAGREGADO'] + campos_beta_def + campos_grano + [f'f_{metrica_name}', f'X_{m_s}T_AGR_NAME', f'X_{m_s}T_DESAGR']].groupby(['AGREGADO', 'DESAGREGADO'] + campos_beta_def + campos_grano + [f'f_{metrica_name}', f'X_{m_s}T_AGR_NAME'], as_index = False).sum().reset_index(drop = True)
        
        #print('A2.7')  
        #print(df_relaciones_duplicidad_i)
        #print(df_relaciones_duplicidad_i_dict_nombres)
        
        df_relaciones_duplicidad_i = df_relaciones_duplicidad_i.merge(df_relaciones_duplicidad_i_dict_nombres, on = f'X_{m_s}T_AGR_NAME', how = 'left')
        
        #print('A3')  
        #print(df_relaciones_duplicidad_i)
        
        df_relaciones_duplicidad_i['NAME'] = ''
        for c in ['AGREGADO', 'DESAGREGADO'] + campos_beta_def + campos_grano:
            df_relaciones_duplicidad_i['NAME'] += (df_relaciones_duplicidad_i[c].astype(str) + '_')
        df_relaciones_duplicidad_i['NAME'] = df_relaciones_duplicidad_i['NAME'].str[:-1]
        
        if activar_holguras:
            df_relaciones_duplicidad_i[f'H5_p_{m}'] = [model.add_var(var_type = 'C', lb = 0, name = f'H5_p_{m}_{name}') for name in df_relaciones_duplicidad_i['NAME'].values]
            df_relaciones_duplicidad_i[f'H5_n_{m}'] = [model.add_var(var_type = 'C', lb = 0, name = f'H5_n_{m}_{name}') for name in df_relaciones_duplicidad_i['NAME'].values]
            conjunto_holguras_contraste.add(f'H5_p')
            conjunto_holguras_contraste.add(f'H5_n')
        
        #print('A4')  
        #print(df_relaciones_duplicidad_i)
        
        #sys.exit()

        for j in tqdm.tqdm(range(len(df_relaciones_duplicidad_i))):
            
            name = df_relaciones_duplicidad_i[f'X_{m_s}T_AGR_NAME'][j]
            
            #print(name, (f'f_{metrica_name}', j), round(df_relaciones_duplicidad_i[f'f_{metrica_name}'][j], 3))
            #print(df_relaciones_duplicidad_i)
            
            df_relaciones_duplicidad_i
            
            
            if activar_holguras:
                #R5 = (df_relaciones_duplicidad_i[f'X_{m_s}T_AGR'][j] == df_relaciones_duplicidad_i[f'f_{metrica_name}'][j] * df_relaciones_duplicidad_i[f'X_{m_s}T_DESAGR'][j] + H5_p[m] - H5_n[m])
                R5 = (df_relaciones_duplicidad_i[f'X_{m_s}T_AGR'][j] == round(df_relaciones_duplicidad_i[f'f_{metrica_name}'][j], 3) * df_relaciones_duplicidad_i[f'X_{m_s}T_DESAGR'][j] + df_relaciones_duplicidad_i[f'H5_p_{m}'][j] - df_relaciones_duplicidad_i[f'H5_n_{m}'][j])
                H += df_relaciones_duplicidad_i[f'H5_p_{m}'][j] + df_relaciones_duplicidad_i[f'H5_n_{m}'][j]
            else:
                R5 = (df_relaciones_duplicidad_i[f'X_{m_s}T_AGR'][j] == round(df_relaciones_duplicidad_i[f'f_{metrica_name}'][j], 3) * df_relaciones_duplicidad_i[f'X_{m_s}T_DESAGR'][j])
            
            model += R5

            if j < escribir_modelo[1]:
                escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R5({name}): {R5}  \n') 

df_times, t0 = medir_tiempo(t0, 'M R5', df_times)
df_times
#sys.exit('M R3c estaba en 4.24 mins')
### Seccion: ## 6. Relación de Visitas & Visitas IN
campos_clacom = list(set(campos_grano) - {campo_last_touch})
campos_clacom

campos_clacom_inicial = []
for c in campos_clacom:
    campos_clacom_inicial.append(f'{c}_INICIAL')
    
campos_clacom_final = []
for c in campos_clacom:
    campos_clacom_final.append(f'{c}_FINAL')
df_visitas_pago = dic_variables['Visitas']['VP'][campos_beta_def + campos_grano  + ['X', 'NAME']].rename(columns = {'X': 'X_VP'})
df_visitas_pago_in = dic_variables['Visitas']['VP_in'][campos_beta_def + campos_grano  + ['X']].rename(columns = {'X': 'X_VP_in'})
df_relacion_visitas = dic_parametros['RELACION_VISITAS']


df_visitas_pago_final = df_visitas_pago.copy()
for c in campos_clacom:
    df_visitas_pago_final = df_visitas_pago_final.rename(columns = {c: c + '_FINAL'})
df_visitas_pago_final
df_visitas_pago_final = df_visitas_pago_final.merge(df_relacion_visitas, on = campos_beta_def + campos_clacom_final + [campo_last_touch], how = 'left')

# Los casos sin familia inicial, no están en df_relacion_visitas...es decir, no hay flujo a estas familias finales -> FLujo = 0

df_visitas_pago_final[['FAMILIA_INICIAL', 'M']] = df_visitas_pago_final[['FAMILIA_INICIAL', 'M']].fillna(0)
df_visitas_pago_final['M'] = round(df_visitas_pago_final['M'], 3)

for c in campos_clacom:
    df_visitas_pago_in = df_visitas_pago_in.rename(columns = {c: c + '_INICIAL'})
    
df_visitas_pago_final = df_visitas_pago_final.merge(df_visitas_pago_in, on = campos_beta_def + campos_clacom_inicial + [campo_last_touch], how = 'left')

df_X_VP_name = df_visitas_pago_final[['X_VP']] # 250116
df_X_VP_name['X_VP_NAME'] = df_X_VP_name['X_VP'].astype(str)
df_X_VP_name = df_X_VP_name.drop_duplicates()

df_visitas_pago_final['X_VP_NAME'] = df_visitas_pago_final['X_VP'].astype(str)
df_visitas_pago_final['M_X_VP_in'] = df_visitas_pago_final['M'] * df_visitas_pago_final['X_VP_in']
df_visitas_pago_final = df_visitas_pago_final[campos_beta_def + campos_clacom_final + [campo_last_touch] + ['X_VP_NAME', 'M_X_VP_in']].groupby(campos_beta_def + campos_clacom_final + [campo_last_touch] + ['X_VP_NAME'], as_index = False).sum().reset_index(drop = True)
df_visitas_pago_final = df_visitas_pago_final.merge(df_X_VP_name, on = 'X_VP_NAME', how = 'left')

print(df_visitas_pago_final)

df_visitas_pago_final['NAME'] = ''
for c in campos_beta_def + campos_clacom_final + [campo_last_touch]:
    df_visitas_pago_final['NAME'] += (df_visitas_pago_final[c].astype(str) + '_')
df_visitas_pago_final['NAME'] = df_visitas_pago_final['NAME'].str[:-1]

if activar_holguras:
    #H6_p = model.add_var(var_type = 'C', lb = 0, name = f'H6_p') # Holgura positiva
    #H6_n = model.add_var(var_type = 'C', lb = 0, name = f'H6_n') # Holgura neg
    
    df_visitas_pago_final['H6_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H6_p_{name}') for name in df_visitas_pago_final['NAME'].values] # Holgura positiva
    df_visitas_pago_final['H6_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H6_n_{name}') for name in df_visitas_pago_final['NAME'].values] # Holgura negativa


for i in tqdm.tqdm(range(len(df_visitas_pago_final))):
    
    name = df_visitas_pago_final['X_VP_NAME'][i]
    
    if activar_holguras:
        #R6 = (df_visitas_pago_final['X_VP'][i] == df_visitas_pago_final['M_X_VP_in'][i] + H6_p - H6_n)
        R6 = (df_visitas_pago_final['X_VP'][i] == df_visitas_pago_final['M_X_VP_in'][i] + df_visitas_pago_final['H6_p'][i] - df_visitas_pago_final['H6_n'][i])
    else:
        R6 = (df_visitas_pago_final['X_VP'][i] == df_visitas_pago_final['M_X_VP_in'][i])
    model += R6
    
    if i <= escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R6({name}): {R6}  \n')

df_times, t0 = medir_tiempo(t0, 'M R6', df_times)
### Seccion: ## 7. Relación de venta neta y colocada
ratio_neta_colocada = dic_parametros['METAS']['RATIO NETA COLOCADA']
ratio_neta_colocada.head()
df_R7 = dic_variables['Venta']['ST'].rename(columns = {'X': 'X_ST'})
df_venta_neta = dic_variables['Venta']['ST_net'][['NAME', 'X']].rename(columns = {'X': 'X_ST_net'})

df_R7 = df_R7.merge(df_venta_neta, on = 'NAME', how = 'left')

if not modo_tactico:
    df_R7['PERIODO'] = df_R7['DATE'].astype(str).str[:7]

df_R7 = df_R7.merge(ratio_neta_colocada, on = ['PAIS', 'PERIODO', 'CANAL', 'FUENTE'], how = 'left')

if activar_holguras:
    H7_p = model.add_var(var_type = 'C', lb = 0, name = 'H7_p') # Holgura positiva
    H7_n = model.add_var(var_type = 'C', lb = 0, name = 'H7_n') # Holgura negativa
    
    df_R7['H7_p'] = [model.add_var(var_type = 'C', lb = 0, name = f'H7_p_{name}') for name in df_R7['NAME'].values] # Holgura positiva
    df_R7['H7_n'] = [model.add_var(var_type = 'C', lb = 0, name = f'H7_n_{name}') for name in df_R7['NAME'].values] # Holgura negativa

df_R7['RATIO_NETA_COLOCADA'] = round(df_R7['RATIO_NETA_COLOCADA'], 3)

#H2 = 0
for i in tqdm.tqdm(range(len(df_R7))):
    
    name = df_R7["NAME"][i]
    
    if activar_holguras:
        R7 = (df_R7['X_ST'][i]  * df_R7['RATIO_NETA_COLOCADA'][i] == df_R7['X_ST_net'][i] + H7_p - H7_n)
        R7 = (df_R7['X_ST'][i]  * df_R7['RATIO_NETA_COLOCADA'][i] == df_R7['X_ST_net'][i] + df_R7['H7_p'][i] - df_R7['H7_n'][i])
    else:
        R7 = (df_R7['X_ST'][i]  * df_R7['RATIO_NETA_COLOCADA'][i] == df_R7['X_ST_net'][i])
    #R7 = (df_R7['X_ST'][i]  * df_R7['RATIO_NETA_COLOCADA'][i] == df_R7['X_ST_net'][i])
    model += R7
    #H2 += (df_R7['H7_p'][i] + df_R7['H7_n'][i])
    
    if i < escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R7({name}): {R7}  \n')

# Este bloque, antes de H2, estaba en 1.20 mins
df_times, t0 = medir_tiempo(t0, 'M R7', df_times)
### Seccion: ## 7.5. Relaciones de inversión 
## ?? Esto ya está cubierto
### Seccion: ## 8. Sensibilizables y cumplimiento de metas
dic_parametros['METAS'].keys()

# Esto es todo lo que está disponible para sensibilizar en GSheets
df_restricciones = df_escenarios[df_escenarios['TIPO'] == 'R'].reset_index(drop = True)
df_restricciones
def depuracion_temporal(filtro, c):
    if c == 'PERIODO':
        periodo0, periodo1 = filtro[0].split('>')
        date0 = pd.to_datetime(periodo0)
        date1 = pd.to_datetime(periodo1)

        df_dates = pd.DataFrame(pd.date_range(date0, date1), columns = ['DATE'])
        df_dates['PERIODO'] = df_dates['DATE'].astype(str).str[:7]
        return list(df_dates['PERIODO'].unique())

    elif c == 'DATE':
        date0, date1 = filtro[0].split('>')
        date0 = pd.to_datetime(date0)
        date1 = pd.to_datetime(date1)
        
        df_dates = pd.DataFrame(pd.date_range(date0, date1), columns = ['DATE'])
        return list(df_dates['DATE'].unique())

    elif c == 'AÑO':
        año0, año1 = filtro[0].split('>')
        año0 = int(año0)
        año1 = int(año1)
        
        return list(range(año0, año1 + 1))

    else:
        return filtro
for i in range(len(df_restricciones)):
    
    if version_simplificada:
        break
    
    #break

    print('Si o si estas restrs tienen que ir con holguras. Aunque no exista ninguna otra holgura en el modelo')
    
    #print('ELIMINAR (2)')
    
    #if i > 4:
    #    break
    
    #if i <= 4: # Hasta aquí están ok revisados
    #    continue
    
    df_i = df_restricciones.iloc[i:i + 1].reset_index(drop = True)
    
    eliminar_cols = ['ESCENARIO', 'Descripción / Comentarios (Opcional)', 'TIPO']
    for c in df_i.columns:
        if df_i[c][0] == '':
            eliminar_cols.append(c)
    
    descripcion = df_i['Descripción / Comentarios (Opcional)'][0]
    
    df_i = df_i.drop(columns = eliminar_cols)
    
    print(f'\n\n\n\n\n i: {i}: {descripcion}')
    print(df_i)
    
    # Primero: Determinar en que nivel de dupicidad / duplicación, aplicar la restricción
    
    lista_duplicacion = []
    if 'Base + Config_FAMILIA' in df_i.columns:
        lista_duplicacion.append('F') 
    if 'Base + Config_SUBFAMILIA' in df_i.columns:
        lista_duplicacion.append('SF')
    if f'Base + Config_{campo_last_touch}' in df_i.columns:
        lista_duplicacion.append('LT')
    
    if lista_duplicacion == []:
        duplicacion = 'TOTALES'
    else:
        duplicacion = '-'.join(lista_duplicacion)
    
    # Selección de variable para la restricción (Métrica)
    
    metrica = df_i['Condiciones de restricción_METRICA'][0]
    ratio = False
    
    if metrica == 'NMV':
        df_base = dic_variables['Venta']['ST_net']
        df_X = df_base[df_base['DUPLICACION'] == duplicacion].reset_index(drop = True)
        df_P_Base = dic_parametros['METAS']['VENTA_NETA_BASE']
        df_P = dic_parametros['METAS']['VENTA_NETA']
        metrica_name = 'VENTA_NETA'
    elif metrica == 'GMV':
        df_base = dic_variables['Venta']['ST']
        df_X = df_base[df_base['DUPLICACION'] == duplicacion].reset_index(drop = True)
        df_P = dic_parametros['METAS']['VENTA_COLOCADA']
        metrica_name = 'VENTA_COLOCADA'
    elif metrica == 'O':
        df_base = dic_variables['Ordenes']['OT']
        df_X = df_base[df_base['DUPLICACION'] == duplicacion].reset_index(drop = True)
        df_P = dic_parametros['METAS']['ORDENES']
        metrica_name = 'ORDENES'
    elif metrica == 'V':
        df_base = dic_variables['Visitas']['VT']
        df_X = df_base[df_base['DUPLICACION'] == duplicacion].reset_index(drop = True)
        df_P = dic_parametros['METAS']['VISITAS']
        metrica_name = 'VISITAS'
    elif metrica == 'TC':
        df_base_ordenes = dic_variables['Ordenes']['OT']
        df_base_ordenes = df_base_ordenes[df_base_ordenes['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_OT'})
        df_base_visitas = dic_variables['Visitas']['VT']
        df_base_visitas = df_base_visitas[df_base_visitas['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_VT'})
        cols_merge = list(set(df_base_ordenes.columns) - {'NAME', 'X_OT'})
        df_X = df_base_ordenes[cols_merge + ['X_OT']].merge(df_base_visitas[cols_merge + ['X_VT']], on = cols_merge, how = 'left')
        
        # Para ratios
        df_ordenes = dic_parametros['METAS']['ORDENES']
        df_visitas = dic_parametros['METAS']['VISITAS']
        df_P = df_ordenes.merge(df_visitas, on = ['PAIS', 'PERIODO', 'CANAL_BASE'], how = 'left')
        
        metricas_aux = [['X_OT', 'X_VT'], ['ORDENES', 'VISITAS']]
        metrica_name = 'TC'
        ratio = True

    elif metrica == 'I':
        df_base = dic_variables['Inversion']['Inv_all']
        df_X = df_base[df_base['DUPLICACION'] == duplicacion].reset_index(drop = True)
        df_P = dic_parametros['METAS']['INVERSION']
        metrica_name = 'INVERSION'
        
    elif metrica == 'TP':
        df_base_venta_colocada = dic_variables['Venta']['ST']
        df_base_venta_colocada = df_base_venta_colocada[df_base_venta_colocada['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_ST'})
        df_base_ordenes = dic_variables['Ordenes']['OT']
        df_base_ordenes = df_base_ordenes[df_base_ordenes['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_OT'})
        cols_merge = list(set(df_base_venta_colocada.columns) - {'NAME', 'X_ST'})
        df_X = df_base_venta_colocada[cols_merge + ['X_ST']].merge(df_base_ordenes[cols_merge + ['X_OT']], on = cols_merge, how = 'left')
        df_P = dic_parametros['METAS']['TP']
        metrica_name = 'TP'
        sys.exit('Ver en TC, para ratios (# Para ratios) y replicar')
        
    elif metrica == 'CV':
        df_base_inversion = dic_variables['Inversion']['Inv_all']
        df_base_inversion = df_base_inversion[df_base_inversion['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_INV'})
        df_base_venta_colocada = dic_variables['Venta']['ST']
        df_base_venta_colocada = df_base_venta_colocada[df_base_venta_colocada['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_ST'})
        cols_merge = list(set(df_base_inversion.columns) - {'NAME', 'X_INV'})
        df_X = df_base_inversion[cols_merge + ['X_INV']].merge(df_base_venta_colocada[cols_merge + ['X_ST']], on = cols_merge, how = 'left')
        df_P_VC = dic_parametros['METAS']['VENTA_COLOCADA']
        df_P_VC = df_P_VC[['PAIS', 'PERIODO', 'VENTA_COLOCADA']].groupby(['PAIS', 'PERIODO'], as_index = False).sum()
        df_P = df_P_VC.merge(dic_parametros['METAS']['INVERSION'], on = ['PAIS', 'PERIODO'], how = 'left')
        df_P['CV'] = df_P['INVERSION'] / df_P['VENTA_COLOCADA']
        df_P = df_P[['PAIS', 'PERIODO', 'CV']]
        metrica_name = 'CV'
        sys.exit('Ver en TC, para ratios (# Para ratios) y replicar')
    elif metrica == 'CVN':
        df_base_inversion = dic_variables['Inversion']['Inv_all']
        df_base_inversion = df_base_inversion[df_base_inversion['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_INV'})
        df_base_venta_neta = dic_variables['Venta']['ST_net']
        df_base_venta_neta = df_base_venta_neta[df_base_venta_neta['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_ST_net'})
        cols_merge = list(set(df_base_inversion.columns) - {'NAME', 'X_INV'})
        df_X = df_base_inversion[cols_merge + ['X_INV']].merge(df_base_venta_neta[cols_merge + ['X_ST_net']], on = cols_merge, how = 'left')
        df_P_VN = dic_parametros['METAS']['VENTA_NETA']
        df_P_VN = df_P_VN[['PAIS', 'PERIODO', 'VENTA_NETA']].groupby(['PAIS', 'PERIODO'], as_index = False).sum()
        df_P = df_P_VN.merge(dic_parametros['METAS']['INVERSION'], on = ['PAIS', 'PERIODO'], how = 'left')
        df_P['CVN'] = df_P['INVERSION'] / df_P['VENTA_NETA']
        df_P = df_P[['PAIS', 'PERIODO', 'CVN']]
        metrica_name = 'CVN'
        sys.exit('Ver en TC, para ratios (# Para ratios) y replicar')
    elif metrica == 'CVis':
        df_base_inversion = dic_variables['Inversion']['Inv_all']
        df_base_inversion = df_base_inversion[df_base_inversion['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_INV'})
        df_base_visitas = dic_variables['Visitas']['VT']
        df_base_visitas = df_base_visitas[df_base_visitas['DUPLICACION'] == duplicacion].reset_index(drop = True).rename(columns = {'X': 'X_VT'})
        cols_merge = list(set(df_base_inversion.columns) - {'NAME', 'X_INV'})
        df_X = df_base_inversion[cols_merge + ['X_INV']].merge(df_base_visitas[cols_merge + ['X_VT']], on = cols_merge, how = 'left')
        df_P_Vis = dic_parametros['METAS']['VISITAS']
        df_P_Vis = df_P_Vis[['PAIS', 'PERIODO', 'VISITAS']].groupby(['PAIS', 'PERIODO'], as_index = False).sum()
        df_P = df_P_Vis.merge(dic_parametros['METAS']['INVERSION'], on = ['PAIS', 'PERIODO'], how = 'left')
        df_P['CVis'] = df_P['INVERSION'] / df_P['VISITAS']
        df_P = df_P[['PAIS', 'PERIODO', 'CVis']]
        metrica_name = 'CVis'
        sys.exit('Ver en TC, para ratios (# Para ratios) y replicar')
    else:
        sys.exit(f'Metrica {metrica} no configurada')
        
    relacion = df_i['Condiciones de restricción_RELACION'][0]
    contraste = df_i['Condiciones de restricción_CONTRASTE'][0]
    para_cada = df_i['Condiciones de restricción_PARA_CADA'][0].split(',')
    cond_share = df_i['Condiciones de restricción_SHARE'][0]
    
    ponderador_contraste = 1 # Por defecto
    if 'Condiciones de restricción_PONDERADOR_CONTRASTE' in df_i.columns:
        ponderador_contraste = float(df_i['Condiciones de restricción_PONDERADOR_CONTRASTE'][0])
    
    # if metrica = NMV, también puede influir df_P_Base
    if ('F' in duplicacion) or (metrica != 'NMV'):
        None
    else:
        df_P = df_P_Base.copy()
        
    df_X['CANAL_BASE'] = np.where(df_X['FUENTE'] == 'SODIMAC', df_X['CANAL'], 'FCOM')
    
    # Aquí deberían ir los filtros en casos puntuales
    # ANCLA 0
    campos_particion = ['PAIS', 'PERIODO', 'CANAL', 'FUENTE', 'FAMILIA', 'SUBFAMILIA', campo_last_touch, 'DATE', 'CANAL_BASE', 'NATURALEZA_MEDIO', 'AÑO']
    
    if 'PERIODO' in para_cada:
        if ('DATE' in df_X.columns) and not ('PERIODO' in df_X.columns):
            df_X['PERIODO'] = df_X['DATE'].astype(str).str[:7]
        # lo mismo para df_P
        if ('DATE' in df_P.columns) and not ('PERIODO' in df_P.columns):
            df_P['PERIODO'] = df_P['DATE'].astype(str).str[:7]
    
    if 'AÑO' in para_cada:
        if ('DATE' in df_X.columns) and not ('AÑO' in df_X.columns):
            df_X['AÑO'] = df_X['DATE'].astype(str).str[:4].astype(int)
        # lo mismo para df_P
        if ('DATE' in df_P.columns) and not ('AÑO' in df_P.columns):
            df_P['AÑO'] = df_P['DATE'].astype(str).str[:4].astype(int)

    existen_filtros = False
    for c in campos_particion:
        if (f'Base + Config_{c}' in df_i.columns) or (f'Aux_{c}' in df_i.columns):
            existen_filtros = True
            break

    if existen_filtros:
        if cond_share == 'TRUE':

            df_X_L, df_X_R = df_X.copy(), df_X.copy() #Left y Right
            df_X_L = df_X_L.rename(columns = {'X': 'X_L'})
            df_X_R = df_X_R.rename(columns = {'X': 'X_R'})
            for c in campos_particion:
                if (f'Base + Config_{c}' in df_i.columns):
                    filtro = df_i[f'Base + Config_{c}'][0].split(',')
                    filtro = depuracion_temporal(filtro, c)
                    df_X_L = df_X_L[df_X_L[c].isin(filtro)]
                    df_i = df_i.drop(columns = [f'Base + Config_{c}'])
                if (f'Aux_{c}' in df_i.columns):
                    filtro = df_i[f'Aux_{c}'][0].split(',')
                    filtro = depuracion_temporal(filtro, c)
                    df_X_R = df_X_R[df_X_R[c].isin(filtro)]
                    df_i = df_i.drop(columns = [f'Aux_{c}'])
        else:
            for c in campos_particion:
                if (f'Base + Config_{c}' in df_i.columns):
                    filtro = df_i[f'Base + Config_{c}'][0].split(',')
                    filtro = depuracion_temporal(filtro, c)
                    df_X = df_X[df_X[c].isin(filtro)]
                    df_P = df_P[df_P[c].isin(filtro)]
                    df_i = df_i.drop(columns = [f'Base + Config_{c}'])


    # Creación del campo periodo si es necesario
    if ('PERIODO' in para_cada) and (cond_share == 'TRUE') and ('DATE' in df_X_L.columns) and ('DATE' in df_X_R.columns):
        df_X_L['PERIODO'] = df_X_L['DATE'].astype(str).str[:7]
        df_X_R['PERIODO'] = df_X_R['DATE'].astype(str).str[:7]
    
    if ('AÑO' in para_cada) and (cond_share == 'TRUE') and ('DATE' in df_X_L.columns) and ('DATE' in df_X_R.columns):
        df_X_L['AÑO'] = df_X_L['DATE'].astype(str).str[:4].astype(int)
        df_X_R['AÑO'] = df_X_R['DATE'].astype(str).str[:4].astype(int)
        
    # Si contraste == M, entonces se usa df_P, si no, no
    if contraste == 'M':
        if cond_share == 'TRUE':
            sys.exit('No se acepta esta combinación')

        #print('df_X')
        #print(df_X)
        #print('df_P')
        #print(df_P)
        
        if ratio:
            df_X = df_X[para_cada + metricas_aux[0]].groupby(para_cada, as_index = False).sum()
            df_P = df_P[para_cada + metricas_aux[1]].groupby(para_cada, as_index = False).sum()
            #df_X['X'] = df_X[metricas_aux[0][0]] / df_X[metricas_aux[0][1]]
            df_P[metrica_name] = df_P[metricas_aux[1][0]] / df_P[metricas_aux[1][1]]
            #df_X = df_X.drop(columns = metricas_aux[0])
            df_P = df_P.drop(columns = metricas_aux[1])
        else:
            df_X = df_X[para_cada + ['X']].groupby(para_cada, as_index = False).sum()
            df_P = df_P[para_cada + [metrica_name]].groupby(para_cada, as_index = False).sum()

        
        #print('df_X')
        #print(df_X)
        #print('df_P')
        #print(df_P)
            
        df_R = df_X.merge(df_P, on = para_cada, how = 'left').reset_index(drop = True)

        #print('df_R')
        #print(df_R)

        #if para_cada != ['PERIODO']:
        #    sys.exit(f'Revisar bien aqui {duplicacion}')
        
        #if metrica == 'TC':
        #    sys.exit('dfP debe estar construido desde ordenes y visitas')
        
    if cond_share == 'TRUE':
        df_X_L = df_X_L[para_cada + ['X_L']].groupby(para_cada, as_index = False).sum()
        df_X_R = df_X_R[para_cada + ['X_R']].groupby(para_cada, as_index = False).sum()
        df_R = df_X_L.merge(df_X_R, on = para_cada, how = 'left').reset_index(drop = True)
        
        
    df_i = df_i.drop(columns = ['Condiciones de restricción_METRICA', 'Condiciones de restricción_RELACION', 'Condiciones de restricción_CONTRASTE', 'Condiciones de restricción_PARA_CADA', 'Condiciones de restricción_SHARE'])
    if 'Condiciones de restricción_PONDERADOR_CONTRASTE' in df_i.columns:
        df_i = df_i.drop(columns = ['Condiciones de restricción_PONDERADOR_CONTRASTE'])
        
    print(duplicacion, metrica, relacion, contraste, para_cada, cond_share, metrica_name, ponderador_contraste)
    
    #print('df_i')
    #print(df_i)
    
    # Restricción

    #print(df_R)
    
    if ratio:
        df_R['X_R'] = df_R[metrica_name] * df_R[metricas_aux[0][0]]
        df_R['X_L'] = df_R[metricas_aux[0][1]]
    
    x_l, x_r = 'X', metrica_name
    if contraste != 'M':
        ponderador_contraste *= float(contraste) # Para casos shares por ejem
        x_l, x_r = 'X_L', 'X_R'
    if ratio:
        x_l, x_r = 'X_L', 'X_R'
    
    #print(x_l, x_r)
    #print(df_R)
    
    print('PC2', ponderador_contraste, type(ponderador_contraste ))
    #sys.exit()

    
    for j in range(len(df_R)):
        if relacion == '==':
            R = (df_R[x_l][j] == ponderador_contraste * df_R[x_r][j])
        elif relacion == '>=':
            R = (df_R[x_l][j] >= ponderador_contraste * df_R[x_r][j])
        elif relacion == '<=':
            R = (df_R[x_l][j] <= ponderador_contraste * df_R[x_r][j])
        else:
            sys.exit(f'Relación {relacion} no configurada') 

        model += R
        
        print(str(R))
        
        if j < escribir_modelo[1]:
            escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R8({i}-{j}): {R}  \n')

    #if i == 1:
    #    sys.exit()
"""
df_configuracion = df_escenarios[df_escenarios['TIPO'] == 'C'].reset_index(drop = True)
df_configuracion

k = 1
# Temporalidad por defecto: Año actual + k Próximos años (Se puede cambiar)
dias_proyeccion = [dt.datetime.today().replace(year = dt.datetime.today().year, month = 1, day = 1).date(), dt.datetime.today().replace(year = dt.datetime.today().year + k, month = 12, day = 31).date()] # declarar arriba, al inicio del código
#dias_proyeccion = [pd.to_datetime(dias_proyeccion[0]), pd.to_datetime(dias_proyeccion[1])]

paises = ['MX', 'CL', 'UY', 'CO', 'BR', 'AR', 'PE'] # Por defecto
familias_ids = ['*'] # Todo
last_touch_lista = ['*'] # Todo
lista_fuentes, lista_canales = ['SODIMAC', 'ES', 'SIS'], ['WEB', 'APP']

for c in df_configuracion.columns:
    if 'Base + Config' in c:
        name = '_'.join(c.split('_')[1:])
        value = df_configuracion[c][0]
        if name == 'MODO_PROYECCION': # Modo proyección
            modo_proyeccion = value
            if modo_proyeccion not in ['PRESUPUESTO', 'LIBRE']:
                sys.exit(f'MODO_PROYECCION {modo_proyeccion} no es correcto')
        
        if (name == 'PAIS') and (value != ''):
            paises = value.split(',')
                        
        if (name == 'DATE') and (value != ''):
            dias_proyeccion_new = pd.to_datetime(value.split('>')[0]).date(), pd.to_datetime(value.split('>')[1]).date()
            dias_proyeccion[0] = max(dias_proyeccion[0], dias_proyeccion_new[0])
            dias_proyeccion[1] = min(dias_proyeccion[1], dias_proyeccion_new[1])
        
        if (name == 'PERIODO') and (value != ''):
            periodo_0, periodo_1 = value.split('>')
            dia0 = dt.datetime(int(periodo_0.split('-')[0]), int(periodo_0.split('-')[1]), 1).date()
            dia1 = dt.datetime(int(periodo_1.split('-')[0]), int(periodo_1.split('-')[1]), 1).date()
            dia1 = (dt.datetime(dia1.year, dia1.month, 1) + pd.DateOffset(months = 1) - pd.DateOffset(days = 1)).date()
            dias_proyeccion[0] = max(dias_proyeccion[0], dia0)
            dias_proyeccion[1] = min(dias_proyeccion[1], dia1)
        
        if (name == 'AÑO') and (value != ''):
            año_0, año_1 = value.split('>')
            dia0 = dt.datetime(int(año_0), 1, 1).date()
            dia1 = dt.datetime(int(año_1), 12, 31).date()
            dias_proyeccion[0] = max(dias_proyeccion[0], dia0)
            dias_proyeccion[1] = min(dias_proyeccion[1], dia1)
        
        if (name == 'FAMILIA') and (value != ''):
            familias_lista = value.split(',')
            familias_ids = [int(f) for f in familias_lista]
        
        if (name == campo_last_touch) and (value != ''):
            last_touch_lista = value.split(',')
        
        if (name == 'FUENTE') and (value != ''):
            lista_fuentes = value.split(',')
        
        if (name == 'CANAL') and (value != ''):
            lista_canales = value
        
        if (name == 'CANAL_BASE') and (value != ''):
            sys.exit('CANAL_BASE en configuración debe ir vacío')
            

dias_proyeccion = [pd.to_datetime(dias_proyeccion[0]), pd.to_datetime(dias_proyeccion[1])]
dias_proyeccion

"""
#sys.exit('Continuar aqui con restricciones')
print('Tiempo hasta ahora [MIN]')
print((time.time() - t_0) / 60)
df_times, t0 = medir_tiempo(t0, 'M R8', df_times)
#sys.exit()
### Seccion:  Ejecución
#Raux = (VN_USD >= 4000)
#model += Raux

#escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'Raux: {Raux}  \n')
df_inversion
df_relaciones_duplicidad
# lista de variables
str(list(model.vars)[0])
conjunto_holguras = set()

lista_vars = list(model.vars)
for i in lista_vars:
    #print(str(i))
    if str(i)[:1] == 'H':
        a, b = str(i).split('_')[0], str(i).split('_')[1]
        if b in ['p', 'n']:
            k = f'{a}_{b}'
        else:
            k = a
        
        conjunto_holguras.add(k)
    
conjunto_holguras

if activar_holguras:
    """
    H = 0
    
    df_R1a['H'] = df_R1a['H1a_p'] + df_R1a['H1a_n']
    for i in tqdm.tqdm(range(len(df_R1a))):
        H += df_R1a['H'][i]
    
    df_R1b['H'] = df_R1b['H1b_p'] + df_R1b['H1b_n']
    for i in tqdm.tqdm(range(len(df_R1b))):
        H += df_R1b['H'][i]
    
    df_R1c['H'] = df_R1c['H1c_p'] + df_R1c['H1c_n']
    for i in tqdm.tqdm(range(len(df_R1c))):
        H += df_R1c['H'][i]
    
    df_R2a['H'] = df_R2a['H2a_p'] + df_R2a['H2a_n']
    for i in tqdm.tqdm(range(len(df_R2a))):
        H += df_R2a['H'][i]
    
    df_R2b['H'] = df_R2b['H2b_p'] + df_R2b['H2b_n']
    for i in tqdm.tqdm(range(len(df_R2b))):
        H += df_R2b['H'][i]
    
    df_R3b['H'] = df_R3b['H3b1'] + df_R3b['H3b2']
    for i in tqdm.tqdm(range(len(df_R3b))):
        H += df_R3b['H'][i]
    
    df_relaciones_duplicidad['H'] = df_relaciones_duplicidad['H5_p_Venta'] + df_relaciones_duplicidad['H5_n_Venta'] + df_relaciones_duplicidad['H5_p_Ordenes'] + df_relaciones_duplicidad['H5_n_Ordenes'] + df_relaciones_duplicidad['H5_p_Visitas'] + df_relaciones_duplicidad['H5_n_Visitas']
    for i in tqdm.tqdm(range(len(df_relaciones_duplicidad))):
        H += df_relaciones_duplicidad['H'][i]
    
    
    df_visitas_pago['H'] = df_visitas_pago['H6_p'] + df_visitas_pago['H6_n']
    for i in tqdm.tqdm(range(len(df_visitas_pago))):
        H += df_visitas_pago['H'][i]
    
    df_R7['H'] = df_R7['H7_p'] + df_R7['H7_n']
    for i in tqdm.tqdm(range(len(df_R7))):
        H += df_R7['H'][i]
    

    """
    #H = 0

    print('A')
    t00 = time.time()
    for c in ['H1a_p', 'H1a_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R1a[c])

    print('B', time.time() - t00, 2 * len(df_R1b))
    for c in ['H1b_p', 'H1b_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R1b[c])
        
    print('C', time.time() - t00, 2 * len(df_R1c))
    for c in ['H1c_p', 'H1c_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R1c[c])

    print('D', time.time() - t00, 2 * len(df_R2a))
    for c in ['H2a_p', 'H2a_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R2a[c])

    print('E', time.time() - t00, 2 * len(df_R2b))
    for c in ['H2b_p', 'H2b_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R2b[c])
        
    print('E2', time.time() - t00, 2 * len(df_y))
    for c in ['H3a_p', 'H3a_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_y[c])

    #H += H2
    print('F', time.time() - t00, 2 * len(df_R3b))
    #sys.exit('Ver tiempos cuando se genera H2')
    for c in ['H3b1', 'H3b2']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R3b[c])
        
    print('F2', time.time() - t00, 2 * len(df_R3b3_agr))
    for c in ['H3b3_p', 'H3b3_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R3b3_agr[c])
        
    print('F3', time.time() - t00, 2 * len(df_R3c))
    for c in ['H3c1', 'H3c2']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R3c[c])

    # H5 se declara todo arriba
    #for c in ['H5_p_Venta', 'H5_n_Venta', 'H5_p_Ordenes', 'H5_n_Ordenes', 'H5_p_Visitas', 'H5_n_Visitas']:
    #    conjunto_holguras_contraste.add(c)
    #    H += sum(df_relaciones_duplicidad[c])
    
    print('G', time.time() - t00, 2 * len(df_visitas_pago_final))
    for c in ['H6_p', 'H6_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_visitas_pago_final[c])
        
    print('H', time.time() - t00, 2 * len(df_R7))
    for c in ['H7_p', 'H7_n']:
        conjunto_holguras_contraste.add(c)
        H += sum(df_R7[c])
        
    #H = H1a_p + H1a_n + H1b_p + H1b_n + H1c_p + H1c_n + H2a_p + H2a_n + H2b_p + H2b_n + H3b1 + H3b2 + H6_p + H6_n + H7_p + H7_n
    
    #H = H1a_p + H1a_n + H1b_p + H1b_n + H1c_p + H1c_n + H2a_p + H2a_n + H2b_p + H2b_n + H3b1 + H3b2 + H6_p + H6_n + H7_p + H7_n
    #for m in ['Venta', 'Ordenes', 'Visitas']:
    #    H += H5_p[m] + H5_n[m]

    Z = VN_USD - INV_USD - (10 ** 10) * H
str(Z)
VN_USD
print('Holguras que se crean pero que no se penalizan')
print(conjunto_holguras - conjunto_holguras_contraste)

print('Holguras que se penalizan pero que no se crean')
print(conjunto_holguras_contraste - conjunto_holguras)
model.objective = mip.maximize(Z)

escribir_modelo_opt(activar = escribir_modelo, archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'Función Objetivo \n')
escribir_modelo_opt(activar = escribir_modelo, archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'----------------  \n\n\n\n\n\n\n\n')
escribir_modelo_opt(activar = escribir_modelo, archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'Z: {Z}  \n\n')
"""
RS = (S == 0)
model += RS

escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'RS: {RS}  \n')
"""
print('Tiempo hasta ahora [MIN]')
print((time.time() - t_0) / 60)
len(model.vars), len(model.constrs) #cantidad de variables y restricciones
len(model.constrs) # Cantidad de restricciones


df_times, t0 = medir_tiempo(t0, 'SOL Prev', df_times)
#sys.exit('Medir memorias')
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Obtener todas las variables almacenadas en la sesión
def get_memory_usage():
    # Diccionario para almacenar el nombre de la variable y su tamaño
    memory_data = []
    for var_name, var_value in globals().items():
        try:
            # Calcula el tamaño en bytes
            size_bytes = sys.getsizeof(var_value)
            memory_data.append({'Variable': var_name, 'Size (MB)': size_bytes / (1024**2)})
        except:
            pass

    # Crear un DataFrame ordenado por tamaño
    df_memory = pd.DataFrame(memory_data).sort_values(by = 'Size (MB)', ascending = False).reset_index(drop=True)
    return df_memory

if False:
    df_memory = get_memory_usage()
    print(df_memory) #['Size (MB)'].sum()
print('Tiempo hast ahora esde inicio (mins)')
round((time.time() - t_inicio_0) / 60, 2)
t1 = time.time()

ejecutar_modelo_relajado = True
resistencia = 0.01 # Parámetro
optimo_encontrado = False
previo = 0

if ejecutar_modelo_relajado:
    for k in range(30): # Fix-and-Optimize
        print(f'Ejecución modelo relajado {k + 1}')
        #status = model.optimize(max_seconds = 300, relax = True) # Primero prueba factibilidad
        status = model.optimize(relax = True) # Primero prueba factibilidad

        print('Relaxed')
        print(status)
        print(model.objective_value)

        print(f'Tiempo en modelo relajado [MIN]: {(time.time() - t1) / 60}')
        df_times, t0 = medir_tiempo(t0, 'SOL Relaxed', df_times)
        
        if k == 0:
            df_Y_aux = dic_variables['Inversion']['Y'].copy()
            df_Y_aux['NUEVO'] = False
            
        df_Y_aux['Y'] = [df_Y_aux['X'][i].x for i in range(len(df_Y_aux))] 
        df_Y_aux['Y'] = np.where(df_Y_aux['Y'] < resistencia, 0, df_Y_aux['Y'])
        df_Y_aux['Y'] = np.where(df_Y_aux['Y'] > 1 - resistencia, 1, df_Y_aux['Y'])
        
        df_Y_aux['BINARIOS'] = np.where(df_Y_aux['Y'].isin([0, 1]), 1, 0)
        df_Y_aux['NUEVO'] = np.where((df_Y_aux['Y'].isin([0, 1])) & (df_Y_aux['NUEVO'] == False), True, False)
        
        proporcion = df_Y_aux['BINARIOS'].sum() / len(df_Y_aux)
        print(f'Proporcion: {100 * proporcion} %') # Proporcion de casos binarios
        
        if proporcion == previo:
            print('Ninguna variable relajada cumple las condiciones')
            break
        
        if proporcion == 1:
            optimo_encontrado = True
            print('Optimo encontrado')
            break
        
        previo = proporcion
        
        df_bins = df_Y_aux[(df_Y_aux['BINARIOS'] == 1) & (df_Y_aux['NUEVO'])].reset_index(drop = True)
        
        # Agregar las restricciones
        for i in range(len(df_bins)):
            R = (df_bins['X'][i] == df_bins['Y'][i])
            model += R
#sys.exit('Revisar en txt por que está unbounded')

if optimo_encontrado:
    sys.exit('Óptimo encontrado en Fix & Optimize. Ver soluciones')
t1 = time.time()

model.max_gap = 0.2 # 0.01 revienta...estuvo unas 48 hrs intentando resolver mx
status = model.optimize() #(max_seconds = 40 * 60) # Luego MIP

print(f'Tiempo en modelo MIP [MIN]: {(time.time() - t1) / 60}')
print('MIP')
print(status)
print(model.objective_value)
df_times, t0 = medir_tiempo(t0, 'SOL MIP', df_times)
sys.exit()
model.write('model.lp')
VN_USD.x, INV_USD.x, INV_USD.x / VN_USD.x
print('Tiempo hasta ahora [MIN]')
print((time.time() - t_0) / 60)
df_times, t0 = medir_tiempo(t0, 'M Ejecución', df_times)
df_times
sys.exit('Otros de aqui hacia abajo')
dic_parametros.keys()
df = dic_parametros['RELACION_VISITAS']
df.M.min(), df.M.max()
df[df['M'].isna()]
from mip import Model, BINARY

# Crear un modelo
model = Model()

# Definir variables
x = model.add_var(name="x", var_type=BINARY)
y = model.add_var(name="y", var_type=BINARY)

# Agregar restricciones
model += x + y <= float('inf'), "Restriccion_1"
model += x - y >= 0, "Restriccion_2"

# Listar restricciones
print("Lista de restricciones:")
for constr in model.constrs:
    print(f"Nombre: {constr.name}, Expresión: {constr.expr}")
if status == mip.OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(model.objective_value))
elif status == mip.OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(model.objective_value, model.objective_bound))
elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(model.objective_bound))
if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
    print('solution:')
    for v in model.vars:
        if abs(v.x) > 1e-6: # only printing non-zeros
            #if 'H' == v.name[:1]:
            print('{} : {}'.format(v.name, v.x))
            #if ('Inv' in v.name) and not ('Inv_rango' in v.name):
            #    print('{} : {}'.format(v.name, v.x))
            #if 'Y_' in v.name:
            #    print('{} : {}'.format(v.name, v.x))
sys.exit('Solo revs abajo')
df_R1c.FAMILIA.unique()
df_R1b
dfk = df_R1c[df_R1c.FAMILIA == familia_sm].reset_index(drop = True)
dfk['X'] = [dfk['X'][i].x for i in range(len(dfk))]
#dfk['X'].sum()
dfk
df_R3c[df_R3c.FAMILIA == "17 - Muebles"]
sys.exit('Abajo solo revisiones')
#(df_R3b['X_Inv'][i] >= df_R3b['MIN_VIS_IN'][i] * df_R3b['X_Y'][i] - df_R3b['H3b1'][i])
dfr = df_R3b[df_R3b['FAMILIA'] == '25 - Promociones - Soporte Tecnico']
dfr = dfr[dfr['TIPO_MEDIO'] == 'SEM - Non Brand']
dfr
sys.exit('Continuar acá')
sys.exit()
# To do

"""

[Extender a otras restricciones > Ocupar en R5] R3b3 Tiene un enfoque con un groupby sum que puede bajar bastante los tiempos en la creación de otras restricciones

Variables sensibilizables: Restricciones R8 asociada a metas

Integrar variable de inversión para distintos niveles de duplicidad (ya creada)
contar pars, vars y restricciones: len(model.vars), len(model.constrs)
Guardar modelo base con pkl
Weighted RLM que permita estimar el tiempo de generación & resolución de un modelo según el tamaño de los conjuntos (cant de días, de paises, etc)

Variables pasadas (hasta ayer) deben tomar si o si los números reales (Vent, Ord, Vis, Inv)

[OK] Variables de Holgura?

# En MIP: Infactible + otras restricciones puede ser factible (infactible a veces puede ser no acotado)
"""

sys.exit('Falta restricción asociada a:')
dic_variables['Inversion']['Inv_all']#.DUPLICACION.unique()
sys.exit('Continuar restricciones acá')
### Seccion:  Otro

"""
    Antigua parte final de R5 (lenta)
    
    for j in tqdm.tqdm(range(len(df_relaciones_duplicidad_i))):
        factor = df_relaciones_duplicidad_i[f'f_{metrica_name}'][j]
        
        df_relaciones_duplicidad_i_j = df_relaciones_duplicidad_i.iloc[j: j + 1]
        df_relaciones_duplicidad_i_j['IN'] = True
        
        if activar_holguras:
            hgp, hgn = df_relaciones_duplicidad_i_j[f'H5_p_{m}'][j], df_relaciones_duplicidad_i_j[f'H5_n_{m}'][j]
        df_relaciones_duplicidad_i_j = df_relaciones_duplicidad_i_j[campos_beta + campos_extra + ['IN']]
        #print(df_relaciones_duplicidad_i_j)
        
        print(df_m)
        sys.exit()
        LHS = df_m[df_m['DUPLICACION'] == agr].reset_index(drop = True)
        LHS = LHS.merge(df_relaciones_duplicidad_i_j, on = campos_beta + campos_extra, how = 'left')
        LHS['IN'] = LHS['IN'].fillna(False)
        LHS = LHS[LHS['IN']].reset_index(drop = True)
        
        RHS = df_m[df_m['DUPLICACION'] == desagr].reset_index(drop = True)
        RHS = RHS.merge(df_relaciones_duplicidad_i_j, on = campos_beta + campos_extra, how = 'left')
        RHS['IN'] = RHS['IN'].fillna(False)
        RHS = RHS[RHS['IN']].reset_index(drop = True)
        
        name = m_s + '_' + agr + '_' + desagr + '_' + '_'.join(LHS['NAME'][0].split('_')[1:])
        
        #print(LHS)
        #print(RHS)
        
        #print(factor)
        #print(name)
        
        rhs = 0
        for k in range(len(RHS)):
            rhs += RHS[f'X_{m_s}T'][k]
        
        if activar_holguras:
            R5 = (LHS[f'X_{m_s}T'][0] == factor * rhs + hgp - hgn)
        else:
            R5 = (LHS[f'X_{m_s}T'][0] == factor * rhs)
            
        model += R5

        if j < escribir_modelo[1]:
            escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R5({name}): {R5}  \n')
"""
#sys.exit('Solo revision')



"""
    Antigua parte final de R6
    
    
    # ELIMINAR BLOQUE


for i in tqdm.tqdm(range(len(df_visitas_pago))):
    df_visitas_pago_i = df_visitas_pago.iloc[i: i + 1]
    df_visitas_pago_i_tupla = df_visitas_pago_i[campos_beta + campos_grano]
    
    name = df_visitas_pago_i["NAME"][i]
    
    for c in campos_clacom:
        df_visitas_pago_i_tupla = df_visitas_pago_i_tupla.rename(columns = {c: c + '_FINAL'})
    df_visitas_pago_i_tupla['IN'] = True
    
    print(df_visitas_pago_i_tupla)
    sys.exit()
    #print('A')
    #print(df_visitas_pago_i)
    
    df_relacion_visitas_i = df_relacion_visitas.merge(df_visitas_pago_i_tupla, on = campos_beta + campos_clacom_final + [campo_last_touch], how = 'left')
    df_relacion_visitas_i['IN'] = df_relacion_visitas_i['IN'].fillna(False)
    df_relacion_visitas_i = df_relacion_visitas_i[df_relacion_visitas_i['IN']].reset_index(drop = True)
    #print(df_relacion_visitas_i)
    
    df_relacion_visitas_i_tupla = df_relacion_visitas_i[campos_beta + campos_clacom_inicial + [campo_last_touch] + ['M', 'IN']]

    for c in campos_clacom:
        df_relacion_visitas_i_tupla = df_relacion_visitas_i_tupla.rename(columns = {c + '_INICIAL': c})
    
    df_visitas_pago_in_i = df_visitas_pago_in.merge(df_relacion_visitas_i_tupla, on = campos_beta + campos_grano, how = 'left')
    df_visitas_pago_in_i['IN'] = df_visitas_pago_in_i['IN'].fillna(False)
    df_visitas_pago_in_i = df_visitas_pago_in_i[df_visitas_pago_in_i['IN']].reset_index(drop = True)

    df_visitas_pago_in_i['M_V_in'] = df_visitas_pago_in_i['M'] * df_visitas_pago_in_i['X_VP_in']
    S = df_visitas_pago_in_i['M_V_in'].sum()
    
    if activar_holguras:
        R6 = (df_visitas_pago_i['X_VP'][i] == S + df_visitas_pago_i['H6_p'][i] - df_visitas_pago_i['H6_n'][i])
    else:
        R6 = (df_visitas_pago_i['X_VP'][i] == S)
    model += R6
    
    if i <= escribir_modelo[1]:
        escribir_modelo_opt(activar = escribir_modelo[0], archivo = f'Modelo de optimizacion.txt', modo = 'a', linea = f'R6({name}): {R6}  \n')
        

"""
import sympy as sp
from sympy import Symbol, integrate #symbols, Q, assume, integrate

def solucion_segmento_representativo():
    
    def func(D, n, i):
        # D = a + c * t
        return D * (i ** n)


    i, A, B, i_0, i_f, D = sp.symbols('i A B i_0 i_f D')
    n = Symbol('n', positive = True, real = True) 

    # D = a + c * t
    # define la funcion
    f = D * (i ** n) 
    g = A * i + B

    # Calcula la integral de (f(i) - g(i)) ** 2

    integral = integrate((f - g) ** 2, (i, i_0, i_f))
    integral.simplify()
    
    # derivar la integral respecto a A y a B
    derivativeA = sp.diff(integral, A)
    derivativeB = sp.diff(integral, B)

    # Resuelve el sistema de ecuaciones
    solution = sp.solve([derivativeA, derivativeB], (A, B))
    
    return solution

def solucion_segmento_representativo_subs(solution, i_0v, i_fv, Dv, nv ):

    i, A, B, i_0, i_f, D = sp.symbols('i A B i_0 i_f D')
    n = Symbol('n', positive = True, real = True) 
    
    A_ = solution[A].subs({i_0: i_0v, i_f: i_fv, D: Dv, n: nv})
    B_ = solution[B].subs({i_0: i_0v, i_f: i_fv, D: Dv, n: nv})
    
    return A_, B_

dict_valores = {'i_0': 20, 'i_f': 40, 'D': 4, 'n': 0.5}
solution = solucion_segmento_representativo()
A, B = solucion_segmento_representativo_subs(solution, dict_valores)
A, B
### Seccion:  Ejemplos
sys.exit()
### Seccion: # Knapsack
from mip import Model, xsum, BINARY, MAXIMIZE, INTEGER

# Datos
values = [10, 15, 7]
weights = [2, 3, 1]
capacity = 4

# Modelo
model = Model("Mochila", sense = MAXIMIZE) # Para maximizar FO

# Variables binarias
x = [model.add_var(var_type=BINARY) for _ in range(len(values))]

# Restricción: Capacidad de la mochila
model += xsum(weights[i] * x[i] for i in range(len(values))) <= capacity

# Función objetivo: Maximizar el valor total
model.objective = xsum(values[i] * x[i] for i in range(len(values)))

# Resolver
model.optimize()

# Resultados
print("Elementos seleccionados:")
for i in range(len(values)):
    if x[i].x >= 0.99:  # Ajuste para precisión numérica
        print(f" - Objeto {i+1} (Valor: {values[i]}, Peso: {weights[i]})")
print(f"Valor total: {model.objective_value}")

### Seccion: # Problema de la Dieta
from mip import Model, xsum, CONTINUOUS

# Datos
costs = [2, 3]
protein = [3, 2]
fat = [2, 4]
req_protein = 10
req_fat = 18

# Modelo
model = Model("Dieta")

# Variables continuas (cantidades de alimentos)
x = [model.add_var(var_type=CONTINUOUS) for _ in range(len(costs))]

# Restricciones nutricionales
model += xsum(protein[i] * x[i] for i in range(len(costs))) >= req_protein
model += xsum(fat[i] * x[i] for i in range(len(costs))) >= req_fat

# Función objetivo: Minimizar el costo
model.objective = xsum(costs[i] * x[i] for i in range(len(costs)))

# Resolver
model.optimize()

# Resultados
print("Cantidades óptimas de alimentos:")
for i in range(len(costs)):
    print(f" - Alimento {chr(65+i)}: {x[i].x:.2f} unidades")
print(f"Costo total: {model.objective_value:.2f}")

