import pandas as pd
import tqdm

def dias_comparacion(paises, dias_proyeccion, df_dias_eventos, dias_historicos):
    
    # Asigna a cada día de la proyección, uno que ya exista (histórico)
    
    df_paises = pd.DataFrame({'PAIS': paises})
    df_paises['AUX'] = 'aux'
    df_dias_predecir = pd.DataFrame({'DATE': pd.date_range(*dias_proyeccion)})
    df_dias_predecir['DIA_SEMANA'] = df_dias_predecir['DATE'].dt.day_name()
    df_dias_predecir['AUX'] = 'aux'
    df_dias_predecir = df_dias_predecir.merge(df_paises, on = 'AUX', how = 'left')
    df_dias_predecir = df_dias_predecir.drop(columns = ['AUX'])
    df_dias_predecir = df_dias_predecir.merge(df_dias_eventos, on = ['DATE', 'PAIS'], how = 'left')
    df_dias_predecir['EVENTO'] = df_dias_predecir['EVENTO'].fillna('') # En blanco, significa sin evento
    
    #dia_desde = (dia_hasta + dt.timedelta(days = 1)).replace(year = dia_hasta.year - 1)
    [dia_desde, dia_hasta] = dias_historicos

    df_dias_historicos = pd.DataFrame({'DATE': pd.date_range(dia_desde, dia_hasta)})
    df_dias_historicos['DIA_SEMANA'] = df_dias_historicos['DATE'].dt.day_name()
    df_dias_historicos['AUX'] = 'aux'
    df_dias_historicos = df_dias_historicos.merge(df_paises, on = 'AUX', how = 'left')
    df_dias_historicos = df_dias_historicos.drop(columns = ['AUX'])
    df_dias_historicos = df_dias_historicos.merge(df_dias_eventos, on = ['DATE', 'PAIS'], how = 'left')
    df_dias_historicos['EVENTO'] = df_dias_historicos['EVENTO'].fillna('') # En blanco, significa sin evento
    df_dias_historicos
    
    df_dias_predecir['DATE_COMPARACION'] = ""
    for i in tqdm.tqdm(range(len(df_dias_predecir))):
        date, dia_de_la_semana, pais, evento = df_dias_predecir.loc[i, ['DATE', 'DIA_SEMANA', 'PAIS', 'EVENTO']] # cada día de la predicción
        
        df_dias_df_dias_existentes_iteracion = df_dias_historicos[df_dias_historicos['EVENTO'] == evento].reset_index(drop = True) # Se filtra el evento del mismo tipo (o no evento)
        
        if evento == "": # Si el día no es evento
            df_dias_df_dias_existentes_iteracion = df_dias_df_dias_existentes_iteracion[df_dias_df_dias_existentes_iteracion['DIA_SEMANA'] == dia_de_la_semana].reset_index(drop = True) # se filtra el mismo día de la semana de la fecha revisada (para NO eventos, en eventos hay menos opciones, asi que se libera esta restriccion)
        else:
            if (len(df_dias_df_dias_existentes_iteracion[df_dias_df_dias_existentes_iteracion['DIA_SEMANA'] == dia_de_la_semana]) > 0) and (evento != 'Festividad'):
                df_dias_df_dias_existentes_iteracion = df_dias_df_dias_existentes_iteracion[df_dias_df_dias_existentes_iteracion['DIA_SEMANA'] == dia_de_la_semana].reset_index(drop = True) # se filtra el mismo día de la semana de la fecha revisada (si existen casos, tiene prioridad que sea el mismo día de la semana)
            else:
                df_dias_df_dias_existentes_iteracion = df_dias_df_dias_existentes_iteracion.copy() # Si hay evento, se toma para este caso la data historica completa

        df_dias_df_dias_existentes_iteracion['DELTA_EN_DIAS'] = (df_dias_df_dias_existentes_iteracion['DATE'] - date).dt.days % 365 # se calcula la diferencia en días con todas las posibiliades históricas % 365
        df_dias_df_dias_existentes_iteracion['DELTA_EN_DIAS_REV'] = (date - df_dias_df_dias_existentes_iteracion['DATE']).dt.days % 365 # lo mismo, pero inverso (puede ser antes o después)
        df_dias_df_dias_existentes_iteracion['DELTA_MIN'] = df_dias_df_dias_existentes_iteracion[['DELTA_EN_DIAS', 'DELTA_EN_DIAS_REV']].min(axis = 1) # se  obtiene el mínimo entre ambos

        df_dias_df_dias_existentes_iteracion = df_dias_df_dias_existentes_iteracion[df_dias_df_dias_existentes_iteracion['DELTA_MIN'] == df_dias_df_dias_existentes_iteracion['DELTA_MIN'].min()].reset_index(drop = True) # obtener el mínimo delta min
        dia_seleccionado = df_dias_df_dias_existentes_iteracion.loc[0, 'DATE'] # se obtiene el mejor dúa
        df_dias_predecir['DATE_COMPARACION'][i] = dia_seleccionado.date() # se asigna

    return df_dias_predecir
