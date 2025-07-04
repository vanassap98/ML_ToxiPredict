# eda.py
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr
from sklearn.metrics import roc_auc_score


# ------------------------------------
# UNIVARIANTE – NUMÉRICAS
# ------------------------------------

def plot_num_univar(df, vars, bins=30):
    """
    Histograma + KDE + Boxplot para variables numéricas
    """
    for var in vars:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.histplot(df[var].dropna(), bins=bins, kde=True, ax=axs[0])
        axs[0].set_title(f'Distribución de {var}')

        sns.boxplot(x=df[var], ax=axs[1])
        axs[1].set_title(f'Boxplot de {var}')

        plt.tight_layout()
        plt.show()


def plot_num_univar_grouped(df, variables, n_cols=3, bins=30):
    """
    Visualiza variables numéricas univariantes agrupadas en subplots:
    - Para cada variable se muestra un histograma+KDE y un boxplot en subplots contiguos.
    - Organiza las variables en una cuadrícula para facilitar la visión general.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    variables : list of str
        Lista de variables numéricas a visualizar.
    n_cols : int
        Número de variables por fila (cada variable ocupa 2 columnas).
    bins : int
        Número de bins para el histograma.

    Retorna:
    --------
    None. Muestra la figura con los subplots.
    """
    n_vars = len(variables)
    n_rows = math.ceil(n_vars / n_cols)

    # Creamos figura con n_rows filas y n_cols*2 columnas (hist + boxplot por variable)
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6 * n_cols * 2, 4 * n_rows))
    
    # Si solo hay una fila, axes no es array 2D, normalizamos
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, var in enumerate(variables):
        row = idx // n_cols
        col = (idx % n_cols) * 2  # 2 columnas por variable

        ax_hist = axes[row, col]
        ax_box = axes[row, col + 1]

        # Histograma + KDE
        sns.histplot(df[var].dropna(), bins=bins, kde=True, ax=ax_hist)
        ax_hist.set_title(f'Hist + KDE: {var}')
        ax_hist.set_ylabel('Frecuencia')
        
        # Boxplot
        sns.boxplot(x=df[var], ax=ax_box)
        ax_box.set_title(f'Boxplot: {var}')
        ax_box.set_xlabel(var)

    # Eliminar ejes sobrantes si los hay
    total_axes = n_rows * n_cols * 2
    for extra_ax in range(n_vars * 2, total_axes):
        row = extra_ax // (n_cols * 2)
        col = extra_ax % (n_cols * 2)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()


# ------------------------------------
# UNIVARIANTE – CATEGÓRICAS
# ------------------------------------

def plot_cat_univar(df, vars, top=None, show_values=True):
    """
    Visualización univariante para variables categóricas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        vars (list): Lista de variables categóricas a graficar.
        top (int, optional): Número máximo de categorías a mostrar por variable (según frecuencia).
        show_values (bool, optional): Mostrar conteos encima de las barras.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    for var in vars:
        counts = df[var].value_counts().head(top) if top else df[var].value_counts()

        plt.figure(figsize=(14, 6))
        ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis")

        if show_values:
            for i, v in enumerate(counts.values):
                ax.text(i, v + max(counts.values) * 0.01, str(v), ha='center', va='bottom', fontsize=9)

        ax.set_title(f'Distribución de {var}')
        ax.set_xlabel(var)
        ax.set_ylabel('Frecuencia')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        plt.show()



# ------------------------------------
# BIVARIANTE – NUMÉRICAS VS TARGET
# ------------------------------------

def plot_bivar_num_vs_target(df, target, vars, test=True, show=True):
    """
    Visualiza relaciones entre variables numéricas y un target binario.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene las variables.
        target (str): Nombre de la variable objetivo binaria.
        vars (list): Lista de variables numéricas a analizar.
        test (bool): Si True, realiza el test Mann-Whitney U.
        show (bool): Si True, muestra las gráficas. Si False, devuelve resumen.
        
    Returns:
        pd.DataFrame: Tabla resumen con p-valores si show=False.
    """
    resumen = []

    for var in vars:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=target, y=var, palette="viridis")
        sns.stripplot(data=df, x=target, y=var, color='black', alpha=0.3, jitter=0.2, size=2)
        plt.title(f"{var} vs {target}")
        plt.tight_layout()
        if show:
            plt.show()
        
        if test:
            try:
                grupo0 = df[df[target] == 0][var].dropna()
                grupo1 = df[df[target] == 1][var].dropna()
                stat, p = mannwhitneyu(grupo0, grupo1, alternative='two-sided')
                resumen.append({
                    'variable': var,
                    'p_valor': p,
                    'mediana_0': grupo0.median(),
                    'mediana_1': grupo1.median()
                })
            except Exception as e:
                resumen.append({
                    'variable': var,
                    'p_valor': None,
                    'error': str(e)
                })

    if not show:
        return pd.DataFrame(resumen).sort_values("p_valor")




def plot_features_num_vs_binary_target(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Visualiza la relación entre variables numéricas y un target binario mediante pairplots.
    Filtra las variables que presentan correlación mínima con el target (Pearson) y, opcionalmente,
    significancia estadística.

    Parámetros:
    - df (pd.DataFrame): Dataset de entrada.
    - target_col (str): Nombre de la columna objetivo (debe ser binaria).
    - columns (list): Lista de columnas numéricas a considerar. Si está vacío, se seleccionan automáticamente.
    - umbral_corr (float): Umbral mínimo de |correlación| para incluir una variable.
    - pvalue (float or None): Umbral de significación estadística (0–1). Si es None, se ignora.

    Return:
    - Listado de variables que cumplen los criterios.
    """

    # Validaciones
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame.")
        return None
    if target_col not in df.columns:
        print(f"Error: columna '{target_col}' no encontrada en el DataFrame.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number) or len(df[target_col].unique()) != 2:
        print("Error: 'target_col' debe ser binaria (dos clases numéricas).")
        return None
    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None
    if pvalue is not None and not (0 < pvalue < 1):
        print("Error: 'pvalue' debe estar entre 0 y 1.")
        return None

    # Selección automática de columnas si no se indican
    if not columns:
        columns = df.select_dtypes(include=np.number).drop(columns=[target_col]).columns.tolist()

    # Filtrado por correlación (y p-valor opcional)
    seleccionadas = []
    for col in columns:
        datos_validos = df[[col, target_col]].dropna()
        if datos_validos.shape[0] < 2:
            continue
        corr, p = pearsonr(datos_validos[col], datos_validos[target_col])
        if abs(corr) >= umbral_corr:
            if pvalue is None or p < pvalue:
                seleccionadas.append(col)

    if not seleccionadas:
        print("No se encontraron variables que cumplan los criterios.")
        return []

    # Pairplots en bloques de 4 + target
    bloque = 4
    for i in range(0, len(seleccionadas), bloque):
        subset = seleccionadas[i:i + bloque]
        columnas_plot = [target_col] + subset
        sns.pairplot(df[columnas_plot].dropna(), hue=target_col, palette="husl", plot_kws={'alpha': 0.5})
        plt.suptitle(f"Pairplot: {', '.join(subset)} vs {target_col}", y=1.03)
        plt.show()

    return seleccionadas





def analisis_bivariante_grupo(df, variables, target="result_conc1_mean_binary", group_name="grupo", plot=False):
    """
    Realiza análisis bivariante de un conjunto de variables numéricas frente a un target binario.

    Para cada variable en la lista proporcionada:
    - Calcula el p-valor del test de Mann-Whitney U para comparar distribuciones entre clases.
    - Calcula el AUC univariado como medida de capacidad predictiva individual.
    - Opcionalmente, genera un gráfico violinplot mostrando la distribución por clase del target.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables y el target.
    variables : list of str
        Lista de nombres de columnas numéricas a analizar.
    target : str, opcional
        Nombre de la columna objetivo binaria. Por defecto es 'result_conc1_mean_binary'.
    group_name : str, opcional
        Nombre del grupo/familia de variables para registrar en los resultados.
    plot : bool, opcional
        Si True, muestra un gráfico violinplot por variable. Por defecto es False.

    Devuelve
    --------
    pd.DataFrame
        Tabla con columnas: Variable, Grupo, p_valor_U_test, AUC_univariado, N (nº de muestras).
    """
    resultados = []

    for var in variables:
        data = df[[var, target]].dropna()

        # Estadísticas
        clases = data[target].unique()
        if len(clases) == 2:
            grupo0 = data[data[target] == clases[0]][var]
            grupo1 = data[data[target] == clases[1]][var]
            u_stat, p_val = mannwhitneyu(grupo0, grupo1, alternative='two-sided')
        else:
            p_val = None

        # AUC
        try:
            auc = roc_auc_score(data[target], data[var])
        except:
            auc = None

        # Visualización si se solicita
        if plot:
            plt.figure(figsize=(10, 5))
            sns.violinplot(x=target, y=var, data=data)
            plt.title(f"{var} vs {target} – {group_name}")
            plt.xlabel("Target")
            plt.ylabel(var)
            plt.tight_layout()
            plt.show()

        resultados.append({
            "Variable": var,
            "Grupo": group_name,
            "p_valor_U_test": round(p_val, 5) if p_val is not None else None,
            "AUC_univariado": round(auc, 3) if auc is not None else None,
            "N": len(data)
        })

    return pd.DataFrame(resultados)




def plot_violin_grouped(df, variables, target, group_name, df_resultados=None, auc_threshold=0.7, pval_threshold=0.05):
    """
    Visualiza violinplots de un grupo de variables numéricas frente a un target binario,
    resaltando con distintos colores según AUC y p-valor.

    Colores:
    - 🔴 darkred: cumple AUC y p-valor
    - 🟡 goldenrod: solo p-valor significativo
    - 🔵 royalblue: solo AUC alto
    - ⚫ black: no destaca

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables y el target.
    variables : list of str
        Lista de nombres de columnas numéricas a graficar.
    target : str
        Nombre de la variable objetivo binaria.
    group_name : str
        Nombre del grupo de variables (usado en el título principal).
    df_resultados : pd.DataFrame, opcional
        DataFrame con columnas ['Variable', 'AUC_univariado', 'p_valor_U_test'].
    auc_threshold : float
        Umbral para AUC alto.
    pval_threshold : float
        Umbral para significancia estadística.

    Returns
    -------
    None. Muestra una figura con los subplots de los violinplots.
    """
    n = len(variables)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        sns.violinplot(data=df, x=target, y=var, inner="box", ax=ax)

        # Extraer métricas si están disponibles
        if df_resultados is not None and var in df_resultados['Variable'].values:
            fila = df_resultados[df_resultados['Variable'] == var].iloc[0]
            auc = fila['AUC_univariado']
            pval = fila['p_valor_U_test']
            cumple_auc = auc is not None and auc >= auc_threshold
            cumple_pval = pval is not None and pval <= pval_threshold

            # Asignar color según combinación de criterios
            if cumple_auc and cumple_pval:
                color = "darkred"
            elif cumple_auc:
                color = "royalblue"
            elif cumple_pval:
                color = "goldenrod"
            else:
                color = "black"

            titulo = f"{var}\nAUC={auc} | p={pval}"
        else:
            titulo = var
            color = "black"

        ax.set_title(titulo, fontsize=10, color=color)
        ax.set_xlabel("Target")
        ax.set_ylabel(var)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Distribuciones por grupo: {group_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()




def analizar_colinealidad(df, variables, umbral=0.9, mostrar_heatmap=True):
    """
    Analiza la colinealidad entre variables numéricas usando la matriz de correlación.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables numéricas a evaluar.
    variables : list of str
        Lista de nombres de columnas numéricas.
    umbral : float, opcional
        Umbral mínimo de correlación para considerar un par como colineal. Por defecto 0.9.
    mostrar_heatmap : bool, opcional
        Si True, muestra un heatmap de la matriz de correlación.

    Devuelve
    --------
    df_pares : pd.DataFrame
        Tabla con los pares de variables altamente correlacionadas y su valor de correlación.
    """
    # 1. Subset de variables numéricas
    X = df[variables].copy()
    
    # 2. Calcular matriz de correlación absoluta
    corr_matrix = X.corr().abs()

    # 3. Mostrar heatmap si se solicita
    if mostrar_heatmap:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap="coolwarm", vmin=0, vmax=1)
        plt.title(f"Matriz de correlación (umbral ≥ {umbral})")
        plt.show()

    # 4. Extraer pares con correlación alta
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    pares = [
        (fila, col, upper.loc[fila, col])
        for fila in upper.index
        for col in upper.columns
        if pd.notnull(upper.loc[fila, col]) and upper.loc[fila, col] >= umbral
    ]
    
    df_pares = pd.DataFrame(pares, columns=["Variable 1", "Variable 2", "Correlación"])
    
    return df_pares.sort_values("Correlación", ascending=False)




def sugerir_variables_a_eliminar_opt(df_pares_correlados, df_resultados, criterio="AUC_univariado"):
    """
    Sugiere variables a eliminar entre pares colineales, priorizando la que tenga menor valor del criterio.
    
    Parámetros
    ----------
    df_pares_correlados : pd.DataFrame
        Contiene columnas ['Variable 1', 'Variable 2', 'Correlación'].
    df_resultados : pd.DataFrame
        Contiene métricas univariantes por variable, incluida la columna 'Variable' y el criterio (p. ej. AUC).
    criterio : str
        Nombre de la columna con la métrica a priorizar (por defecto 'AUC_univariado').

    Devuelve
    --------
    df_eliminadas : pd.DataFrame
        Tabla con columnas: Variable_1, Variable_2, Correlación, Criterio_var1, Criterio_var2, A_eliminar.
    lista_final : list
        Lista única de variables sugeridas para eliminar.
    """
    registros = []
    eliminadas = set()

    for _, row in df_pares_correlados.iterrows():
        var1, var2, corr = row["Variable 1"], row["Variable 2"], row["Correlación"]

        # Obtener valores del criterio
        val1 = df_resultados[df_resultados["Variable"] == var1][criterio].values
        val2 = df_resultados[df_resultados["Variable"] == var2][criterio].values

        if len(val1) == 0 or len(val2) == 0 or pd.isna(val1[0]) or pd.isna(val2[0]):
            continue  # Ignora si falta AUC

        val1, val2 = val1[0], val2[0]
        if val1 < val2:
            eliminar = var1
        else:
            eliminar = var2

        if eliminar not in eliminadas:  # evitar duplicados
            eliminadas.add(eliminar)
            registros.append({
                "Variable_1": var1,
                "Variable_2": var2,
                "Correlación": corr,
                f"{criterio}_{var1}": val1,
                f"{criterio}_{var2}": val2,
                "A_eliminar": eliminar
            })

    df_eliminadas = pd.DataFrame(registros)
    lista_final = list(eliminadas)
    return df_eliminadas, lista_final






# ------------------------------------
# BIVARIANTE – CATEGÓRICAS VS TARGET
# ------------------------------------


def plot_cat_grouped(df, variables, target, n_cols=3, figsize_per_plot=(4, 3),
                     colors=["#4C72B0", "#C44E52"], max_categories=20):
    """
    Visualiza variables categóricas frente a un target binario mediante gráficos de barras proporcionales apiladas.
    Incluye p-valor del test chi-cuadrado y agrupa categorías poco frecuentes si exceden max_categories.

    Parámetros
    ----------
    df : pd.DataFrame
    variables : list[str]
        Variables categóricas a analizar.
    target : str
        Variable binaria objetivo.
    n_cols : int
        Columnas por fila.
    figsize_per_plot : tuple
        Tamaño por subplot.
    colors : list[str]
        Colores para clases del target.
    max_categories : int
        Número máximo de categorías distintas antes de agrupar como "otros".

    Returns
    -------
    pd.DataFrame
        Tabla con p-valores ordenados por variable.
    """
    resultados = []

    n_vars = len(variables)
    n_rows = math.ceil(n_vars / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_plot[0], n_rows * figsize_per_plot[1]))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        df_filtered = df[[var, target]].dropna()

        # Agrupación de categorías poco frecuentes si excede el límite
        n_categorias = df_filtered[var].nunique()
        if n_categorias > max_categories:
            top_categorias = df_filtered[var].value_counts().nlargest(max_categories).index
            df_filtered[var] = df_filtered[var].apply(lambda x: x if x in top_categorias else "otros")

        # Tabla de proporciones
        counts = df_filtered.groupby([var, target]).size().unstack(fill_value=0)
        total = counts.sum(axis=1)
        prop_df = counts.divide(total, axis=0).reset_index()
        prop_df_melted = prop_df.melt(id_vars=var, var_name="target", value_name="proporcion")
        pivot = prop_df_melted.pivot(index=var, columns="target", values="proporcion").fillna(0)

        # Chi-cuadrado
        try:
            tabla = pd.crosstab(df_filtered[var], df_filtered[target])
            chi2, p, dof, expected = chi2_contingency(tabla)
        except:
            p = float('nan')
        resultados.append({"Variable": var, "p_valor": p})

        # Plot
        ax = axes[i]
        pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.9, edgecolor="black")
        ax.set_title(f"{var}\np-valor = {p:.3e}" if not pd.isna(p) else f"{var}\np-valor = N/A")
        ax.set_xlabel("")
        ax.set_ylabel("Proporción")

        # Rotación automática
        if len(pivot) > 5:
            ax.tick_params(axis="x", rotation=90)

        # Leyenda solo una vez
        if i == 0:
            ax.legend(title=target, loc="upper right")
        else:
            ax.get_legend().remove()

    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(resultados).sort_values(by="p_valor", ascending=True).reset_index(drop=True)






def plot_num_grouped(df, variables, target,
                     n_cols=3, figsize_per_plot=(4, 3),
                     bins=30, use_kde=True,
                     clip_outliers=True, clip_quantile=0.99,
                     return_stats=True):
    """
    Análisis bivariante entre variables numéricas y un target binario.
    Incluye visualización por clase y test U de Mann-Whitney.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset completo.
    variables : list[str]
        Lista de variables numéricas.
    target : str
        Variable binaria (0/1).
    n_cols : int
        Columnas por fila.
    figsize_per_plot : tuple
        Tamaño (ancho, alto) por subplot.
    bins : int
        Bins para histograma.
    use_kde : bool
        Si True, usa KDE. Si False, usa histogramas.
    clip_outliers : bool
        Si True, recorta outliers extremos.
    clip_quantile : float
        Percentil de corte para outliers (solo si clip_outliers = True).
    return_stats : bool
        Si True, retorna tabla de p-valores.

    Retorna
    -------
    pd.DataFrame (opcional)
        Tabla con p-valores ordenados si return_stats = True.
    """
    resultados = []

    n_vars = len(variables)
    n_rows = math.ceil(n_vars / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_plot[0], n_rows * figsize_per_plot[1]))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        if var not in df.columns:
            print(f"⚠️ Variable no encontrada: {var}")
            continue

        df_filtered = df[[var, target]].dropna()

        if clip_outliers:
            upper = df_filtered[var].quantile(clip_quantile)
            lower = df_filtered[var].quantile(1 - clip_quantile)
            df_filtered = df_filtered[(df_filtered[var] >= lower) & (df_filtered[var] <= upper)]

        grupo_0 = df_filtered[df_filtered[target] == 0][var]
        grupo_1 = df_filtered[df_filtered[target] == 1][var]

        # Test estadístico
        try:
            stat, p = mannwhitneyu(grupo_0, grupo_1, alternative='two-sided')
        except:
            p = np.nan
        resultados.append({"Variable": var, "p_valor": p})

        # Gráfico
        ax = axes[i]
        if use_kde:
            sns.kdeplot(grupo_0, ax=ax, label="0", fill=True, linewidth=1)
            sns.kdeplot(grupo_1, ax=ax, label="1", fill=True, linewidth=1)
        else:
            ax.hist(grupo_0, bins=bins, alpha=0.5, label="0", color="#4C72B0", edgecolor="black")
            ax.hist(grupo_1, bins=bins, alpha=0.5, label="1", color="#C44E52", edgecolor="black")

        ax.set_title(f"{var}\np-valor = {p:.3e}" if not pd.isna(p) else f"{var}\np-valor = N/A")
        ax.set_xlabel(var)
        ax.set_ylabel("Densidad" if use_kde else "Frecuencia")

        # Leyenda solo una vez
        if i == 0:
            ax.legend(title=target)
        else:
            ax.get_legend().remove()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    if return_stats:
        return pd.DataFrame(resultados).sort_values(by="p_valor", ascending=True).reset_index(drop=True)




# ------------------------------------
# UTILIDAD – RANKING DE VARIABLES
# ------------------------------------

def rank_features(df_plantilla, top_n=10, metodo='rf'):
    """
    Extrae las top_n variables según importancia por RF o KBest
    """
    if metodo == 'rf':
        return df_plantilla.sort_values("rf_importancia", ascending=False).head(top_n)["nombre_variable"].tolist()
    elif metodo == 'kbest':
        return df_plantilla.sort_values("kbest_score", ascending=False).head(top_n)["nombre_variable"].tolist()
    else:
        raise ValueError("Método no válido. Usa 'rf' o 'kbest'.")





