import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Período da análise
inicio = '2020-01-01'
fim = '2024-12-31'

# Importação dos dados

retornos = pd.read_parquet('retornos.parquet')
retornos = retornos[(retornos.index >= inicio) & (retornos.index <= fim)]
retornos = retornos.dropna(axis=1, how='any')

fundamentus = pd.read_parquet('dados_fundamentus.parquet')
fundamentus = fundamentus[fundamentus['pl'] > 0]
fundamentus['ult_preco'] = pd.to_datetime(fundamentus['ult_preco'], errors='coerce') # transformando as colunas de data em datetime
fundamentus['ult_balanco'] = pd.to_datetime(fundamentus['ult_balanco'], errors='coerce')
fundamentus = fundamentus[(fundamentus['ult_balanco'] >= '2024-12-31') & (fundamentus['ult_preco'] > '2024-12-31')]

taxas = pd.read_parquet('taxas.parquet')
taxas = taxas[(taxas.index >= inicio) & (taxas.index <= fim)]
taxas = taxas.dropna(axis=1, how='any')

# Funções auxiliares exatamente como no seu código

# Filtrando as empresas que serão analisadas

def filtrar_fundamentus(fundamentus, retornos):
    df_fund = fundamentus[fundamentus['Papel'].isin(retornos.columns)]
    return df_fund

# Processamento dos dados

def book_to_market(df_fund):
    df_fund['btm'] = df_fund['pl'] / df_fund['market_cap']
    return df_fund

def calc_avg_size(df_fund): 
    avg_size = df_fund['market_cap'].mean()
    return avg_size

def calc_agv_btm(df_fund):
    avg_btm = df_fund['btm'].mean()
    return avg_btm

def classificar_empresas(df_fund):
    avg_size = calc_avg_size(df_fund)
    avg_btm = calc_agv_btm(df_fund)
    big = df_fund[df_fund['market_cap'] > avg_size]['Papel'].tolist()
    small = df_fund[df_fund['market_cap'] <= avg_size]['Papel'].tolist()
    high = df_fund[df_fund['btm'] > avg_btm]['Papel'].tolist()
    low = df_fund[df_fund['btm'] <= avg_btm]['Papel'].tolist()
    return big, small, high, low

def fatores(retornos, big, small, high, low):
    retornos['smb'] = retornos[small].mean(axis=1) - retornos[big].mean(axis=1)
    retornos['hml'] = retornos[high].mean(axis=1) - retornos[low].mean(axis=1)
    return retornos

def calc_dados_mercado(taxas):
    taxas['mkt'] = taxas['ibov'] - taxas['selic']
    return taxas

def base_modelo(retornos, dados_mercado, papel):
    dados = retornos[[papel,'smb','hml']].join(dados_mercado[['selic','mkt']], how='inner')
    dados = dados.dropna()
    dados['excesso'] = dados[papel] - dados['selic']
    return dados

def estimar_modelo(dados):
    X = dados[['mkt', 'smb', 'hml']]
    X = sm.add_constant(X)
    y = dados['excesso']
    modelo = sm.OLS(y, X).fit()
    estimado = modelo.predict(X)
    return estimado, y, modelo

def plotar_resultados(estimado, y, papel):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(estimado, y, alpha=0.6)
    ax.plot([min(estimado), max(estimado)], [min(estimado), max(estimado)], 'r--')
    ax.set_xlabel("Estimado")
    ax.set_ylabel("Observado")
    ax.grid(True)
    ax.set_title(f"Fama-French para {papel}")
    return fig

def interpretar_resultado(resultados, papel):
    r2 = resultados.rsquared
    coef = resultados.params
    pval = resultados.pvalues

    texto = f"**Análise do modelo Fama-French para o ativo {papel}:**\n\n"
    texto += f"O modelo apresenta um R-quadrado de {r2:.3f}, indicando que aproximadamente {r2*100:.1f}% da variação dos retornos é explicada pelos fatores.\n\n"

    alpha = coef['const']
    p_alpha = pval['const']
    signif_alpha = 'significativo' if p_alpha < 0.05 else 'não significativo'
    direcao_alpha = 'positivo' if alpha > 0 else 'negativo'
    texto += f"- O **alpha** estimado é {alpha:.4f} (p-valor = {p_alpha:.3f}), indicando um retorno anormal **{direcao_alpha}** e **{signif_alpha}**. Esse valor representa o retorno que não é explicado pelos fatores de risco do modelo.\n\n"

    for fator in ['mkt', 'smb', 'hml']:
        sinal = 'positiva' if coef[fator] > 0 else 'negativa'
        signif = 'significativa' if pval[fator] < 0.05 else 'não significativa'
        texto += f"- O coeficiente do fator {fator.upper()} é {coef[fator]:.4f} (p-valor = {pval[fator]:.3f}), indicando uma relação {sinal} e {signif} com o retorno do ativo.\n"

    return texto


# --- Streamlit interface ---

st.title("Modelo Fama-French 3 Fatores")

df_fund = filtrar_fundamentus(fundamentus, retornos)
df_fund = book_to_market(df_fund)
big, small, high, low = classificar_empresas(df_fund)
retornos = fatores(retornos, big, small, high, low)
dados_mercado = calc_dados_mercado(taxas)

papel = st.sidebar.selectbox("Selecione o papel para análise", retornos.columns.tolist())


if papel:
    dados = base_modelo(retornos, dados_mercado, papel)
    if dados.empty:
        st.warning("Não há dados suficientes para o papel selecionado nesse período.")
    else:
        estimado, y, modelo = estimar_modelo(dados)


        tab1, tab2 = st.tabs(["Resultados e Gráfico", "Explicação do Modelo"])

        # Aba 1: Resultado e Gráfico
        with tab1:
            col1, col2 = st.columns([3, 2])  # Ajuste a proporção conforme necessário (2:3)

            with col1:
                st.subheader("Resumo do Modelo")
                st.code(modelo.summary().as_text(), language='text')  # Exibe o resumo do modelo

            with col2:
                st.subheader("Gráfico dos Resultados")
                fig = plotar_resultados(estimado, y, papel)  # Exibe o gráfico
                st.pyplot(fig)

        # Aba 2: Explicação do Modelo
        with tab2:
            st.subheader("Modelo Fama-French de 3 Fatores")

            st.markdown(r"""
            O **modelo Fama-French de 3 fatores** é uma extensão do modelo tradicional de precificação de ativos, o **CAPM (Capital Asset Pricing Model)**. Desenvolvido por Eugene Fama e Kenneth French, o modelo busca explicar os retornos de um ativo a partir de três fatores principais:

            1. **MKT (Mercado)**: Refere-se ao retorno do mercado em relação à taxa livre de risco, geralmente representada pela **Selic** ou outro benchmark de taxa de juros. Esse fator captura o risco sistemático, que afeta todos os ativos de forma geral.
               
            2. **SMB (Small Minus Big)**: Este fator representa a diferença entre o retorno de empresas de **pequena** e **grande** capitalização. A teoria por trás do SMB é que **empresas pequenas** (com baixa capitalização de mercado) tendem a gerar retornos maiores em relação às **grandes empresas** (high market cap), devido a um maior potencial de crescimento e risco associado.
                
            3. **HML (High Minus Low)**: Refere-se à diferença de retorno entre **ações de valor** e **ações de crescimento**, baseando-se no **índice book-to-market**. O fator **HML** sugere que **ações de valor** (alto índice book-to-market) geralmente geram retornos mais altos do que **ações de crescimento** (baixo índice book-to-market).

            ### Como o Modelo Funciona:
                
            O modelo utiliza esses três fatores para explicar a **variação** no retorno de um ativo ou portfólio. Os coeficientes estimados para cada fator indicam a **sensibilidade** do ativo a cada um dos fatores. A equação do modelo é:
            """)
            
            # Exibindo a equação em LaTeX
            st.latex(r'''
            E(R_i) = R_f + \beta_{\text{MKT}} \cdot (E(R_m) - R_f) + \beta_{\text{SMB}} \cdot SMB + \beta_{\text{HML}} \cdot HML
            ''')

            st.markdown(r"""
            Onde:
            - **$E(R_i)$**: Esperado retorno do ativo $i$.
            - **$R_f$**: Taxa livre de risco.
            - **$E(R_m)$**: Retorno esperado do mercado.
            - **SMB**: Fator de porte (small minus big).
            - **HML**: Fator de valor (high minus low).

            ### Interpretação dos Coeficientes:

            - **Intercepto** ($\alpha$): Representa o retorno anormal do ativo, ou seja, o retorno que não é explicado pelos fatores de risco do modelo. Um valor positivo indica que o ativo teve um desempenho melhor do que o esperado, dados os riscos assumidos. Um valor negativo sugere um desempenho inferior ao previsto pelo modelo.

            - **Coeficiente MKT ($\beta_{\text{MKT}}$)**: Um valor positivo indica que o ativo tem uma **correlação positiva** com o mercado, ou seja, o ativo tende a acompanhar o movimento do mercado. Um coeficiente negativo sugere que o ativo se comporta de maneira **contrária** ao mercado.
            
            - **Coeficiente SMB ($\beta_{\text{SMB}}$)**: Um coeficiente positivo sugere que o ativo se comporta mais como uma ação de **pequena capitalização**, enquanto um valor negativo indica que o ativo segue mais o comportamento das **grandes empresas**.
            
            - **Coeficiente HML ($\beta_{\text{HML}}$)**: Coeficientes positivos indicam que o ativo tende a se comportar como uma ação de **valor**, ou seja, com um **alto índice book-to-market**. Coeficientes negativos indicam que o ativo segue o comportamento das **ações de crescimento**.

            ### Métricas Estatísticas:
            
            - **R-quadrado (R²)**: Mede a proporção da variabilidade do retorno do ativo que é explicada pelos fatores do modelo. Quanto mais próximo de 1, maior é o poder explicativo do modelo.
            
            - **P-valores**: Usados para testar a **significância estatística** dos coeficientes. Se o p-valor for menor que 0.05, o coeficiente é considerado **significativo**, ou seja, o fator tem um impacto estatisticamente relevante no retorno do ativo.

            ### Utilidade do Modelo:
            
            O modelo Fama-French de 3 fatores é amplamente utilizado em finanças para **avaliar** a relação entre os retornos dos ativos e os fatores de risco do mercado. Além disso, é fundamental para **análises de alocação de portfólio**, ajudando os investidores a entender os riscos e retornos associados a diferentes tipos de ativos.            """)
        
        # Exibe interpretação na sidebar
        interpretacao = interpretar_resultado(modelo, papel)
        st.sidebar.markdown(interpretacao)
