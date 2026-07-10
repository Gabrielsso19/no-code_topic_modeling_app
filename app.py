import streamlit as st
import pandas as pd

#LDA
import numpy as np
import re
import nltk
import pyLDAvis
import pyLDAvis.lda_model
import streamlit.components.v1 as components
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

#BERTopic
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

@st.cache_resource
def download_nltk_deps():
    nltk.download('stopwords')
download_nltk_deps()

# Configuração da página
st.set_page_config(page_title="No-Code Topic Modeling", layout="wide")
st.title("🧙‍♂️ No-Code Topic Modeling Platform")
st.sidebar.header("Configurações do Projeto")

# 1. Upload da Base de Dados
uploaded_file = st.sidebar.file_uploader("Importe sua base de dados (CSV ou Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Leitura dos dados
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
    st.write("### Prévia dos Dados", df.head(3))
    
    # Seleção da coluna de texto
    text_column = st.sidebar.selectbox("Selecione a coluna com os textos:", df.columns)
    docs = [
        str(text).strip() 
        for text in df[text_column].dropna() 
        if str(text).strip() != ""
    ]

    # Métrica informativa para o usuário saber quantos documentos válidos restaram
    st.sidebar.metric(label="Documentos válidos para treino", value=len(docs))

    st.divider()

    if len(docs) == 0:
        st.error("⚠️ A coluna selecionada não possui nenhum texto válido após a filtragem das linhas em branco!")
    
    # 2. Seleção do Modelo
    model_choice = st.sidebar.radio("Escolha o algoritmo de Modelagem:", ["LDA (Scikit-Learn)", "BERTopic"])
    
    # 3. Parâmetros Dinâmicos
    if model_choice == "LDA (Scikit-Learn)":
        st.sidebar.title("Configurações do LDA")

        # 1. Limpeza de texto e stopwords
        remover_pontuacao = st.sidebar.checkbox("Remover Pontuação e Caracteres Especiais", value=True)
        remover_numeros = st.sidebar.checkbox("Remover Números", value=True)
        converter_minusculo = st.sidebar.checkbox("Converter tudo para minúsculo", value=True)
        
        idiomas_stopwords = st.sidebar.multiselect(
            "Selecione os idiomas para Stopwords:",
            options=["portuguese", "english"],
            default=["portuguese"]
        )
        
        stopwords_customizadas = st.sidebar.text_area(
            "Stopwords customizadas (separadas por vírgula):",
            help="Exemplo: ola, voce, tambem"
        )

        if st.sidebar.button("🧼 Executar Limpeza de Texto", type="primary"):
            with st.spinner("A limpar e a filtrar a base de dados..."):
                # Processar lista final de stopwords
                lista_stopwords = []
                for lang in idiomas_stopwords:
                    lista_stopwords.extend(stopwords.words(lang))
                    
                if stopwords_customizadas:
                    custom_words = [w.strip().lower() for w in stopwords_customizadas.split(",") if w.strip()]
                    lista_stopwords.extend(custom_words)
                    
                # Eliminar duplicados das stopwords
                st.session_state.lista_stopwords = list(set(lista_stopwords))

                # Função aplicada para limpar os documentos válidos
                def limpar_texto(texto):
                    if converter_minusculo:
                        texto = texto.lower()
                    if remover_numeros:
                        texto = re.sub(r'\d+', '', texto)
                    if remover_pontuacao:
                        texto = re.sub(r'[^\w\s]', '', texto)
                    # Remover espaços múltiplos gerados pelas limpezas
                    texto = re.sub(r'\s+', ' ', texto).strip()
                    return texto
        
                # Filtragem e limpeza preliminar dos documentos
                st.session_state.docs_limpos = [limpar_texto(doc) for doc in docs if limpar_texto(doc) != ""]
                st.success("Texto limpo com sucesso!")

        # Exibir métrica se a limpeza já tiver sido realizada
        if st.session_state.docs_limpos is not None:
            st.sidebar.metric(label="Documentos prontos para treino", value=len(st.session_state.docs_limpos))
        else:
            st.sidebar.info("👉 Clique no botão 'Executar Limpeza de Texto' acima para começar.")
        
        # Só avança se a base de dados já tiver sido limpa pelo utilizador
        if st.session_state.docs_limpos:
            docs_para_treino = st.session_state.docs_limpos
            stopwords_finais = st.session_state.lista_stopwords

            # 2. TESTE DE NÚMERO DE TÓPICOS (MÉTRICAS)
            st.write("## ⚙️ Otimização e Ajuste de Tópicos (LDA)")
            
            with st.expander("📊 Rodar Teste de Coerência e Perplexidade", expanded=False):
                st.markdown("""
                A **Perplexidade** mede o quão bem o modelo prevê a amostra (quanto menor, melhor). 
                A **Coerência** mede o grau de similaridade semântica entre as palavras de alta pontuação no tópico (quanto maior, melhor).
                """)

                min_k = st.number_input("Mínimo de Tópicos para teste", min_value=2, max_value=10, value=2)
                max_k = st.number_input("Máximo de Tópicos para teste", min_value=3, max_value=30, value=8)
                
                if st.button("📈 Calcular Métricas de Avaliação"):
                    lista_k = list(range(int(min_k), int(max_k) + 1))
                    perplexidades = []
                    coerencias = []
                    
                    # Vetorização temporária para o teste
                    tf_vectorizer_test = CountVectorizer(stop_words=stopwords_finais)
                    tf_test = tf_vectorizer_test.fit_transform(docs_para_treino)
                    
                    # Preparar dados para o cálculo de coerência via Gensim
                    # (Precisamos quebrar os textos em tokens/palavras)
                    textos_tokenizados = [doc.split() for doc in docs_para_treino]
                    dicionario_gensim = Dictionary(textos_tokenizados)
                    
                    barra_progresso = st.progress(0)
                    
                    for idx, k in enumerate(lista_k):
                        # Treinar LDA temporário do scikit-learn
                        lda_teste = LatentDirichletAllocation(n_components=k, max_iter=5, random_state=42, n_jobs=-1)
                        lda_teste.fit(tf_test)
                        
                        # 1. Perplexidade (Scikit-Learn)
                        perplexidades.append(lda_teste.perplexity(tf_test))
                        
                        # 2. Coerência (Gensim adaptado com as principais palavras do Scikit-Learn)
                        # Extrair as top 10 palavras de cada tópico gerado pelo Sklearn
                        nomes_features = tf_vectorizer_test.get_feature_names_out()
                        topicos_palavras = []
                        for topic_idx, topic in enumerate(lda_teste.components_):
                            top_palavras_idx = topic.argsort()[:-11:-1]
                            topicos_palavras.append([nomes_features[i] for i in top_palavras_idx])

                        # Calcular Coerência C_V usando os tópicos extraídos
                        cm = CoherenceModel(topics=topicos_palavras, texts=textos_tokenizados, dictionary=dicionario_gensim, coherence='c_v')
                        coerencias.append(cm.get_coherence())
                        
                        barra_progresso.progress((idx + 1) / len(lista_k))
                    
                    # Exibir gráficos comparativos
                    df_metricas = pd.DataFrame({
                        'Número de Tópicos (K)': lista_k,
                        'Perplexidade (Menor é Melhor)': perplexidades,
                        'Coerência (Maior é Melhor)': coerencias
                    }).set_index('Número de Tópicos (K)')
                    
                    st.line_chart(df_metricas['Perplexidade (Menor é Melhor)'])
                    st.line_chart(df_metricas['Coerência (Maior é Melhor)'])
                    st.dataframe(df_metricas)
            
            # 3. CONFIGURAÇÃO FINAL E EXECUÇÃO
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎯 Parâmetros de Treino Final")
            n_components_final = st.sidebar.slider("Número Final de Tópicos (K)", min_value=2, max_value=50, value=5)
            max_iter_final = st.sidebar.number_input("Máximo de Iterações", value=10, min_value=1)
            lda_seed = st.sidebar.number_input("LDA Seed (Reprodutibilidade)", value=42)

    elif model_choice == "BERTopic":
        st.sidebar.title("Configurações do BERTopic")

        min_topic_size = st.sidebar.number_input("Tamanho Mínimo do Tópico", value=10)

        # 1. Seleção do Modelo de Embedding
        embedding_model = st.sidebar.selectbox("Modelo de Embedding", ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"])

        # 2. Configurações do UMAP (Redução de Dimensionalidade)
        st.sidebar.markdown("---")
        st.sidebar.subheader("📐 Parâmetros do UMAP")
        
        umap_seed = st.sidebar.number_input("UMAP Seed (Reprodutibilidade)", value=42, step=1)
        n_neighbors = st.sidebar.slider("Vizinhos (n_neighbors)", min_value=2, max_value=100, value=15, help="Controla como o UMAP equilibra a estrutura local vs. global.")
        n_components = st.sidebar.slider("Componentes de Saída", min_value=2, max_value=10, value=5, help="Dimensões finais para onde os dados serão reduzidos antes do agrupamento.")
        min_dist = st.sidebar.slider("Distância Mínima (min_dist)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, help="Controla o quão próximos os pontos podem ficar na redução.")
        
        # 3. Configurações do HDBSCAN (Agrupamento/Clustering)
        st.sidebar.markdown("---")
        st.sidebar.subheader("⬢ Parâmetros do HDBSCAN")
        
        min_cluster_size = st.sidebar.number_input("Tamanho Mínimo do Tópico (min_cluster_size)", value=10, min_value=2, help="O menor tamanho de grupo que o algoritmo considerará um tópico.")
        min_samples = st.sidebar.number_input("Amostras Mínimas (min_samples)", value=5, min_value=1, help="Controla a sensibilidade ao ruído. Valores maiores tornam o modelo mais conservador.")
        
    # 4. Botão de Execução
    if st.sidebar.button("🚀 Executar Modelagem de Tópicos"):
        with st.spinner("Treinando o modelo... Isso pode levar alguns minutos dependendo do tamanho da base."):
            
            if model_choice == "LDA (Scikit-Learn)":
                st.info("Processando LDA...")
                # Vetorização (CountVectorizer)
                tf_vectorizer = CountVectorizer(stop_words=stopwords_finais)
                tf_data = tf_vectorizer.fit_transform(docs_para_treino)

                # Instanciação e Treino do LDA
                lda_model = LatentDirichletAllocation(
                    n_components=n_components_final,
                    max_iter=max_iter_final,
                    random_state=lda_seed,
                    n_jobs=-1
                )
                
                # Matriz Documento-Tópico
                doc_topic_dist = lda_model.fit_transform(tf_data)
                
                # Descobrir o tópico predominante de cada linha treinada
                topico_predominante = np.argmax(doc_topic_dist, axis=1)
                
                def limpar_validador(texto):
                    if converter_minusculo: texto = str(texto).lower()
                    if remover_numeros: texto = re.sub(r'\d+', '', str(texto))
                    if remover_pontuacao: texto = re.sub(r'[^\w\s]', '', str(texto))
                    return re.sub(r'\s+', ' ', str(texto)).strip()
                
                # --- ALINHAMENTO COM O DATAFRAME ORIGINAL ---
                df_cleaned_lda = df.copy()
                df_cleaned_lda['__texto_limpo__'] = df_cleaned_lda[text_column].apply(limpar_validador)
                mascara_validos_lda = (df_cleaned_lda['__texto_limpo__'] != "") & (df_cleaned_lda[text_column].notna())
                
                df['LDA_Topic_ID'] = "Ignorado/Vazio"
                df.loc[mascara_validos_lda, 'LDA_Topic_ID'] = topico_predominante

                st.success("LDA Concluído!")

                # Download dos Resultados
                st.write("### 📂 Base de Dados Classificada (LDA)")
                st.dataframe(df[[text_column, 'LDA_Topic_ID']].head(10), use_container_width=True)
                
                csv_data_lda = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Baixar Base de Dados com Tópicos (LDA)",
                    data=csv_data_lda,
                    file_name="base_com_topicos_lda.csv",
                    mime="text/csv",
                )

                st.divider()
                st.write("### 📊 Painel Interativo pyLDAvis")
                st.markdown("Use o painel abaixo para explorar a distribuição de termos e relevância dos tópicos criados:")
                
                # Preparar os dados para o pyLDAvis
                painel_dados = pyLDAvis.lda_model.prepare(
                    lda_model, 
                    tf_data, 
                    tf_vectorizer, 
                    mds='tsne', # t-SNE costuma funcionar muito bem para projeção 2D do LDA
                    sort_topics=False
                )
                
                # Renderizar o pyLDAvis como componente HTML dentro do Streamlit
                html_painel = pyLDAvis.prepared_data_to_html(painel_dados)
                components.html(html_painel, height=850, scrolling=True)
                st.download_button(
                    label="🌐 Baixar Painel pyLDAvis (HTML Interativo)",
                    data=html_painel,
                    file_name="visualizacao_lda_interativa.html",
                    mime="text/html",
                    help="Baixe o gráfico interativo para abrir no seu navegador a qualquer momento, mesmo sem o Streamlit rodando."
                )

            elif model_choice == "BERTopic":
                st.info("Processando BERTopic...")
                # Passo 1: Configurar UMAP com a Seed para garantir reprodutibilidade
                umap_model = UMAP(
                    n_neighbors=n_neighbors, 
                    n_components=n_components, 
                    min_dist=min_dist, 
                    metric='cosine', 
                    random_state=umap_seed
                )
                
                # Passo 2: Configurar HDBSCAN
                hdbscan_model = HDBSCAN(
                    min_cluster_size=min_cluster_size, 
                    min_samples=min_samples, 
                    metric='euclidean', 
                    cluster_selection_method='eom',
                    prediction_data=True
                )
                
                # Passo 3: Inicializar e treinar o BERTopic acoplando os submodelos modificados
                topic_model = BERTopic(
                    min_topic_size=min_topic_size,
                    embedding_model=embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    calculate_probabilities=True
                )

                #Treino
                topics, probs = topic_model.fit_transform(docs)
                st.success("BERTopic Concluído!")
                
                # --- APRESENTAÇÃO DOS RESULTADOS ---
                st.write("### 📋 Visão Geral dos Tópicos Encontrados")
                df_topics = topic_model.get_topic_info()
                st.dataframe(df_topics, use_container_width=True)
                csv_data = df_topics.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Baixar Tópicos (CSV)",
                    data=csv_data,
                    file_name="topicos_bertopic.csv",
                    mime="text/csv",
                )

                # --- MAPEAMENTO DE VOLTA PARA O DATAFRAME ORIGINAL ---
            
                # Criamos uma máscara idêntica à que usamos para filtrar os docs para saber exatamente quais índices do DataFrame original receberam tópicos
                df_cleaned = df.copy()
                df_cleaned['__texto_limpo__'] = df_cleaned[text_column].astype(str).str.strip()
                mascara_validos = (df_cleaned['__texto_limpo__'] != "") & (df_cleaned[text_column].notna())
                
                # Inicializa as colunas de resultado com um valor padrão para as linhas vazias
                df['BERTopic_ID'] = "Ignorado (Linha em Branco)"
                df['BERTopic_Nome'] = "Ignorado (Linha em Branco)"
                
                # Atribui os IDs dos tópicos gerados apenas às linhas que foram para o treino
                df.loc[mascara_validos, 'BERTopic_ID'] = topics
                
                # Cria um dicionário mapeando o ID do tópico para o nome/palavras-chave dele
                nomes_topicos = topic_model.get_topic_info().set_index('Topic')['Name'].to_dict()
                
                # Atribui os nomes legíveis dos tópicos no DataFrame
                df.loc[mascara_validos, 'BERTopic_Nome'] = [nomes_topicos[t] for t in topics]
                
                # --- EXIBIÇÃO E EXPORTAÇÃO ---
                st.write("### 📂 Base de Dados Classificada")
                st.write("Aqui está uma prévia do seu arquivo original com as novas colunas de tópicos:")
                
                # Mostra as últimas colunas criadas junto com a coluna de texto original para conferência
                colunas_exibicao = [text_column, 'BERTopic_ID', 'BERTopic_Nome']
                st.dataframe(df[colunas_exibicao].head(10), use_container_width=True)
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Baixar Base de Dados com Tópicos (CSV)",
                    data=csv_data,
                    file_name="base_com_topicos_bertopic.csv",
                    mime="text/csv",
                )

                # Visualizações Interativas do BERTopic
                st.write("### 📊 Gráficos Interativos")
                
                tab1, tab2, tab3 = st.tabs(["Mapa de Distância", "Hierarquia de Tópicos", "Palavras-Chave"])
                
                with tab1:
                    st.subheader("Mapa Interativo de Distância Intertópica")
                    fig_distance = topic_model.visualize_topics()
                    st.plotly_chart(fig_distance, use_container_width=True)
                    
                with tab2:
                    st.subheader("Agrupamento Hierárquico dos Tópicos")
                    # Evita erro caso tenha gerado poucos tópicos para criar uma hierarquia
                    if len(df_topics) > 2:
                        fig_hierarchy = topic_model.visualize_hierarchy()
                        st.plotly_chart(fig_hierarchy, use_container_width=True)
                    else:
                        st.info("Tópicos insuficientes para gerar um gráfico hierárquico.")
                        
                with tab3:
                    st.subheader("Importância das Palavras por Tópico (c-TF-IDF)")
                    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
                    st.plotly_chart(fig_barchart, use_container_width=True)

else:
    st.info("👋 Por favor, faça o upload de uma base de dados na barra lateral para começar.")