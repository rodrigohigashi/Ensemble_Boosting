📊 Análise Preditiva de Filmes - Classificação Binária de Avaliações 🎬
Este repositório contém um modelo de classificação binária para prever a classificação de filmes (1 para "bom" e 0 para "ruim") com base em um conjunto de dados sobre filmes do IMDB. A análise é realizada utilizando os algoritmos AdaBoost e Gradient Boosting. O código também inclui a manipulação de dados, análise exploratória e avaliação do modelo.

🚀 Estrutura do Código
🔧 Importação de Bibliotecas

As bibliotecas pandas, numpy, matplotlib, seaborn, entre outras, são utilizadas para manipulação de dados, visualização e modelagem preditiva.

O scikit-learn é usado para a construção e avaliação dos modelos de Machine Learning.

📂 Fonte dos Dados
Os dados utilizados neste projeto estão disponíveis publicamente e podem ser acessados no seguinte link:
https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data/data

📂 Carregamento e Limpeza de Dados

O dataset do IMDB é carregado a partir de um arquivo CSV.

O código trata valores ausentes nas colunas Revenue (Millions) (preenchendo com a mediana) e Metascore (preenchendo com a média).

O código também remove colunas desnecessárias e divide múltiplos gêneros de filmes em variáveis dummies.

📈 Análise Exploratória de Dados

São gerados gráficos de correlação para visualizar a relação entre as variáveis e a variável-alvo (Binary_Rating).

Relatórios detalhados das características do dataset são gerados utilizando as bibliotecas pandas_profiling e sweetviz.

🤖 Criação de Modelos

O código utiliza os algoritmos AdaBoostClassifier e GradientBoostingClassifier para prever a classificação binária.

Os dados são divididos em treino e teste com uma proporção de 70%/30%, respectivamente.

O desempenho dos modelos é avaliado por métricas como Acurácia, AUROC, Precision, Recall e F1.

📊 Avaliação de Desempenho

São gerados relatórios de desempenho para os dois modelos com as métricas mencionadas.

Além disso, a importância das variáveis é analisada para o modelo de Gradient Boosting.

📝 Como Usar
📋 Pré-requisitos

Para rodar este código, você precisa de um ambiente Python com as seguintes bibliotecas instaladas:

numpy

pandas

matplotlib

seaborn

scikit-learn

pandas_profiling

sweetviz

preditiva (Biblioteca personalizada para cálculos de desempenho e relatórios)

Para instalar as dependências, basta rodar o comando:

pip install -r requirements.txt
▶️ Execução

Após instalar as dependências, basta rodar o código Python para carregar o dataset, realizar a análise exploratória, treinar os modelos e avaliar o desempenho.

Os resultados serão apresentados em gráficos 📈 e métricas 📊.

⚙️ Personalização

Você pode personalizar os hiperparâmetros do modelo, como o número de estimadores (n_estimators), a taxa de aprendizado (learning_rate), entre outros, para otimizar os resultados.

Também é possível adicionar ou remover variáveis explicativas conforme necessário.

🔍 Observações
O dataset contém informações sobre filmes, como Rating, Votes, Revenue (Millions), entre outras. A variável de interesse para a classificação binária é a Rating, transformada em Binary_Rating (1 para filmes com nota >= 7 e 0 para filmes com nota < 7).

A análise pode ser facilmente adaptada para outros datasets de filmes ou outros tipos de classificação binária.