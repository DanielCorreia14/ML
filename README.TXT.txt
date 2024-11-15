Análise de Séries Temporais com MLPRegressor
Este projeto tem como objetivo a previsão de valores de séries temporais utilizando um modelo de Multilayer Perceptron Regressor (MLPRegressor), uma rede neural artificial do scikit-learn, para prever os valores futuros de um índice de ações (Ibovespa).

Descrição do Código
O código realiza os seguintes passos:

1. Leitura e Limpeza dos Dados
O arquivo ibovespa.csv é lido e processado.
Os valores da coluna "ibovespa" são convertidos para numéricos, com a função pd.to_numeric, e valores inválidos são convertidos para NaN e removidos com dropna().
2. Normalização dos Dados
Os dados são normalizados para um intervalo de 0 a 1, utilizando a classe MinMaxScaler do scikit-learn. Isso é importante para garantir que os valores de entrada estejam dentro de um intervalo que ajude o treinamento do modelo.
3. Criação do Dataset Janelado com TimeseriesGenerator
A série temporal é dividida em "janelas" de observação, onde cada janela contém 10 valores consecutivos da série.
O TimeseriesGenerator cria os pares de entradas (X) e saídas (y), sendo que as entradas representam os valores anteriores e as saídas representam o valor futuro.
4. Treinamento do Modelo
O modelo de rede neural MLPRegressor é configurado com duas camadas ocultas (com 100 e 50 neurônios) e usa a função de ativação ReLU.
O modelo é treinado utilizando as janelas criadas no passo anterior.
5. Avaliação do Modelo
Após o treinamento, o modelo é avaliado utilizando duas métricas principais:
Mean Squared Error (MSE): Mede o erro quadrático médio entre os valores reais e as previsões.
Mean Absolute Error (MAE): Mede o erro absoluto médio entre os valores reais e as previsões.
R² (Coeficiente de Determinação): Mede a qualidade da previsão, ou seja, a proporção da variabilidade dos dados explicada pelo modelo.
6. Visualização dos Resultados
O código gera vários gráficos para ajudar a visualizar os resultados do modelo:
Curva de Perda: Mostra como o erro da rede diminui durante o treinamento.
Dados de Treinamento: Um gráfico comparando os valores reais e as previsões para o conjunto de treinamento.
Resíduos: Um gráfico de dispersão mostrando a diferença entre os valores reais e as previsões no conjunto de teste.
7. Gráficos Salvos
Todos os gráficos são salvos como arquivos PNG no diretório onde o script é executado:
primeira_janela.png: Exibe a primeira janela de dados.
curva_perda.png: Exibe a curva de perda durante o treinamento.
dados_treinamento.png: Exibe os dados reais e previstos no treinamento.
residuos.png: Exibe os resíduos do modelo.
Requisitos
Python 3.x
Bibliotecas:
pandas
numpy
scikit-learn
tensorflow
matplotlib
Você pode instalar as dependências necessárias utilizando o pip:

bash
Copiar código
pip install -r requirements.txt
O arquivo requirements.txt pode ser gerado utilizando o comando pip freeze > requirements.txt caso não o tenha.

Como Executar
Para executar o código, basta rodar o seguinte comando:

bash
Copiar código
python3 main.py
O código irá gerar os gráficos e mostrar as métricas de avaliação no terminal.