
# Dashboard de Análise de Diabetes

Este projeto é um dashboard interativo desenvolvido com [Dash](https://dash.plotly.com/) para análise e visualização de dados de diabetes. Ele inclui a criação de um modelo de regressão logística para previsão da presença de diabetes, visualizações gráficas e a geração de relatórios em PDF.

---

## Funcionalidades

- Visualização da distribuição das classes (diabético ou não)
- Histogramas das features selecionadas, segmentados por diagnóstico
- Matriz de correlação das variáveis
- Modelo de Regressão Logística treinado e avaliação da acurácia
- Exibição do relatório de classificação e matriz de confusão
- Download do relatório completo em PDF

---

## Tecnologias e Bibliotecas

- Python 3.x
- [Dash](https://dash.plotly.com/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [FPDF](https://pyfpdf.github.io/fpdf2/)

---

## Como rodar localmente

1. Clone o repositório:

```bash
git clone https://github.com/wendelYHWH/dashboard-diabetes.git
cd dashboard-diabetes
```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

> Caso não tenha o arquivo `requirements.txt`, as principais libs são:
> ```bash
> pip install dash pandas numpy scikit-learn plotly fpdf
> ```

4. Execute o app:

```bash
python app_diabetes.py
```

5. Acesse o dashboard no navegador:

```
http://127.0.0.1:8050
```

---

## Estrutura dos arquivos

- `app_diabetes.py`: Código principal do dashboard e análise
- `diabetes.csv`: Dataset usado para análise (não incluído no repositório, coloque-o na raiz)
- `README.md`: Este arquivo

---

## Contato

Criado por Wendel - [GitHub](https://github.com/wendelYHWH)

---

## Licença

Este projeto está sob a licença MIT.
