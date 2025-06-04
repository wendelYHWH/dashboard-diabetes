import dash
from dash import dcc, html, Input, Output, callback
from dash.dcc import send_bytes
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.graph_objects as go
from fpdf import FPDF

# --- Carregando dados e tratamento inicial ---
df = pd.read_csv("diabetes.csv")

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

# --- Treinamento modelo ---
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report_txt = classification_report(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# --- Função para gerar PDF ---
def create_pdf(buffer):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Relatório de Análise de Diabetes", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Acurácia do modelo: {accuracy:.2f}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, "Relatório de Classificação:", ln=True)
    pdf.set_font("Courier", size=10)
    for line in report_txt.split('\n'):
        pdf.cell(0, 6, line.strip(), ln=True)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    buffer.write(pdf_bytes)

# --- Inicializando App ---
app = dash.Dash(__name__)

# --- Layout ---
app.layout = html.Div([
    html.H1("Análise de Diabetes - Dashboard Interativo", style={'textAlign': 'center'}),

    html.H2("Distribuição das Classes"),
    dcc.Graph(
        id='dist-classes',
        figure=px.histogram(df, x='Outcome', title='Distribuição das classes (0=Não Diabético, 1=Diabético)')
    ),

    html.H2("Selecione a feature para visualizar"),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']],
        value='Glucose'
    ),
    dcc.Graph(id='hist-feature'),

    html.H2("Matriz de Correlação"),
    dcc.Graph(
        id='heatmap-corr',
        figure=px.imshow(df.corr(), text_auto=True, color_continuous_scale='RdBu_r',
                         title='Matriz de Correlação')
    ),

    html.H2("Resultado do Modelo de Regressão Logística"),
    html.P(f"Acurácia: {accuracy:.2f}"),

    html.H3("Relatório de Classificação"),
    html.Pre(report_txt),

    html.H3("Matriz de Confusão"),
    dcc.Graph(id='conf-matrix'),

    html.Hr(),
    html.Button("📄 Baixar Relatório em PDF", id="btn-pdf"),
    dcc.Download(id="download-pdf")
])

# --- Callbacks ---
@app.callback(
    Output('hist-feature', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_hist(selected_feature):
    fig = px.histogram(df, x=selected_feature, color='Outcome', barmode='overlay',
                       title=f'Distribuição de {selected_feature} por Diagnóstico',
                       labels={selected_feature: selected_feature, 'count': 'Frequência'})
    return fig

@app.callback(
    Output('conf-matrix', 'figure'),
    Input('feature-dropdown', 'value')  # usado apenas para trigger
)
def update_conf_matrix(_):
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predito 0', 'Predito 1'],
        y=['Verdadeiro 0', 'Verdadeiro 1'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        showscale=True))
    fig.update_layout(title='Matriz de Confusão')
    return fig

@app.callback(
    Output("download-pdf", "data"),
    Input("btn-pdf", "n_clicks"),
    prevent_initial_call=True
)
def gerar_pdf(n_clicks):
    return send_bytes(create_pdf, filename="relatorio_diabetes.pdf")

# --- Rodar App ---
if __name__ == '__main__':
    app.run(debug=True)