from ultralytics import YOLO
from functools import partial
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from pdf_report import PDF
from lib.facial_emotions.facial_emotions import HSEmotionRecognizer

import os
import cv2
import time
import csv
import dlib
import pytz
import smtplib
import datetime
import schedule
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import io

saopaulo_timezone = pytz.timezone('America/Sao_Paulo')
tempo_ultima_salvacao = datetime.datetime.now(saopaulo_timezone)


emotions_count = {
    "Anger": 0,
    "Contempt": 0,
    "Disgust": 0,
    "Fear": 0,
    "Happiness": 0,
    "Neutral": 0,
    "Sadness": 0,
    "Surprise": 0
}

age_count = {
    '(0-2)': 0,
    '(4-6)': 0,
    '(8-12)': 0,
    '(15-20)': 0,
    '(25-32)': 0,
    '(38-43)': 0,
    '(48-53)': 0,
    '(60-100)': 0
}

emotions = {
    "Anger": {
        "color": (193, 69, 42)
    },
    "Contempt": {
        "color": (164, 175, 49)
    },
    "Disgust": {
        "emotion": "Disgust",
        "color": (40, 52, 155)
    },
    "Fear": {
        "color": (128, 0, 128)
    },
    "Happiness": {
        "color": (88, 158, 38)
    },
    "Neutral": {
        "color": (218, 229, 97)
    },
    "Sadness": {
        "emotion": "Sadness",
        "color": (108, 72, 200)
    },
    "Surprise": {
        "color": (164, 93, 23)
    }
}


# Load the YOLOv8 model
model = YOLO('resources/yolov8n-face.pt')

model_name = 'enet_b0_8_best_afew'
fer = HSEmotionRecognizer(model_name=model_name)

predictor = dlib.shape_predictor(
    "resources/shape_predictor_68_face_landmarks.dat")
facial_recognition_model = dlib.face_recognition_model_v1(
    "resources/dlib_face_recognition_resnet_model_v1.dat")

jitters = 11
max_embeddings = 10
persons_counter = 0
embeddings_global_list = []

padding = 10
font = cv2.FONT_HERSHEY_SIMPLEX

# Store the track history and track start times
track_history = defaultdict(lambda: [])
track_start_times = defaultdict(lambda: 0)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageProto = "resources/age_deploy.prototxt"
ageModel = "resources/age_net.caffemodel"
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']

ageNet = cv2.dnn.readNet(ageModel, ageProto)

# init PDF class
csv_data = None

def plot_sum_of_columns_to_bytes(df, filename, title):
    if df.empty:
        print("DataFrame está vazio.")
        return None
    
    # Remove as colunas 'Timestamp' e 'Total_pessoas' do cálculo da soma
    cols_to_sum = [col for col in df.columns if col not in ['Timestamp', 'Total_pessoas']]
    
    # Calcula a soma de cada coluna
    sum_of_columns = df[cols_to_sum].sum()
    
    # Define uma paleta de cores personalizada
    num_cols = len(sum_of_columns)
    color_palette = plt.get_cmap('Pastel2')(range(num_cols))
    
    # Cria o gráfico de barras
    plt.figure(figsize=(5, 4))
    bars = sum_of_columns.plot(kind='bar', color=color_palette)
    plt.title(f'{title}', fontsize=10)

    plt.xticks(rotation=45, fontsize=6)
    plt.yticks(fontsize=6) 
    plt.grid(axis='y')
    
    # Adiciona os valores em cima de cada barra
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, round(bar.get_height(), 2), ha='center', va='bottom', fontsize=6)
    
    
    plt.savefig(f"data/{filename}.png", format='png')


def plot_time_graph(df, x_column, y_columns, filename):
    # Converter a coluna de timestamp para o formato datetime
    df[x_column] = pd.to_datetime(df[x_column])

    # Extrair apenas as horas da coluna de timestamp
    df['Hora'] = df[x_column].dt.strftime('%H:%M')

    # Cores pastel para os gráficos
    pastel_colors = ['#B0E0E6']

    # Configurar o gráfico
    plt.figure(figsize=(5, 4))
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8) 
    plt.title('Detections per time', fontsize=12)

    # Plotar cada coluna do DataFrame no gráfico com cores pastel
    for i, column in enumerate(y_columns):
        plt.plot(df['Hora'], df[column], label=column, color=pastel_colors[i])

    legend_names = {'Total_Pessoas': 'Total Detections'}

    # Adicionar legenda com os nomes personalizados
    plt.legend(labels=[legend_names.get(col, col) for col in y_columns])

    # Salvar o gráfico como um arquivo PNG
    plt.tight_layout()  # Para evitar cortes nos rótulos dos eixos
    plt.savefig(f"data/{filename}.png", format='png')

def zerar_contadores():
    global emotions_count, age_count
    emotions_count = {
        "Anger": 0,
        "Contempt": 0,
        "Disgust": 0,
        "Fear": 0,
        "Happiness": 0,
        "Neutral": 0,
        "Sadness": 0,
        "Surprise": 0
    }

    age_count = {
        '(0-2)': 0,
        '(4-6)': 0,
        '(8-12)': 0,
        '(15-20)': 0,
        '(25-32)': 0,
        '(38-43)': 0,
        '(48-53)': 0,
        '(60-100)': 0
    }


def age_classifier(face):
    # blob = cv2.dnn.blobFromImages([face], 1.5, (227, 227), MODEL_MEAN_VALUES, swapRB=False) # FIXME: batch
    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return age


def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)


def extract_face_embeddings(frame, faceBox):
    embeddings_list = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceBoxDlib = dlib.rectangle(
        faceBox[0], faceBox[1], faceBox[2], faceBox[3])
    landmarks = predictor(gray, faceBoxDlib)

    face_embedding = facial_recognition_model.compute_face_descriptor(
        frame, landmarks)

    face_embedding_np = np.array(face_embedding)

    embeddings_list.append(face_embedding_np)

    return embeddings_list


def process_faces(frame, box, track_id, track_history):
    global persons_counter, embeddings_global_list

    x, y, w, h = box.cpu().numpy()
    track = track_history[track_id]
    track.append((float(x), float(y)))  # x, y center point

    # Check if it's a new track, and record the start time
    if len(track) == 1:
        track_start_times[track_id] = time.time()

    if len(track) > 30:  # retain 90 tracks for 90 frames
        track.pop(0)

    # Crop and store the detected face
    tly, bry, tlx, brx = int(y - h/2), int(y + h/2), int(x - w/2), int(x + w/2)
    face = frame[max(0, tly - padding):min(bry + padding, frame.shape[0] - 1),
                 max(0, tlx - padding):min(brx + padding, frame.shape[1] - 1)]

    faceBox = [tly, bry, tlx, brx]
    embeddings = extract_face_embeddings(frame, faceBox)
    # embeddings = fr.face_encodings(cv2.cvtColor(face, cv2.COLOR_BGR2RGB),
    #                                  boxes, num_jitters=jitters)
    ages = age_classifier(face)

    emotion_prediction, emotion_probability = fer.predict_emotions(
        face, logits=False)

    for embedding in embeddings:
        if len(embeddings_global_list) == 0:
            embeddings_global_list.append(embedding)
            persons_counter += 1
            for emotion in emotions_count.keys():
                if (np.max(emotion_probability) > 0.36):
                    if (emotion == emotion_prediction):
                        emotions_count[emotion] += 1
                        break
            for age_option in age_count.keys():
                if (ages == age_option):
                    age_count[age_option] += 1
        else:
            new_face = True
            for stored_embedding in embeddings_global_list:
                distance = euclidean_distance(embedding, stored_embedding)
                if distance < 0.4:
                    new_face = False
                    break
            if new_face:
                if len(embeddings_global_list) == max_embeddings:
                    embeddings_global_list.pop(0)
                embeddings_global_list.append(embedding)
                persons_counter += 1
                for emotion in emotions_count.keys():
                    if (np.max(emotion_probability) > 0.36):
                        if (emotion == emotion_prediction):
                            emotions_count[emotion] += 1
                            break
                for age_option in age_count.keys():
                    if (ages == age_option):
                        age_count[age_option] += 1
    if (np.max(emotion_probability) > 0.36):

        age_text = f'{ages[1:-1]} anos'
        color = emotions[emotion_prediction]['color']

        frame = cv2.rectangle(
            frame, (tlx, tly), (tlx + int(w), tly + int(h)), color, 2)
        frame = cv2.line(frame, (tlx, tly + int(h)), (tlx + 20, tly + int(h) + 20),
                         color,
                         thickness=2)

        frame = cv2.putText(frame, f'{emotion_prediction}',
                            (tlx + 25, tly + int(h) +
                             36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        frame = cv2.putText(frame, age_text,
                            (tlx + 25, tly + int(h) +
                             56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Calculate the time duration the detection stayed on the screen
        current_time = time.time()
        duration = current_time - track_start_times[track_id]

        time_text = f"{duration:.2f}s"
        frame = cv2.putText(frame, time_text,
                            (tlx + 25, tly + int(h) +
                             76), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
    else:
        color = (255, 255, 255)
        frame = cv2.rectangle(
            frame, (tlx, tly), (tlx + int(w), tly + int(h)), color, 2)

    return frame


def salvar_csv(age_count, emotions_count, persons_counter):
    global df

    nome_arquivo = "data/dados.csv"

    arquivo_existe = os.path.isfile(nome_arquivo)

    with open(nome_arquivo, mode='a' if arquivo_existe else 'w', newline='') as arquivo_csv:
        colunas = ['Timestamp'] + list(age_count.keys()) + \
            list(emotions_count.keys()) + ['Total_Pessoas']
        escritor_csv = csv.DictWriter(arquivo_csv, fieldnames=colunas)

        if not arquivo_existe:
            escritor_csv.writeheader()

        timestamp = datetime.datetime.now(
            saopaulo_timezone).strftime("%Y-%m-%d %H:%M:%S")

        csv_data = {'Timestamp': timestamp}
        csv_data.update(age_count)
        csv_data.update(emotions_count)
        csv_data['Total_Pessoas'] = persons_counter

        escritor_csv.writerow(csv_data)
        zerar_contadores()

    df = pd.read_csv("data/dados.csv")


def enviar_email():
    global df, pdf_report

    pdf_report = PDF(pdf_title='IA Recognition Report')
    pdf_report.add_page()
    image_w, image_h = 188, 50  # Ajuste as dimensões conforme necessário
    
    # Calcular as dimensões da página
    page_w = pdf_report.w
    page_h = pdf_report.h
    
    # Calcular a posição x e y para centralizar a imagem
    x = (page_w - image_w) / 2
    y = (page_h - image_h) / 2
    
    # Adicionar a imagem
    pdf_report.image("data/iracema-merged.png", x, y, w=image_w, h=image_h)

    pdf_report.add_page()
    pdf_report.set_font("Arial", size=10)
    pdf_report.cell(200, 5, txt=f"Datetime: {df['Timestamp'].tail(1).values[0]}", ln=True)
    pdf_report.cell(200, 5, txt=f'\t \t \t \t Detected Ages: {df.iloc[:,1:9].sum().sum()}', ln=True)
    pdf_report.cell(200, 5, txt=f'\t \t \t \t Detected Emotions: {df.iloc[:,10:-1].sum().sum()}', ln=True)
    pdf_report.cell(200, 5, txt=f"\t \t \t \t Detected People: {df['Total_Pessoas'].tail(1).values[0]}", ln=True)
    plot_sum_of_columns_to_bytes(df.iloc[:,1:9], 'age', 'Detected Ages') # ou ler uma imagem
    plot_sum_of_columns_to_bytes(df.iloc[:,10:-1], 'emotion', 'Detected Emotions') 
    plot_time_graph(df, 'Timestamp', ['Total_Pessoas'], 'grafico_tempo')


    pdf_report.image("data/age.png")

    pdf_report.image("data/emotion.png")

    pdf_report.image("data/grafico_tempo.png")

    pdf_name = f"data/report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    pdf_report.output(pdf_name)

    if df.empty:
        print('[INFO]: DF data is None')
    else:
        try:
            from_email = "pedropedrosa@lapisco.ifce.edu.br"
            to_email = "juliomacedochaves@gmail.com"
            senha = "@lapisco2024"

            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
            server.starttls()
            server.login(from_email, senha)

            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = "IA Recognition Report Facial"

            body = "Olá,\n\nSegue relatório(s) referente(s) aos dados registrados pelos analíticos faciais durante os últimos 15 minutos.\n\nAtenciosamente,\nEquipe do Lapisco/Instituto Iracema."
            msg.attach(MIMEText(body, 'plain'))



            for key, value in df.items():
                pdf_report.cell(200, 10, txt=f'{key}: {value}', ln=True)


            filename = pdf_name
            attachment = open(filename, "rb")
            part = MIMEBase("application", "octet-stream")
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment; filename= " + filename)
            msg.attach(part)

            server.send_message(msg)
            print("E-mail enviado com sucesso para", to_email)

            server.quit()
            # os.rename(pdf_name, f"data/{datetime.datetime.now(saopaulo_timezone).strftime('%Y-%m-%d_%H-%M-%S')}.csv")

            df = None
        except:
            # os.rename(
                # pdf_name, f"data/{datetime.datetime.now(saopaulo_timezone).strftime('%Y-%m-%d_%H-%M-%S')}.csv")

            df = None



intervalo = datetime.timedelta(minutes=15)
schedule.every(3).hours.do(enviar_email)

while True:

    if datetime.datetime.now(saopaulo_timezone) - tempo_ultima_salvacao >= intervalo:
        salvar_csv(age_count, emotions_count, persons_counter)
        tempo_ultima_salvacao = datetime.datetime.now(saopaulo_timezone)


    schedule.run_pending()

    frame = cv2.imread("../stream/frame.jpg")
    frame = cv2.resize(frame, (640, 480))
    # frame = cv2.resize(frame, None, fx=0.75, fy=0.75)

    # Start timer
    new_frame_time = time.time()

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, conf=0.50, verbose=False)

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Process faces in parallel
        with ThreadPoolExecutor() as executor:
            processed_frames = list(executor.map(
                partial(process_faces, track_history=track_history),
                [frame] * len(track_ids), boxes, track_ids))

        frame = processed_frames[0]

    prev_frame_time = time.time()
    fps = 1/(prev_frame_time-new_frame_time)

    cv2.putText(frame, "FPS: {:.2f}".format(fps), (5, 25), font,
                1, (255, 0, 0), 1, cv2.LINE_AA)

    text = f"Visitantes: {persons_counter}"
    text_size = cv2.getTextSize(text, font, 1, 1)[0]
    text_width, text_height = text_size[0], text_size[1]
    text_x = (frame.shape[1] - text_width) // 2
    cv2.putText(frame, text, (text_x, 25), font,
                1, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite("./frame_temp.jpg", frame)
    os.system("mv frame_temp.jpg frame.jpg")
