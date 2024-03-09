import os
import cv2
import csv
import datetime
import cvzone
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import smtplib
import datetime
import pytz

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from lib.tracker.tracker import *
from lib.heatmap.heatmap import HeatMap
from lib.pdf_report.pdf_report import PDF

from ultralytics import YOLO
import re

saopaulo_timezone = pytz.timezone('America/Sao_Paulo')


def plot_time_graph(df, x_column, y_columns, filename, title):
    # Converter a coluna de timestamp para o formato datetime
    df[x_column] = pd.to_datetime(df[x_column])

    # Extrair apenas as horas da coluna de timestamp
    df['Hora'] = df[x_column].dt.strftime('%H:%M:%S')

    # Cores pastel para os gráficos
    pastel_colors = ['#98FB98', '#FFB6C1']

    # Configurar o gráfico
    plt.figure(figsize=(5, 4))
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8) 
    plt.title(title, fontsize=12)

    # Plotar cada coluna do DataFrame no gráfico com cores pastel
    for i, column in enumerate(y_columns):
        plt.plot(df['Hora'], df[column], label=column, color=pastel_colors[i])

    legend_names = {'Saiu': 'Lefted', 'Entrou': 'Entered', 'Ocupação': 'Capacity'}

    # Adicionar legenda com os nomes personalizados
    plt.legend(labels=[legend_names.get(col, col) for col in y_columns])

    # Salvar o gráfico como um arquivo PNG
    plt.tight_layout()  # Para evitar cortes nos rótulos dos eixos
    plt.savefig(f"data/{filename}.png", format='png')

def main():
    desired_width = 840
    desired_height = 560
    saopaulo_timezone = pytz.timezone('America/Sao_Paulo')
    heatmap_paths = []
#excluir csv
    if os.path.exists('./data/dados.csv'):
        os.remove('./data/dados.csv')
    else:
        pass

#excluir csv2
    if os.path.exists('./data/heatmap.csv'):
        os.remove('./data/heatmap.csv')
    else:
        pass

    # Load model
    verbose = False
    model = YOLO('./resources/yolov8s.pt')

    # Load Tracker class
    tracker = Tracker()

    # Load HeatMap class
    heat_map = HeatMap()

    # Load initial frame
    initial_frame = cv2.imread("../monitor/frame.jpg")
    initial_frame = cv2.resize(initial_frame, (840, 560))
    frame_shape = np.shape(initial_frame)

    # Get object classes
    file = open('./resources/coco.names', 'r')
    data = file.read()
    class_list = data.split('\n')

    # Variables
    count = 0
    counter1 = []
    persondown = {}

    personup = {}
    counter2 = []

    cy1 = 400
    cy2 = 425

    offset = 6
    alpha = 0.1

    black_image = np.zeros(frame_shape, dtype=np.uint64)
    accumulated_image = np.zeros(
        (frame_shape[0], frame_shape[1]), dtype=np.uint64)

    output_heatmap_path = './data/heatmap_images'
    output_csv_path = './data/dados.csv'

    time_to_save_heatmap = 1800  # in seconds
    last_increment_time = datetime.datetime.now(saopaulo_timezone)
    last_csv_update_time = datetime.datetime.now(saopaulo_timezone)

    csv_update_counter = 0

    while True:
        frame = cv2.imread("../monitor/frame.jpg")


        count += 1
        if (count % 3 != 0):
            continue

        results = model.predict(frame, verbose=verbose)

        a = results[0].boxes.data
        try:
            px = pd.DataFrame(a.cpu().numpy()).astype("float")
        except Exception as e:
            print(f"Error: {e}")

        list = []

        for index, row in px.iterrows():
            d = int(row[5])
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            c = class_list[d]
            if 'person' in c:
                list.append([x1, y1, x2, y2])

        bbox_id = tracker.update(list)
        for bbox in bbox_id:

            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

            # for down going
            if (cy1 < (cy + offset) and (cy1 > cy - offset)):

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                persondown[id] = (cx, cy)

            if (id in persondown):
                if (cy2 < (cy+offset) and (cy2 > cy-offset)):
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                    if counter1.count(id) == 0:
                        counter1.append(id)

            # for up going
            if (cy2 < (cy+offset) and (cy2 > cy-offset)):

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                personup[id] = (cx, cy)

            if (id in personup):
                if (cy1 < (cy+offset) and (cy1 > cy-offset)):
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
                    cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 2)
                    if counter2.count(id) == 0:
                        counter2.append(id)

            # Create heat map
            accumulated_image[y3:y4, x3:x4] += 1

            current_time = datetime.datetime.now(saopaulo_timezone)
            if (current_time - last_increment_time).total_seconds() >= time_to_save_heatmap:
                HEATMAP = heat_map.plot_heatmap(
                    accumulated_image, black_image, alpha, color_map=cv2.COLORMAP_JET)
                

                #transforma a imagem para o tipo panda dataframe
                df = pd.DataFrame(accumulated_image)
                df.to_csv('data/heatmap.csv', index=False)

                heatmap_filename = f'data/heatmap_{current_time.strftime("%Y-%m-%d %H:%M:%S")}.png'
                #salva o heatmap em png
                cv2.imwrite(heatmap_filename, HEATMAP)

                heatmap_paths.append(heatmap_filename)

                last_increment_time = current_time

        cv2.line(frame, (0, cy1), (frame_shape[1], cy1), (0, 255, 0), 2)
        cv2.line(frame, (0, cy2), (frame_shape[1], cy2), (0, 255, 255), 2)

        downcount = len(counter1)
        upcount = len(counter2)
        inside = upcount - downcount

        if inside < 0:
            inside = 0

        cvzone.putTextRect(frame, f'Down: {downcount}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'Up: {upcount}', (50, 160), 2, 2)

        # Atualizar o arquivo CSV a cada hora
        current_time = datetime.datetime.now(saopaulo_timezone)

        if (current_time - last_csv_update_time).total_seconds() >= 1800:  # 3600 segundos = 1 hora
            with open(output_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)

                #verifica/insere a linha de parametros
                if file.tell() == 0:
                    print("linha zero")
                    writer.writerow(['Data', 'Horário', 'Saiu', 'Entrou', 'Ocupação'])
                
                
                writer.writerow([current_time.strftime(
                    '%Y-%m-%d'), current_time.strftime('%H:%M:%S'), downcount, upcount, inside])
            
            #att o timer
            last_csv_update_time = current_time

            # incrementa o contador de informções da planilha
            csv_update_counter += 1

            # Email
            if csv_update_counter == 6:
                df_dados = pd.read_csv("data/dados.csv")
                pdf_report = PDF(pdf_title='IA People Count Report')
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
                # pdf_report.set_font("Arial", size=10)
                # pdf_report.cell(200, 5, txt=f"Datetime: {df['Timestamp'].tail(1).values[0]}", ln=True)
                # pdf_report.cell(200, 5, txt=f'\t \t \t \t Detected Ages: {df.iloc[:,1:9].sum().sum()}', ln=True)
                # pdf_report.cell(200, 5, txt=f'\t \t \t \t Detected Emotions: {df.iloc[:,10:-1].sum().sum()}', ln=True)
                # pdf_report.cell(200, 5, txt=f"\t \t \t \t Detected People: {df['Total_Pessoas'].tail(1).values[0]}", ln=True)

                plot_time_graph(df_dados, 'Horário', ['Saiu','Entrou'], 'grafico_entrada_saida', "Exit and entry in relation to time")

                plot_time_graph(df_dados, 'Horário', ['Ocupação'], 'grafico_ocupacao', "Capacity in relation to time")

            
                pdf_report.image("data/grafico_entrada_saida.png")

                pdf_report.image("data/grafico_ocupacao.png")

                for filename in heatmap_paths:
                    # Dimensões da página
                    page_width = pdf_report.w
                    page_height = pdf_report.h

                    # Calcula a proporção da imagem
                    image_ratio = desired_width / desired_height

                    # Calcula a proporção da página
                    page_ratio = page_width / page_height

                    # Redimensiona a imagem mantendo as proporções
                    if image_ratio > page_ratio:
                        # A largura da imagem é limitante
                        new_width = page_width - 20 # Ajuste conforme necessário
                        new_height = new_width / image_ratio
                    else:
                        # A altura da imagem é limitante
                        new_height = page_height # Ajuste conforme necessário
                        new_width = new_height * image_ratio

                    # Calcula as coordenadas x e y para centralizar a imagem
                    x = (page_width - new_width) / 2
                    y = (page_height - new_height) / 2

                    # Adiciona a página e a imagem redimensionada centralizada
                    pdf_report.add_page()
                    # String com o nome do arquivo

                    # Padroniza a expressão regular para encontrar o timestamp
                    regex_pattern = r'heatmap_(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.png'

                    # Encontra o padrão na string
                    match = re.search(regex_pattern, filename)

                    # Se encontrou um padrão, extrai o timestamp
                    if match:
                        timestamp_string = match.group(1)
                        pdf_report.cell(190, 5, txt=f"Datetime: {timestamp_string}", ln=True)
                    else:
                        print("Padrão não encontrado na string.")
                    pdf_report.image(filename, x=x, y=y, w=new_width, h=new_height)

                heatmap_paths = []

                pdf_name = f"data/report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
                pdf_report.output(pdf_name)
                try:
                    # Email setup
                    sender_email = "pedropedrosa@lapisco.ifce.edu.br"
                    sender_password = "@lapisco2024"
                    receiver_email = "juliomacedochaves@gmail.com"

                    print("A1")
                    msg = MIMEMultipart()
                    msg['From'] = sender_email
                    msg['To'] = receiver_email
                    msg['Subject'] = "IA People Counting Report"
                    print("A2")
                    # Add text message to email body
                    body = "Olá,\n\nSegue relatório(s) referente(s) ao mapa de calor do local e/ou presença do público.\n\nAtenciosamente,\nEquipe do Lapisco/Instituto Iracema."
                    msg.attach(MIMEText(body, 'plain'))
                    print("A3")

                    # Attach CSV file
                    with open(pdf_name, 'rb') as pdf_file:
                        print("A4")
                        attachment = MIMEApplication(pdf_file.read(), _subtype="pdf")
                        attachment.add_header('Content-Disposition', 'attachment', filename=f'data/{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.pdf')
                        msg.attach(attachment)
                        print("A5")
                    
                    # Attach heatmap CSV file
                    with open('data/heatmap.csv', 'rb') as heatmap_file:
                        attachment_heatmap = MIMEApplication(heatmap_file.read(), _subtype="csv")
                        attachment_heatmap.add_header('Content-Disposition', 'attachment', filename=f'data/heatmap_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.csv')
                        msg.attach(attachment_heatmap)

                    # Connect to the SMTP server and send email
                    with smtplib.SMTP('smtp.gmail.com', 587, timeout=10) as server:
                        print("A6")
                        server.starttls()
                        print("A7")
                        server.login(sender_email, sender_password)
                        print("A8")
                        server.sendmail(sender_email, receiver_email, msg.as_string())
                        print("Email enviado")
                except:
                    os.rename("data/dados.csv", f"data/dados_{datetime.datetime.now(saopaulo_timezone).strftime('%Y-%m-%d_%H-%M-%S')}.csv")
                    os.rename("data/heatmap.csv", f"data/heatmap_{datetime.datetime.now(saopaulo_timezone).strftime('%Y-%m-%d_%H-%M-%S')}.csv")
                    print("Falha ao enviar Email")
                    #lógica para salvar csv q não foi enviado

                # Reset CSV 
                with open(output_csv_path, mode='w', newline='') as file:
                    #apagar  csv
                    os.remove(output_csv_path)                       
                    csv_update_counter = 0 #zera o contador 
            

        cv2.imwrite("./frame_temp.jpg", frame)
        os.system("mv frame_temp.jpg frame.jpg")

    


if __name__ == '__main__':
    main()
