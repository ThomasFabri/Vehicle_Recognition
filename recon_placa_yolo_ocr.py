import cv2
from ultralytics import YOLO

# Carregar o modelo YOLOv8 pré-treinado
modelo = YOLO('yolov8n.pt')  # Usando o modelo YOLOv8

# Configurar classes que queremos identificar
classes_desejadas = ['car', 'motorcycle', 'bus', 'truck']

# Captura de vídeo
cap = cv2.VideoCapture(0)  # ou o caminho para o arquivo de vídeo

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a detecção com o YOLO
    resultados = modelo(frame)

    # Inicializar a variável para armazenar o nome do veículo
    nome_veiculo = "Nenhum veiculo detectado"

    # Iterar pelas detecções
    for det in resultados[0].boxes:
        conf = det.conf[0]  # Confiança da detecção
        class_id = int(det.cls[0])  # ID da classe detectada
        nome_classe = modelo.names[class_id]  # Nome da classe

        # Verificar se é uma das classes desejadas
        if nome_classe in classes_desejadas and conf > 0.5:  # Threshold de confiança
            nome_veiculo = nome_classe.capitalize()  # Armazenar o nome do veículo detectado
            print(f"Detectado um(a) {nome_veiculo} com {conf * 100:.2f}% de confiança.")

            # Desenhar a caixa de detecção
            x1, y1, x2, y2 = det.xywh[0]  # Coordenadas da caixa delimitadora
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Desenhar a caixa

    # Exibir o nome do veículo na tela (GUI)
    cv2.putText(frame, nome_veiculo, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar a imagem com as detecções e o nome do veículo
    cv2.imshow("Deteccao de Veiculos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
