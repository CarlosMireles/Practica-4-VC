# Detección y Reconocimiento de Matrículas en Video con YOLOv8/v11 y EasyOCR

Este proyecto utiliza modelos de detección y reconocimiento óptico de caracteres (OCR) para identificar vehículos y matrículas en un archivo de video. Se basa en el modelo YOLOv11 para detección de objetos y EasyOCR para el reconocimiento de matrículas, permitiendo almacenar cada detección en un archivo CSV con los detalles de los objetos.

## Tabla de Contenidos

- [Preparación del entorno](#preparación-del-entorno)
- [Entrenamiento de un modelo para detectar matrículas de coches](#entrenamiento-de-un-modelo-para-detectar-matrículas-de-coches)
- [Proceso de detección y lectura de Matrículas](#proceso-de-detección-y-lectura-de-matrículas)

## Preparación del entorno

1. Entorno
```
conda create --name VC_P4 python=3.9.5
conda activate VC_P4
```
2. Librerías a instalar:
```
pip install lapx
```
```
pip install  ultralytics
```
```
pip install easyocr
```
Sin embargo, surge un problema con OpenCV, ya que algunas funciones de visualización, como `imshow`, dejan de estar disponibles. Al analizar la incompatibilidad entre la instalación de YOLO y easyOCR, es posible que OpenCV no se instale completamente en nuestro entorno de trabajo. Hemos logrado resolverlo utilizando:
```
pip uninstall opencv-python opencv-python-headless
pip install opencv-python --upgrade
```
3. Habilitar GPU:

Para utilizar la potencia de cálculo de la GPU en el entorno, se requiere tener CUDA instalado y seguir los pasos para instalar los paquetes compatibles necesarios. Esta [guía](https://pytorch.org/get-started/locally/) proporciona los comandos listos para ejecutar. A continuación, se muestra un ejemplo de configuración con CUDA v11.6.
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

## Entrenamiento de un modelo para detectar matrículas de coches

### Entrenamiento del Modelo
   - Este código utiliza el modelo YOLO v8 para entrenarlo en la detección de matrículas. 
   
   - Primero, carga el modelo preentrenado YOLOv8n. 
   - Luego, entrena el modelo con un conjunto de datos específico de matrículas, usando 50 épocas, un tamaño de imagen de 500 píxeles y un lote de 15 imágenes en la GPU (device='0'). 
   - Finalmente, realiza una validación del modelo entrenado.

     ```python
        model = YOLO('yolov8n.pt')  
        model.train(
            data='license_plates.yaml',
            epochs=50, 
            imgsz=500,
            batch=15,
            device='0' 
        )
        model.val()
     ```

### Imagenes del Dataset

<img src="images\Cars94_annotated.png" width="300" height="300" style="display:inline-block;" />
<img src="images\Cars389_annotated.png" width="300" height="300" style="display:inline-block;" />

### Gráficas de Entrenamiento y Validación

![Métricas Dataset](model\results.png)

La imagen presenta una serie de gráficas que muestran las métricas y pérdidas clave en el proceso de entrenamiento y validación del modelo.

- **train/box_loss, train/cls_loss, train/dfl_loss**: Estas gráficas representan las pérdidas en diferentes aspectos durante el entrenamiento. Todas muestran una tendencia a la baja, indicando que el modelo mejora en la detección y clasificación de las características.

- **metrics/precision(B) y metrics/recall(B)**: Las métricas de precisión y recall incrementan de forma sostenida, lo que indica que el modelo es capaz de hacer predicciones acertadas y de capturar la mayoría de las instancias de interés.

- **val/box_loss, val/cls_loss, val/dfl_loss**: Las pérdidas en el conjunto de validación también muestran una reducción general, aunque con algo de variabilidad, lo que sugiere ligeras fluctuaciones en la validación.

- **metrics/mAP50(B) y metrics/mAP50-95(B)**: Estas métricas de mAP (mean Average Precision) muestran una alta precisión en diferentes niveles de IoU (Intersection over Union), indicando un buen desempeño general del modelo en la detección de objetos.


## Proceso de detección y lectura de Matrículas

Este proyecto desarrolla un sistema para la detección y lectura de matrículas en video utilizando modelos avanzados de detección de objetos y reconocimiento óptico de caracteres (OCR). El proceso se lleva a cabo en varias etapas:

1. **Detección de Objetos**: Se emplea el modelo **YOLOv11** para identificar objetos en cada fotograma del video, incluyendo personas y vehículos. Esto se logra extrayendo las coordenadas de los objetos detectados.

2. **Detección de Matrículas**: Tras identificar los vehículos, se recorta la región de interés y se aplica un modelo especializado de YOLO para localizar las matrículas en esos vehículos.

3. **Lectura de Matrículas**: Las áreas recortadas de las matrículas se procesan utilizando **EasyOCR**, que reconoce y extrae el texto. Los resultados obtenidos se almacenan en un archivo CSV para su posterior análisis y referencia.

Este enfoque permite identificar, localizar y leer matrículas de manera eficiente en tiempo real.

![Métricas Dataset](images\imagen_prueba_1.png)

## Detalles extra

El vídeo con las detecciones se puede encontrar en este *[link](video/video1final.mp4)*.

El archivo CSV con la información de las detecciones puede ser encontrado en este *[link](output.csv)*.

---
Saúl Antonio Cruz Pérez  
Carlos Mireles Rodríguez

Universidad de las Palmas de Gran Canaria  
Escuela de Ingeniería en Informática  
Grado de Ingeniería Informática  
Visión por Computador  
Curso 2024/2025
