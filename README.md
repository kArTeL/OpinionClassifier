# OpinionClassifier
Opinion classifier, MSc. Natural language Processing (Universidad de Costa Rica)



## Como correr?
El proyecto esta usando la versión 2.7 de python.
El proyecto ya tiene el modelo creado utilizado para mostrar resultados, estan en archivos de extension `.pickle` es necesario unzipear el archivo `Pickles.zip` para correr el programa.

Utilizá una serie de dependencias que se puede instalar con pip, si no tiene pip instalado puede usar el siguiente comando para instalarlo: `python get-pip.py`.
Una vez instalado `pip` podemos instalar las dependencias del proyecto corriendo los siguientes comandos.

### Instalando dependencias
* `pip install ntlk`
* `pip install numpy`
* `pip install scipy`
* `pip install pandas`
* `pip install scikit-learn`

## Entrenamiento
El proyecto ya tiene el modelo entrenado pero se puede entrenar de la siguiente manera:
`python main.py` esto dejara dentro del archivo `classifier.pickle` el modelo probabilistico.

## Testear un archivo
El programa permite recibir un archivo para analizar con el formato
1. ID_del_comentario
2. seguido del texto del comentario.

El archivo se tendrá que llamar `Test.csv` y para utilizar probarlo se debe de correr el siguiente comando:
`python main_test.py` y el resultado final quedará en un archivo llamado `result_file.csv`.

## Evaluador del clasificador
  El clasificador permite ser evaluado si se cuenta con un archivo previamente clasificado por OPINION o no OPINION que tenga la misma forma que el archivo que se uso para el entrenamiento.
Para evaluar el clasificador el archivo con el que cual se desea evaluar se debe de llamar `Test.scv`, y luego correr el siguiente comando: `python report.py`.

Al final se mostrará el reporte que muestra la precisión, recall y F1Score.
