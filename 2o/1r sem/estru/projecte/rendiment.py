import time
import psutil
import subprocess

def ejecutar_programa():
    # Reemplaza 'tu_programa.py' con el nombre de tu programa o comando a ejecutar
    proceso = subprocess.Popen(['python', 'main.py'])
    proceso.wait()

def medir_rendimiento():
    # Medir el tiempo de inicio
    tiempo_inicio = time.time()

    # Ejecutar el programa
    ejecutar_programa()

    # Medir el tiempo de finalización
    tiempo_fin = time.time()

    # Calcular el tiempo total de ejecución
    tiempo_total = tiempo_fin - tiempo_inicio

    # Obtener el uso de memoria
    uso_memoria = psutil.Process().memory_info().rss / 1024  # en kilobytes

    # Mostrar los resultados
    print(f'\nTiempo de ejecución: {tiempo_total:.4f} segundos')
    print(f'Uso de memoria: {uso_memoria:.2f} KB')

if __name__ == "__main__":
    medir_rendimiento()
