class Hash:
    def __init__(self):
        pass
    
def llegir_arxiu(nom_fitxer):
    import os
    
    dir = os.getcwd()
    
    print("Directorio actual:", dir)
    print("Archivos en el directorio:", os.listdir(dir))
    
    #fitxer = "main.py"
    
    with open(nom_fitxer, "r") as arxiu:
        contenido = arxiu.read()
        print(f"\nContenido de {nom_fitxer}:\n")
        print(contenido)

def get_info_sys():

    import platform
    import os
    import psutil
    import socket
    
    # Información general del sistema
    print("Sistema operativo:", platform.system())
    print("Nombre del nodo:", platform.node())
    print("Versión del sistema operativo:", platform.version())
    print("Arquitectura:", platform.machine())
    print("Procesador:", platform.processor())
    print("Plataforma:", platform.platform())
    print("Nombre de la red:", socket.gethostname())
    
    # Información de la CPU
    print("\nInformación de la CPU:")
    print("Número de núcleos físicos:", psutil.cpu_count(logical=False))
    print("Número de núcleos lógicos:", psutil.cpu_count(logical=True))
    print("Frecuencia máxima de la CPU:", psutil.cpu_freq().max, "MHz")
    print("Frecuencia actual de la CPU:", psutil.cpu_freq().current, "MHz")
    print("Uso de CPU por núcleo:", psutil.cpu_percent(percpu=True), "%")
    print("Uso total de CPU:", psutil.cpu_percent(), "%")
    
    # Información de la memoria
    print("\nInformación de la memoria:")
    mem = psutil.virtual_memory()
    print("Total de memoria:", mem.total / (1024 ** 3), "GB")
    print("Memoria disponible:", mem.available / (1024 ** 3), "GB")
    print("Memoria usada:", mem.used / (1024 ** 3), "GB")
    print("Porcentaje de memoria usada:", mem.percent, "%")
    
    # Información de la memoria swap
    swap = psutil.swap_memory()
    print("\nInformación de la memoria swap:")
    print("Total de memoria swap:", swap.total / (1024 ** 3), "GB")
    print("Memoria swap usada:", swap.used / (1024 ** 3), "GB")
    print("Memoria swap libre:", swap.free / (1024 ** 3), "GB")
    print("Porcentaje de memoria swap usada:", swap.percent, "%")
    
    # Información de los discos
    print("\nInformación de los discos:")
    particiones = psutil.disk_partitions()
    for particion in particiones:
        print(f"\nPartición: {particion.device}")
        uso = psutil.disk_usage(particion.mountpoint)
        print("Punto de montaje:", particion.mountpoint)
        print("Sistema de archivos:", particion.fstype)
        print("Tamaño total:", uso.total / (1024 ** 3), "GB")
        print("Usado:", uso.used / (1024 ** 3), "GB")
        print("Libre:", uso.free / (1024 ** 3), "GB")
        print("Porcentaje usado:", uso.percent, "%")
    
    # Información de la red
    print("\nInformación de la red:")
    interfaces = psutil.net_if_addrs()
    for interfaz, direcciones in interfaces.items():
        print(f"\nInterfaz: {interfaz}")
        for direccion in direcciones:
            if direccion.family == socket.AF_INET:
                print("Dirección IP:", direccion.address)
                print("Máscara de subred:", direccion.netmask)
            elif direccion.family == socket.AF_PACKET:
                print("Dirección MAC:", direccion.address)
    
def instalar_paquete(paquete):
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])

def ejecutar_comando_sudo():
    import subprocess
    import os
    
    try:
        # Ejecuta 'sudo echo "Hola, mundo!"'
        resultado = subprocess.run(["touch", "prova.txt"], check=True, text=True, capture_output=True)
        print("Archivos en el directorio:", os.listdir(os.getcwd()))
        if "touch.txt" in os.listdir(os.getcwd()):
            print("Archivo creado")
            resultado = subprocess.run(["rm", "prova.txt"], check=True, text=True, capture_output=True)
        if "touch.txt" not in os.listdir(os.getcwd()):
            print("Archivo eliminado")
        #print("Salida del comando:\n", resultado.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e}")
        print("Salida de error:\n", e.stderr)

#instalar_paquete("pyjoke")

#llegir_arxiu("common_script.sh")

#ejecutar_comando_sudo()

#get_info_sys()

raise # per alguna raó, necessari per a que s'imprimeixi la sortida quan es valida"""
