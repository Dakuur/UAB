import os

def contar_lineas_python(directorio):
    total_lineas = 0
    for root, _, files in os.walk(directorio):
        for file in files:
            if file.endswith('.py'):
                ruta_archivo = os.path.join(root, file)
                with open(ruta_archivo, 'r', encoding='utf-8') as f:
                    lineas = f.readlines()
                    total_lineas += len(lineas)
    return total_lineas

if __name__ == "__main__":
    directorio = "."
    lineas = contar_lineas_python(directorio)
    print(f"Total de líneas en archivos .py: {lineas}")
