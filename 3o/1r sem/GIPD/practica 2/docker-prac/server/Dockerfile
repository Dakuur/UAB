# Dockerfile para el servidor de modelo
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt y los archivos de la aplicación
COPY requirements-server.txt /app/
COPY main-server.py /app/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements-server.txt

# Comando por defecto
CMD ["flask", "--app", "main-server.py", "run", "--host=0.0.0.0"]

