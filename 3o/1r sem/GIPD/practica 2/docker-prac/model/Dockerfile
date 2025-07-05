# Dockerfile para el modelo de entrenamiento
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt y el script al contenedor
COPY requirements-train.txt /app/
COPY main-train.py /app/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements-train.txt

# Comando por defecto
CMD ["python", "main-train.py"]

