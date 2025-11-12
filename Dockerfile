# Dockerfile

# 1. Usa una imagen base oficial de Python. La versión 'slim' es más ligera.
FROM python:3.11-slim

# 2. Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app

# 3. Copia primero el archivo de requerimientos y luego instala las dependencias.
#    Esto aprovecha el sistema de caché de Docker para acelerar futuras construcciones.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia el resto del código de tu aplicación al directorio de trabajo.
COPY . .

# 5. Expone el puerto que la aplicación usará. Cloud Run espera el 8080 por defecto.
EXPOSE 8080

# 6. El comando para iniciar la aplicación cuando el contenedor se ejecute.
#    --host 0.0.0.0 es crucial para que sea accesible desde fuera del contenedor.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]