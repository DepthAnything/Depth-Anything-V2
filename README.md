🚀 Instrucciones para Ejecutar el Proyecto
1️⃣ Crear un entorno virtual (recomendado)
bash

python -m venv .venv

Activa el entorno virtual:

En Windows:
.venv\Scripts\activate

2️⃣ Instalar dependencias

pip install -r requirements.txt

3️⃣ Levantar el servidor backend (API) con Uvicorn

uvicorn api:app --host 0.0.0.0 --port 8000 --reload

🔹 Cambia api:app por el módulo y objeto correctos según tu estructura.

4️⃣ Iniciar la aplicación Streamlit

streamlit run dashboard.py --server.port 8501