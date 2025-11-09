
# ASL Classifier — MVP

MVP funcional basado en el dataset de imágenes de lenguaje de señas (A–Z). Incluye:
- `train_mvp.py`: descarga el dataset, entrena un modelo ligero (5 epochs) y guarda `artifacts/model.h5` y `artifacts/labels.json`.
- `app.py`: aplicación Streamlit para clasificar imágenes cargadas por el usuario.
- `requirements.txt`: dependencias mínimas.

## Dataset utilizado: https://zenodo.org/records/14635573

## Cómo correr (local)
1) Crear venv y activar (opcional):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
2) Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3) Entrenar modelo (descarga automática de dataset ~ decenas de MB):
   ```bash
   python train_mvp.py
   ```
4) Ejecutar la app:
   ```bash
   streamlit run app.py
   ```
5) Abrir el link local que muestra Streamlit.

## Notas
- Si la URL del dataset deja de estar disponible, reemplace `DATA_URL` en `train_mvp.py` por un espejo alternativo del mismo dataset (estructura Train/Test por carpetas).
- Para acelerar más el entrenamiento, reduzca `epochs` o use un subconjunto del set.
- Este MVP procesa la imagen a 64×64 en escala de grises para rapidez.
