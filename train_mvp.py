
# train_mvp.py  (parcheado)
# - Soporta múltiples mirrors de dataset.
# - Permite ruta local (carpetas Train/Test) o un ZIP local.
# - Maneja HTTP 403 con mensaje claro.
import os, zipfile, io, json, sys, pathlib
from urllib.request import urlopen, Request, URLError, HTTPError
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Lista de mirrors conocidos (reemplaza si alguno cae)
MIRRORS = [
    # Kaggle requiere autenticación; se deja documentado pero no se usa directo.
    # "https://www.kaggle.com/datasets/grassknoted/asl-alphabet/download?datasetVersionNumber=2",
    # Zenodo (24 letras, con Train/Test). Si cambia, descargar manualmente desde la página.
    # Coloca aquí un enlace directo al ZIP si dispones de él.
]

def try_download(url, target_dir):
    print(f"Descargando dataset desde: {url}")
    req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urlopen(req) as resp:
        data = resp.read()
    return data

def download_and_unzip_any(mirrors, target_dir):
    target_dir = pathlib.Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in mirrors:
        try:
            data = try_download(url, target_dir)
            print("Descomprimiendo...")
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(target_dir)
            print("Listo.")
            return True
        except HTTPError as e:
            last_err = e
            print(f"[WARN] Descarga falló ({e.code}): {url}")
        except URLError as e:
            last_err = e
            print(f"[WARN] Descarga falló (URLError): {url}")
    if last_err:
        print(f"[ERROR] No fue posible descargar automáticamente. Motivo: {last_err}")
    return False

def prepare_local_data(local_path: pathlib.Path):
    # Acepta carpeta con Train/Test o un archivo ZIP
    if local_path.is_dir():
        train = local_path / "Train"
        test  = local_path / "Test"
        if train.exists() and test.exists():
            return True
    elif local_path.is_file() and local_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(local_path.parent)
        return True
    return False

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(64,64,1)),
        layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    out_dir = pathlib.Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    # Variables de entorno opcionales
    env_zip = os.getenv("DATA_ZIP")   # ruta a ZIP local
    env_dir = os.getenv("DATA_DIR")   # ruta a carpeta con Train/Test

    data_root = pathlib.Path("data")
    dataset_dir = data_root / "sign-language-img"

    if not dataset_dir.exists():
        ok = False
        # 1) ¿El usuario pasó DATA_DIR / DATA_ZIP?
        if env_dir:
            ok = prepare_local_data(pathlib.Path(env_dir))
        elif env_zip:
            ok = prepare_local_data(pathlib.Path(env_zip))
        # 2) Intentar mirrors
        if not ok and MIRRORS:
            ok = download_and_unzip_any(MIRRORS, data_root)

        if not ok:
            msg = (
                "###########################################\n"
                "NO SE ENCONTRÓ/ DESCARGÓ EL DATASET.\n"
                "Soluciones:\n"
                "  A) Descarga manualmente un dataset con estructura:\n"
                "     data/sign-language-img/Train/<clase>/*.jpg\n"
                "     data/sign-language-img/Test/<clase>/*.jpg\n\n"
                "     Ejemplos públicos:\n"
                '       - Kaggle: "ASL Alphabet" (87k imágenes). Requiere login.\n'
                "       - Zenodo/Mendeley (variantes con 24-29 clases).\n\n"
                "  B) O define variables de entorno antes de ejecutar:\n"
                "       set DATA_DIR=C:\\ruta\\a\\sign-language-img    (Windows)\n"
                "       export DATA_DIR=/ruta/a/sign-language-img      (macOS/Linux)\n"
                "     (También puedes usar DATA_ZIP apuntando a un .zip)\n\n"
                "Luego vuelve a ejecutar:  python train_mvp.py\n"
                "###########################################\n"
            )
            print(msg)
            sys.exit(1)

    train_dir = dataset_dir / "Train"
    test_dir  = dataset_dir / "Test"

    if not train_dir.exists() or not test_dir.exists():
        print("[ERROR] La carpeta no tiene subcarpetas Train/Test. Verifica el dataset.")
        sys.exit(1)

    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        str(train_dir), target_size=(64,64), color_mode="grayscale",
        class_mode="categorical", batch_size=64, shuffle=True)
    val_gen = datagen.flow_from_directory(
        str(test_dir), target_size=(64,64), color_mode="grayscale",
        class_mode="categorical", batch_size=64, shuffle=False)

    num_classes = len(train_gen.class_indices)
    with open(out_dir / "labels.json","w") as f:
        json.dump({v:k for k,v in train_gen.class_indices.items()}, f, indent=2)

    model = build_model(num_classes)
    model.fit(train_gen, epochs=5, validation_data=val_gen)

    model.save(out_dir / "model.h5")
    print("\nModelo guardado en", out_dir / "model.h5")
    print("Etiquetas guardadas en", out_dir / "labels.json")

    loss, acc = model.evaluate(val_gen)
    print(f"Accuracy de validación: {acc:.3f}")

if __name__ == "__main__":
    # Si te aparece un warning de OneDNN y quieres desactivarlo:
    #   Windows: set TF_ENABLE_ONEDNN_OPTS=0
    #   macOS/Linux: export TF_ENABLE_ONEDNN_OPTS=0
    main()
