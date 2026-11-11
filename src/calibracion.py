
# ============================================================
# SMART INVENTORY - CALIBRACION DE CAMARA CON IRIUN
# Tablero: 7 x 10 cuadrados = 6 x 9 esquinas interiores
# ============================================================

# LIBRERIAS

import cv2
from typing import List
import numpy as np
import copy
import os
import glob

# ============================================================
# CONFIGURACION
# ============================================================

CHESSBOARD_SHAPE = (9, 6)

# IMPORTANTE: medir el lado REAL de un cuadrado
# Ejemplo: 30 mm = 0.030 metros
SQUARE_SIZE = 0.030

NUM_IMAGES = 19

CAMERA_INDEX = 1  # Cambiar si Iriun utiliza otro indice

IMAGE_FOLDER = "calibration_chess_images"


# ============================================================
# FUNCIONES
# ============================================================

def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]


def get_chessboard_points(chessboard_shape, dx, dy):

    cols, rows = chessboard_shape

    points = []

    for i in range(rows):
        for j in range(cols):

            x = j * dx
            y = i * dy
            z = 0.0

            points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def show_image(img):

    cv2.imshow("Calibracion", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# 1. CAPTURA DE FOTOGRAFIAS CON IRIUN
# ============================================================

def capturar_imagenes():

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("ERROR: No se puede abrir Iriun")
        return False

    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    contador = 0

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01
    )

    while contador < NUM_IMAGES:

        ret, frame = cap.read()

        if not ret:
            print("Error leyendo la camara")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar tablero de 6 x 9 esquinas
        found, corners = cv2.findChessboardCorners(
            gray,
            CHESSBOARD_SHAPE,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        display = frame.copy()

        if found:

            # Refinar esquinas
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (4, 4),
                (-1, -1),
                criteria
            )

            cv2.drawChessboardCorners(
                display,
                CHESSBOARD_SHAPE,
                corners_refined,
                found
            )

            texto = "TABLERO DETECTADO - ESPACIO PARA GUARDAR"

            color = (0, 255, 0)

        else:

            texto = "BUSCANDO TABLERO..."

            color = (0, 0, 255)

        cv2.putText(
            display,
            texto,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        cv2.putText(
            display,
            f"Imagenes: {contador}/{NUM_IMAGES}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.imshow("Captura de calibracion Iriun", display)

        key = cv2.waitKey(1) & 0xFF

        # Guardar imagen pulsando ESPACIO
        if key == ord(' ') and found:

            filename = os.path.join(
                IMAGE_FOLDER,
                f"{contador:03d}.jpg"
            )

            cv2.imwrite(filename, frame)

            contador += 1

            print("Imagen guardada:", filename)

        # Salir con ESC
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Capturadas {contador} imagenes")

    return contador == NUM_IMAGES


# ============================================================
# 2. CALIBRACION DE LA CAMARA
# ============================================================

def calibracion():

    # Cargar imagenes
    imgs_path = sorted(
        glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg"))
    )

    imgs = load_images(imgs_path)

    if len(imgs) < 10:
        raise ValueError(
            "Se necesitan al menos 10 imagenes validas"
        )

    if any(img is None for img in imgs):
        raise ValueError("Error cargando imagenes")

    # Comprobar resolucion
    image_size = imgs[0].shape[1], imgs[0].shape[0]

    if any(
        (img.shape[1], img.shape[0]) != image_size
        for img in imgs
    ):
        raise ValueError(
            "Todas las imagenes deben tener la misma resolucion"
        )

    # Criterio de refinamiento
    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01
    )

    object_points = []
    valid_corners = []
    corners_refined_all = []

    # Puntos 3D del tablero
    pts_3d = get_chessboard_points(
        CHESSBOARD_SHAPE,
        SQUARE_SIZE,
        SQUARE_SIZE
    )

    # Detectar esquinas en cada fotografia
    for img in imgs:

        img_gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        ret, corners = cv2.findChessboardCorners(
            img_gray,
            CHESSBOARD_SHAPE,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:

            refined = cv2.cornerSubPix(
                img_gray,
                corners,
                (4, 4),
                (-1, -1),
                criteria
            )

            object_points.append(
                pts_3d.copy()
            )

            valid_corners.append(
                refined
            )

            corners_refined_all.append(
                refined
            )

        else:

            corners_refined_all.append(None)

    print(
        f"Fotografias validas: {len(object_points)}"
    )

    if len(object_points) < 10:
        raise ValueError(
            "No hay suficientes fotografias validas"
        )

    # Calibrar camara
    rms, intrinsics, dist_coeffs, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            valid_corners,
            image_size,
            None,
            None
        )
    )

    # Calcular matrices extrinsecas
    extrinsics = []

    for rvec, tvec in zip(rvecs, tvecs):

        R, _ = cv2.Rodrigues(rvec)

        extrinsic = np.hstack((R, tvec))

        extrinsics.append(extrinsic)

    # Mostrar resultados
    print("\n================================")
    print("RESULTADOS DE CALIBRACION")
    print("================================")

    print("\nError RMS:")
    print(rms)

    print("\nMatriz intrinseca:")
    print(intrinsics)

    print("\nCoeficientes de distorsion:")
    print(dist_coeffs)

    # Guardar parametros
    os.makedirs("calibration", exist_ok=True)

    np.savez(
        "calibration/camera_params.npz",
        camera_matrix=intrinsics,
        dist_coeffs=dist_coeffs,
        rms=rms
    )

    print("\nParametros guardados correctamente")

    # Mostrar dos imagenes con esquinas
    right_imgs_copy = copy.deepcopy(imgs)

    shown = 0

    for i, corners in enumerate(corners_refined_all):

        if corners is not None:

            cv2.drawChessboardCorners(
                right_imgs_copy[i],
                CHESSBOARD_SHAPE,
                corners,
                True
            )

            if shown < 2:

                show_image(right_imgs_copy[i])

                shown += 1

    return rms, intrinsics, dist_coeffs, extrinsics


# ============================================================
# 3. EJECUCION
# ============================================================

if __name__ == "__main__":

    print("SMART INVENTORY - CALIBRACION")

    if capturar_imagenes():

        calibracion()

    else:

        print(
            "Captura incompleta. No se ejecuta la calibracion."
        )
