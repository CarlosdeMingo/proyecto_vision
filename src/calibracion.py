#LIBRERIAS

import cv2
from typing import List
import numpy as np
import imageio.v3 as imageio
import copy
import os
import glob

#=====================================================================================

#FUNCIONES
def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]


def get_chessboard_points(chessboard_shape, dx, dy):
    cols, rows = chessboard_shape  # cols = tuple[0], rows = tuple[1]

    points = []
    for i in range(rows):
        for j in range(cols):
            x = i * dx
            y = j * dy
            z = 0.0
            points.append([x, y, z])
            
    return np.array(points, dtype=np.float32)

def show_image(img):
    cv2.imshow("nombre",img)
    cv2.waitKey()
    cv2.destroyAllWindows()

#=====================================================================================
def calibracion():
    # 1. Carga de imágenes
    imgs_path = [f"../calibration_chess_images/{i:03d}.jpg" for i in range(19)]
    imgs = load_images(imgs_path)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    # 2. Detección y refinamiento de esquinas asegurando sincronización
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    object_points = []
    valid_corners = []
    corners_refined_all = []

    for img_gray in imgs_gray:
        ret, corners = cv2.findChessboardCorners(img_gray, (8, 6))
        if ret:
            # Refinamos las esquinas
            refined = cv2.cornerSubPix(img_gray, corners, (4, 4), (-1, -1), criteria)
            
            # Añadimos los puntos 3D y 2D sincronizados
            pts_3d = get_chessboard_points((8, 6), 30 * 10**-3, 30 * 10**-3)
            object_points.append(np.array(pts_3d, dtype=np.float32))
            valid_corners.append(np.array(refined, dtype=np.float32))
            corners_refined_all.append(refined)
        else:
            corners_refined_all.append(None)

    # 3. Calibración de la cámara derecha con cv2.calibrateCamera()
    image_size = imgs_gray[0].shape[::-1]
    rms_r, intrinsics_r, dist_coeffs_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        object_points, valid_corners, image_size, None, None
    )

    # Cálculo de extrínsecos para la cámara derecha
    extrinsics_r = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs_r, tvecs_r))

    # 4. Impresión por pantalla de los resultados solicitados
    print("=== RESULTADOS CALIBRACIÓN CÁMARA DERECHA ===")
    print("Error RMS de reproyección:\n", rms_r)
    print("\nMatriz de Intrínsecos:\n", intrinsics_r)
    print("\nCoeficientes de Distorsión:\n", dist_coeffs_r)

    # 5. Dibujar y mostrar al menos 2 imágenes con la detección de esquinas
    right_imgs_copy = copy.deepcopy(imgs)
    valid_idx = 0
    for i, ref in enumerate(corners_refined_all):
        if ref is not None:
            cv2.drawChessboardCorners(right_imgs_copy[i], (8, 6), ref, True)

    print("\nMostrando las 2 primeras imágenes con esquinas detectadas...")
    shown = 0
    for i, ref in enumerate(corners_refined_all):
        if ref is not None and shown < 2:
            show_image(right_imgs_copy[i])
            shown += 1