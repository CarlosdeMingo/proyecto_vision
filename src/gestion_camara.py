"""Uso de la camara calibrada, sin editar calibracion.py."""
from pathlib import Path
import os
import cv2
import numpy as np
import calibracion

BASE = Path(__file__).resolve().parent.parent
RUTA_CALIBRACION = BASE / 'calibration' / 'camera_params.npz'
CAMERA_INDEX = calibracion.CAMERA_INDEX
ANCHO, ALTO = 640, 480  # Solo si no consta la resolucion original.


def cargar_parametros():
    ruta = RUTA_CALIBRACION
    if not ruta.is_file():
        ruta = BASE/'camera_params.npz'
    if not ruta.is_file():
        ruta = Path(__file__).resolve().parent/'camera_params.npz'
    if not ruta.is_file():
        raise RuntimeError('No hay parametros. Ejecuta calibracion.py primero.')
    with np.load(ruta) as datos:
        K = datos['camera_matrix'].astype(np.float64)
        dist = datos['dist_coeffs'].astype(np.float64)
        resolucion = tuple(map(int, datos['image_size'])) if 'image_size' in datos else None
    if K.shape != (3,3) or not np.isfinite(K).all() or not np.isfinite(dist).all() or K[0,0]<=0 or K[1,1]<=0:
        raise RuntimeError('Parametros de calibracion invalidos')
    fotos = sorted((BASE / calibracion.IMAGE_FOLDER).glob('*.jpg')) if resolucion is None else []
    for foto in fotos:
        img = cv2.imread(str(foto))
        if img is not None:
            resolucion = (img.shape[1],img.shape[0])
            break
    if resolucion is None:
        resolucion = (ANCHO,ALTO)
        print(f'AVISO: resolucion original desconocida. Se supone {ANCHO}x{ALTO}. Confirma estos valores.')
    if len(resolucion) != 2 or min(resolucion) <= 0:
        raise ValueError("Resolucion de calibracion invalida")
    return K,dist,resolucion,ruta


def abrir_camara(resolucion):
    backend = cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX,backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'No se puede abrir la camara {CAMERA_INDEX}. Comprueba CAMERA_INDEX.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,resolucion[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,resolucion[1])
    ok,frame = cap.read()
    if not ok or (frame.shape[1],frame.shape[0])!=resolucion:
        real = (frame.shape[1],frame.shape[0]) if ok else 'sin imagen'
        cap.release()
        raise RuntimeError(f'Resolucion real {real}; se necesita {resolucion} para usar la calibracion. Ajusta la camara o recalibra.')
    return cap


def crear_mapas(K, dist, resolucion):
    """Precalcula la correccion de distorsion una vez por sesion."""
    return cv2.initUndistortRectifyMap(K, dist, None, K, resolucion, cv2.CV_16SC2)


def leer_frame(cap, mapas):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('No se recibe imagen de la camara')
    if frame.shape[:2] != mapas[0].shape[:2]:
        raise RuntimeError('La resolucion de la camara ha cambiado')
    # Equivalente a undistort; precalcula las tablas para todos los frames.
    return cv2.remap(frame,mapas[0],mapas[1],cv2.INTER_LINEAR)


