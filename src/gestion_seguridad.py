"""Integracion y registro opcional, sin modificar sistema_seguridad.py."""
import importlib
import time
from pathlib import Path
import cv2
from gestion_camara import leer_frame
from interfaz import dibujar_fps


def comprobar_acceso(cap, mapas, ruta_calibracion):
    # La version de GitHub lee contrasena.txt en el momento de importar.
    # Si falta, se explica el requisito original en lugar de crear una clave
    # por defecto o modificar el sistema de seguridad.
    try:
        seguridad = importlib.import_module('sistema_seguridad')
    except FileNotFoundError as error:
        if error.filename and Path(error.filename).name == 'contrasena.txt':
            raise RuntimeError('Falta src/contrasena.txt. Crealo con cuatro figuras, una por linea, como indica el README original.') from error
        raise

    # Cambia solo el valor en memoria: el archivo original queda intacto.
    seguridad.CALIBRACION = str(ruta_calibracion)
    if not configurar_seguridad(cap, mapas, seguridad):
        return False
    print('Introduce la contrasena mediante las cuatro figuras, retirando cada tarjeta.')
    # Se pasan frames sin corregir: la funcion original ya aplica undistort.
    return seguridad.sistema_seguridad(cap)


def configurar_seguridad(cap, mapas, sistema_seguridad):
    """Opcional: registrar cuatro figuras usando el detector ya existente."""
    respuesta = input('Crear nueva contrasena con figuras? (s/N): ').strip().lower()
    if respuesta != 's':
        return True
    secuencia = []
    ultima = None
    repetidas = vacios = 0
    esperar = False
    print('Ensenar cuatro figuras una a una, retirando cada tarjeta despues de confirmarla.')
    anterior_t = time.perf_counter()
    while len(secuencia)<4:
        frame = leer_frame(cap,mapas)
        alto,ancho = frame.shape[:2]
        lado = int(sistema_seguridad.TAMANO_RECUADRO*alto)
        x,y = (ancho-lado)//2,(alto-lado)//2
        recorte = frame[y:y+lado,x:x+lado]
        figura,contorno,_ = sistema_seguridad.detectar_figura(recorte)
        if figura is None:
            repetidas = 0
            ultima = None
            vacios += 1
            if vacios>=sistema_seguridad.FRAMES_VACIOS:
                esperar = False
        else:
            vacios = 0
            repetidas = repetidas+1 if figura==ultima else 1
            ultima = figura
            if repetidas>=sistema_seguridad.FRAMES_PARA_CONFIRMAR and not esperar:
                secuencia.append(figura)
                esperar = True
        cv2.rectangle(frame,(x,y),(x+lado,y+lado),(0,255,0),2)
        if contorno is not None:
            cv2.drawContours(recorte,[contorno],-1,(0,0,255),2)
        cv2.putText(frame,f'Registro: {len(secuencia)}/4 | '+('Retira tarjeta' if esperar else 'Muestra figura'),(10,25),0,0.6,(0,255,255),2)
        ahora = time.perf_counter()
        dibujar_fps(frame, 1/max(ahora-anterior_t, 1e-6))
        anterior_t = ahora
        cv2.imshow('Crear contrasena',frame)
        if cv2.waitKey(1)&255 in (27,ord('q')):
            cv2.destroyAllWindows()
            return False
    sistema_seguridad.CONTRASENA = secuencia
    cv2.destroyAllWindows()
    print('Contrasena registrada para esta sesion. Introducela de nuevo para acceder.')
    return True


