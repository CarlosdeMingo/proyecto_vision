"""Comprueba la clave existente sin modificar el sistema original."""
import importlib
from pathlib import Path


def comprobar_acceso(cap, mapas, ruta_calibracion):
    try:
        seguridad = importlib.import_module('sistema_seguridad')
    except FileNotFoundError as error:
        if error.filename and Path(error.filename).name == 'contrasena.txt':
            raise RuntimeError('Falta el fichero de contrasena del sistema original. Conserva tu contrasena.txt.') from error
        raise
    # La funcion original ya corrige la distorsion. Solo ajustamos su ruta
    # en memoria, sin cambiar el fichero ni la contrasena.
    seguridad.CALIBRACION = str(ruta_calibracion)
    print('Introduce la contrasena existente mediante las cuatro figuras.')
    return seguridad.sistema_seguridad(cap)
