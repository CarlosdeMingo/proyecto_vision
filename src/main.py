"""Punto de entrada: camara, seguridad, detector y maquina."""
import cv2
from gestion_camara import cargar_parametros, abrir_camara, crear_mapas
from gestion_seguridad import comprobar_acceso
from deteccion_objetos import preparar_detector
from maquina_expendedora import ejecutar_maquina


def main():
    cap = None
    try:
        K, dist, resolucion, ruta = cargar_parametros()
        cap = abrir_camara(resolucion)
        mapas = crear_mapas(K, dist, resolucion)

        if not comprobar_acceso(cap, mapas, ruta):
            return
        cv2.destroyAllWindows()

        detector = preparar_detector(cap, mapas)
        if detector is not None:
            ejecutar_maquina(cap, mapas, detector)

    except (RuntimeError, ValueError, OSError, KeyError, cv2.error) as error:
        print('ERROR:', error)
    except (KeyboardInterrupt, EOFError):
        print('Programa finalizado.')
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()