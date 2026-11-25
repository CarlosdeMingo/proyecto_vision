"""
SISTEMA DE SEGURIDAD (version sencilla)

Hay que ensenar a la camara 4 figuras en el orden correcto.
La figura se coloca dentro del recuadro verde del centro de la pantalla.

Pasos para cada frame:
  1. Corregir la distorsion con la calibracion
  2. Recortar el recuadro central
  3. Pasar a gris, suavizar y umbralizar (Otsu)
  4. Buscar contornos y quedarse con el mas grande
  5. Clasificar la figura contando sus vertices
  6. Si la misma figura se ve varios frames seguidos, se anade a la secuencia
  7. Con 4 figuras se compara con la contrasena

Teclas: r = borrar la secuencia | q = salir
"""

import time

import cv2
import numpy as np
import os

# ------------------------------------------------------------
# CONFIGURACION
# ------------------------------------------------------------
CAMERA_INDEX=0
CALIBRACION = "calibration/camera_params.npz"

FRAMES_PARA_CONFIRMAR = 10   # frames seguidos viendo la misma figura
FRAMES_VACIOS = 5            # frames sin figura para poder ensenar la siguiente
TAMANO_RECUADRO = 0.5        # lado del recuadro central (fraccion del alto)

# ------------------------------------------------------------
# CONTRASEÑA
# ------------------------------------------------------------
def leer_contrasena():
    """Lee contrasena.txt (una figura por linea) y devuelve una lista de 4."""
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contrasena.txt")

    contrasena = []
    with open(ruta, "r") as f:
        for linea in f:
            figura = linea.strip().upper()   # quita espacios / saltos y pasa a mayusculas
            if figura != "":                 # ignora lineas vacias
                contrasena.append(figura)

    if len(contrasena) != 4:
        raise ValueError(f"La contrasena debe tener 4 figuras y tiene {len(contrasena)}")
    for figura in contrasena:
        if figura not in FIGURAS_VALIDAS:
            raise ValueError(f"Figura no valida en contrasena.txt: {figura}")

    return contrasena

FIGURAS_VALIDAS = ["TRIANGULO", "CUADRADO", "PENTAGONO", "HEXAGONO",
                   "CIRCULO", "ESTRELLA", "CRUZ"]
CONTRASENA = leer_contrasena()
# ------------------------------------------------------------
# 1. CALIBRACION
# ------------------------------------------------------------
def cargar_calibracion():
    try:
        datos = np.load(CALIBRACION)
        print("Calibracion cargada")
        return datos["camera_matrix"], datos["dist_coeffs"]
    except FileNotFoundError:
        print("No hay calibracion: se trabaja sin corregir la distorsion")
        return None, None


# ------------------------------------------------------------
# 2. DETECCION DE LA FIGURA
# ------------------------------------------------------------

def clasificar(contorno):
    """Devuelve el nombre de la figura segun su numero de vertices."""
    perimetro = cv2.arcLength(contorno, True)
    vertices = len(cv2.approxPolyDP(contorno, 0.03 * perimetro, True))
    area = cv2.contourArea(contorno)
    circularidad = 4 * np.pi * area / (perimetro * perimetro)

    if vertices == 3:
        return "TRIANGULO"
    if vertices == 4:
        return "CUADRADO"
    if vertices == 5:
        return "PENTAGONO"
    if vertices == 6:
        return "HEXAGONO"
    if vertices == 10:
        return "ESTRELLA"
    if vertices == 12:
        return "CRUZ"
    if circularidad > 0.8:
        return "CIRCULO"
    return None


def detectar_figura(recorte):
    """Busca la figura negra mas grande del recorte. Devuelve (nombre, contorno)."""
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    # si el recuadro es casi uniforme no hay ninguna tarjeta delante
    if gris.std() < 20:
        return None, None, np.zeros(gris.shape, np.uint8)
    # Otsu elige solo el umbral; INV para que la figura negra quede en blanco
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # RETR_LIST devuelve todos los contornos (tambien los que estan dentro del marco)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    alto, ancho = binaria.shape
    mejor = None
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        # si toca el borde del recuadro no es la figura (es el marco, la mano...)
        if x <= 1 or y <= 1 or x + w >= ancho - 1 or y + h >= alto - 1:
            continue
        # descartar manchas pequenas y contornos enormes (borde interior del marco)
        area = cv2.contourArea(c)
        if area < 0.03 * alto * ancho or area > 0.5 * alto * ancho:
            continue
        if mejor is None or cv2.contourArea(c) > cv2.contourArea(mejor):
            mejor = c

    if mejor is None:
        return None, None, binaria
    return clasificar(mejor), mejor, binaria


# ------------------------------------------------------------
# 3. BUCLE PRINCIPAL
# ------------------------------------------------------------
def sistema_seguridad(cap):
    """Devuelve True si se introduce la contraseña correcta, False si se sale."""
    K, dist = cargar_calibracion()

    secuencia = []          # figuras introducidas
    ultima = None           # ultima figura vista
    contador = 0            # frames seguidos viendo la misma figura
    vacios = 0              # frames seguidos sin ver ninguna figura
    esperar_vacio = False   # hay que retirar la figura antes de ensenar otra
    mensaje = ""
    t_mensaje = 0
    t_anterior = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se recibe imagen de la camara")
            return False

        if K is not None:
            frame = cv2.undistort(frame, K, dist)

        # --- recuadro central ---
        alto, ancho = frame.shape[:2]
        lado = int(TAMANO_RECUADRO * alto)
        x0, y0 = (ancho - lado) // 2, (alto - lado) // 2
        recorte = frame[y0:y0 + lado, x0:x0 + lado]

        figura, contorno, binaria = detectar_figura(recorte)

        # --- confirmar la figura si se ve varios frames seguidos ---
        if figura is None:
            vacios += 1
            if vacios >= FRAMES_VACIOS:     # la tarjeta se ha retirado de verdad
                contador = 0
                ultima = None
                esperar_vacio = False
        else:
            vacios = 0
            if figura == ultima:
                contador += 1
            else:
                contador = 1
                ultima = figura

        if contador == FRAMES_PARA_CONFIRMAR and not esperar_vacio:
            secuencia.append(figura)
            esperar_vacio = True
            print("Figura anadida:", figura, secuencia)

        # --- comprobar la contrasena ---
        if len(secuencia) == 4:
            if secuencia == CONTRASENA:
                mensaje = "ACCESO CONCEDIDO"
            else:
                mensaje = "ACCESO DENEGADO"
            t_mensaje = time.time()
            print(mensaje)
            secuencia = []

        # --- dibujar ---
        cv2.rectangle(frame, (x0, y0), (x0 + lado, y0 + lado), (0, 255, 0), 2)
        if contorno is not None:
            cv2.drawContours(recorte, [contorno], -1, (0, 0, 255), 3)
            if figura is not None:
                cv2.putText(frame, figura, (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "Secuencia: " + " ".join(secuencia), (10, alto - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        t = time.time()
        fps = 1 / max(t - t_anterior, 1e-6)
        t_anterior = t
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if time.time() - t_mensaje < 2:
            color = (0, 200, 0) if mensaje == "ACCESO CONCEDIDO" else (0, 0, 255)
            cv2.putText(frame, mensaje, (40, alto // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        elif mensaje == "ACCESO CONCEDIDO":
            return True     # tras 2 s de mensaje, pasar al tracker

        cv2.imshow("Sistema de seguridad", frame)
        cv2.imshow("Binaria", binaria)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            return False
        if tecla == ord("r"):
            secuencia = []
            print("Secuencia borrada")


# ------------------------------------------------------------
# 4. EJECUCION
# ------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se puede abrir la camara")
    else:
        if sistema_seguridad(cap):
            print("ACCESO CONCEDIDO -> lanzar tracker")
        cap.release()
        cv2.destroyAllWindows()
