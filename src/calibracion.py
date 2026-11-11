import cv2
import os

def capturar_imagenes():

    # Cambiar el índice si Iriun está en otra cámara
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: no se encuentra la cámara Iriun")
        return

    # Crear carpeta para guardar las fotografías
    os.makedirs("calibration_chess_images", exist_ok=True)

    contador = 0
    total_imagenes = 19

    while contador < total_imagenes:

        ret, frame = cap.read()

        if not ret:
            print("Error capturando imagen")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar esquinas del tablero
        found, corners = cv2.findChessboardCorners(
            gray, (9, 6), None
        )

        imagen = frame.copy()

        if found:

            cv2.drawChessboardCorners(
                imagen, (9, 6), corners, found
            )

            cv2.putText(
                imagen,
                "TABLERO DETECTADO - Pulsa ESPACIO",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                imagen,
                "Buscando tablero...",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.putText(
            imagen,
            f"Imagenes guardadas: {contador}/{total_imagenes}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.imshow("Calibracion Iriun", imagen)

        key = cv2.waitKey(1) & 0xFF

        # Guardar imagen pulsando espacio
        if key == ord(' ') and found:

            filename = (
                f"calibration_chess_images/{contador:03d}.jpg"
            )

            cv2.imwrite(filename, frame)

            contador += 1

            print(f"Imagen guardada: {filename}")

        # Salir pulsando ESC
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Captura finalizada: {contador} imagenes")

    if contador < total_imagenes:
        print("No se han capturado las 19 imagenes.")

if __name__ == "__main__":

    # PASO 1: Capturar las 19 fotografías
    capturar_imagenes()

    # PASO 2: Calibrar la cámara
    calibracion()