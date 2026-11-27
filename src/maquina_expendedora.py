"""Bucle de deteccion y estados de la maquina expendedora."""
import queue
import threading
import time
import cv2

from gestion_camara import leer_frame
from gestion_pedidos import GestorPedidos, leer_pedido, preguntar_nuevo_pedido
from interfaz import mostrar


def pedir_en_terminal(resultados, pregunta):
    # Solo input se ejecuta en otro hilo. La camara sigue leyendo frames,
    # incluso mientras se espera la respuesta y/n de un pedido completado.
    try:
        respuesta = leer_pedido() if pregunta == 'pedido' else preguntar_nuevo_pedido()
        resultados.put((pregunta, respuesta))
    except (EOFError, KeyboardInterrupt):
        resultados.put((pregunta, None))


def ejecutar_maquina(cap,mapas,detector):
    gestor = GestorPedidos()
    resultados = queue.Queue()
    # Estados posibles:
    # pedido -> en_curso -> continuar -> esperar_zona -> pedido
    # Solo se pide un nuevo codigo despues de completar el anterior y elegir y.
    estado = 'pedido'
    threading.Thread(target=pedir_en_terminal,args=(resultados,'pedido'),daemon=True).start()
    ver_mascara = False
    tiempo = time.perf_counter()
    fps = 0.0
    vacios = 0
    while True:
        frame = leer_frame(cap,mapas)
        # Detectar siempre: incluso tras completar el pedido, las bebidas de
        # mas pasan a sobrantes, mientras el usuario contesta y/n.
        for bebida in detector.actualizar(frame):
            gestor.registrar(bebida)

        if detector.zona_vacia():
            vacios += 1
        else:
            vacios = 0

        # El hilo de terminal nunca modifica el gestor ni usa OpenCV.
        if estado in ('pedido','continuar'):
            try:
                pregunta,respuesta = resultados.get_nowait()
            except queue.Empty:
                pass
            else:
                if respuesta is None:
                    return
                if pregunta == 'pedido' and estado == 'pedido':
                    gestor.iniciar(respuesta)
                    estado = 'en_curso'
                elif pregunta == 'continuar' and estado == 'continuar':
                    if respuesta == 'n':
                        print('Programa finalizado.')
                        return
                    estado = 'esperar_zona'
                    if not detector.zona_vacia():
                        print('Esperando a que salgan las bebidas restantes...')

        # Solo se pregunta una vez por pedido, tambien si se completo usando
        # exclusivamente las bebidas sobrantes de pedidos anteriores.
        if estado == 'en_curso' and gestor.completo():
            print('\nPEDIDO COMPLETADO.')
            estado = 'continuar'
            threading.Thread(target=pedir_en_terminal,args=(resultados,'continuar'),daemon=True).start()

        # Mantener el detector activo hasta que la zona lleve varios frames
        # vacia. Evita asignar el final del pedido anterior al siguiente.
        if estado == 'esperar_zona' and vacios >= 5:
            gestor.cerrar()
            estado = 'pedido'
            threading.Thread(target=pedir_en_terminal,args=(resultados,'pedido'),daemon=True).start()

        ahora = time.perf_counter()
        instantaneo = 1/max(ahora-tiempo,1e-6)
        fps = instantaneo if fps == 0 else 0.85*fps+0.15*instantaneo
        tiempo = ahora
        mostrar(frame,detector,gestor,estado,fps)
        if ver_mascara:
            cv2.imshow('Mascara de deteccion',detector.mascara)
        tecla = cv2.waitKey(1)&255
        if tecla in (27,ord('q')):
            return
        if tecla == ord('l'):
            detector.cambiar_linea()
        if tecla==ord('m'):
            ver_mascara = not ver_mascara
            if not ver_mascara:
                cv2.destroyWindow('Mascara de deteccion')
