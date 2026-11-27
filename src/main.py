"""Maquina expendedora: seguridad, camara, pedidos y stock de salida."""
from pathlib import Path
import os
import queue
import threading
import time
import cv2
import numpy as np
import calibracion
import sistema_seguridad
from deteccion_objetos import DetectorObjetos, BEBIDAS, COLORES

BASE = Path(__file__).resolve().parent
CAMERA_INDEX = calibracion.CAMERA_INDEX
# El NPZ original no indica la resolucion. Usar las fotos de calibracion
# si existen. En su ausencia, 640x480 es una suposicion configurable.
ANCHO, ALTO = 640, 480
IZQUIERDA_DERECHA = True
VENTANA = 'Maquina expendedora'


class GestorPedidos:
    def __init__(self):
        self.sobrantes = dict.fromkeys(BEBIDAS,0)
        self.pedido = None
        self.entregadas = dict.fromkeys(BEBIDAS,0)
        self.previas = dict.fromkeys(BEBIDAS,0)
        self.nuevas = dict.fromkeys(BEBIDAS,0)

    def completo(self):
        return self.pedido is not None and all(
            self.entregadas[b] == self.pedido[b] for b in BEBIDAS)

    def iniciar(self, pedido):
        if self.pedido is not None and not self.completo():
            raise ValueError('Hay un pedido pendiente')
        if set(pedido) != set(BEBIDAS) or any(not isinstance(n,int) or n<0 for n in pedido.values()) or not any(pedido.values()):
            raise ValueError('Pedido invalido')
        self.pedido = pedido.copy()
        for b in BEBIDAS:
            disponibles = min(self.pedido[b],self.sobrantes[b])
            self.previas[b] = disponibles
            self.entregadas[b] = disponibles
            self.nuevas[b] = 0
            self.sobrantes[b] -= disponibles
            if disponibles:
                print(f'Ya hay {disponibles} de {b} en salida. Faltan {self.pendientes(b)}.')

    def pendientes(self, b):
        return 0 if self.pedido is None else self.pedido[b]-self.entregadas[b]

    def registrar(self, bebida):
        if self.pedido is not None and self.pendientes(bebida)>0:
            self.entregadas[bebida] += 1
            self.nuevas[bebida] += 1
            print(f'{bebida}: {self.entregadas[bebida]}/{self.pedido[bebida]}')
        else:
            self.sobrantes[bebida] += 1
            print(f'Sobrante de {bebida}: {self.sobrantes[bebida]} en salida.')

    def cerrar(self):
        if not self.completo():
            raise ValueError('No se puede cerrar un pedido incompleto')
        self.pedido = None
        self.entregadas = dict.fromkeys(BEBIDAS,0)
        self.previas = dict.fromkeys(BEBIDAS,0)
        self.nuevas = dict.fromkeys(BEBIDAS,0)

def leer_pedido():
    print('\nOrden: Coca-Cola, Powerade, Fanta naranja, Agua.')
    print('Ejemplo: 1111 = 1 Coca-Cola, 1 Powerade, 1 Fanta y 1 Agua.')
    while True:
        texto = input('Pedido (4 digitos, q para salir): ').strip().lower()
        if texto == 'q':
            return None
        if len(texto)==4 and all(c in '0123456789' for c in texto) and texto!='0000':
            return dict(zip(BEBIDAS,map(int,texto)))
        print('Introduce exactamente cuatro digitos y al menos una bebida.')


def preguntar_nuevo_pedido():
    while True:
        respuesta = input('Quieres hacer nuevo pedido? (y/n): ').strip().lower()
        if respuesta in ('y', 'n'):
            return respuesta
        print('Introduce y (yes) o n (no).')


def pedir_en_terminal(resultados, pregunta):
    # Solo input se ejecuta en otro hilo. La camara sigue leyendo frames,
    # incluso mientras se espera la respuesta y/n de un pedido completado.
    try:
        respuesta = leer_pedido() if pregunta == 'pedido' else preguntar_nuevo_pedido()
        resultados.put((pregunta, respuesta))
    except (EOFError, KeyboardInterrupt):
        resultados.put((pregunta, None))


def dibujar_fps(frame, fps):
    texto = f'FPS: {fps:.1f}'
    (w,h),base = cv2.getTextSize(texto,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
    x = max(4,frame.shape[1]-w-12)
    cv2.rectangle(frame,(x-4,5),(x+w+5,16+h+base),(0,0,0),-1)
    cv2.putText(frame,texto,(x,12+h),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)


def cargar_parametros():
    ruta = BASE/'calibration'/'camera_params.npz'
    if not ruta.is_file():
        ruta = BASE/'camera_params.npz'
    if not ruta.is_file():
        raise RuntimeError('No hay parametros. Ejecuta calibracion.py primero.')
    with np.load(ruta) as datos:
        K = datos['camera_matrix'].astype(np.float64)
        dist = datos['dist_coeffs'].astype(np.float64)
    if K.shape != (3,3) or not np.isfinite(K).all() or not np.isfinite(dist).all() or K[0,0]<=0 or K[1,1]<=0:
        raise RuntimeError('Parametros de calibracion invalidos')
    fotos = sorted((BASE/calibracion.IMAGE_FOLDER).glob('*.jpg'))
    resolucion = None
    for foto in fotos:
        img = cv2.imread(str(foto))
        if img is not None:
            resolucion = (img.shape[1],img.shape[0])
            break
    if resolucion is None:
        resolucion = (ANCHO,ALTO)
        print(f'AVISO: resolucion original desconocida. Se supone {ANCHO}x{ALTO}. Confirma estos valores.')
    # La funcion de seguridad usa el mismo fichero de calibracion.
    sistema_seguridad.CALIBRACION = str(ruta)
    return K,dist,resolucion


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


def leer_frame(cap, mapas):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('No se recibe imagen de la camara')
    if frame.shape[:2] != mapas[0].shape[:2]:
        raise RuntimeError('La resolucion de la camara ha cambiado')
    # Equivalente a undistort; precalcula las tablas para todos los frames.
    return cv2.remap(frame,mapas[0],mapas[1],cv2.INTER_LINEAR)


def preparar_detector(cap,mapas,anterior=None):
    print('Retira las bebidas y deja la camara fija. Pulsa ESPACIO con la zona vacia.')
    ventana = 'Preparar zona vacia'
    anterior_t = time.perf_counter()
    while True:
        frame = leer_frame(cap,mapas)
        vista = frame.copy()
        cv2.putText(vista,'ESPACIO: capturar fondo | Q: cancelar',(10,25),0,0.55,(0,255,255),2)
        ahora = time.perf_counter()
        dibujar_fps(vista, 1/max(ahora-anterior_t, 1e-6))
        anterior_t = ahora
        cv2.imshow(ventana,vista)
        tecla = cv2.waitKey(1)&255
        if tecla in (27,ord('q')):
            cv2.destroyWindow(ventana)
            return None
        if tecla==32:
            break
    # Mediana temporal: reduce ruido y fluctuaciones puntuales del sensor.
    muestras = []
    for i in range(15):
        muestras.append(leer_frame(cap,mapas))
        cv2.waitKey(1)
    fondo = np.median(np.stack(muestras),axis=0).astype(np.uint8)
    cv2.destroyWindow(ventana)
    print('Selecciona toda la zona por donde ruedan las bebidas, de entrada a salida. ENTER para aceptar.')
    zona = tuple(map(int,cv2.selectROI('Zona de paso',fondo,False,False)))
    cv2.destroyWindow('Zona de paso')
    if zona[2]<60 or zona[3]<30:
        print('Zona cancelada o demasiado pequena.')
        return None
    detector = DetectorObjetos(fondo,zona,IZQUIERDA_DERECHA)
    if anterior is not None:
        detector.siguiente_id = anterior.siguiente_id
    return detector


def configurar_seguridad(cap,mapas):
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


def mostrar(frame, detector, gestor, estado_pedido, fps):
    detector.dibujar(frame)
    dibujar_fps(frame, fps)
    ancho = max(780,frame.shape[1])
    panel = np.full((235,ancho,3),26,np.uint8)
    columnas = [12,190,280,380,480,590,690]
    titulos = ['Bebida','Pedido','Previas','Nuevas','Total','Faltan','Sobra']
    for x,t in zip(columnas,titulos):
        cv2.putText(panel,t,(x,22),0,0.49,(220,220,220),1)
    for i,b in enumerate(BEBIDAS):
        y = 49+i*28
        cv2.putText(panel,b,(columnas[0],y),0,0.63,COLORES[b],2)
        datos = [gestor.pedido[b] if gestor.pedido else 0,gestor.previas[b],
                 gestor.nuevas[b],gestor.entregadas[b],gestor.pendientes(b),gestor.sobrantes[b]]
        for j,n in enumerate(datos):
            color = (0,220,255) if j==4 and n else (230,230,230)
            cv2.putText(panel,str(n),(columnas[j+1],y),0,0.58,color,1)
    if estado_pedido == 'continuar':
        estado,color = 'PEDIDO COMPLETADO | Responde y/n en terminal',(0,240,0)
    elif estado_pedido == 'esperar_zona':
        estado,color = 'PEDIDO COMPLETADO | Esperando zona libre',(0,240,0)
    elif estado_pedido == 'pedido':
        estado,color = 'Introduce el pedido en la terminal',(0,230,255)
    else:
        estado,color = 'Pedido en curso',(0,230,255)
    cv2.putText(panel,estado,(12,159),0,0.63,color,2)
    cv2.putText(panel,detector.aviso or f'Objetos seguidos: {len(detector.tracks)}',(12,185),0,0.5,(200,200,200),1)
    cv2.putText(panel,'Q: salir | R: recapturar fondo/ROI | M: mascara de movimiento',(12,214),0,0.51,(200,200,200),1)
    if ancho>frame.shape[1]:
        frame = cv2.copyMakeBorder(frame,0,0,0,ancho-frame.shape[1],cv2.BORDER_CONSTANT)
    cv2.imshow(VENTANA,np.vstack((frame,panel)))


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
            cv2.imshow('Mascara de movimiento',detector.mascara)
        tecla = cv2.waitKey(1)&255
        if tecla in (27,ord('q')):
            return
        if tecla==ord('m'):
            ver_mascara = not ver_mascara
            if not ver_mascara:
                cv2.destroyWindow('Mascara de movimiento')
        if tecla==ord('r'):
            if estado != 'en_curso' or not detector.zona_vacia():
                print('Solo se puede recapturar durante un pedido, con la zona vacia.')
            else:
                cv2.destroyAllWindows()
                nuevo = preparar_detector(cap,mapas,detector)
                if nuevo is not None:
                    detector = nuevo
                    vacios = 0
                ver_mascara = False
                tiempo = time.perf_counter()
                fps = 0.0


def main():
    cap = None
    try:
        K,dist,resolucion = cargar_parametros()
        cap = abrir_camara(resolucion)
        mapas = cv2.initUndistortRectifyMap(K,dist,None,K,resolucion,cv2.CV_16SC2)
        if not configurar_seguridad(cap,mapas):
            return
        print('Introduce la contrasena mediante las cuatro figuras, retirando cada tarjeta.')
        if not sistema_seguridad.sistema_seguridad(cap):
            return
        cv2.destroyAllWindows()
        detector = preparar_detector(cap,mapas)
        if detector is not None:
            ejecutar_maquina(cap,mapas,detector)
    except (RuntimeError,ValueError,OSError,KeyError,cv2.error) as error:
        print('ERROR:',error)
    except (KeyboardInterrupt,EOFError):
        print('\nPrograma finalizado.')
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
