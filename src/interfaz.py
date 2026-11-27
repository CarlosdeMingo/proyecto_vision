"""Presentacion de datos de la maquina; no modifica los pedidos."""
import cv2
import numpy as np

VENTANA = 'Maquina expendedora'


def dibujar_fps(frame, fps):
    texto = f'FPS: {fps:.1f}'
    (w,h),base = cv2.getTextSize(texto,cv2.FONT_HERSHEY_SIMPLEX,0.65,2)
    x = max(4,frame.shape[1]-w-12)
    cv2.rectangle(frame,(x-4,5),(x+w+5,16+h+base),(0,0,0),-1)
    cv2.putText(frame,texto,(x,12+h),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)


def mostrar(frame, detector, gestor, estado_pedido, fps):
    from deteccion_objetos import BEBIDAS, COLORES

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
