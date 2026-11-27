"""Deteccion, seguimiento y conteo de bebidas (version sencilla).

1. Al empezar se guarda una foto del FONDO (zona vacia).
2. En cada frame, lo que cambia respecto al fondo es "primer plano".
3. Dentro del primer plano se busca el color de cada bebida (HSV).
4. Cada mancha grande de un color es una bebida; se sigue por cercania.
5. Cuando una bebida cruza la linea amarilla, se cuenta una vez.
"""
import cv2
import numpy as np

from gestion_camara import leer_frame

# El orden coincide con los cuatro digitos que introduce el usuario.
BEBIDAS = ('Coca-Cola', 'Powerade', 'Fanta', 'Agua')
COLORES = {'Coca-Cola': (0, 0, 255), 'Powerade': (255, 210, 0),
           'Fanta': (0, 145, 255), 'Agua': (255, 120, 100)}

ESCALA = 0.5           # se procesa a mitad de tamano (320x240): 4 veces mas rapido
UMBRAL_FONDO = 35      # diferencia con el fondo para considerar que algo ha cambiado
AREA_MIN = 1500        # area minima de una bebida (px, en la imagen de 640x480)
AREA_MIN_AGUA = 4000   # el agua necesita mas: asi no salen etiquetas o reflejos sueltos
MAX_DISTANCIA = 120    # px que puede moverse una bebida entre dos frames
MAX_PERDIDOS = 10      # frames sin ver una bebida antes de olvidarla
MARGEN_LINEA = 10      # zona muerta alrededor de la linea (evita dobles conteos)


def mascara_color(hsv, bebida):
    """Pixeles con el color caracteristico de cada bebida.
    H: 0-179, S y V: 0-255 (valores medidos con la camara real)."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if bebida == 'Coca-Cola':   # liquido muy oscuro o etiqueta roja
        m = (v < 80) | (((h <= 10) | (h >= 170)) & (s >= 110) & (v >= 80))
    elif bebida == 'Fanta':     # naranja muy saturado (la mesa tiene S ~ 90)
        m = (h >= 12) & (h <= 30) & (s >= 125) & (v >= 100)
    elif bebida == 'Powerade':  # azul cian saturado
        m = (h >= 88) & (h <= 112) & (s >= 110) & (v >= 60)
    else:                       # agua: clara y casi sin color
        m = (s < 60) & (v >= 120)
    return m.astype(np.uint8) * 255


class DetectorObjetos:
    def __init__(self, fondo):
        self.h, self.w = fondo.shape[:2]
        self.fondo = self.reducir(fondo)
        self.vertical = True               # linea vertical: paso izquierda <-> derecha
        self.linea = self.w // 2
        self.tracks = []                   # bebidas que se estan siguiendo
        self.siguiente_id = 1
        self.mascara = np.zeros((self.h, self.w), np.uint8)
        self.aviso = ''

    # ---------- 1. primer plano ----------
    def reducir(self, frame):
        pequeno = cv2.resize(frame, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(pequeno, (3, 3), 0)

    def primer_plano(self, frame):
        frame = self.reducir(frame)
        diferencia = cv2.absdiff(frame, self.fondo).max(axis=2)
        # compensar cambios de luz de toda la imagen (exposicion automatica del movil)
        diferencia = cv2.subtract(diferencia, int(np.median(diferencia)))
        mascara = (diferencia > UMBRAL_FONDO).astype(np.uint8) * 255
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return mascara, frame

    # ---------- 2. detecciones ----------
    def detectar(self, frame):
        primer_plano, suave = self.primer_plano(frame)
        hsv = cv2.cvtColor(suave, cv2.COLOR_BGR2HSV)
        self.mascara = cv2.resize(primer_plano, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        detecciones = []
        ocupado = np.zeros_like(primer_plano)   # zona de las bebidas de color ya encontradas
        # El agua va la ultima: es "lo claro que queda" y no debe confundirse
        # con los tapones o etiquetas blancas de las otras bebidas.
        for bebida in BEBIDAS:
            m = mascara_color(hsv, bebida) & primer_plano & ~ocupado
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
            contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contornos:
                minimo = AREA_MIN_AGUA if bebida == 'Agua' else AREA_MIN
                if cv2.contourArea(c) < minimo * ESCALA ** 2:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                # reservar la caja de esta bebida (+8 px) para que nada mas se detecte encima
                cv2.rectangle(ocupado, (x - 8, y - 8), (x + w + 8, y + h + 8), 255, -1)
                # volver a coordenadas de la imagen completa
                x, y, w, h = (int(v / ESCALA) for v in (x, y, w, h))
                detecciones.append({'bebida': bebida, 'caja': (x, y, w, h),
                                    'centro': (x + w / 2, y + h / 2)})
        return detecciones

    # ---------- 3. seguimiento y conteo ----------
    def lado(self, centro):
        p = (centro[0] if self.vertical else centro[1]) - self.linea
        if p > MARGEN_LINEA:
            return 1
        if p < -MARGEN_LINEA:
            return -1
        return 0

    def actualizar(self, frame):
        """Devuelve la lista de bebidas que acaban de cruzar la linea."""
        detecciones = self.detectar(frame)
        for t in self.tracks:
            t['perdidos'] += 1

        eventos = []
        for d in detecciones:
            # la bebida ya seguida mas cercana, del mismo tipo
            candidatos = [t for t in self.tracks if t['bebida'] == d['bebida'] and t['perdidos'] > 0]
            t = min(candidatos, key=lambda t: np.hypot(t['centro'][0] - d['centro'][0],
                                                       t['centro'][1] - d['centro'][1]),
                    default=None)
            if t is None or np.hypot(t['centro'][0] - d['centro'][0],
                                     t['centro'][1] - d['centro'][1]) > MAX_DISTANCIA:
                t = {'id': self.siguiente_id, 'bebida': d['bebida'], 'lado': 0, 'contado': False}
                self.siguiente_id += 1
                self.tracks.append(t)
            t.update(centro=d['centro'], caja=d['caja'], perdidos=0)

            # ha cruzado la linea si pasa de un lado al otro
            lado = self.lado(d['centro'])
            if lado != 0:
                if t['lado'] != 0 and lado != t['lado'] and not t['contado']:
                    t['contado'] = True
                    eventos.append(t['bebida'])
                t['lado'] = lado

        self.tracks = [t for t in self.tracks if t['perdidos'] <= MAX_PERDIDOS]
        self.aviso = '' if detecciones else 'Sin bebidas en la imagen'
        return eventos

    def zona_vacia(self):
        return not self.tracks

    def cambiar_linea(self):
        if self.tracks:
            print('Retira las bebidas antes de cambiar la linea.')
            return
        self.vertical = not self.vertical
        self.linea = self.w // 2 if self.vertical else self.h // 2

    # ---------- 4. dibujo ----------
    def dibujar(self, frame):
        if self.vertical:
            cv2.line(frame, (self.linea, 0), (self.linea, self.h), (0, 255, 255), 2)
        else:
            cv2.line(frame, (0, self.linea), (self.w, self.linea), (0, 255, 255), 2)
        for t in self.tracks:
            if t['perdidos']:
                continue
            x, y, w, h = t['caja']
            color = COLORES[t['bebida']]
            texto = f"{t['bebida']} #{t['id']}" + (' OK' if t['contado'] else '')
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, texto, (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


def preparar_detector(cap, mapas, anterior=None):
    """Guarda la foto del fondo (zona vacia) cuando se pulsa ESPACIO."""
    print('Deja la zona VACIA (sin bebidas ni manos) y pulsa ESPACIO.')
    while True:
        frame = leer_frame(cap, mapas)
        vista = frame.copy()
        cv2.putText(vista, 'Zona VACIA y pulsa ESPACIO', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Fondo', vista)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(' '):
            cv2.destroyWindow('Fondo')
            detector = DetectorObjetos(frame)
            if anterior is not None:
                detector.siguiente_id = anterior.siguiente_id
            print('Fondo guardado. Pasa las bebidas cruzando la linea amarilla.')
            return detector
        if tecla == ord('q'):
            return None
