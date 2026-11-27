"""Deteccion y seguimiento de bebidas mediante OpenCV y NumPy.

Se analiza una superficie lisa en cada frame, sin guardar una imagen de fondo.
Las bebidas ruedan en un unico sentido y la camara puede moverse ligeramente.
"""
import itertools
import cv2
import numpy as np

from gestion_camara import leer_frame

IZQUIERDA_DERECHA = True

# El orden coincide con los cuatro digitos que introduce el usuario.
BEBIDAS = ('Coca-Cola', 'Powerade', 'Fanta', 'Agua')
# H: 0..179; S y V: 0..255 en OpenCV.
# Evidencias del liquido y de la etiqueta, no un color unico por bebida.
# El negro no tiene una tonalidad H fiable: se detecta por luminosidad V.
RANGOS = {
    'rojo': [((0, 110, 65), (8, 255, 255)),
             ((170, 110, 65), (179, 255, 255))],
    'cian': [((90, 150, 85), (107, 255, 255))],
    'amarillo': [((17, 160, 100), (35, 255, 255))],
    'naranja': [((9, 170, 100), (16, 255, 255))],
    'azul_oscuro': [((108, 40, 40), (145, 255, 255))],
    'oscuro': [((0, 0, 0), (179, 255, 95))],
}
COLORES = {'Coca-Cola': (0, 0, 255), 'Powerade': (255, 210, 0),
           'Fanta': (0, 145, 255), 'Agua': (255, 120, 100)}
# Fracciones minimas de la silueta. Ajustables con la camara real.
MIN_CIAN = 0.14
MIN_AMARILLO = 0.14
MIN_NARANJA = 0.19
MIN_ROJO = 0.045
MIN_OSCURO = 0.20
MIN_AZUL_AGUA = 0.055
UMBRAL_CROMA = 19  # Diferencia de color en Lab frente a la superficie.
UMBRAL_LUZ = 39    # Diferencia de luminosidad frente a la superficie.
MIN_VOTOS = 3
MAX_PERDIDOS = 12
MAX_OBJETOS = 2


class DetectorObjetos:
    def __init__(self, frame, izquierda_derecha=True):
        # Toda la imagen es la zona de deteccion. No hay seleccion manual.
        self.h, self.w = frame.shape[:2]
        self.x = self.y = 0
        self.sentido = 1 if izquierda_derecha else -1
        self.linea = int(self.w * 0.50)
        self.margen = max(4, int(self.w * 0.035))  # Evitar cruces por pequenos temblores.
        self.area_min = max(45, self.w*self.h*0.0007)
        self.tracks = []
        self.siguiente_id = 1
        self.mascara = np.zeros((self.h, self.w), np.uint8)
        self.ocupada = False
        self.aviso = ''

    def recortar(self, frame):
        return frame[self.y:self.y+self.h, self.x:self.x+self.w]

    def posicion(self, x):
        # Negativo antes de la linea y positivo despues, en ambos sentidos.
        return self.sentido * (x-self.linea)

    def color_superficie(self, lab):
        """Estima el color dominante del borde en el frame actual.

        Una superficie lisa suele aparecer en varios bordes aunque la camara
        cambie de posicion. Se usa la moda de colores cuantizados, no la
        media de toda la imagen (que incluiria las botellas).
        """
        h, w = lab.shape[:2]
        borde = max(2, min(h, w)//16)
        muestras = np.concatenate((lab[:borde:3, ::3].reshape(-1, 3),
                                   lab[-borde::3, ::3].reshape(-1, 3),
                                   lab[::3, :borde:3].reshape(-1, 3),
                                   lab[::3, -borde::3].reshape(-1, 3)))
        # Cuantizacion para agrupar pequenas variaciones de iluminacion.
        q = muestras.astype(np.int32)//16
        claves = q[:, 0]*256 + q[:, 1]*16 + q[:, 2]
        moda = np.bincount(claves, minlength=4096).argmax()
        centro = np.median(muestras[claves == moda], axis=0)
        cercanas = np.max(np.abs(muestras.astype(np.float32)-centro), axis=1) < 24
        return np.median(muestras[cercanas], axis=0).astype(np.float32)

    def segmentar(self, frame):
        recorte = cv2.GaussianBlur(self.recortar(frame), (5, 5), 0)
        lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB)
        superficie = self.color_superficie(lab)
        diferencia = lab.astype(np.float32)-superficie
        croma = np.hypot(diferencia[:, :, 1], diferencia[:, :, 2])
        luz = np.abs(diferencia[:, :, 0])
        # El fondo se estima de nuevo cada vez: mover el movil no compara
        # coordenadas actuales con las de una fotografia antigua.
        primer_plano = ((croma > UMBRAL_CROMA) | (luz > UMBRAL_LUZ)).astype(np.uint8)*255
        kernel3 = np.ones((3, 3), np.uint8)
        primer_plano = cv2.morphologyEx(primer_plano, cv2.MORPH_OPEN, kernel3)
        primer_plano = cv2.morphologyEx(primer_plano, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # Las botellas ruedan horizontalmente: cerrar huecos pequenos en
        # direccion perpendicular ayuda a unir etiqueta y liquido.
        vertical = max(5, min(25, int(self.h*0.035)//2*2+1))
        primer_plano = cv2.morphologyEx(primer_plano, cv2.MORPH_CLOSE,
                                        np.ones((vertical, 3), np.uint8))
        hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
        mascaras = {}
        for nombre, rangos in RANGOS.items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for inferior, superior in rangos:
                m |= cv2.inRange(hsv, np.array(inferior), np.array(superior))
            mascaras[nombre] = cv2.bitwise_and(m, primer_plano)
        self.mascara = primer_plano
        return primer_plano, mascaras

    def clasificar_color(self, silueta, mascaras):
        """Clasifica por evidencias de liquido y etiqueta dentro de la silueta.

        Los pixeles blancos de las letras no son un requisito ni un fallo.
        Powerade: cian saturado del liquido (no la etiqueta negra).
        Fanta: amarillo/naranja del liquido, con apoyo de su etiqueta.
        Coca-Cola: oscuridad del liquido y rojo de la etiqueta.
        Agua: azul de etiqueta/tapon y ausencia de evidencia de otros liquidos.
        No se identifica como agua cualquier objeto de color desconocido.
        """
        area = cv2.countNonZero(silueta)
        if area == 0:
            return None, 0.0
        f = {nombre: cv2.countNonZero(m & silueta)/area
             for nombre, m in mascaras.items()}
        rojo = f['rojo']
        cian = f['cian']
        amarillo = f['amarillo']
        naranja = f['naranja']
        azul = f['azul_oscuro']
        oscuro = f['oscuro']
        # Cada bebida tiene su propia evidencia. La etiqueta blanca o negra
        # del Powerade no compite como si fuera el color de otra bebida.
        puntuaciones = {}
        if cian >= MIN_CIAN:
            puntuaciones['Powerade'] = cian
        if amarillo >= MIN_AMARILLO or naranja >= MIN_NARANJA:
            puntuaciones['Fanta'] = amarillo + 0.65*naranja
        if rojo >= MIN_ROJO and oscuro >= MIN_OSCURO:
            puntuaciones['Coca-Cola'] = min(1., 2*rojo + 0.4*oscuro)
        elif rojo >= 0.20 and amarillo+naranja < 0.10 and cian < 0.08:
            # Apoyo de la etiqueta si el liquido aparece demasiado iluminado.
            puntuaciones['Coca-Cola'] = rojo

        if puntuaciones:
            orden = sorted(puntuaciones, key=puntuaciones.get, reverse=True)
            mejor = orden[0]
            # Si dos tipos tienen evidencias comparables, esperar mas frames.
            if len(orden) > 1 and puntuaciones[mejor] < 1.35*puntuaciones[orden[1]]:
                return None, 0.0
            confianza = puntuaciones[mejor]/sum(puntuaciones.values())
            return mejor, confianza

        # La transparencia no es un color HSV. Se exige azul de la etiqueta
        # y poca evidencia de liquido coloreado; nunca basta la ausencia de color.
        if (azul >= MIN_AZUL_AGUA and cian < 0.08 and
                amarillo+naranja < 0.10 and rojo < 0.045 and oscuro < 0.25):
            return 'Agua', min(1., azul/MIN_AZUL_AGUA/2)
        return None, 0.0

    def candidato(self, silueta, mascaras, estimado=False):
        puntos = cv2.findNonZero(silueta)
        if puntos is None:
            return None
        x, y, w, h = cv2.boundingRect(puntos)
        area = cv2.countNonZero(silueta)
        if area < self.area_min or min(w, h) < 7:
            return None
        if area > self.w*self.h*0.92 or max(w,h)/min(w,h) > 12:
            return None
        if area/(w*h) < 0.23:
            return None
        bebida, confianza = self.clasificar_color(silueta, mascaras)
        # Centroide geometrico de la silueta: momentos espaciales de orden 1.
        # No es simplemente el centro de su rectangulo delimitador.
        momentos = cv2.moments(silueta, binaryImage=True)
        centro = (momentos['m10']/momentos['m00'],
                  momentos['m01']/momentos['m00'])
        return {'caja': (x,y,w,h), 'centro': centro,
                'area': area, 'bebida': bebida, 'confianza': confianza,
                'estimado': estimado}

    def predecir(self, t):
        pasos = t['perdidos']+1
        return tuple(c+v*pasos for c,v in zip(t['centro'],t['velocidad']))

    def separar_unidos(self, silueta, mascaras):
        """Divide un contorno unido usando dos trayectorias ya conocidas.

        Si no hay evidencia suficiente, no inventa una segunda bebida.
        La division es aproximada: no aporta nuevos votos de color.
        """
        if len(self.tracks) != 2:
            return None
        if any(t['frames'] < 2 for t in self.tracks):
            return None
        puntos = cv2.findNonZero(silueta)
        if puntos is None:
            return None
        x,y,w,h = cv2.boundingRect(puntos)
        orden = sorted(self.tracks, key=lambda t:self.predecir(t)[0])
        a,b = orden
        pa,pb = self.predecir(a),self.predecir(b)
        wa,wb = a['tamano'][0],b['tamano'][0]
        # Debe abarcar ambos centros y ser mas ancho que una sola bebida.
        margen = max(8, int(0.18*max(wa,wb)))
        if not all(x-margen <= p[0] <= x+w+margen and
                   y-margen <= p[1] <= y+h+margen for p in (pa,pb)):
            return None
        if w < 1.30*max(wa,wb) or pb[0]-pa[0] < 0.35*min(wa,wb):
            return None
        if cv2.countNonZero(silueta) < 0.65*(a['area']+b['area']):
            return None
        # Frontera entre las dos cajas esperadas; funciona con tamanos distintos.
        corte = int(((pa[0]+wa/2)+(pb[0]-wb/2))/2)
        if not x+8 <= corte <= x+w-8:
            return None
        izquierda = silueta.copy(); izquierda[:,corte:] = 0
        derecha = silueta.copy(); derecha[:,:corte] = 0
        partes = [self.candidato(m, mascaras, True) for m in (izquierda,derecha)]
        if any(d is None for d in partes):
            return None
        for d,t in zip(partes,orden):
            if not 0.40*t['area'] <= d['area'] <= 1.8*t['area']:
                return None
        self.aviso = 'Dos objetos unidos: seguimiento estimado'
        return partes

    def agrupar_fragmentos(self, contornos):
        """Une piezas alineadas de una misma botella (tapon y etiqueta).

        En agua, el plastico transparente puede no diferenciarse de la mesa.
        Las bebidas van una detras de otra en horizontal; solo se agrupan
        fragmentos separados en vertical, no dos botellas lado a lado.
        """
        grupos = [c for c in contornos if cv2.contourArea(c) >= self.area_min*0.45]
        cambio = True
        while cambio:
            cambio = False
            for i, a in enumerate(grupos):
                x,y,w,h = cv2.boundingRect(a)
                for j in range(i+1, len(grupos)):
                    b = grupos[j]
                    xx,yy,ww,hh = cv2.boundingRect(b)
                    solape = min(x+w,xx+ww)-max(x,xx)
                    hueco = max(yy-(y+h),y-(yy+hh))
                    if (solape >= 0.65*min(w,ww) and
                        abs((x+w/2)-(xx+ww/2)) <= 0.35*max(w,ww) and
                        0 <= hueco <= min(self.h*0.35, 2.2*max(w,ww)) and
                        min(cv2.contourArea(a),cv2.contourArea(b)) <=
                        0.65*max(cv2.contourArea(a),cv2.contourArea(b))):
                        grupos[i] = cv2.convexHull(np.vstack((a,b)))
                        grupos.pop(j)
                        cambio = True
                        break
                if cambio:
                    break
        return grupos

    def detectar(self, frame):
        movimiento, mascaras = self.segmentar(frame)
        contornos, _ = cv2.findContours(movimiento, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        detecciones = []
        self.ocupada = False
        self.aviso = ''
        for c in self.agrupar_fragmentos(contornos):
            if cv2.contourArea(c) < self.area_min*0.7:
                continue
            silueta = np.zeros(movimiento.shape, np.uint8)
            cv2.drawContours(silueta, [c], -1, 255, -1)
            partes = self.separar_unidos(silueta, mascaras)
            if partes is not None:
                detecciones.extend(partes)
                continue
            d = self.candidato(silueta, mascaras)
            if d is not None:
                # No seguir sombras ni manchas sin evidencia de bebida, salvo
                # que correspondan a una trayectoria que ya conocemos.
                if d['bebida'] is None and not any(self.coste(t, d) is not None for t in self.tracks):
                    continue
                # Si un solo contorno tapa dos tracks, no asignarlo a uno
                # arbitrariamente: esperar a que vuelvan a separarse.
                if len(self.tracks) == 2:
                    x,y,w,h = d['caja']
                    dentro = [x <= self.predecir(t)[0] <= x+w and
                              y <= self.predecir(t)[1] <= y+h for t in self.tracks]
                    if all(dentro) and w > 1.25*max(t['tamano'][0] for t in self.tracks):
                        self.aviso = 'Solapamiento ambiguo: esperando separacion'
                        continue
                detecciones.append(d)
        self.ocupada = bool(detecciones)
        return detecciones

    def clasificacion(self, t):
        orden = sorted(BEBIDAS, key=t['votos'].get, reverse=True)
        a,b = orden[:2]
        if t['votos'][a] >= MIN_VOTOS and t['votos'][a] > t['votos'][b]*1.3:
            return a
        return None

    def coste(self, t, d):
        px,py = self.predecir(t)
        dx,dy = d['centro']
        distancia = float(np.hypot(dx-px,dy-py))
        limite = max(30, self.w*0.18, t['tamano'][0]*0.8)
        limite *= 1+0.12*min(t['perdidos'],5)
        if distancia > limite or abs(dy-py) > max(25,self.h*0.12,t['tamano'][1]*0.8):
            return None
        if self.sentido*(dx-t['centro'][0]) < -max(12,self.w*0.055,t['tamano'][0]*0.4):
            return None
        # El color solo penaliza asociaciones dudosas, no define la identidad.
        tipo = self.clasificacion(t)
        penalizacion = 0
        if tipo and d['bebida'] and tipo != d['bebida'] and not d['estimado']:
            penalizacion = limite*0.45
        return distancia+penalizacion

    def asociar(self, detecciones):
        """Asignacion global: con dos tracks se pueden probar todas las parejas."""
        mejor = []
        clave_mejor = (0,0)
        opciones = []
        for i,t in enumerate(self.tracks):
            opciones.append([-1]+[j for j,d in enumerate(detecciones)
                                  if self.coste(t,d) is not None])
        for asignacion in itertools.product(*opciones):
            usados = [j for j in asignacion if j >= 0]
            if len(set(usados)) != len(usados):
                continue
            pares = [(i,j) for i,j in enumerate(asignacion) if j >= 0]
            # No intercambiar el orden de bebidas que circulan una tras otra.
            if len(pares) == 2:
                (i,j),(k,l) = pares
                antes = self.tracks[i]['centro'][0]-self.tracks[k]['centro'][0]
                ahora = detecciones[j]['centro'][0]-detecciones[l]['centro'][0]
                if antes*ahora < 0:
                    continue
            coste = sum(self.coste(self.tracks[i],detecciones[j]) for i,j in pares)
            clave = (-len(pares),coste)
            if clave < clave_mejor:
                mejor,clave_mejor = pares,clave
        return mejor

    def nuevo_track(self, d):
        votos = dict.fromkeys(BEBIDAS,0)
        if d['bebida']:
            votos[d['bebida']] = 1
        t = {'id': self.siguiente_id, 'centro': d['centro'], 'caja': d['caja'],
             'tamano': d['caja'][2:], 'area': d['area'], 'velocidad': (0.,0.),
             'perdidos': 0, 'frames': 1, 'votos': votos, 'contado': False,
             'antes': self.posicion(d['centro'][0]) <= -self.margen,
             'cruzado': False, 'estimado': False}
        self.siguiente_id += 1
        self.tracks.append(t)

    def actualizar_track(self, t, d):
        dt = t['perdidos']+1
        medida = tuple((a-b)/dt for a,b in zip(d['centro'],t['centro']))
        peso = 1.0 if t['frames'] == 1 else 0.65
        t['velocidad'] = tuple(peso*a+(1-peso)*b for a,b in zip(medida,t['velocidad']))
        t['centro'],t['caja'] = d['centro'],d['caja']
        t['perdidos'] = 0
        t['frames'] += 1
        t['estimado'] = d['estimado']
        if not d['estimado']:
            if d['bebida']:
                t['votos'][d['bebida']] += 1
            t['area'] = 0.7*t['area']+0.3*d['area']
            t['tamano'] = tuple(0.7*a+0.3*b for a,b in zip(t['tamano'],d['caja'][2:]))
        posicion = self.posicion(t['centro'][0])
        if posicion <= -self.margen:
            t['antes'] = True
            if not t['contado']:
                t['cruzado'] = False
        if t['antes'] and posicion >= self.margen:
            t['cruzado'] = True

    def actualizar(self, frame):
        """Devuelve una lista de bebidas nuevas que han cruzado la linea."""
        detecciones = self.detectar(frame)
        pares = self.asociar(detecciones)
        usados_t = {i for i,j in pares}
        usados_d = {j for i,j in pares}
        for i,j in pares:
            self.actualizar_track(self.tracks[i],detecciones[j])
        for i,t in enumerate(self.tracks):
            if i not in usados_t:
                t['perdidos'] += 1
        # Un track perdido se conserva para recuperar su ID y estado de conteo.
        self.tracks = [t for t in self.tracks if t['perdidos'] <= MAX_PERDIDOS]
        libres = [d for j,d in enumerate(detecciones) if j not in usados_d]
        libres.sort(key=lambda d:(d['bebida'] is None,self.posicion(d['centro'][0]) > -self.margen,-d['area']))
        for d in libres:
            if len(self.tracks) >= MAX_OBJETOS:
                break
            # No crear IDs sobre una division estimada o una mancha desconocida.
            # Un objeto nacido tras la linea se muestra, pero no se cuenta.
            if d['estimado'] or d['bebida'] is None:
                continue
            self.nuevo_track(d)
        eventos = []
        for t in self.tracks:
            bebida = self.clasificacion(t)
            if t['cruzado'] and not t['contado'] and not t['perdidos']:
                if bebida and self.posicion(t['centro'][0]) >= self.margen:
                    eventos.append(bebida)
                    t['contado'] = True
                elif not bebida:
                    self.aviso = 'Bebida sin clasificacion segura; no contabilizada'
        return eventos

    def zona_vacia(self):
        return not self.ocupada and not self.tracks

    def dibujar(self, frame):
        xlinea = self.x+self.linea
        cv2.line(frame,(xlinea,self.y),(xlinea,self.y+self.h-1),(0,255,255),2)
        inicio = (self.x+int(self.w*(0.35 if self.sentido==1 else 0.65)),self.y+12)
        fin = (inicio[0]+self.sentido*max(15,self.w//8),inicio[1])
        cv2.arrowedLine(frame,inicio,fin,(0,255,255),2)
        for t in self.tracks:
            if t['perdidos']:
                continue
            bebida = self.clasificacion(t)
            nombre = bebida or 'Identificando'
            color = COLORES.get(bebida,(255,255,255))
            x,y,w,h = t['caja']; x += self.x; y += self.y
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            texto = f"{nombre} #{t['id']}"
            if t['contado']:
                texto += ' OK'
            if t['estimado']:
                texto += ' ~'
            cv2.putText(frame,texto,(x,max(18,y-7)),0,0.5,color,2)
            # Las coordenadas del seguimiento son relativas a la ROI.
            # Dibujar un punto rojo visible para cada objeto, sea del color que sea.
            cx = self.x + int(round(t['centro'][0]))
            cy = self.y + int(round(t['centro'][1]))
            cv2.circle(frame,(cx,cy),7,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(frame,(cx,cy),5,(0,0,255),-1,cv2.LINE_AA)
        return frame


def preparar_detector(cap, mapas, anterior=None):
    """Arranque directo, sin capturar el fondo ni pedir una ROI."""
    frame = leer_frame(cap, mapas)
    print('Detector listo: toda la imagen. Pasa las bebidas de izquierda a derecha.'
          if IZQUIERDA_DERECHA else
          'Detector listo: toda la imagen. Pasa las bebidas de derecha a izquierda.')
    detector = DetectorObjetos(frame, IZQUIERDA_DERECHA)
    if anterior is not None:
        detector.siguiente_id = anterior.siguiente_id
    return detector
