"""Deteccion y seguimiento de bebidas mediante OpenCV y NumPy.

La camara debe estar fija y las bebidas deben rodar en un unico sentido.
Las coordenadas del detector son relativas a la zona de paso (ROI).
"""
import itertools
import cv2
import numpy as np

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
UMBRAL_FONDO = 25
MIN_VOTOS = 3
MAX_PERDIDOS = 12
MAX_OBJETOS = 2


class DetectorObjetos:
    def __init__(self, fondo, zona, izquierda_derecha=True):
        self.x, self.y, self.w, self.h = map(int, zona)
        if self.w < 60 or self.h < 30 or self.x < 0 or self.y < 0:
            raise ValueError('Zona de paso demasiado pequena o invalida')
        if self.x+self.w > fondo.shape[1] or self.y+self.h > fondo.shape[0]:
            raise ValueError('Zona fuera de la imagen')
        self.sentido = 1 if izquierda_derecha else -1
        self.linea = int(self.w * (0.70 if izquierda_derecha else 0.30))
        self.margen = max(3, int(self.w * 0.012))
        self.area_min = max(110, self.w*self.h*0.0025)
        self.tracks = []
        self.siguiente_id = 1
        self.fondo = cv2.GaussianBlur(self.recortar(fondo), (5, 5), 0)
        self.mascara = np.zeros((self.h, self.w), np.uint8)
        self.ocupada = False
        self.aviso = ''

    def recortar(self, frame):
        return frame[self.y:self.y+self.h, self.x:self.x+self.w]

    def posicion(self, x):
        # Negativo antes de la linea y positivo despues, en ambos sentidos.
        return self.sentido * (x-self.linea)

    def segmentar(self, frame):
        recorte = cv2.GaussianBlur(self.recortar(frame), (5, 5), 0)
        # Compensar cambios globales moderados de luminosidad, no los locales.
        diferencia = recorte.astype(np.int16)-self.fondo.astype(np.int16)
        muestra = diferencia[::5, ::5]
        cambio = float(np.clip(np.median(muestra), -22, 22))
        distancia = np.max(np.abs(diferencia-cambio), axis=2)
        movimiento = (distancia > UMBRAL_FONDO).astype(np.uint8)*255
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)
        movimiento = cv2.morphologyEx(movimiento, cv2.MORPH_OPEN, kernel3)
        # Cerrar huecos de letras, reflejos y zonas transparentes.
        movimiento = cv2.morphologyEx(movimiento, cv2.MORPH_CLOSE, kernel5)
        hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
        mascaras = {}
        for bebida, rangos in RANGOS.items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for inferior, superior in rangos:
                m |= cv2.inRange(hsv, np.array(inferior), np.array(superior))
            # Solo los pixeles que han cambiado respecto al fondo.
            mascaras[bebida] = cv2.bitwise_and(m, movimiento)
        self.mascara = movimiento
        return movimiento, mascaras

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
        if area < self.area_min or min(w, h) < 10:
            return None
        if area > self.w*self.h*0.75 or max(w,h)/min(w,h) > 6:
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

    def detectar(self, frame):
        movimiento, mascaras = self.segmentar(frame)
        contornos, _ = cv2.findContours(movimiento, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        detecciones = []
        self.ocupada = False
        self.aviso = ''
        for c in contornos:
            if cv2.contourArea(c) < self.area_min*0.7:
                continue
            self.ocupada = True
            silueta = np.zeros(movimiento.shape, np.uint8)
            cv2.drawContours(silueta, [c], -1, 255, -1)
            partes = self.separar_unidos(silueta, mascaras)
            if partes is not None:
                detecciones.extend(partes)
                continue
            d = self.candidato(silueta, mascaras)
            if d is not None:
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
        limite = max(30, self.w*0.12, t['tamano'][0]*0.8)
        limite *= 1+0.12*min(t['perdidos'],5)
        if distancia > limite or abs(dy-py) > max(20,t['tamano'][1]*0.8):
            return None
        if self.sentido*(dx-t['centro'][0]) < -max(12,t['tamano'][0]*0.4):
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
        libres.sort(key=lambda d:(d['bebida'] is None,self.posicion(d['centro'][0])))
        for d in libres:
            if len(self.tracks) >= MAX_OBJETOS:
                break
            # No crear IDs sobre una division estimada ni despues de la linea.
            if d['estimado'] or self.posicion(d['centro'][0]) > -self.margen:
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
        cv2.rectangle(frame,(self.x,self.y),(self.x+self.w-1,self.y+self.h-1),(255,255,255),2)
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
