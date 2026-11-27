"""Deteccion y seguimiento de bebidas mediante OpenCV y NumPy.

Se analiza una superficie lisa en cada frame, sin guardar una imagen de fondo.
Se detecta primero el color y se cuenta en ambos sentidos.
"""
import itertools
import cv2
import numpy as np

from gestion_camara import leer_frame

# La camara frontal puede estar espejada: se cuentan ambos sentidos.
# 'vertical': paso izquierda/derecha. 'horizontal': paso arriba/abajo.
ORIENTACION_LINEA = 'vertical'
POSICION_LINEA = 0.50

# El orden coincide con los cuatro digitos que introduce el usuario.
BEBIDAS = ('Coca-Cola', 'Powerade', 'Fanta', 'Agua')
# H: 0..179; S y V: 0..255 en OpenCV.
# Evidencias del liquido y de la etiqueta, no un color unico por bebida.
# El negro no tiene una tonalidad H fiable: se detecta por luminosidad V.
RANGOS = {
    'rojo': [((0, 85, 40), (13, 255, 255)),
             ((165, 85, 40), (179, 255, 255))],
    'cian': [((78, 85, 45), (109, 255, 255))],
    'amarillo': [((19, 125, 55), (39, 255, 255))],
    'naranja': [((10, 155, 60), (18, 255, 255))],
    'azul_oscuro': [((110, 40, 30), (150, 255, 255))],
    'oscuro': [((0, 0, 0), (179, 255, 105))],
}
COLORES = {'Coca-Cola': (0, 0, 255), 'Powerade': (255, 210, 0),
           'Fanta': (0, 145, 255), 'Agua': (255, 120, 100)}
# El area minima se refiere a la mancha de color, no a toda la botella.
MIN_VOTOS = 3
MAX_PERDIDOS = 12
MAX_OBJETOS = 2


class DetectorObjetos:
    def __init__(self, frame, izquierda_derecha=True):
        self.h, self.w = frame.shape[:2]
        self.x = self.y = 0
        self.orientacion = ORIENTACION_LINEA
        self.configurar_linea()
        self.area_min = max(28, self.w*self.h*0.00045)
        self.tracks = []
        self.siguiente_id = 1
        self.mascara = np.zeros((self.h, self.w), np.uint8)
        self.ocupada = False
        self.aviso = ''

    def configurar_linea(self):
        self.eje = 0 if self.orientacion == 'vertical' else 1
        dimension = self.w if self.eje == 0 else self.h
        self.linea = int(dimension * POSICION_LINEA)
        self.margen = max(4, int(dimension * 0.018))

    def cambiar_linea(self):
        # No borrar identificadores ni recontar bebidas al cambiar la linea.
        if self.tracks or self.ocupada:
            print('Retira las bebidas antes de cambiar la linea.')
            return
        self.orientacion = 'horizontal' if self.eje == 0 else 'vertical'
        self.configurar_linea()
        print('Linea de conteo:', self.orientacion)

    def recortar(self, frame):
        return frame[self.y:self.y+self.h, self.x:self.x+self.w]

    def posicion(self, centro):
        return centro[self.eje] - self.linea

    def lado(self, centro):
        p = self.posicion(centro)
        return -1 if p <= -self.margen else (1 if p >= self.margen else 0)

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
        """HSV primero. El fondo solo ayuda a completar la silueta.

        Nunca se exige que un pixel de color supere un umbral de diferencia
        respecto al fondo para detectar una bebida.
        """
        recorte = cv2.GaussianBlur(self.recortar(frame), (3, 3), 0)
        hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB)
        mascaras = {}
        for nombre, rangos in RANGOS.items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for inferior, superior in rangos:
                m |= cv2.inRange(hsv, inferior, superior)
            mascaras[nombre] = m
        # El fondo se estima de nuevo, pero solo se usa como apoyo geometrico.
        superficie = self.color_superficie(lab)
        diferencia = lab.astype(np.float32) - superficie
        croma = np.hypot(diferencia[:, :, 1], diferencia[:, :, 2])
        luz = np.abs(diferencia[:, :, 0])
        apoyo = ((croma > 13) | (luz > 24)).astype(np.uint8)*255
        apoyo = cv2.morphologyEx(apoyo, cv2.MORPH_CLOSE,
                                  np.ones((5, 5), np.uint8))
        # Evitar que una mesa amarilla/naranja uniforme se convierta en Fanta.
        # Solo se elimina una region si ocupa gran parte del fotograma y
        # conecta con varios bordes; no se descarta una botella cercana.
        semillas = {
            'Coca-Cola': mascaras['rojo'],
            'Powerade': mascaras['cian'],
            'Fanta': mascaras['amarillo'] | mascaras['naranja'],
            'Agua': mascaras['azul_oscuro'],
        }
        self.mascara = np.zeros(hsv.shape[:2], np.uint8)
        for nombre, m in semillas.items():
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            semillas[nombre] = m
        return semillas, apoyo

    def candidato(self, silueta, bebida, estimado=False):
        puntos = cv2.findNonZero(silueta)
        if puntos is None:
            return None
        x,y,w,h = cv2.boundingRect(puntos)
        area = cv2.countNonZero(silueta)
        if area < self.area_min or min(w,h) < 4:
            return None
        momentos = cv2.moments(silueta, binaryImage=True)
        centro = (momentos['m10']/momentos['m00'], momentos['m01']/momentos['m00'])
        return {'caja': (x,y,w,h), 'centro': centro, 'area': area,
                'bebida': bebida, 'confianza': 1.0, 'estimado': estimado}

    def predecir(self, t):
        pasos = t['perdidos']+1
        return tuple(c+v*pasos for c,v in zip(t['centro'],t['velocidad']))

    def agrupar_fragmentos(self, contornos, bebida):
        """Une etiqueta/tapon a lo largo del eje de una botella tumbada.

        No une dos regiones lado a lado en el sentido de paso.
        """
        grupos = [c for c in contornos if cv2.contourArea(c) >= self.area_min*0.35]
        cambio = True
        while cambio:
            cambio = False
            for i,a in enumerate(grupos):
                ax,ay,aw,ah = cv2.boundingRect(a)
                for j in range(i+1,len(grupos)):
                    b = grupos[j]
                    bx,by,bw,bh = cv2.boundingRect(b)
                    u,v = (ax,ay),(bx,by)
                    du,dv = (aw,ah),(bw,bh)
                    e, k = self.eje, 1-self.eje
                    solape = min(u[e]+du[e],v[e]+dv[e])-max(u[e],v[e])
                    hueco = max(v[k]-(u[k]+du[k]),u[k]-(v[k]+dv[k]))
                    alineados = (solape >= 0.50*min(du[e],dv[e]) and
                                  abs(u[e]+du[e]/2-v[e]-dv[e]/2) <= 0.45*max(du[e],dv[e]))
                    # En agua, el tapon y la etiqueta de una botella inclinada
                    # pueden tener menos solape. Exigir separacion longitudinal.
                    agua_inclinada = (bebida == 'Agua' and hueco > 0 and
                        solape >= 0.15*min(du[e],dv[e]) and
                        abs(u[e]+du[e]/2-v[e]-dv[e]/2) <= 0.80*max(du[e],dv[e]) and
                        min(cv2.contourArea(a),cv2.contourArea(b)) <= 0.78*max(cv2.contourArea(a),cv2.contourArea(b)))
                    if ((alineados or agua_inclinada) and
                        hueco <= min((self.w,self.h)[k]*0.26, 2.4*max(du[e],dv[e]))):
                        grupos[i] = cv2.convexHull(np.vstack((a,b)))
                        grupos.pop(j)
                        cambio = True
                        break
                if cambio:
                    break
        return grupos

    def detectar(self, frame):
        semillas, apoyo = self.segmentar(frame)
        contornos_apoyo,_ = cv2.findContours(apoyo,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # Candidatos de color antes de cualquier filtro de fondo.
        candidatos = []
        self.aviso = ''
        for bebida, mascara in semillas.items():
            contornos,_ = cv2.findContours(mascara,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in self.agrupar_fragmentos(contornos,bebida):
                x,y,w,h = cv2.boundingRect(c)
                bordes = sum((x<=2, y<=2, x+w>=self.w-2, y+h>=self.h-2))
                area = cv2.contourArea(c)
                if (bordes >= 3 and area > self.w*self.h*0.28) or (bordes >= 2 and area > self.w*self.h*0.55):
                    continue
                if area < self.area_min*0.65:
                    continue
                region = np.zeros_like(mascara)
                cv2.drawContours(region,[c],-1,255,-1)
                pixeles = cv2.countNonZero(region & mascara)
                if pixeles < self.area_min:
                    continue
                d = self.candidato(region,bebida)
                if d is not None:
                    d['pixeles'] = pixeles
                    d['region'] = region
                    candidatos.append(d)
        # Suprimir pequenos tapones/fragmentos de otra bebida en la misma
        # trayectoria (ej. tapon azul de Fanta o etiqueta multicolor).
        candidatos.sort(key=lambda d: d['pixeles']*(0.65 if d['bebida']=='Agua' else 1), reverse=True)
        validos = []
        for d in candidatos:
            x,y,w,h = d['caja']
            e,k = self.eje,1-self.eje
            u,du = (x,y),(w,h)
            descartar = False
            for otro in validos:
                xx,yy,ww,hh = otro['caja']
                v,dv = (xx,yy),(ww,hh)
                solape = min(u[e]+du[e],v[e]+dv[e])-max(u[e],v[e])
                hueco = max(v[k]-(u[k]+du[k]),u[k]-(v[k]+dv[k]),0)
                if (d['bebida'] != otro['bebida'] and
                    solape >= 0.55*min(du[e],dv[e]) and
                    abs(d['centro'][e]-otro['centro'][e]) <= 0.55*max(du[e],dv[e]) and
                    hueco <= min((self.w,self.h)[k]*0.25,2.1*max(du[e],dv[e])) and
                    d['pixeles'] <= 0.45*otro['pixeles']):
                    descartar = True
                    break
            if not descartar:
                validos.append(d)
        # Intenta recuperar toda la silueta a partir del color. Solo se usa
        # un contorno de apoyo si cubre exactamente una semilla y su tamano
        # es razonable; de lo contrario se conserva el centroide del color.
        detecciones = []
        for d in validos:
            mejor = None
            for c in contornos_apoyo:
                x,y,w,h = cv2.boundingRect(c)
                a = cv2.contourArea(c)
                if a < d['area']*0.8 or a > min(self.w*self.h*0.85,d['area']*10):
                    continue
                if not (x <= d['centro'][0] <= x+w and y <= d['centro'][1] <= y+h):
                    continue
                region = np.zeros_like(apoyo)
                cv2.drawContours(region,[c],-1,255,-1)
                if cv2.countNonZero(region & d['region']) < d['area']*0.70:
                    continue
                if any(o is not d and cv2.countNonZero(region & o['region']) > o['area']*0.35 for o in validos):
                    continue
                if mejor is None or a > mejor[0]:
                    mejor = (a,region)
            region = mejor[1] if mejor else d['region']
            nuevo = self.candidato(region,d['bebida'])
            if nuevo:
                nuevo['pixeles'] = d['pixeles']
                nuevo['region'] = region
                nuevo['estimado'] = False
                detecciones.append(nuevo)
                self.mascara |= region
        detecciones = self.separar_trayectorias(detecciones)
        self.ocupada = bool(detecciones)
        # Limitar candidatos para que la asociacion no crezca con el ruido.
        detecciones.sort(key=lambda d:d['pixeles'],reverse=True)
        if not detecciones:
            self.aviso = 'Sin color de bebida detectado | M: ver mascara'
        elif len(detecciones)>2:
            self.aviso = 'Hay mas de 2 regiones de color; revisar fondo/luz'
        return detecciones[:6]

    def separar_trayectorias(self, detecciones):
        """Si dos objetos conocidos se juntan, conservar sus identidades.

        No se inventan dos objetos a partir de una sola mancha sin historial.
        Las partes estimadas no votan ni provocan un conteo por si solas.
        """
        if len(self.tracks) != 2 or len(detecciones) != 1:
            return detecciones
        d = detecciones[0]
        e = self.eje
        a,b = sorted(self.tracks,key=lambda t:self.predecir(t)[e])
        if any(self.clasificacion(t) != d['bebida'] for t in (a,b)):
            return detecciones
        pa,pb = self.predecir(a),self.predecir(b)
        x,y,w,h = d['caja']
        u,du = (x,y),(w,h)
        wa,wb = a['tamano'][e],b['tamano'][e]
        if not all(x-5 <= p[0] <= x+w+5 and y-5 <= p[1] <= y+h+5 for p in (pa,pb)):
            return detecciones
        if (du[e] < 1.25*max(wa,wb) or pb[e]-pa[e] < 0.35*min(wa,wb)
                or d['area'] < 0.65*(a['area']+b['area'])):
            return detecciones
        corte = int((pa[e]+pb[e])/2)
        if not u[e]+4 < corte < u[e]+du[e]-4:
            return detecciones
        m1,m2 = d['region'].copy(),d['region'].copy()
        if e == 0:
            m1[:,corte:] = 0
            m2[:,:corte] = 0
        else:
            m1[corte:,:] = 0
            m2[:corte,:] = 0
        partes = [self.candidato(m,d['bebida'],True) for m in (m1,m2)]
        if any(p is None for p in partes):
            return detecciones
        for parte in partes:
            parte['pixeles'] = parte['area']
        self.aviso = 'Dos bebidas juntas: seguimiento estimado'
        return partes

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
        limite = max(24, min(self.w,self.h)*0.12, max(t['tamano'])*0.50)
        limite *= 1+0.12*min(t['perdidos'],5)
        if distancia > limite:
            return None
        tipo = self.clasificacion(t) or max(BEBIDAS,key=t['votos'].get)
        if tipo and d['bebida'] and tipo != d['bebida']:
            return None
        return distancia

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
                antes = self.tracks[i]['centro'][self.eje]-self.tracks[k]['centro'][self.eje]
                ahora = detecciones[j]['centro'][self.eje]-detecciones[l]['centro'][self.eje]
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
             'lado_inicial': self.lado(d['centro']),
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
        lado = self.lado(t['centro'])
        if lado and not t['lado_inicial']:
            t['lado_inicial'] = lado
        elif lado and lado != t['lado_inicial']:
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
        libres.sort(key=lambda d:(d['bebida'] is None,-d['pixeles']))
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
            if t['cruzado'] and not t['contado'] and not t['perdidos'] and not t['estimado']:
                if bebida:
                    eventos.append(bebida)
                    t['contado'] = True
                elif not bebida:
                    self.aviso = 'Bebida sin clasificacion segura; no contabilizada'
        return eventos

    def zona_vacia(self):
        return not self.ocupada and not self.tracks

    def dibujar(self, frame):
        if self.eje == 0:
            cv2.line(frame,(self.linea,0),(self.linea,self.h-1),(0,255,255),2)
        else:
            cv2.line(frame,(0,self.linea),(self.w-1,self.linea),(0,255,255),2)
        cv2.putText(frame,'Conteo: ambos sentidos',(8,22),0,0.5,(0,255,255),2)
        for t in self.tracks:
            if t['perdidos']:
                continue
            bebida = self.clasificacion(t)
            nombre = bebida or max(BEBIDAS,key=t['votos'].get) + ' ?'
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
    print('Detector listo: pasa las bebidas atravesando la linea amarilla en cualquier sentido.')
    print('L: cambia entre linea vertical y horizontal con la zona vacia. M: mascara.')
    detector = DetectorObjetos(frame)
    if anterior is not None:
        detector.siguiente_id = anterior.siguiente_id
    return detector
