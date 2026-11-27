"""Deteccion y seguimiento de bebidas mediante OpenCV y NumPy.

Se analiza una superficie lisa en cada frame, sin guardar una imagen de fondo.
Se buscan siluetas de botella y se verifica el color del liquido y etiqueta.
No se identifica una bebida solamente por una mancha de color.
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
    'rojo': [((0, 95, 78), (13, 255, 255)),
             ((165, 95, 78), (179, 255, 255))],
    # El cian del liquido es mas saturado y claro que la etiqueta de agua.
    'cian': [((85, 145, 85), (108, 255, 255))],
    'amarillo': [((15, 145, 85), (37, 255, 255))],
    'naranja': [((8, 140, 80), (20, 255, 255))],
    'azul_oscuro': [((109, 40, 35), (145, 255, 255)),
                    ((100, 50, 35), (108, 145, 160))],
    'oscuro': [((0, 0, 0), (179, 255, 100))],
}
# Parametros relativos a la resolucion: ajustar solo con capturas reales.
AREA_RELATIVA_MIN = 0.0025
ASPECTO_MIN = 1.48
ASPECTO_MAX = 6.5
MIN_CIAN = 0.23
MIN_AMARILLO = 0.23
MIN_AZUL_AGUA = 0.035
COLORES = {'Coca-Cola': (0, 0, 255), 'Powerade': (255, 210, 0),
           'Fanta': (0, 145, 255), 'Agua': (255, 120, 100)}
# El area minima se refiere a la silueta, no a una mancha de color.
MIN_VOTOS = 4
MAX_PERDIDOS = 12
MAX_OBJETOS = 2


class DetectorObjetos:
    def __init__(self, frame, izquierda_derecha=True):
        self.h, self.w = frame.shape[:2]
        self.x = self.y = 0
        self.orientacion = ORIENTACION_LINEA
        self.configurar_linea()
        self.area_min = max(100, self.w*self.h*AREA_RELATIVA_MIN)
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
        validas = muestras[:,0] > 8
        if np.count_nonzero(validas) >= len(muestras)*0.25:
            muestras = muestras[validas]
        q = muestras.astype(np.int32)//16
        claves = q[:, 0]*256 + q[:, 1]*16 + q[:, 2]
        moda = np.bincount(claves, minlength=4096).argmax()
        centro = np.median(muestras[claves == moda], axis=0)
        cercanas = np.max(np.abs(muestras.astype(np.float32)-centro), axis=1) < 24
        return np.median(muestras[cercanas], axis=0).astype(np.float32)

    def segmentar(self, frame):
        """Mascara de contraste actual + evidencias de color independientes.

        Se estima el fondo por frame, nunca se necesita capturarlo antes.
        Las zonas negras del borde introducidas por undistort no cuentan.
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
        superficie = self.color_superficie(lab)
        diferencia = lab.astype(np.float32) - superficie
        croma = np.hypot(diferencia[:, :, 1], diferencia[:, :, 2])
        luz = np.abs(diferencia[:, :, 0])
        apoyo = ((croma > 14) | (luz > 24)).astype(np.uint8)*255
        # Las sombras suaves no suelen tener cromaticidad distinta del fondo.
        # Se conservan las zonas oscuras porque son utiles en Coca-Cola.
        k = max(3, min(9, (min(self.w,self.h)//90)|1))
        apoyo = cv2.morphologyEx(apoyo, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
        apoyo = cv2.morphologyEx(apoyo, cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        # Eliminar componentes de negro puro pegadas al borde (calibracion).
        negro = cv2.inRange(recorte,(0,0,0),(5,5,5))
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(negro)
        for i in range(1,n):
            x,y,w,h,area = stats[i]
            if x==0 or y==0 or x+w==self.w or y+h==self.h:
                apoyo[etiquetas==i] = 0
        return mascaras, apoyo

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

    def geometria(self, region):
        """Proporciones en cualquier orientacion; no depende del eje de paso."""
        puntos = cv2.findNonZero(region)
        if puntos is None:
            return None
        (cx,cy),(a,b),angulo = cv2.minAreaRect(puntos)
        largo,corto = max(a,b),min(a,b)
        area = cv2.countNonZero(region)
        if corto < 8 or area < self.area_min or area > self.w*self.h*0.78:
            return None
        if not ASPECTO_MIN <= largo/corto <= ASPECTO_MAX:
            return None
        if area/max(1.,largo*corto) < 0.38:
            return None
        x,y,w,h = cv2.boundingRect(puntos)
        bordes = sum((x<=2,y<=2,x+w>=self.w-2,y+h>=self.h-2))
        if bordes >= 3:
            return None
        angulo = np.deg2rad(angulo + (90 if b>a else 0))
        eje = np.array([np.cos(angulo),np.sin(angulo)])
        return np.array([cx,cy]),eje,largo,corto

    def extension(self, mascara, region, geo):
        """Fraccion longitudinal ocupada por un color (ignora pocos reflejos)."""
        puntos = cv2.findNonZero(mascara & region)
        if puntos is None or len(puntos)<10:
            return 0.
        centro,eje,largo,_ = geo
        proy = (puntos[:,0,:]-centro) @ eje
        return float((np.percentile(proy,95)-np.percentile(proy,5))/max(largo,1))

    def evidencia_agua(self, mascara, region, geo):
        """Dos zonas azules separadas a lo largo de la botella (tapon/etiqueta)."""
        m = cv2.morphologyEx(mascara & region, cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        n,_,stats,centros = cv2.connectedComponentsWithStats(m)
        centro,eje,largo,corto = geo
        minimo = max(12, cv2.countNonZero(region)*0.008)
        ids = [i for i in range(1,n) if stats[i,4]>=minimo]
        perpendicular = np.array([-eje[1],eje[0]])
        for i,j in itertools.combinations(ids,2):
            delta = centros[i]-centros[j]
            if (0.25*largo <= abs(delta@eje) <= 0.85*largo and
                abs(delta@perpendicular) <= 0.45*corto):
                return True
        return False

    def evaluar(self, region, mascaras, origen):
        geo = self.geometria(region)
        if geo is None:
            return None
        area = cv2.countNonZero(region)
        p = {k:cv2.countNonZero(m & region)/area for k,m in mascaras.items()}
        span_cian = self.extension(mascaras['cian'],region,geo)
        amarilla = mascaras['amarillo'] | mascaras['naranja']
        span_amarillo = self.extension(amarilla,region,geo)
        caliente = cv2.countNonZero(amarilla & region)/area
        # Las reglas se aplican a la silueta completa, no a cada etiqueta.
        # Un trozo azul aislado no es Powerade: se exige liquido extenso.
        bebida,confianza = None,0.
        if (MIN_CIAN<=p['cian']<=0.88 and span_cian>=0.48 and
            p['cian']>=1.8*p['azul_oscuro'] and p['cian']>=1.5*caliente):
            bebida,confianza = 'Powerade', p['cian'] + 0.35*span_cian
        elif (p['oscuro']>=0.23 and p['rojo']>=0.045 and
              self.extension(mascaras['oscuro'],region,geo)>=0.45 and
              p['cian']<0.15 and caliente<0.35):
            bebida,confianza = 'Coca-Cola',p['oscuro']+0.9*p['rojo']
        elif (MIN_AMARILLO<=p['amarillo']<=0.88 and 0.32<=caliente<=0.95 and
              span_amarillo>=0.48 and p['cian']<0.15):
            bebida,confianza = 'Fanta',caliente+0.25*span_amarillo
        elif (p['azul_oscuro']>=MIN_AZUL_AGUA and p['cian']<0.12 and
              p['amarillo']<0.16 and p['rojo']<0.10 and
              self.evidencia_agua(mascaras['azul_oscuro'],region,geo)):
            bebida,confianza = 'Agua',0.45+min(p['azul_oscuro'],0.3)
        if bebida is None:
            return None
        d = self.candidato(region,bebida)
        if d is not None:
            d.update(region=region,pixeles=area,confianza=confianza,origen=origen)
        return d

    def regiones(self, mascara):
        contornos,_ = cv2.findContours(mascara,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # Limitar el numero de regiones a analizar aunque haya mucho ruido.
        for c in sorted(contornos,key=cv2.contourArea,reverse=True)[:20]:
            if cv2.contourArea(c)<self.area_min*0.8:
                continue
            region = np.zeros_like(mascara)
            cv2.drawContours(region,[c],-1,255,-1)
            yield region

    def propuestas_agua(self, azul, apoyo):
        """Recupera agua transparente cuando el fondo se ve a traves de ella.

        Exige dos piezas azules alineadas y una envolvente alargada. Una
        etiqueta aislada nunca crea una propuesta de agua o de Powerade.
        """
        m = cv2.morphologyEx(azul,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        cs,_ = cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in sorted(cs,key=cv2.contourArea,reverse=True)[:12]
              if cv2.contourArea(c)>=max(18,self.area_min*0.055)]
        for a,b in itertools.combinations(cs,2):
            aa,ab = cv2.contourArea(a),cv2.contourArea(b)
            if min(aa,ab)/max(aa,ab)<0.12:
                continue
            # Un tapon y una etiqueta son de tamanos distintos.
            if min(aa,ab)/max(aa,ab)>0.88:
                continue
            hull = cv2.convexHull(np.vstack((a,b)))
            region = np.zeros_like(azul)
            cv2.drawContours(region,[hull],-1,255,-1)
            geo = self.geometria(region)
            if geo is None:
                continue
            # No unir dos etiquetas de botellas distintas, alineadas con el paso.
            _,eje,_,_ = geo
            if abs(eje[self.eje])>0.70:
                continue
            if cv2.countNonZero(region & apoyo)<0.12*cv2.countNonZero(region):
                continue
            yield region

    def detectar(self, frame):
        mascaras,apoyo = self.segmentar(frame)
        self.aviso = ''
        self.mascara = np.zeros_like(apoyo)
        candidatos = []
        for region in self.regiones(apoyo):
            d = self.evaluar(region,mascaras,'silueta')
            if d:
                candidatos.append(d)
        for region in self.propuestas_agua(mascaras['azul_oscuro'],apoyo):
            d = self.evaluar(region,mascaras,'agua')
            if d and d['bebida']=='Agua':
                candidatos.append(d)
        # Preferir una silueta completa a una parte de la misma botella.
        candidatos.sort(key=lambda d:(d['origen']=='silueta',d['confianza'],d['area']),reverse=True)
        validos = []
        for d in candidatos:
            if any(cv2.countNonZero(d['region'] & o['region']) > 0.30*min(d['area'],o['area'])
                   for o in validos):
                continue
            # Suprimir una posible pareja de azul que pertenece a Fanta/Powerade.
            if d['origen']=='agua' and any(o['bebida']!='Agua' and
                   cv2.countNonZero(d['region'] & o['region'])>0.10*d['area'] for o in validos):
                continue
            validos.append(d)
            self.mascara |= d['region']
        detecciones = self.separar_trayectorias(validos)
        self.ocupada = bool(detecciones)
        if not detecciones:
            # Muestra el contraste aunque la clasificacion no sea valida.
            self.mascara = apoyo
            self.aviso = 'Sin botella confirmada | M: revisar segmentacion'
        elif len(detecciones)>2:
            self.aviso = 'Demasiadas regiones validas; despejar superficie'
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
            if self.clasificacion(t) is not None:
                return None
            distancia += limite*0.4
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
             'cruzado': False, 'estimado': False, 'inicio': d['centro'][self.eje]}
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
            recorrido = abs(t['centro'][self.eje]-t['inicio'])
            minimo = max(2*self.margen,0.06*(self.w if self.eje==0 else self.h))
            if t['cruzado'] and recorrido>=minimo and not t['contado'] and not t['perdidos'] and not t['estimado']:
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
