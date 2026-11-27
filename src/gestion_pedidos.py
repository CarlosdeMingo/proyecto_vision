"""Cantidades solicitadas, entregadas y sobrantes. Entrada por terminal."""
from deteccion_objetos import BEBIDAS


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
