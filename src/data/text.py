"""
text.py

Gestiona la conversión entre texto y números.
El modelo no entiende letras — necesita números.
Este módulo define el vocabulario y las funciones
de conversión en ambas direcciones.
"""

# ── Vocabulario ──────────────────────────────────────────────
# todos los caracteres que el modelo puede manejar
# si un carácter no está aquí, se eliminará del texto

PAD   = '_'   # padding — relleno para igualar longitudes en un batch
EOS   = '~'   # end of sequence — marca el final de una frase

VOCABULARIO = (
    PAD                          # índice 0 — padding
    + EOS                        # índice 1 — fin de frase
    + ' '                        # índice 2 — espacio
    + 'abcdefghijklmnñopqrstuvwxyz'
    + 'áéíóúüà'
    + ',.;:!?¿¡-\''
)

# longitud del vocabulario — cuántos caracteres distintos hay
VOCAB_SIZE = len(VOCABULARIO)

# diccionario carácter → índice
# {'_': 0, '~': 1, ' ': 2, 'a': 3, ...}
CHAR_A_IDX = {char: idx for idx, char in enumerate(VOCABULARIO)}

# diccionario índice → carácter
# {0: '_', 1: '~', 2: ' ', 3: 'a', ...}
IDX_A_CHAR = {idx: char for idx, char in enumerate(VOCABULARIO)}

# índices especiales — los usaremos frecuentemente
PAD_IDX = CHAR_A_IDX[PAD]   # 0
EOS_IDX = CHAR_A_IDX[EOS]   # 1


def normalizar(texto):
    """
    Limpia el texto antes de convertirlo a números.

    - Convierte a minúsculas
    - Elimina caracteres que no están en el vocabulario
    - Elimina espacios duplicados

    Ejemplo:
        "¡HOLA!  ¿Qué tal?"  →  "¡hola! ¿qué tal?"
    """
    # minúsculas
    texto = texto.lower()

    # eliminar caracteres fuera del vocabulario
    texto = ''.join(c for c in texto if c in CHAR_A_IDX)

    # eliminar espacios duplicados
    while '  ' in texto:
        texto = texto.replace('  ', ' ')

    # eliminar espacios al inicio y al final
    texto = texto.strip()

    return texto


def texto_a_indices(texto, normalizar_texto=True):
    """
    Convierte una cadena de texto en una lista de índices numéricos.
    Añade el token EOS al final para que el modelo sepa dónde terminar.

    Ejemplo:
        "hola"  →  [10, 17, 14, 3, 1]
                    h    o    l   a  ~
    """
    if normalizar_texto:
        texto = normalizar(texto)

    # convertir cada carácter a su índice
    indices = [CHAR_A_IDX[c] for c in texto]

    # añadir token de fin de frase
    indices.append(EOS_IDX)

    return indices


def indices_a_texto(indices):
    """
    Convierte una lista de índices de vuelta a texto.
    Para depuración — ver qué está procesando el modelo.

    Ejemplo:
        [10, 17, 14, 3, 1]  →  "hola~"
    """
    return ''.join(IDX_A_CHAR[idx] for idx in indices)


def info_vocabulario():
    """
    Muestra información del vocabulario por consola.
    Útil para verificar que todo está bien configurado.
    """
    print(f"Vocabulario completo: '{VOCABULARIO}'")
    print(f"Tamaño del vocabulario: {VOCAB_SIZE} caracteres")
    print(f"Índice de PAD  '{PAD}': {PAD_IDX}")
    print(f"Índice de EOS  '{EOS}': {EOS_IDX}")


if __name__ == '__main__':

    info_vocabulario()
    print()

    frases = [
        "Hola, ¿en qué te puedo ayudar?",
        "Buenos días, bienvenido.",
        "TEXTO EN MAYÚSCULAS con símbolos raros ###",
    ]

    for frase in frases:
        normalizada = normalizar(frase)
        indices     = texto_a_indices(frase)
        recuperada  = indices_a_texto(indices)

        print(f"Original:    '{frase}'")
        print(f"Normalizada: '{normalizada}'")
        print(f"Índices:     {indices}")
        print(f"Recuperada:  '{recuperada}'")
        print()