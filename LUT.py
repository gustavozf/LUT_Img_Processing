import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
    Author: Gustavo Zanoni Felipe
    Date:   June 2nd, 2019
'''
# tom de cinza maximo
k = 255

# ----------------------------------------------------- Look Up Table (LUT)
def identidade():
    global k
    return np.array(range(k+1))

def constante(c):
    global k
    return np.array([c for _ in range(k+1)])

def negacao():
    global k
    return np.array([k - i for i in range(k+1)])

def brilho(b):
    global k

    # cria o lut adicionando o valor b.
    # se positivo, aumenta o brilho, se for negativo diminui
    lut = np.array([i + b for i in range(k+1)])

    # coloca todos os valore entre 0 e k
    return np.clip(lut, 0, k)

def aumento_contraste(min_k, max_k):
    global k
    return np.array([ ((i-min_k)/(max_k - min_k))*k for i in range(k+1)])

# thresholding ou limiarizacao ou binarizacao
def thresholding(c):
    global k

    lut = np.zeros(k+1).astype(np.int32)
    # todos antes de 'c' sao zero, todos apos sao 1
    lut[c:] = k

    return lut

def gamma_correc(gamma):
    global k

    lut = np.array([i**gamma for i in range(k+1)]).astype(np.float64)
    lut /= lut.max()
    lut *= k

    return lut

# ----------------------------------------------------- histograma
def hist(img):
    global k

    return np.array([len(np.where(img == i)[0]) for i in np.arange(k+1)])

def hist_acum(h):
    global k
    hac = np.zeros(k+1).astype(np.int32)

    # se i == 0
    hac[0] = h[0]
    # else
    for i in range(1, k+1):
        hac[i] = hac[i-1] + h[i]

    return hac

def hist_acum_norm(hac):
    global k
    # hac[k] == tamanho da imagem
    return hac / hac[k]

def hist_equalizado(img):
    global k

    # primeiro calcula-se o histograma
    # isto eh, a frequencia de cada tom de cinza na imagem
    h = hist(img)
    # calcula-se o histograma acumulado
    hac = hist_acum(h)
    # normaliza-se o histograma acumulado
    # o dividindo pelo tamanho da imagem
    hac_nom = hist_acum_norm(hac)
    # o histograma equalizado eh construido pela multiplicacao
    # do histograma acumulado normalizado e 'k'
    hist_eq = hac_nom * k

    return hist_eq

# -------------------------------------------------------------------------- Main
img = cv2.imread('teste.png', 0)

# constante
LUT = constante(128)
img_mod = LUT[img]
cv2.imwrite('./outputs/constante.png', img_mod)

# negacao
LUT = negacao()
img_mod = LUT[img]
cv2.imwrite('./outputs/negacao.png', img_mod)

# brilho
LUT = brilho(128)
img_mod = LUT[img]
cv2.imwrite('./outputs/brilho+128.png', img_mod)

LUT = brilho(-128)
img_mod = LUT[img]
cv2.imwrite('./outputs/brilho-128.png', img_mod)

# aumento de contraste
LUT = aumento_contraste(128, 192)
img_mod = LUT[img]
cv2.imwrite('./outputs/aum_contraste.png', img_mod)

# thresholding
LUT = thresholding(128)
img_mod = LUT[img]
cv2.imwrite('./outputs/thresholding.png', img_mod)

# histograma
plt.clf()
h = hist(img)
plt.bar(range(len(h)), h)
plt.savefig('./outputs/histograma.png')

plt.clf()
heq = hist_equalizado(img)
plt.bar(range(len(heq)), heq)
plt.savefig('./outputs/histograma_equalizado.png')

# correcao de contrastes pelo hist. equal.
img_mod = heq[img]
cv2.imwrite('./outputs/contraste_heq.png', img_mod)

# correcao de gamma
LUT = gamma_correc(0.5)
img_mod = LUT[img]
cv2.imwrite('./outputs/contraste_gamma_05.png', img_mod)

plt.clf()
plt.bar(range(len(LUT)), LUT)
plt.savefig('./outputs/gamma_05.png')

LUT = gamma_correc(2)
img_mod = LUT[img]
cv2.imwrite('./outputs/contraste_gamma_20.png', img_mod)

plt.clf()
plt.bar(range(len(LUT)), LUT)
plt.savefig('./outputs/gamma_20.png')