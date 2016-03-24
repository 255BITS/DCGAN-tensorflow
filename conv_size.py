from math import floor
def size_after_conv(iW, iH, kW, kH, dW, dH, padW, padH):
    oW  = floor((iW  + 2*padW - kW) / dW + 1)
    oH = floor((iH + 2*padH - kH) / dH + 1)
    return [oW,oH]
kw=14
#kh=8
kh=kw
dw=kw
dh=dw
padh=int((kh-1)/2)
padw=int((kw-1)/2)

print(size_after_conv(441, 100, kw, kh, dw, dh, padw, padh))
