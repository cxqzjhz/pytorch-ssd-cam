from ssd import SSD
from PIL import Image

ssd = SSD()

while True:
    img = input('Input image filename:')  # img/street.jpg
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:

        # print('打印信息起始'.center(50,'-'))
        # print(image.format, image.size, image.mode)  # JPEG (1330, 1330) RGB
        # print('打印信息结束'.center(50,'-'))

        image = image.convert('RGB')
        r_image = ssd.detect_image(image)

        # print('打印信息起始'.center(50,'-'))
        # print(type(r_image))  # <class 'PIL.Image.Image'>
        # print(r_image.format, r_image.size, r_image.mode)  # None (1330, 1330) RGB
        # print('打印信息结束'.center(50,'-'))

        r_image.show()
