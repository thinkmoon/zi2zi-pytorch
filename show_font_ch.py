from fontTools.ttLib import TTFont

font = TTFont('font/c.ttf')
unicode_map = font['cmap'].tables[0].ttFont.getBestCmap()
glyf_map = font['glyf']
words = '一二龍三四'

for char_code in unicode_map.keys():
    if len(glyf_map[unicode_map[char_code]].getCoordinates(0)[0]) > 0:
        print(chr(char_code))
        with open("text.txt","a",encoding="utf-8") as file:
            file.write(chr(char_code))