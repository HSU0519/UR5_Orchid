def hex2rgb(hex):
    rgb = []
    for i in (4, 2, 0):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)
    
    return tuple(rgb)

if __name__ == "__main__":
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']     
    
    color = hex2rgb(HEX[4 % 20])
    print(color)
    
