import os

def load_cpr_coord_map(file_path):
    f = open(file_path, mode='rb')
    a = f.read()
    w, h, c, t = struct.unpack('iiii', a[:16])
#    x, y, z, c = struct.unpack('iiii', a[:16])
#    print(x,y,z,c)
    assert c == 3 and t == 10, 'The third and fourth items of cpr coor map should be 3 and 10'
#    assert c == 2 and t == 10
    maps = struct.unpack('f' * w * h * c, a[16:])
    maps = numpy.float32(maps).reshape(h, w, c)

 #   maps = struct.unpack('f' * x * y * z * c, a[20:])
#    maps = numpy.float32(maps).reshape(z, y, x, c)

  #  print(maps)
    return maps

if __name__ == '__main__':
    path = '/data1/zhangyongming/cta_new'#'/data1/zhangyongming/cta/b4_lz493/cpr_scpr_lumen_s18_n10/'
    bat_list = os.listdir(path)
    for bat in bat_list:
        load_cpr_coord_map(bat+'/cpr')