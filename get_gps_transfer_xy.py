from osgeo import gdal, osr
import numpy as np
import cv2

def get_tif_info(tif_path):
    # if tif_path.endswith('.tif') or tif_path.endswith('.TIF'):
    dataset = gdal.Open(tif_path)
    pcs = osr.SpatialReference()
    pcs.ImportFromWkt(dataset.GetProjection())
    gcs = pcs.CloneGeogCS()
    extend = dataset.GetGeoTransform()
    # im_width = dataset.RasterXSize #栅格矩阵的列数
    # im_height = dataset.RasterYSize #栅格矩阵的行数
    shape = (dataset.RasterYSize, dataset.RasterXSize)
    # else:
    #     raise "Unsupported file format"
    img = dataset.GetRasterBand(1).ReadAsArray()  # (height, width)
    # img(ndarray), gdal数据集、地理空间坐标系、投影坐标系、栅格影像大小
    return img, dataset, gcs, pcs, extend, shape

def longlat_to_xy(gcs, pcs, lon, lat):
    ct = osr.CoordinateTransformation(gcs, pcs)
    coordinates = ct.TransformPoint(lon, lat)
    return coordinates[0], coordinates[1], coordinates[2]


def xy_to_lonlat(gcs, pcs, x, y):
    ct = osr.CoordinateTransformation(gcs, pcs)
    lon, lat, _ = ct.TransformPoint(x, y)
    return lon, lat


def xy_to_rowcol(extend, x, y):
    a = np.array([[extend[1], extend[2]], [extend[4], extend[5]]])
    b = np.array([x - extend[0], y - extend[3]])

    row_col = np.linalg.solve(a, b)
    row = int(np.floor(row_col[1]))
    col = int(np.floor(row_col[0]))

    return row, col


def rowcol_to_xy(extend, row, col):
    x = extend[0] + col * extend[1] + row * extend[2]
    y = extend[3] + col * extend[4] + row * extend[5]
    return x, y

def get_value_by_coordinates(tif_pah, coordinates, coordinate_type='rowcol'):
    img, dataset, gcs, pcs, extend, shape = get_tif_info(tif_pah)

    if coordinate_type == 'rowcol':
        value = img[coordinates[0], coordinates[1]]
    elif coordinate_type == 'lonlat':
        x, y, _ = longlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
        row, col = xy_to_rowcol(extend, x, y)
        value = [row, col]
    elif coordinate_type == 'xy':
        row, col = xy_to_rowcol(extend, coordinates[0], coordinates[1])
        value = [row, col]
    else:
        raise 'coordinated_type error'
    return value

def get_xy(gcs, pcs, extend, gpsloc):
    xyxy = []
    for block in gpsloc:
        lat, lon = block
        lat, lon = eval(lat), eval(lon)
        x, y, _ = longlat_to_xy(gcs, pcs, lat, lon)
        row, col = xy_to_rowcol(extend, x, y)
        xyxy.append([row, col])
    return xyxy
def get_lonlat(gcs, pcs, extend, xyloc):
    # gpsloc = []
    
    x1, y1, x2, y2, x3, y3, x4, y4 = xyloc
    row1, col1 = rowcol_to_xy(extend, x1, y1)
    lon1, lat1 = xy_to_lonlat(pcs, gcs, row1, col1)

    row2, col2 = rowcol_to_xy(extend, x2, y2)
    lon2, lat2 = xy_to_lonlat(pcs, gcs, row2, col2)

    row3, col3 = rowcol_to_xy(extend, x3, y3)
    lon3, lat3 = xy_to_lonlat(pcs, gcs, row3, col3)

    row4, col4 = rowcol_to_xy(extend, x4, y4)
    lon4, lat4 = xy_to_lonlat(pcs, gcs, row4, col4)

    # gpsloc.append([lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4])   

    return [lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4]


if __name__ == '__main__':
    # img, dataset, gcs, pcs, extend, shape = get_tif_info('/home/qzq/下载/6593b3d9fdd1d95ccac8370b.odm_orthophoto.tif') # (-18.92076444440943, -48.06534681204571)
    inPath = '/home/qzq/下载/65a8dbcafdd1d92052704364.odm_orthophoto.tif'
    img, dataset, gcs, pcs, extend, shape = get_tif_info(inPath)
    coordinates = (-18.919259971944363, -48.06449323723823)
    x, y, _ = longlat_to_xy(gcs, pcs, coordinates[0], coordinates[1])
    row, col = xy_to_rowcol(extend, x, y) # y, x
    print(row, col)

    # x, y = 48527, 7919
    x, y = 329, 5033
    row, col = rowcol_to_xy(extend, x, y)
    lon, lat = xy_to_lonlat(pcs, gcs, int(row), int(col))
    print(row, col, lon, lat)