from osgeo import ogr
import numpy as np
from shapely.geos import lgeos
import pandas as pd
from numba import jit
from rtree import index

GEOS_HANDLE = lgeos.geos_handle
GEOSGeomFromWKT = lgeos._lgeos.GEOSGeomFromWKT_r
GEOSWithin_ctypes = lgeos._lgeos.GEOSWithin_r
GEOSContains_ctypes = lgeos._lgeos.GEOSContains_r
GEOSCoordSeq_getX_ctypes = lgeos._lgeos.GEOSCoordSeq_getX_r
GEOSCoordSeq_getY_ctypes = lgeos._lgeos.GEOSCoordSeq_getY_r
GEOSCoordSeq_getSize_ctypes = lgeos._lgeos.GEOSCoordSeq_getSize_r
GEOSGeom_getCoordSeq_ctypes = lgeos._lgeos.GEOSGeom_getCoordSeq_r
GEOSEnvelope_ctypes = lgeos._lgeos.GEOSEnvelope_r
GEOSGetExteriorRing_ctypes = lgeos._lgeos.GEOSGetExteriorRing_r
GEOSGetCentroid = lgeos._lgeos.GEOSGetCentroid_r


def shape_to_geodata(fields_list, shapefile):
    # load shapefile
    ds = ogr.Open(shapefile)
    layer = ds.GetLayer()

    # feat_list stores features from fields list, geom_list stores geometry as point
    features_list = []
    geom_list = []
    for f in layer:
        geom = f.GetGeometryRef()
        if geom is not None:
            wkt = geom.ExportToWkt()
            geom_list.append(
                GEOSGeomFromWKT(GEOS_HANDLE, wkt.encode('ascii')))  # create C_pointers corresponding to each geom
            features_list.append([f.GetField(item) for item in fields_list])

    pandas_df = pd.DataFrame(features_list, columns=fields_list)
    pandas_df['geometry'] = geom_list
    return pandas_df


@jit(nopython=True)
def get_coord(array_geom):
    cseq = GEOSGeom_getCoordSeq_ctypes(GEOS_HANDLE, int(array_geom))
    dx = np.zeros(0)
    dy = np.zeros(0)
    c_len = np.zeros(1, dtype=np.intc)
    GEOSCoordSeq_getSize_ctypes(GEOS_HANDLE, cseq, c_len.ctypes.data)
    for j in np.arange(c_len[0]):
        GEOSCoordSeq_getX_ctypes(GEOS_HANDLE, cseq, j, dx.ctypes.data)
        GEOSCoordSeq_getY_ctypes(GEOS_HANDLE, cseq, j, dy.ctypes.data)

    return (dx[0], dy[0])


@jit(nopython=True)
def get_coord_array(array_geom):
    n = array_geom.shape[0]
    out_array = np.empty((n, 2))
    for i in np.arange(n):
        coord = get_coord(array_geom[i])
        out_array[i, 0] = coord[0]
        out_array[i, 1] = coord[1]
    return out_array


def get_centroid(p_geom):
    n = p_geom.shape[0]
    centroid = np.zeros(n, dtype=np.intc)

    for i in np.arange(n):
        centroid[i] = GEOSGetCentroid(GEOS_HANDLE, int(p_geom[i]))

    return centroid


@jit(nopython=True)
def get_bounds(pol_geom):
    envelope = GEOSEnvelope_ctypes(GEOS_HANDLE, int(pol_geom))
    g = GEOSGetExteriorRing_ctypes(GEOS_HANDLE, envelope)
    cseq = GEOSGeom_getCoordSeq_ctypes(GEOS_HANDLE, g)

    cs_len = np.zeros(1, dtype=np.intc)
    GEOSCoordSeq_getSize_ctypes(GEOS_HANDLE, cseq, cs_len.ctypes.data)
    minx = 1.e+20
    maxx = -1e+20
    miny = 1.e+20
    maxy = -1e+20
    temp = np.zeros(0)
    for i in range(cs_len[0]):
        GEOSCoordSeq_getX_ctypes(GEOS_HANDLE, cseq, i, temp.ctypes.data)
        x = temp[0]
        if x < minx: minx = x
        if x > maxx: maxx = x
        GEOSCoordSeq_getY_ctypes(GEOS_HANDLE, cseq, i, temp.ctypes.data)
        y = temp[0]
        if y < miny: miny = y
        if y > maxy: maxy = y
    return (minx, miny, maxx, maxy)


def spatial_index(pol_geom):
    idx = index.Index()
    m = pol_geom.shape[0]
    for i in np.arange(m):
        idx.insert(i, get_bounds(pol_geom[i]))
    return idx


def point_in_boundingbox(a_geom, idx):
    n = a_geom.shape[0]
    coord_array = get_coord_array(a_geom)
    res = np.zeros((n, 10))
    for i in np.arange(n):
        count = 0
        for j in idx.intersection((coord_array[i, 0], coord_array[i, 1])):
            res[i, count] = j + 1
            count += 1
    return res


@jit(nopython=True)
def within_numba(pol_geom, pt_geom, intersection):
    n = pt_geom.shape[0]
    res = np.zeros(n) - 1

    for i in np.arange(n):
        pol = intersection[i]
        m = pol.shape[0]

        for j in np.arange(m):
            if pol[j] > 0:
                if GEOSWithin_ctypes(GEOS_HANDLE, int(pt_geom[i]), int(pol_geom[int(pol[j] - 1)])):
                    res[i] = pol[j] - 1
                    break
    return res


def sjoin_within(points, polygon, feature_polygon, how=None):
    polygon = polygon.set_index(np.arange(len(polygon)))

    pol_geom = np.array(polygon.geometry)
    idx = spatial_index(pol_geom)

    pt_geom = np.array(points.geometry)
    res = point_in_boundingbox(pt_geom, idx)
    point_in_polygon = within_numba(pol_geom, pt_geom, res)
    points['pol_number'] = point_in_polygon

    if how == 'left':
        df = pd.merge(points, polygon[feature_polygon], left_on='pol_number', right_index=True, how='left')
    elif how == 'outer':
        df = pd.merge(points, polygon[feature_polygon], left_on='pol_number', right_index=True, how='outer')
    else:
        df = pd.merge(points, polygon[feature_polygon], left_on='pol_number', right_index=True, how='inner')
    return df.drop('pol_number', axis=1)

def centroid_table(table):
    centroid = get_centroid(np.array(table.geometry))
    table = table.drop('geometry', axis=1)
    table['geometry'] = centroid

    return table

