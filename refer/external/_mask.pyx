# distutils: language = c
# distutils: sources = external/maskApi.c

#**************************************************************************
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]
#**************************************************************************

__author__ = 'tsungyi'

# import both Python-level and C-level symbols of Numpy
# the API uses Numpy to interface C and Python
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# intialized Numpy. must do.
np.import_array()

# import numpy C function
# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memoery management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

# Declare the prototype of the C functions in MaskApi.h
cdef extern from "maskApi.h":
    ctypedef unsigned int uint
    ctypedef unsigned long siz
    ctypedef unsigned char byte
    ctypedef double* BB
    ctypedef struct RLE:
        siz h,
        siz w,
        siz m,
        uint* cnts,
    void rlesInit( RLE **R, siz n )
    void rleEncode( RLE *R, const byte *M, siz h, siz w, siz n )
    void rleDecode( const RLE *R, byte *mask, siz n )
    void rleMerge( const RLE *R, RLE *M, siz n, bint intersect )
    void rleArea( const RLE *R, siz n, uint *a )
    void rleIou( RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o )
    void bbIou( BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o )
    void rleToBbox( const RLE *R, BB bb, siz n )
    void rleFrBbox( RLE *R, const BB bb, siz h, siz w, siz n )
    void rleFrPoly( RLE *R, const double *xy, siz k, siz h, siz w )
    char* rleToString( const RLE *R )
    void rleFrString( RLE *R, char *s, siz h, siz w )

# python class to wrap RLE array in C
# the class handles the memory allocation and deallocation
cdef class RLEs:
    cdef RLE *_R
    cdef siz _n

    def __cinit__(self, siz n =0):
        rlesInit(&self._R, n)
        self._n = n

    # free the RLE array here
    def __dealloc__(self):
        if self._R is not NULL:
            for i in range(self._n):
                free(self._R[i].cnts)
            free(self._R)
    def __getattr__(self, key):
        if key == 'n':
            return self._n
        raise AttributeError(key)

# python class to wrap Mask array in C
# the class handles the memory allocation and deallocation
cdef class Masks:
    cdef byte *_mask
    cdef siz _h
    cdef siz _w
    cdef siz _n

    def __cinit__(self, h, w, n):
        self._mask = <byte*> malloc(h*w*n* sizeof(byte))
        self._h = h
        self._w = w
        self._n = n
    # def __dealloc__(self):
        # the memory management of _mask has been passed to np.ndarray
        # it doesn't need to be freed here

    # called when passing into np.array() and return an np.ndarray in column-major order
    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self._h*self._w*self._n
        # Create a 1D array, and reshape it to fortran/Matlab column-major array
        ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self._mask).reshape((self._h, self._w, self._n), order='F')
        # The _mask allocated by Masks is now handled by ndarray
        PyArray_ENABLEFLAGS(ndarray, np.NPY_OWNDATA)
        return ndarray

# internal conversion from Python RLEs object to compressed RLE format
def _toString(RLEs Rs):
    cdef siz n = Rs.n
    cdef bytes py_string
    cdef char* c_string
    objs = []
    for i in range(n):
        c_string = rleToString( <RLE*> &Rs._R[i] )
        py_string = c_string
        objs.append({
            'size': [Rs._R[i].h, Rs._R[i].w],
            'counts': py_string
        })
        free(c_string)
    return objs

# internal conversion from compressed RLE format to Python RLEs object
def _frString(rleObjs):
    cdef siz n = len(rleObjs)
    Rs = RLEs(n)
    cdef bytes py_string
    cdef char* c_string
    for i, obj in enumerate(rleObjs):
        py_string = str(obj['counts'])
        c_string = py_string
        rleFrString( <RLE*> &Rs._R[i], <char*> c_string, obj['size'][0], obj['size'][1] )
    return Rs

# encode mask to RLEs objects
# list of RLE string can be generated by RLEs member function
def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask):
    h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
    cdef RLEs Rs = RLEs(n)
    rleEncode(Rs._R,<byte*>mask.data,h,w,n)
    objs = _toString(Rs)
    return objs

# decode mask from compressed list of RLE string or RLEs object
def decode(rleObjs):
    cdef RLEs Rs = _frString(rleObjs)
    h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
    masks = Masks(h, w, n)
    rleDecode( <RLE*>Rs._R, masks._mask, n );
    return np.array(masks)

def merge(rleObjs, bint intersect=0):
    cdef RLEs Rs = _frString(rleObjs)
    cdef RLEs R = RLEs(1)
    rleMerge(<RLE*>Rs._R, <RLE*> R._R, <siz> Rs._n, intersect)
    obj = _toString(R)[0]
    return obj

def area(rleObjs):
    cdef RLEs Rs = _frString(rleObjs)
    cdef uint* _a = <uint*> malloc(Rs._n* sizeof(uint))
    rleArea(Rs._R, Rs._n, _a)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> Rs._n
    a = np.array((Rs._n, ), dtype=np.uint8)
    a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)
    PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
    return a

# iou computation. support function overload (RLEs-RLEs and bbox-bbox).
def iou( dt, gt, pyiscrowd ):
    def _preproc(objs):
        if len(objs) == 0:
            return objs
        if type(objs) == np.ndarray:
            if len(objs.shape) == 1:
                objs = objs.reshape((objs[0], 1))
            # check if it's Nx4 bbox
            if not len(objs.shape) == 2 or not objs.shape[1] == 4:
                raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
            objs = objs.astype(np.double)
        elif type(objs) == list:
            # check if list is in box format and convert it to np.ndarray
            isbox = np.all(np.array([(len(obj)==4) and ((type(obj)==list) or (type(obj)==np.ndarray)) for obj in objs]))
            isrle = np.all(np.array([type(obj) == dict for obj in objs]))
            if isbox:
                objs = np.array(objs, dtype=np.double)
                if len(objs.shape) == 1:
                    objs = objs.reshape((1,objs.shape[0]))
            elif isrle:
                objs = _frString(objs)
            else:
                raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')
        else:
            raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
        return objs
    def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t,  ndim=1] _iou):
        rleIou( <RLE*> dt._R, <RLE*> gt._R, m, n, <byte*> iscrowd.data, <double*> _iou.data )
    def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
        bbIou( <BB> dt.data, <BB> gt.data, m, n, <byte*> iscrowd.data, <double*>_iou.data )
    def _len(obj):
        cdef siz N = 0
        if type(obj) == RLEs:
            N = obj.n
        elif len(obj)==0:
            pass
        elif type(obj) == np.ndarray:
            N = obj.shape[0]
        return N
    # convert iscrowd to numpy array
    cdef np.ndarray[np.uint8_t, ndim=1] iscrowd = np.array(pyiscrowd, dtype=np.uint8)
    # simple type checking
    cdef siz m, n
    dt = _preproc(dt)
    gt = _preproc(gt)
    m = _len(dt)
    n = _len(gt)
    if m == 0 or n == 0:
        return []
    if not type(dt) == type(gt):
        raise Exception('The dt and gt should have the same data type, either RLEs, list or np.ndarray')

    # define local variables
    cdef double* _iou = <double*> 0
    cdef np.npy_intp shape[1]
    # check type and assign iou function
    if type(dt) == RLEs:
        _iouFun = _rleIou
    elif type(dt) == np.ndarray:
        _iouFun = _bbIou
    else:
        raise Exception('input data type not allowed.')
    _iou = <double*> malloc(m*n* sizeof(double))
    iou = np.zeros((m*n, ), dtype=np.double)
    shape[0] = <np.npy_intp> m*n
    iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)
    PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
    _iouFun(dt, gt, iscrowd, m, n, iou)
    return iou.reshape((m,n), order='F')

def toBbox( rleObjs ):
    cdef RLEs Rs = _frString(rleObjs)
    cdef siz n = Rs.n
    cdef BB _bb = <BB> malloc(4*n* sizeof(double))
    rleToBbox( <const RLE*> Rs._R, _bb, n )
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> 4*n
    bb = np.array((1,4*n), dtype=np.double)
    bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb).reshape((n, 4))
    PyArray_ENABLEFLAGS(bb, np.NPY_OWNDATA)
    return bb

def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w ):
    cdef siz n = bb.shape[0]
    Rs = RLEs(n)
    rleFrBbox( <RLE*> Rs._R, <const BB> bb.data, h, w, n )
    objs = _toString(Rs)
    return objs

def frPoly( poly, siz h, siz w ):
    cdef np.ndarray[np.double_t, ndim=1] np_poly
    n = len(poly)
    Rs = RLEs(n)
    for i, p in enumerate(poly):
        np_poly = np.array(p, dtype=np.double, order='F')
        rleFrPoly( <RLE*>&Rs._R[i], <const double*> np_poly.data, <unsigned long>(len(np_poly) / 2), h, w )
    objs = _toString(Rs)
    return objs

def frUncompressedRLE(ucRles, siz h, siz w):
    cdef np.ndarray[np.uint32_t, ndim=1] cnts
    cdef RLE R
    cdef uint *data
    n = len(ucRles)
    objs = []
    for i in range(n):
        Rs = RLEs(1)
        cnts = np.array(ucRles[i]['counts'], dtype=np.uint32)
        # time for malloc can be saved here but it's fine
        data = <uint*> malloc(len(cnts)* sizeof(uint))
        for j in range(len(cnts)):
            data[j] = <uint> cnts[j]
        R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), <uint*> data)
        Rs._R[0] = R
        objs.append(_toString(Rs)[0])
    return objs

def frPyObjects(pyobj, siz h, w):
    if type(pyobj) == np.ndarray:
        objs = frBbox(pyobj, h, w )
    elif type(pyobj) == list and len(pyobj[0]) == 4:
        objs = frBbox(pyobj, h, w )
    elif type(pyobj) == list and len(pyobj[0]) > 4:
        objs = frPoly(pyobj, h, w )
    elif type(pyobj) == list and type(pyobj[0]) == dict:
        objs = frUncompressedRLE(pyobj, h, w)
    else:
        raise Exception('input type is not supported.')
    return objs
