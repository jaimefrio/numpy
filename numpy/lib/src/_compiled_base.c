#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <Python.h>
#include <structmember.h>
#include <string.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_3kcompat.h"
#include "npy_config.h"
#include "numpy/ufuncobject.h"


/*
 * Returns -1 if the array is monotonic decreasing,
 * +1 if the array is monotonic increasing,
 * and 0 if the array is not monotonic.
 */
static int
check_array_monotonic(const double *a, npy_int lena)
{
    npy_intp i;
    double next;
    double last = a[0];

    /* Skip repeated values at the beginning of the array */
    for (i = 1; (i < lena) && (a[i] == last); i++);

    if (i == lena) {
        /* all bin edges hold the same value */
        return 1;
    }

    next = a[i];
    if (last < next) {
        /* Possibly monotonic increasing */
        for (i += 1; i < lena; i++) {
            last = next;
            next = a[i];
            if (last > next) {
                return 0;
            }
        }
        return 1;
    }
    else {
        /* last > next, possibly monotonic decreasing */
        for (i += 1; i < lena; i++) {
            last = next;
            next = a[i];
            if (last < next) {
                return 0;
            }
        }
        return -1;
    }
}

/* Find the minimum and maximum of an integer array */
static void
minmax(const npy_intp *data, npy_intp data_len, npy_intp *mn, npy_intp *mx)
{
    npy_intp min = *data;
    npy_intp max = *data;

    while (--data_len) {
        const npy_intp val = *(++data);
        if (val < min) {
            min = val;
        }
        else if (val > max) {
            max = val;
        }
    }

    *mn = min;
    *mx = max;
}

/*
 * arr_bincount is registered as bincount.
 *
 * bincount accepts one, two or three arguments. The first is an array of
 * non-negative integers The second, if present, is an array of weights,
 * which must be promotable to double. Call these arguments list and
 * weight. Both must be one-dimensional with len(weight) == len(list). If
 * weight is not present then bincount(list)[i] is the number of occurrences
 * of i in list.  If weight is present then bincount(self,list, weight)[i]
 * is the sum of all weight[j] where list [j] == i.  Self is not used.
 * The third argument, if present, is a minimum length desired for the
 * output array.
 */
static PyObject *
arr_bincount(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyArray_Descr *type;
    PyObject *list = NULL, *weight=Py_None, *mlength=Py_None;
    PyArrayObject *lst=NULL, *ans=NULL, *wts=NULL;
    npy_intp *numbers, *ians, len , mx, mn, ans_size, minlength;
    int i;
    double *weights , *dans;
    static char *kwlist[] = {"list", "weights", "minlength", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO",
                kwlist, &list, &weight, &mlength)) {
            goto fail;
    }

    lst = (PyArrayObject *)PyArray_ContiguousFromAny(list, NPY_INTP, 1, 1);
    if (lst == NULL) {
        goto fail;
    }
    len = PyArray_SIZE(lst);
    type = PyArray_DescrFromType(NPY_INTP);

    if (mlength == Py_None) {
        minlength = 0;
    }
    else {
        minlength = PyArray_PyIntAsIntp(mlength);
        if (minlength <= 0) {
            if (!PyErr_Occurred()) {
                PyErr_SetString(PyExc_ValueError,
                                "minlength must be positive");
            }
            goto fail;
        }
    }

    /* handle empty list */
    if (len == 0) {
        if (!(ans = (PyArrayObject *)PyArray_Zeros(1, &minlength, type, 0))){
            goto fail;
        }
        Py_DECREF(lst);
        return (PyObject *)ans;
    }

    numbers = (npy_intp *) PyArray_DATA(lst);
    minmax(numbers, len, &mn, &mx);
    if (mn < 0) {
        PyErr_SetString(PyExc_ValueError,
                "The first argument of bincount must be non-negative");
        goto fail;
    }
    ans_size = mx + 1;
    if (mlength != Py_None) {
        if (ans_size < minlength) {
            ans_size = minlength;
        }
    }
    if (weight == Py_None) {
        ans = (PyArrayObject *)PyArray_Zeros(1, &ans_size, type, 0);
        if (ans == NULL) {
            goto fail;
        }
        ians = (npy_intp *)(PyArray_DATA(ans));
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < len; i++)
            ians [numbers [i]] += 1;
        NPY_END_ALLOW_THREADS;
        Py_DECREF(lst);
    }
    else {
        wts = (PyArrayObject *)PyArray_ContiguousFromAny(
                                                weight, NPY_DOUBLE, 1, 1);
        if (wts == NULL) {
            goto fail;
        }
        weights = (double *)PyArray_DATA (wts);
        if (PyArray_SIZE(wts) != len) {
            PyErr_SetString(PyExc_ValueError,
                    "The weights and list don't have the same length.");
            goto fail;
        }
        type = PyArray_DescrFromType(NPY_DOUBLE);
        ans = (PyArrayObject *)PyArray_Zeros(1, &ans_size, type, 0);
        if (ans == NULL) {
            goto fail;
        }
        dans = (double *)PyArray_DATA(ans);
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < len; i++) {
            dans[numbers[i]] += weights[i];
        }
        NPY_END_ALLOW_THREADS;
        Py_DECREF(lst);
        Py_DECREF(wts);
    }
    return (PyObject *)ans;

fail:
    Py_XDECREF(lst);
    Py_XDECREF(wts);
    Py_XDECREF(ans);
    return NULL;
}

/*
 * digitize(x, bins, right=False) returns an array of integers the same length
 * as x. The values i returned are such that bins[i - 1] <= x < bins[i] if
 * bins is monotonically increasing, or bins[i - 1] > x >= bins[i] if bins
 * is monotonically decreasing.  Beyond the bounds of bins, returns either
 * i = 0 or i = len(bins) as appropriate. If right == True the comparison
 * is bins [i - 1] < x <= bins[i] or bins [i - 1] >= x > bins[i]
 */
static PyObject *
arr_digitize(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj_x = NULL;
    PyObject *obj_bins = NULL;
    PyArrayObject *arr_x = NULL;
    PyArrayObject *arr_bins = NULL;
    PyObject *ret = NULL;
    npy_intp len_bins;
    int monotonic, right = 0;
    NPY_BEGIN_THREADS_DEF

    static char *kwlist[] = {"x", "bins", "right", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|i", kwlist,
                                     &obj_x, &obj_bins, &right)) {
        goto fail;
    }

    /* PyArray_SearchSorted will make `x` contiguous even if we don't */
    arr_x = (PyArrayObject *)PyArray_FROMANY(obj_x, NPY_DOUBLE, 0, 0,
                                             NPY_ARRAY_CARRAY_RO);
    if (arr_x == NULL) {
        goto fail;
    }

    /* TODO: `bins` could be strided, needs change to check_array_monotonic */
    arr_bins = (PyArrayObject *)PyArray_FROMANY(obj_bins, NPY_DOUBLE, 1, 1,
                                               NPY_ARRAY_CARRAY_RO);
    if (arr_bins == NULL) {
        goto fail;
    }

    len_bins = PyArray_SIZE(arr_bins);
    if (len_bins == 0) {
        PyErr_SetString(PyExc_ValueError, "bins must have non-zero length");
        goto fail;
    }

    NPY_BEGIN_THREADS_THRESHOLDED(len_bins)
    monotonic = check_array_monotonic((const double *)PyArray_DATA(arr_bins),
                                      len_bins);
    NPY_END_THREADS

    if (monotonic == 0) {
        PyErr_SetString(PyExc_ValueError,
                        "bins must be monotonically increasing or decreasing");
        goto fail;
    }

    /* PyArray_SearchSorted needs an increasing array */
    if (monotonic == - 1) {
        PyArrayObject *arr_tmp = NULL;
        npy_intp shape = PyArray_DIM(arr_bins, 0);
        npy_intp stride = -PyArray_STRIDE(arr_bins, 0);
        void *data = (void *)(PyArray_BYTES(arr_bins) - stride * (shape - 1));

        arr_tmp = (PyArrayObject *)PyArray_New(&PyArray_Type, 1, &shape,
                                               NPY_DOUBLE, &stride, data, 0,
                                               PyArray_FLAGS(arr_bins), NULL);
        if (!arr_tmp) {
            goto fail;
        }

        if (PyArray_SetBaseObject(arr_tmp, (PyObject *)arr_bins) < 0) {

            Py_DECREF(arr_tmp);
            goto fail;
        }
        arr_bins = arr_tmp;
    }

    ret = PyArray_SearchSorted(arr_bins, (PyObject *)arr_x,
                               right ? NPY_SEARCHLEFT : NPY_SEARCHRIGHT, NULL);
    if (!ret) {
        goto fail;
    }

    /* If bins is decreasing, ret has bins from end, not start */
    if (monotonic == -1) {
        npy_intp *ret_data =
                        (npy_intp *)PyArray_DATA((PyArrayObject *)ret);
        npy_intp len_ret = PyArray_SIZE((PyArrayObject *)ret);

        NPY_BEGIN_THREADS_THRESHOLDED(len_ret)
        while (len_ret--) {
            *ret_data = len_bins - *ret_data;
            ret_data++;
        }
        NPY_END_THREADS
    }

    fail:
        Py_XDECREF(arr_x);
        Py_XDECREF(arr_bins);
        return ret;
}

static char arr_insert__doc__[] = "Insert vals sequentially into equivalent 1-d positions indicated by mask.";

/*
 * Insert values from an input array into an output array, at positions
 * indicated by a mask. If the arrays are of dtype object (indicated by
 * the objarray flag), take care of reference counting.
 *
 * This function implements the copying logic of arr_insert() defined
 * below.
 */
static void
arr_insert_loop(char *mptr, char *vptr, char *input_data, char *zero,
                char *avals_data, int melsize, int delsize, int objarray,
                int totmask, int numvals, int nd, npy_intp *instrides,
                npy_intp *inshape)
{
    int mindx, rem_indx, indx, i, copied;

    /*
     * Walk through mask array, when non-zero is encountered
     * copy next value in the vals array to the input array.
     * If we get through the value array, repeat it as necessary.
     */
    copied = 0;
    for (mindx = 0; mindx < totmask; mindx++) {
        if (memcmp(mptr,zero,melsize) != 0) {
            /* compute indx into input array */
            rem_indx = mindx;
            indx = 0;
            for (i = nd - 1; i > 0; --i) {
                indx += (rem_indx % inshape[i]) * instrides[i];
                rem_indx /= inshape[i];
            }
            indx += rem_indx * instrides[0];
            /* fprintf(stderr, "mindx = %d, indx=%d\n", mindx, indx); */
            /* Copy value element over to input array */
            memcpy(input_data+indx,vptr,delsize);
            if (objarray) {
                Py_INCREF(*((PyObject **)vptr));
            }
            vptr += delsize;
            copied += 1;
            /* If we move past value data.  Reset */
            if (copied >= numvals) {
                vptr = avals_data;
            }
        }
        mptr += melsize;
    }
}

/*
 * Returns input array with values inserted sequentially into places
 * indicated by the mask
 */
static PyObject *
arr_insert(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{
    PyObject *mask = NULL, *vals = NULL;
    PyArrayObject *ainput = NULL, *amask = NULL, *avals = NULL, *tmp = NULL;
    int numvals, totmask, sameshape;
    char *input_data, *mptr, *vptr, *zero = NULL;
    int melsize, delsize, nd, objarray, k;
    npy_intp *instrides, *inshape;

    static char *kwlist[] = {"input", "mask", "vals", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "O&OO", kwlist,
                PyArray_Converter, &ainput,
                &mask, &vals)) {
        goto fail;
    }

    amask = (PyArrayObject *)PyArray_FROM_OF(mask, NPY_ARRAY_CARRAY);
    if (amask == NULL) {
        goto fail;
    }
    /* Cast an object array */
    if (PyArray_DESCR(amask)->type_num == NPY_OBJECT) {
        tmp = (PyArrayObject *)PyArray_Cast(amask, NPY_INTP);
        if (tmp == NULL) {
            goto fail;
        }
        Py_DECREF(amask);
        amask = tmp;
    }

    sameshape = 1;
    if (PyArray_NDIM(amask) == PyArray_NDIM(ainput)) {
        for (k = 0; k < PyArray_NDIM(amask); k++) {
            if (PyArray_DIMS(amask)[k] != PyArray_DIMS(ainput)[k]) {
                sameshape = 0;
            }
        }
    }
    else {
        /* Test to see if amask is 1d */
        if (PyArray_NDIM(amask) != 1) {
            sameshape = 0;
        }
        else if ((PyArray_SIZE(ainput)) != PyArray_SIZE(amask)) {
            sameshape = 0;
        }
    }
    if (!sameshape) {
        PyErr_SetString(PyExc_TypeError,
                        "mask array must be 1-d or same shape as input array");
        goto fail;
    }

    avals = (PyArrayObject *)PyArray_FromObject(vals,
                                        PyArray_DESCR(ainput)->type_num, 0, 1);
    if (avals == NULL) {
        goto fail;
    }
    numvals = PyArray_SIZE(avals);
    nd = PyArray_NDIM(ainput);
    input_data = PyArray_DATA(ainput);
    mptr = PyArray_DATA(amask);
    melsize = PyArray_DESCR(amask)->elsize;
    vptr = PyArray_DATA(avals);
    delsize = PyArray_DESCR(avals)->elsize;
    zero = PyArray_Zero(amask);
    if (zero == NULL) {
        goto fail;
    }
    objarray = (PyArray_DESCR(ainput)->type_num == NPY_OBJECT);

    /* Handle zero-dimensional case separately */
    if (nd == 0) {
        if (memcmp(mptr,zero,melsize) != 0) {
            /* Copy value element over to input array */
            memcpy(input_data,vptr,delsize);
            if (objarray) {
                Py_INCREF(*((PyObject **)vptr));
            }
        }
        Py_DECREF(amask);
        Py_DECREF(avals);
        PyDataMem_FREE(zero);
        Py_DECREF(ainput);
        Py_INCREF(Py_None);
        return Py_None;
    }

    totmask = (int) PyArray_SIZE(amask);
    instrides = PyArray_STRIDES(ainput);
    inshape = PyArray_DIMS(ainput);
    if (objarray) {
        /* object array, need to refcount, can't release the GIL */
        arr_insert_loop(mptr, vptr, input_data, zero, PyArray_DATA(avals),
                        melsize, delsize, objarray, totmask, numvals, nd,
                        instrides, inshape);
    }
    else {
        /* No increfs take place in arr_insert_loop, so release the GIL */
        NPY_BEGIN_ALLOW_THREADS;
        arr_insert_loop(mptr, vptr, input_data, zero, PyArray_DATA(avals),
                        melsize, delsize, objarray, totmask, numvals, nd,
                        instrides, inshape);
        NPY_END_ALLOW_THREADS;
    }

    Py_DECREF(amask);
    Py_DECREF(avals);
    PyDataMem_FREE(zero);
    Py_DECREF(ainput);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    PyDataMem_FREE(zero);
    Py_XDECREF(ainput);
    Py_XDECREF(amask);
    Py_XDECREF(avals);
    return NULL;
}

/** @brief Use bisection on a sorted array to find first entry > key.
 *
 * Use bisection to find an index i s.t. arr[i] <= key < arr[i + 1]. If there is
 * no such i the error returns are:
 *     key < arr[0] -- -1
 *     key == arr[len - 1] -- len - 1
 *     key > arr[len - 1] -- len
 * The array is assumed contiguous and sorted in ascending order.
 *
 * @param key key value.
 * @param arr contiguous sorted array to be searched.
 * @param len length of the array.
 * @return index
 */
static npy_intp
binary_search(double key, double arr [], npy_intp len)
{
    npy_intp imin = 0;
    npy_intp imax = len;

    if (key > arr[len - 1]) {
        return len;
    }
    while (imin < imax) {
        npy_intp imid = imin + ((imax - imin) >> 1);
        if (key >= arr[imid]) {
            imin = imid + 1;
        }
        else {
            imax = imid;
        }
    }
    return imin - 1;
}

static PyObject *
arr_interp(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwdict)
{

    PyObject *fp, *xp, *x;
    PyObject *left = NULL, *right = NULL;
    PyArrayObject *afp = NULL, *axp = NULL, *ax = NULL, *af = NULL;
    npy_intp i, lenx, lenxp;
    double lval, rval;
    double *dy, *dx, *dz, *dres, *slopes;

    static char *kwlist[] = {"x", "xp", "fp", "left", "right", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwdict, "OOO|OO", kwlist,
                                     &x, &xp, &fp, &left, &right)) {
        return NULL;
    }

    afp = (PyArrayObject *)PyArray_ContiguousFromAny(fp, NPY_DOUBLE, 1, 1);
    if (afp == NULL) {
        return NULL;
    }
    axp = (PyArrayObject *)PyArray_ContiguousFromAny(xp, NPY_DOUBLE, 1, 1);
    if (axp == NULL) {
        goto fail;
    }
    ax = (PyArrayObject *)PyArray_ContiguousFromAny(x, NPY_DOUBLE, 1, 0);
    if (ax == NULL) {
        goto fail;
    }
    lenxp = PyArray_DIMS(axp)[0];
    if (lenxp == 0) {
        PyErr_SetString(PyExc_ValueError,
                "array of sample points is empty");
        goto fail;
    }
    if (PyArray_DIMS(afp)[0] != lenxp) {
        PyErr_SetString(PyExc_ValueError,
                "fp and xp are not of the same length.");
        goto fail;
    }

    af = (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(ax),
                                        PyArray_DIMS(ax), NPY_DOUBLE);
    if (af == NULL) {
        goto fail;
    }
    lenx = PyArray_SIZE(ax);

    dy = (double *)PyArray_DATA(afp);
    dx = (double *)PyArray_DATA(axp);
    dz = (double *)PyArray_DATA(ax);
    dres = (double *)PyArray_DATA(af);

    /* Get left and right fill values. */
    if ((left == NULL) || (left == Py_None)) {
        lval = dy[0];
    }
    else {
        lval = PyFloat_AsDouble(left);
        if ((lval == -1) && PyErr_Occurred()) {
            goto fail;
        }
    }
    if ((right == NULL) || (right == Py_None)) {
        rval = dy[lenxp-1];
    }
    else {
        rval = PyFloat_AsDouble(right);
        if ((rval == -1) && PyErr_Occurred()) {
            goto fail;
        }
    }

    /* only pre-calculate slopes if there are relatively few of them. */
    if (lenxp <= lenx) {
        slopes = (double *) PyArray_malloc((lenxp - 1)*sizeof(double));
        if (! slopes) {
            goto fail;
        }
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < lenxp - 1; i++) {
            slopes[i] = (dy[i + 1] - dy[i])/(dx[i + 1] - dx[i]);
        }
        for (i = 0; i < lenx; i++) {
            const double x = dz[i];
            npy_intp j;

            if (npy_isnan(x)) {
                dres[i] = x;
                continue;
            }

            j = binary_search(x, dx, lenxp);
            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else {
                dres[i] = slopes[j]*(x - dx[j]) + dy[j];
            }
        }
        NPY_END_ALLOW_THREADS;
        PyArray_free(slopes);
    }
    else {
        NPY_BEGIN_ALLOW_THREADS;
        for (i = 0; i < lenx; i++) {
            const double x = dz[i];
            npy_intp j;

            if (npy_isnan(x)) {
                dres[i] = x;
                continue;
            }

            j = binary_search(x, dx, lenxp);
            if (j == -1) {
                dres[i] = lval;
            }
            else if (j == lenxp - 1) {
                dres[i] = dy[j];
            }
            else if (j == lenxp) {
                dres[i] = rval;
            }
            else {
                const double slope = (dy[j + 1] - dy[j])/(dx[j + 1] - dx[j]);
                dres[i] = slope*(x - dx[j]) + dy[j];
            }
        }
        NPY_END_ALLOW_THREADS;
    }

    Py_DECREF(afp);
    Py_DECREF(axp);
    Py_DECREF(ax);
    return (PyObject *)af;

fail:
    Py_XDECREF(afp);
    Py_XDECREF(axp);
    Py_XDECREF(ax);
    Py_XDECREF(af);
    return NULL;
}

/*
 * Converts a Python sequence into 'count' PyArrayObjects
 *
 * seq       - Input Python object, usually a tuple but any sequence works.
 * op        - Where the arrays are placed.
 * count     - How many arrays there should be (errors if it doesn't match).
 * paramname - The name of the parameter that produced 'seq'.
 */
static int sequence_to_arrays(PyObject *seq,
                                PyArrayObject **op, int count,
                                char *paramname)
{
    int i;

    if (!PySequence_Check(seq) || PySequence_Size(seq) != count) {
        PyErr_Format(PyExc_ValueError,
                "parameter %s must be a sequence of length %d",
                paramname, count);
        return -1;
    }

    for (i = 0; i < count; ++i) {
        PyObject *item = PySequence_GetItem(seq, i);
        if (item == NULL) {
            while (--i >= 0) {
                Py_DECREF(op[i]);
                op[i] = NULL;
            }
            return -1;
        }

        op[i] = (PyArrayObject *)PyArray_FromAny(item, NULL, 0, 0, 0, NULL);
        if (op[i] == NULL) {
            while (--i >= 0) {
                Py_DECREF(op[i]);
                op[i] = NULL;
            }
            Py_DECREF(item);
            return -1;
        }

        Py_DECREF(item);
    }

    return 0;
}

/* Inner loop for unravel_index */
static int
ravel_multi_index_loop(int ravel_ndim, npy_intp *ravel_dims,
                        npy_intp *ravel_strides,
                        npy_intp count,
                        NPY_CLIPMODE *modes,
                        char **coords, npy_intp *coords_strides)
{
    int i;
    char invalid;
    npy_intp j, m;

    NPY_BEGIN_ALLOW_THREADS;
    invalid = 0;
    while (count--) {
        npy_intp raveled = 0;
        for (i = 0; i < ravel_ndim; ++i) {
            m = ravel_dims[i];
            j = *(npy_intp *)coords[i];
            switch (modes[i]) {
                case NPY_RAISE:
                    if (j < 0 || j >= m) {
                        invalid = 1;
                        goto end_while;
                    }
                    break;
                case NPY_WRAP:
                    if (j < 0) {
                        j += m;
                        if (j < 0) {
                            j = j % m;
                            if (j != 0) {
                                j += m;
                            }
                        }
                    }
                    else if (j >= m) {
                        j -= m;
                        if (j >= m) {
                            j = j % m;
                        }
                    }
                    break;
                case NPY_CLIP:
                    if (j < 0) {
                        j = 0;
                    }
                    else if (j >= m) {
                        j = m - 1;
                    }
                    break;

            }
            raveled += j * ravel_strides[i];

            coords[i] += coords_strides[i];
        }
        *(npy_intp *)coords[ravel_ndim] = raveled;
        coords[ravel_ndim] += coords_strides[ravel_ndim];
    }
end_while:
    NPY_END_ALLOW_THREADS;
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
              "invalid entry in coordinates array");
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/* ravel_multi_index implementation - see add_newdocs.py */
static PyObject *
arr_ravel_multi_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    int i, s;
    PyObject *mode0=NULL, *coords0=NULL;
    PyArrayObject *ret = NULL;
    PyArray_Dims dimensions={0,0};
    npy_intp ravel_strides[NPY_MAXDIMS];
    NPY_ORDER order = NPY_CORDER;
    NPY_CLIPMODE modes[NPY_MAXDIMS];

    PyArrayObject *op[NPY_MAXARGS];
    PyArray_Descr *dtype[NPY_MAXARGS];
    npy_uint32 op_flags[NPY_MAXARGS];

    NpyIter *iter = NULL;

    char *kwlist[] = {"multi_index", "dims", "mode", "order", NULL};

    memset(op, 0, sizeof(op));
    dtype[0] = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                        "OO&|OO&:ravel_multi_index", kwlist,
                     &coords0,
                     PyArray_IntpConverter, &dimensions,
                     &mode0,
                     PyArray_OrderConverter, &order)) {
        goto fail;
    }

    if (dimensions.len+1 > NPY_MAXARGS) {
        PyErr_SetString(PyExc_ValueError,
                    "too many dimensions passed to ravel_multi_index");
        goto fail;
    }

    if (!PyArray_ConvertClipmodeSequence(mode0, modes, dimensions.len)) {
       goto fail;
    }

    switch (order) {
        case NPY_CORDER:
            s = 1;
            for (i = dimensions.len-1; i >= 0; --i) {
                ravel_strides[i] = s;
                s *= dimensions.ptr[i];
            }
            break;
        case NPY_FORTRANORDER:
            s = 1;
            for (i = 0; i < dimensions.len; ++i) {
                ravel_strides[i] = s;
                s *= dimensions.ptr[i];
            }
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                            "only 'C' or 'F' order is permitted");
            goto fail;
    }

    /* Get the multi_index into op */
    if (sequence_to_arrays(coords0, op, dimensions.len, "multi_index") < 0) {
        goto fail;
    }


    for (i = 0; i < dimensions.len; ++i) {
        op_flags[i] = NPY_ITER_READONLY|
                      NPY_ITER_ALIGNED;
    }
    op_flags[dimensions.len] = NPY_ITER_WRITEONLY|
                               NPY_ITER_ALIGNED|
                               NPY_ITER_ALLOCATE;
    dtype[0] = PyArray_DescrFromType(NPY_INTP);
    for (i = 1; i <= dimensions.len; ++i) {
        dtype[i] = dtype[0];
    }

    iter = NpyIter_MultiNew(dimensions.len+1, op, NPY_ITER_BUFFERED|
                                                  NPY_ITER_EXTERNAL_LOOP|
                                                  NPY_ITER_ZEROSIZE_OK,
                                                  NPY_KEEPORDER,
                                                  NPY_SAME_KIND_CASTING,
                                                  op_flags, dtype);
    if (iter == NULL) {
        goto fail;
    }

    if (NpyIter_GetIterSize(iter) != 0) {
        NpyIter_IterNextFunc *iternext;
        char **dataptr;
        npy_intp *strides;
        npy_intp *countptr;

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            goto fail;
        }
        dataptr = NpyIter_GetDataPtrArray(iter);
        strides = NpyIter_GetInnerStrideArray(iter);
        countptr = NpyIter_GetInnerLoopSizePtr(iter);

        do {
            if (ravel_multi_index_loop(dimensions.len, dimensions.ptr,
                        ravel_strides, *countptr, modes,
                        dataptr, strides) != NPY_SUCCEED) {
                goto fail;
            }
        } while(iternext(iter));
    }

    ret = NpyIter_GetOperandArray(iter)[dimensions.len];
    Py_INCREF(ret);

    Py_DECREF(dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);
    return PyArray_Return(ret);

fail:
    Py_XDECREF(dtype[0]);
    for (i = 0; i < dimensions.len; ++i) {
        Py_XDECREF(op[i]);
    }
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);
    return NULL;
}

/* C-order inner loop for unravel_index */
static int
unravel_index_loop_corder(int unravel_ndim, npy_intp *unravel_dims,
                        npy_intp unravel_size, npy_intp count,
                        char *indices, npy_intp indices_stride,
                        npy_intp *coords)
{
    int i;
    char invalid;
    npy_intp val;

    NPY_BEGIN_ALLOW_THREADS;
    invalid = 0;
    while (count--) {
        val = *(npy_intp *)indices;
        if (val < 0 || val >= unravel_size) {
            invalid = 1;
            break;
        }
        for (i = unravel_ndim-1; i >= 0; --i) {
            coords[i] = val % unravel_dims[i];
            val /= unravel_dims[i];
        }
        coords += unravel_ndim;
        indices += indices_stride;
    }
    NPY_END_ALLOW_THREADS;
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
              "invalid entry in index array");
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/* Fortran-order inner loop for unravel_index */
static int
unravel_index_loop_forder(int unravel_ndim, npy_intp *unravel_dims,
                        npy_intp unravel_size, npy_intp count,
                        char *indices, npy_intp indices_stride,
                        npy_intp *coords)
{
    int i;
    char invalid;
    npy_intp val;

    NPY_BEGIN_ALLOW_THREADS;
    invalid = 0;
    while (count--) {
        val = *(npy_intp *)indices;
        if (val < 0 || val >= unravel_size) {
            invalid = 1;
            break;
        }
        for (i = 0; i < unravel_ndim; ++i) {
            *coords++ = val % unravel_dims[i];
            val /= unravel_dims[i];
        }
        indices += indices_stride;
    }
    NPY_END_ALLOW_THREADS;
    if (invalid) {
        PyErr_SetString(PyExc_ValueError,
              "invalid entry in index array");
        return NPY_FAIL;
    }
    return NPY_SUCCEED;
}

/* unravel_index implementation - see add_newdocs.py */
static PyObject *
arr_unravel_index(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *indices0 = NULL, *ret_tuple = NULL;
    PyArrayObject *ret_arr = NULL;
    PyArrayObject *indices = NULL;
    PyArray_Descr *dtype = NULL;
    PyArray_Dims dimensions={0,0};
    NPY_ORDER order = NPY_CORDER;
    npy_intp unravel_size;

    NpyIter *iter = NULL;
    int i, ret_ndim;
    npy_intp ret_dims[NPY_MAXDIMS], ret_strides[NPY_MAXDIMS];

    char *kwlist[] = {"indices", "dims", "order", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&|O&:unravel_index",
                    kwlist,
                    &indices0,
                    PyArray_IntpConverter, &dimensions,
                    PyArray_OrderConverter, &order)) {
        goto fail;
    }

    if (dimensions.len == 0) {
        PyErr_SetString(PyExc_ValueError,
                "dims must have at least one value");
        goto fail;
    }

    unravel_size = PyArray_MultiplyList(dimensions.ptr, dimensions.len);

    if (!PyArray_Check(indices0)) {
        indices = (PyArrayObject*)PyArray_FromAny(indices0,
                                                    NULL, 0, 0, 0, NULL);
        if (indices == NULL) {
            goto fail;
        }
    }
    else {
        indices = (PyArrayObject *)indices0;
        Py_INCREF(indices);
    }

    dtype = PyArray_DescrFromType(NPY_INTP);
    if (dtype == NULL) {
        goto fail;
    }

    iter = NpyIter_New(indices, NPY_ITER_READONLY|
                                NPY_ITER_ALIGNED|
                                NPY_ITER_BUFFERED|
                                NPY_ITER_ZEROSIZE_OK|
                                NPY_ITER_DONT_NEGATE_STRIDES|
                                NPY_ITER_MULTI_INDEX,
                                NPY_KEEPORDER, NPY_SAME_KIND_CASTING,
                                dtype);
    if (iter == NULL) {
        goto fail;
    }

    /*
     * Create the return array with a layout compatible with the indices
     * and with a dimension added to the end for the multi-index
     */
    ret_ndim = PyArray_NDIM(indices) + 1;
    if (NpyIter_GetShape(iter, ret_dims) != NPY_SUCCEED) {
        goto fail;
    }
    ret_dims[ret_ndim-1] = dimensions.len;
    if (NpyIter_CreateCompatibleStrides(iter,
                dimensions.len*sizeof(npy_intp), ret_strides) != NPY_SUCCEED) {
        goto fail;
    }
    ret_strides[ret_ndim-1] = sizeof(npy_intp);

    /* Remove the multi-index and inner loop */
    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
        goto fail;
    }
    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
        goto fail;
    }

    ret_arr = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                            ret_ndim, ret_dims, ret_strides, NULL, 0, NULL);
    dtype = NULL;
    if (ret_arr == NULL) {
        goto fail;
    }

    if (order == NPY_CORDER) {
        if (NpyIter_GetIterSize(iter) != 0) {
            NpyIter_IterNextFunc *iternext;
            char **dataptr;
            npy_intp *strides;
            npy_intp *countptr, count;
            npy_intp *coordsptr = (npy_intp *)PyArray_DATA(ret_arr);

            iternext = NpyIter_GetIterNext(iter, NULL);
            if (iternext == NULL) {
                goto fail;
            }
            dataptr = NpyIter_GetDataPtrArray(iter);
            strides = NpyIter_GetInnerStrideArray(iter);
            countptr = NpyIter_GetInnerLoopSizePtr(iter);

            do {
                count = *countptr;
                if (unravel_index_loop_corder(dimensions.len, dimensions.ptr,
                            unravel_size, count, *dataptr, *strides,
                            coordsptr) != NPY_SUCCEED) {
                    goto fail;
                }
                coordsptr += count*dimensions.len;
            } while(iternext(iter));
        }
    }
    else if (order == NPY_FORTRANORDER) {
        if (NpyIter_GetIterSize(iter) != 0) {
            NpyIter_IterNextFunc *iternext;
            char **dataptr;
            npy_intp *strides;
            npy_intp *countptr, count;
            npy_intp *coordsptr = (npy_intp *)PyArray_DATA(ret_arr);

            iternext = NpyIter_GetIterNext(iter, NULL);
            if (iternext == NULL) {
                goto fail;
            }
            dataptr = NpyIter_GetDataPtrArray(iter);
            strides = NpyIter_GetInnerStrideArray(iter);
            countptr = NpyIter_GetInnerLoopSizePtr(iter);

            do {
                count = *countptr;
                if (unravel_index_loop_forder(dimensions.len, dimensions.ptr,
                            unravel_size, count, *dataptr, *strides,
                            coordsptr) != NPY_SUCCEED) {
                    goto fail;
                }
                coordsptr += count*dimensions.len;
            } while(iternext(iter));
        }
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "only 'C' or 'F' order is permitted");
        goto fail;
    }

    /* Now make a tuple of views, one per index */
    ret_tuple = PyTuple_New(dimensions.len);
    if (ret_tuple == NULL) {
        goto fail;
    }
    for (i = 0; i < dimensions.len; ++i) {
        PyArrayObject *view;

        view = (PyArrayObject *)PyArray_New(&PyArray_Type, ret_ndim-1,
                                ret_dims, NPY_INTP,
                                ret_strides,
                                PyArray_BYTES(ret_arr) + i*sizeof(npy_intp),
                                0, 0, NULL);
        if (view == NULL) {
            goto fail;
        }
        Py_INCREF(ret_arr);
        if (PyArray_SetBaseObject(view, (PyObject *)ret_arr) < 0) {
            Py_DECREF(view);
            goto fail;
        }
        PyTuple_SET_ITEM(ret_tuple, i, PyArray_Return(view));
    }

    Py_DECREF(ret_arr);
    Py_XDECREF(indices);
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);

    return ret_tuple;

fail:
    Py_XDECREF(ret_tuple);
    Py_XDECREF(ret_arr);
    Py_XDECREF(dtype);
    Py_XDECREF(indices);
    PyDimMem_FREE(dimensions.ptr);
    NpyIter_Deallocate(iter);
    return NULL;
}

typedef enum {
    NPY_MERGE_JOIN = 0,
    NPY_MERGE_UNION,
    NPY_MERGE_INTERSECT,
    NPY_MERGE_DIFFERENCE,
    NPY_MERGE_SYMMETRIC_DIFFERENCE
} NPY_MERGEKIND;

static int
_mergekind_converter(PyObject *obj, void *addr)
{
    NPY_MERGEKIND *kind = (NPY_MERGEKIND *)addr;
    char *str;
    PyObject *tmp = NULL;

    if (PyUnicode_Check(obj)) {
        obj = tmp = PyUnicode_AsASCIIString(obj);
    }

    str = PyBytes_AsString(obj);
    if (!str || strlen(str) < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "expected non-empty string for keyword 'kind'");
        Py_XDECREF(tmp);
        return NPY_FAIL;
    }

    if (str[0] == 'j' || str[0] == 'J') {
        *kind = NPY_MERGE_JOIN;
    }
    else if (str[0] == 'u' || str[0] == 'U') {
        *kind = NPY_MERGE_UNION;
    }
    else if (str[0] == 'i' || str[0] == 'I') {
        *kind = NPY_MERGE_INTERSECT;
    }
    else if (str[0] == 'd' || str[0] == 'D') {
        *kind = NPY_MERGE_DIFFERENCE;
    }
    else if (str[0] == 's' || str[0] == 'S') {
        *kind = NPY_MERGE_SYMMETRIC_DIFFERENCE;
    }
    else {
        PyErr_Format(PyExc_ValueError,
                     "'%s' is an invalid value for keyword 'kind'", str);
        Py_XDECREF(tmp);
        return NPY_FAIL;
    }

    Py_XDECREF(tmp);
    return NPY_SUCCEED;
}

#define INIT_MERGE \
    npy_intp len_ar1 = PyArray_SIZE(ar1); \
    npy_intp len_ar2 = ar2 ? PyArray_SIZE(ar2) : 0; \
    npy_intp len_ret = 0; \
    npy_intp stride_ar1 = PyArray_STRIDE(ar1, 0); \
    npy_intp stride_ar2 = ar2 ? PyArray_STRIDE(ar2, 0) : 0; \
    npy_intp stride_ret = ret ? PyArray_STRIDE(ret, 0) : 0; \
    npy_intp stride_idx_ar1 = idx_ar1 ? PyArray_STRIDE(idx_ar1, 0) : 0; \
    npy_intp stride_idx_ar2 = idx_ar2 ? PyArray_STRIDE(idx_ar2, 0) : 0; \
    const char *data_ar1 = (const char *)PyArray_DATA(ar1); \
    const char *data_ar2 = ar2 ? (const char *)PyArray_DATA(ar2) : NULL; \
    char *data_ret = ret ? (char *)PyArray_DATA(ret) : NULL; \
    char *data_idx_ar1 = idx_ar1 ? (char *)PyArray_DATA(idx_ar1) : NULL; \
    char *data_idx_ar2 = idx_ar2 ? (char *)PyArray_DATA(idx_ar2) : NULL; \
    const char *repeat; \
    PyArray_Descr *dtype = PyArray_DESCR(ar1); \
    PyArray_CompareFunc *cmp = dtype->f->compare; \
    PyArray_CopySwapFunc *cpy = dtype->f->copyswap; \
    PyArray_CopySwapNFunc *cpyn = dtype->f->copyswapn;

#define CMP(a, b) \
    cmp((const void *)data_##a, (void *)data_##b, (void *)a)

#define COPY_FROM(ar) \
    if (ret) { \
        cpy((void *)data_ret, (void *)data_##ar, 0, (void *)ret); \
        data_ret += stride_ret; \
    } \
    if (idx_##ar) { \
        *(npy_intp *)data_idx_##ar = len_ret; \
        data_idx_##ar += stride_idx_##ar; \
    } \
    repeat = (const char *)data_##ar; \
    data_##ar += stride_##ar; \
    len_##ar--;\

#define COPY_ALL_FROM(ar) \
    if (ret && len_##ar) { \
        cpyn((void *)data_ret, stride_ret, (void *)data_##ar, \
             stride_##ar, len_##ar, 0, (void *)ret); \
    } \
    if (idx_##ar) { \
        while(len_##ar--) { \
            *(npy_intp *)data_idx_##ar = len_ret++; \
            data_idx_##ar += stride_idx_##ar; \
        } \
    } \
    else { \
        len_ret += len_##ar; \
        len_##ar = 0; \
    }

#define DO_NOT_COPY_FROM(ar, idx) \
    len_##ar--; \
    data_##ar += stride_##ar; \
    if (idx_##ar) { \
        *(npy_intp *)data_idx_##ar = idx; \
        data_idx_##ar += stride_idx_##ar; \
    }

#define DO_NOT_COPY_REPEATS_FROM(ar, idx) \
    while (len_##ar && \
           cmp((void *)data_##ar, (void *)repeat, (void *)ar) == 0) { \
        DO_NOT_COPY_FROM(ar, idx); \
    }

#define SKIP_FROM(ar) DO_NOT_COPY_FROM(ar, len_ret)

#define SKIP_REPEATS_FROM(ar) DO_NOT_COPY_REPEATS_FROM(ar, len_ret)

#define DISCARD_FROM(ar) DO_NOT_COPY_FROM(ar, -1)

#define DISCARD_REPEATS_FROM(ar) DO_NOT_COPY_REPEATS_FROM(ar, -1)

#define COPY_ALL_UNIQUE_FROM(ar) \
    while (len_##ar) { \
        COPY_FROM(ar) \
        SKIP_REPEATS_FROM(ar) \
        len_ret++; \
    }

#define DISCARD_ALL_FROM(ar) \
    if (idx_##ar) { \
        while (len_##ar--) { \
            *(npy_intp *)data_idx_##ar = -1; \
            data_idx_##ar += stride_idx_##ar; \
        } \
    }

/*
 * Merge two sorted, strided, 1d arrays of the same type into a single sorted,
 * strided, 1d array.
 * Returns the number of items written to the ret array.
 */
static npy_intp
merge_join(PyArrayObject *ar1, PyArrayObject *ar2, PyArrayObject *ret,
           PyArrayObject *idx_ar1, PyArrayObject *idx_ar2)
{
    INIT_MERGE

    while (len_ar1 && len_ar2) {
        if (CMP(ar2, ar1) < 0) {
            COPY_FROM(ar2)
        }
        else {
            COPY_FROM(ar1)
        }
        len_ret++;
    }
    COPY_ALL_FROM(ar1)
    COPY_ALL_FROM(ar2)

    return len_ret;
}

/*
 * Merge two sorted, strided, 1d arrays of the same type into a single sorted,
 * strided, 1d array of unique items.
 * Returns the number of items written to the ret array.
 */
static npy_intp
merge_union(PyArrayObject *ar1, PyArrayObject *ar2, PyArrayObject *ret,
            PyArrayObject *idx_ar1, PyArrayObject *idx_ar2)
{
    INIT_MERGE

    while (len_ar1 && len_ar2) {
        if (CMP(ar2, ar1) < 0) {
            COPY_FROM(ar2)
        }
        else {
            COPY_FROM(ar1)
        }
        SKIP_REPEATS_FROM(ar1);
        SKIP_REPEATS_FROM(ar2);
        len_ret++;
    }
    COPY_ALL_UNIQUE_FROM(ar1)
    COPY_ALL_UNIQUE_FROM(ar2)

    return len_ret;
}


/*
 * Merge two sorted, strided, 1d arrays of the same type into a single sorted,
 * strided, 1d array, including only unique items found in both arrays.
 * Returns the number of items written to the ret array.
 */
static npy_intp
merge_intersect(PyArrayObject *ar1, PyArrayObject *ar2, PyArrayObject *ret,
                PyArrayObject *idx_ar1, PyArrayObject *idx_ar2)
{
    INIT_MERGE

    while (len_ar1 && len_ar2) {
        int comp_val = CMP(ar1, ar2);
        if (comp_val < 0) {
            DISCARD_FROM(ar1)
        }
        else if (comp_val > 0) {
            DISCARD_FROM(ar2)
        }
        else {
            COPY_FROM(ar1)
            SKIP_REPEATS_FROM(ar1)
            SKIP_REPEATS_FROM(ar2)
            len_ret++;
        }
    }
    DISCARD_ALL_FROM(ar1)
    DISCARD_ALL_FROM(ar2)

    return len_ret;
}

/*
 * Merge two sorted, strided, 1d arrays of the same type into a single sorted,
 * strided, 1d array, including only unique items found in the first, but not
 * the second array.
 * Returns the number of items written to the ret array.
 */
static npy_intp
merge_difference(PyArrayObject *ar1, PyArrayObject *ar2, PyArrayObject *ret,
                 PyArrayObject *idx_ar1, PyArrayObject *idx_ar2)
{
    INIT_MERGE

    while (len_ar1 && len_ar2) {
        int comp_val = CMP(ar1, ar2);
        if (comp_val < 0) {
            COPY_FROM(ar1)
            SKIP_REPEATS_FROM(ar1)
            len_ret++;
        }
        else if (comp_val > 0) {
            DISCARD_FROM(ar2)
        }
        else {
            repeat = data_ar1;
            DISCARD_REPEATS_FROM(ar1)
            DISCARD_REPEATS_FROM(ar2)
        }
    }
    COPY_ALL_UNIQUE_FROM(ar1)
    DISCARD_ALL_FROM(ar2)

    return len_ret;
}

/*
 * Merge two sorted, strided, 1d arrays of the same type into a single sorted,
 * strided, 1d array, including only unique items found in one array, but not
 * in both.
 * Returns the number of items written to the ret array.
 */
static npy_intp
merge_symmetric_difference(PyArrayObject *ar1, PyArrayObject *ar2,
                           PyArrayObject *ret, PyArrayObject *idx_ar1,
                           PyArrayObject *idx_ar2)
{
    INIT_MERGE

    while (len_ar1 && len_ar2) {
        int comp_val = CMP(ar1, ar2);
        if (comp_val < 0) {
            COPY_FROM(ar1)
            SKIP_REPEATS_FROM(ar1)
            len_ret++;
        }
        else if (comp_val > 0) {
            COPY_FROM(ar2)
            SKIP_REPEATS_FROM(ar2)
            len_ret++;
        }
        else {
            repeat = (char *)data_ar1;
            SKIP_REPEATS_FROM(ar1)
            SKIP_REPEATS_FROM(ar2)
        }
    }
    COPY_ALL_UNIQUE_FROM(ar1)
    COPY_ALL_UNIQUE_FROM(ar2)

    return len_ret;
}

#undef DISCARD_ALL_FROM
#undef COPY_ALL_UNIQUE_FROM
#undef DISCARD_REPEATS_FROM
#undef DISCARD_FROM
#undef SKIP_REPEATS_FROM
#undef SKIP_FROM
#undef DO_NOT_COPY_REPEATS_FROM
#undef DO_NOT_COPY_FROM
#undef COPY_ALL_FROM
#undef COPY_FROM
#undef CMP
#undef INIT_MERGE

/*
 * mergesorted(ar1, ar2, argsort=False) takes two sorted arrays, ar1 and
 * ar2, and merges them into a single sorted array. If argsort == True,
 * the return is a tuple, the first item being the merged sorted array, the
 * second an array of indices that sort the concatenation of ar1 and ar2.
 */
static PyObject*
arr_mergesorted(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *ar1 = NULL;
    PyObject *ar2 = NULL;
    PyObject *index_flag = NULL;
    PyObject *merge_flag = NULL;
    PyArrayObject *arr_ar1 = NULL;
    PyArrayObject *arr_ar2 = NULL;
    PyArrayObject *arr_merge = NULL;
    PyArrayObject *arr_idx_ar1 = NULL;
    PyArrayObject *arr_idx_ar2 = NULL;
    NPY_MERGEKIND kind = NPY_MERGE_JOIN;
    npy_bool do_indices[2] = {0};
    npy_intp len_merge, len_total, len_ar1, len_ar2, num_rets = 0;
    PyArray_Descr *dtype = NULL, *dtype1 = NULL;
    char *kwlist[] = {"ar1", "ar2", "kind", "return_indices",
                      "return_merge", NULL};
    NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO&OO", kwlist,
                                     &ar1, &ar2, _mergekind_converter,
                                     &kind, &index_flag, &merge_flag)) {
        goto fail;
    }

    /* Find a common dtype for both arrays */
    dtype1 = PyArray_DescrFromObject(ar1, NULL);
    if (dtype1 == NULL) {
        goto fail;
    }

    if (ar2 == NULL) {
        dtype = dtype1;
    }
    else {
        dtype = PyArray_DescrFromObject(ar2, dtype1);
        Py_DECREF(dtype1);
        if (dtype == NULL) {
            goto fail;
        }
    }

    /* Make sure the common dtype is in native byte order */
    if (PyDataType_ISBYTESWAPPED(dtype)) {
        PyArray_DESCR_REPLACE(dtype);
        dtype->byteorder = NPY_NATIVE;
    }

    /* check that the common dtype has a comparison function */
    if (dtype->f->compare == NULL) {
        PyErr_SetString(PyExc_TypeError, "compare not supported for type");
        Py_DECREF(dtype);
        goto fail;
    }

    /* Convert the inputs to aligned arrays of the same non-swapped type */
    Py_INCREF(dtype);
    arr_ar1 = (PyArrayObject *)PyArray_FromAny(ar1, dtype, 1, 1,
                                               NPY_ARRAY_ALIGNED, NULL);
    if (arr_ar1 == NULL) {
        Py_DECREF(dtype);
        goto fail;
    }
    len_ar1 = PyArray_SIZE(arr_ar1);

    if (ar2 != NULL) {
        Py_INCREF(dtype);
        arr_ar2 = (PyArrayObject *)PyArray_FromAny(ar2, dtype, 1, 1,
                                                   NPY_ARRAY_ALIGNED, NULL);
        if (arr_ar2 == NULL) {
            Py_DECREF(dtype);
            goto fail;
        }
        len_ar2 = PyArray_SIZE(arr_ar2);
    }

    /* Default value for 'return_merge' is True */
    if (merge_flag == NULL || PyObject_IsTrue(merge_flag)) {
        len_merge = len_ar1;
        if (arr_ar2 != NULL && kind != NPY_MERGE_DIFFERENCE) {
                len_merge += len_ar2;
        }
        /* arr_merge consumes the last reference to dtype */
        arr_merge = (PyArrayObject *)PyArray_SimpleNewFromDescr(1, &len_merge,
                                                                dtype);
        if (arr_merge == NULL) {
            goto fail;
        }
        num_rets++;
    }
    else {
        Py_DECREF(dtype);
    }

    /* Default value for 'return_indices' is both False */
    if (index_flag) {
        if (PyTuple_Check(index_flag)) {
            if (PyTuple_Size(index_flag) != 2) {
                goto fail;
            }
            do_indices[0] = PyObject_IsTrue(PyTuple_GET_ITEM(index_flag, 0));
            do_indices[1] = PyObject_IsTrue(PyTuple_GET_ITEM(index_flag, 1));
        }
        else {
            do_indices[0] = do_indices[1] = PyObject_IsTrue(index_flag);
        }
    }

    if (do_indices[0]) {
        arr_idx_ar1 = (PyArrayObject *)PyArray_SimpleNew(1, &len_ar1,
                                                         NPY_INTP);
        if (arr_idx_ar1 == NULL) {
            goto fail;
        }
        num_rets++;
    }

    if (arr_ar2 != NULL && do_indices[1]) {
        arr_idx_ar2 = (PyArrayObject *)PyArray_SimpleNew(1, &len_ar2,
                                                         NPY_INTP);
        if (arr_idx_ar2 == NULL) {
            goto fail;
        }
        if (kind == NPY_MERGE_DIFFERENCE) {
            /* The second index array of a difference is all -1s */
            PyObject *fill = PyInt_FromLong(-1);
            if (!PyArray_FillWithScalar(arr_idx_ar2, fill)) {
                Py_DECREF(fill);
                goto fail;
            }
            Py_DECREF(fill);
        }
        num_rets++;
    }

    /* Call the actual merging function */
    NPY_BEGIN_THREADS_DESCR(PyArray_DESCR(arr_ar1));
    switch (kind) {
        case NPY_MERGE_JOIN:
            len_total = merge_join(arr_ar1, arr_ar2, arr_merge,
                                   arr_idx_ar1, arr_idx_ar2);
            break;
        case NPY_MERGE_UNION:
            len_total = merge_union(arr_ar1, arr_ar2, arr_merge,
                                    arr_idx_ar1, arr_idx_ar2);
            break;
        case NPY_MERGE_INTERSECT:
            len_total = merge_intersect(arr_ar1, arr_ar2, arr_merge,
                                        arr_idx_ar1, arr_idx_ar2);
            break;
        case NPY_MERGE_DIFFERENCE:
            len_total = merge_difference(arr_ar1, arr_ar2, arr_merge,
                                         arr_idx_ar1, NULL);
            break;
        case NPY_MERGE_SYMMETRIC_DIFFERENCE:
            len_total = merge_symmetric_difference(arr_ar1, arr_ar2,
                                                   arr_merge,
                                                   arr_idx_ar1, arr_idx_ar2);
            break;
        default:
            len_total = 0;
    }
    NPY_END_THREADS_DESCR(PyArray_DESCR(arr_ar1));

    if (arr_merge != NULL && len_total < len_merge) {
        PyArray_Dims new_shape = {&len_total, 1};
        PyObject *tmp;
        tmp = PyArray_Resize(arr_merge, &new_shape, 0, 0);
        if (tmp == NULL) {
            goto fail;
        }
        Py_DECREF(tmp);
    }

    Py_DECREF(arr_ar1);
    Py_XDECREF(arr_ar2);
    if (num_rets == 0) {
         Py_RETURN_NONE;
    }
    else if (num_rets == 1) {
        if (arr_merge) {
            return arr_merge;
        }
        if (arr_idx_ar1) {
            return arr_idx_ar1;
        }
        if (arr_idx_ar2) {
            return arr_idx_ar2;
        }
    }
    else {
        PyObject *ret = PyTuple_New(num_rets);
        Py_ssize_t j = 0;
        if (arr_merge) {
            PyTuple_SetItem(ret, j, arr_merge);
            j++;
        }
        if (arr_idx_ar1) {
            PyTuple_SetItem(ret, j, arr_idx_ar1);
            j++;
        }
        if (arr_idx_ar2) {
            PyTuple_SetItem(ret, j, arr_idx_ar2);
        }
        return ret;
    }

    fail:
        Py_XDECREF(arr_ar1);
        Py_XDECREF(arr_ar2);
        Py_XDECREF(arr_merge);
        Py_XDECREF(arr_idx_ar1);
        Py_XDECREF(arr_idx_ar2);
        return NULL;
}

static PyTypeObject *PyMemberDescr_TypePtr = NULL;
static PyTypeObject *PyGetSetDescr_TypePtr = NULL;
static PyTypeObject *PyMethodDescr_TypePtr = NULL;

/* Can only be called if doc is currently NULL */
static PyObject *
arr_add_docstring(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyObject *obj;
    PyObject *str;
    char *docstr;
    static char *msg = "already has a docstring";

    /* Don't add docstrings */
    if (Py_OptimizeFlag > 1) {
        Py_INCREF(Py_None);
        return Py_None;
    }
#if defined(NPY_PY3K)
    if (!PyArg_ParseTuple(args, "OO!", &obj, &PyUnicode_Type, &str)) {
        return NULL;
    }

    docstr = PyBytes_AS_STRING(PyUnicode_AsUTF8String(str));
#else
    if (!PyArg_ParseTuple(args, "OO!", &obj, &PyString_Type, &str)) {
        return NULL;
    }

    docstr = PyString_AS_STRING(str);
#endif

#define _TESTDOC1(typebase) (Py_TYPE(obj) == &Py##typebase##_Type)
#define _TESTDOC2(typebase) (Py_TYPE(obj) == Py##typebase##_TypePtr)
#define _ADDDOC(typebase, doc, name) do {                               \
        Py##typebase##Object *new = (Py##typebase##Object *)obj;        \
        if (!(doc)) {                                                   \
            doc = docstr;                                               \
        }                                                               \
        else {                                                          \
            PyErr_Format(PyExc_RuntimeError, "%s method %s", name, msg); \
            return NULL;                                                \
        }                                                               \
    } while (0)

    if (_TESTDOC1(CFunction)) {
        _ADDDOC(CFunction, new->m_ml->ml_doc, new->m_ml->ml_name);
    }
    else if (_TESTDOC1(Type)) {
        _ADDDOC(Type, new->tp_doc, new->tp_name);
    }
    else if (_TESTDOC2(MemberDescr)) {
        _ADDDOC(MemberDescr, new->d_member->doc, new->d_member->name);
    }
    else if (_TESTDOC2(GetSetDescr)) {
        _ADDDOC(GetSetDescr, new->d_getset->doc, new->d_getset->name);
    }
    else if (_TESTDOC2(MethodDescr)) {
        _ADDDOC(MethodDescr, new->d_method->ml_doc, new->d_method->ml_name);
    }
    else {
        PyObject *doc_attr;

        doc_attr = PyObject_GetAttrString(obj, "__doc__");
        if (doc_attr != NULL && doc_attr != Py_None) {
            PyErr_Format(PyExc_RuntimeError, "object %s", msg);
            return NULL;
        }
        Py_XDECREF(doc_attr);

        if (PyObject_SetAttrString(obj, "__doc__", str) < 0) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot set a docstring for that object");
            return NULL;
        }
        Py_INCREF(Py_None);
        return Py_None;
    }

#undef _TESTDOC1
#undef _TESTDOC2
#undef _ADDDOC

    Py_INCREF(str);
    Py_INCREF(Py_None);
    return Py_None;
}


/* docstring in numpy.add_newdocs.py */
static PyObject *
add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args)
{
    PyUFuncObject *ufunc;
    PyObject *str;
    char *docstr, *newdocstr;

#if defined(NPY_PY3K)
    if (!PyArg_ParseTuple(args, "O!O!", &PyUFunc_Type, &ufunc,
                                        &PyUnicode_Type, &str)) {
        return NULL;
    }
    docstr = PyBytes_AS_STRING(PyUnicode_AsUTF8String(str));
#else
    if (!PyArg_ParseTuple(args, "O!O!", &PyUFunc_Type, &ufunc,
                                         &PyString_Type, &str)) {
         return NULL;
    }
    docstr = PyString_AS_STRING(str);
#endif

    if (NULL != ufunc->doc) {
        PyErr_SetString(PyExc_ValueError,
                "Cannot change docstring of ufunc with non-NULL docstring");
        return NULL;
    }

    /*
     * This introduces a memory leak, as the memory allocated for the doc
     * will not be freed even if the ufunc itself is deleted. In practice
     * this should not be a problem since the user would have to
     * repeatedly create, document, and throw away ufuncs.
     */
    newdocstr = malloc(strlen(docstr) + 1);
    strcpy(newdocstr, docstr);
    ufunc->doc = newdocstr;

    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * This function packs boolean values in the input array into the bits of a
 * byte array. Truth values are determined as usual: 0 is false, everything
 * else is true.
 */
static NPY_INLINE void
pack_inner(const char *inptr,
           npy_intp element_size,   /* in bytes */
           npy_intp n_in,
           npy_intp in_stride,
           char *outptr,
           npy_intp n_out,
           npy_intp out_stride)
{
    /*
     * Loop through the elements of inptr.
     * Determine whether or not it is nonzero.
     *  Yes: set corresponding bit (and adjust build value)
     *  No:  move on
     * Every 8th value, set the value of build and increment the outptr
     */
    npy_intp index;
    int remain = n_in % 8;              /* uneven bits */

    if (remain == 0) {                  /* assumes n_in > 0 */
        remain = 8;
    }
    for (index = 0; index < n_out; index++) {
        char build = 0;
        int i, maxi;
        npy_intp j;

        maxi = (index == n_out - 1) ? remain : 8;
        for (i = 0; i < maxi; i++) {
            build <<= 1;
            for (j = 0; j < element_size; j++) {
                build |= (inptr[j] != 0);
            }
            inptr += in_stride;
        }
        if (index == n_out - 1) {
            build <<= 8 - remain;
        }
        *outptr = build;
        outptr += out_stride;
    }
}

static PyObject *
pack_bits(PyObject *input, int axis)
{
    PyArrayObject *inp;
    PyArrayObject *new = NULL;
    PyArrayObject *out = NULL;
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    NPY_BEGIN_THREADS_DEF;

    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        return NULL;
    }
    if (!PyArray_ISBOOL(inp) && !PyArray_ISINTEGER(inp)) {
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of integer or boolean data type");
        goto fail;
    }

    new = (PyArrayObject *)PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        return NULL;
    }
    /* Handle empty array separately */
    if (PyArray_SIZE(new) == 0) {
        return PyArray_Copy(new);
    }

    if (PyArray_NDIM(new) == 0) {
        char *optr, *iptr;

        out = (PyArrayObject *)PyArray_New(Py_TYPE(new), 0, NULL, NPY_UBYTE,
                NULL, NULL, 0, 0, NULL);
        if (out == NULL) {
            goto fail;
        }
        optr = PyArray_DATA(out);
        iptr = PyArray_DATA(new);
        *optr = 0;
        for (i = 0; i < PyArray_ITEMSIZE(new); i++) {
            if (*iptr != 0) {
                *optr = 1;
                break;
            }
            iptr++;
        }
        goto finish;
    }


    /* Setup output shape */
    for (i = 0; i < PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    /*
     * Divide axis dimension by 8
     * 8 -> 1, 9 -> 2, 16 -> 2, 17 -> 3 etc..
     */
    outdims[axis] = ((outdims[axis] - 1) >> 3) + 1;

    /* Create output array */
    out = (PyArrayObject *)PyArray_New(Py_TYPE(new),
                        PyArray_NDIM(new), outdims, NPY_UBYTE,
                        NULL, NULL, 0, PyArray_ISFORTRAN(new), NULL);
    if (out == NULL) {
        goto fail;
    }
    /* Setup iterators to iterate over all but given axis */
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)new, &axis);
    ot = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)out, &axis);
    if (it == NULL || ot == NULL) {
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }

    NPY_BEGIN_THREADS_THRESHOLDED(PyArray_DIM(out, axis));
    while (PyArray_ITER_NOTDONE(it)) {
        pack_inner(PyArray_ITER_DATA(it), PyArray_ITEMSIZE(new),
                   PyArray_DIM(new, axis), PyArray_STRIDE(new, axis),
                   PyArray_ITER_DATA(ot), PyArray_DIM(out, axis),
                   PyArray_STRIDE(out, axis));
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    NPY_END_THREADS;

    Py_DECREF(it);
    Py_DECREF(ot);

finish:
    Py_DECREF(new);
    return (PyObject *)out;

fail:
    Py_XDECREF(new);
    Py_XDECREF(out);
    return NULL;
}

static PyObject *
unpack_bits(PyObject *input, int axis)
{
    PyArrayObject *inp;
    PyArrayObject *new = NULL;
    PyArrayObject *out = NULL;
    npy_intp outdims[NPY_MAXDIMS];
    int i;
    PyArrayIterObject *it, *ot;
    npy_intp n_in, in_stride, out_stride;
    NPY_BEGIN_THREADS_DEF;

    inp = (PyArrayObject *)PyArray_FROM_O(input);

    if (inp == NULL) {
        return NULL;
    }
    if (PyArray_TYPE(inp) != NPY_UBYTE) {
        PyErr_SetString(PyExc_TypeError,
                "Expected an input array of unsigned byte data type");
        goto fail;
    }

    new = (PyArrayObject *)PyArray_CheckAxis(inp, &axis, 0);
    Py_DECREF(inp);
    if (new == NULL) {
        return NULL;
    }
    /* Handle zero-dim array separately */
    if (PyArray_SIZE(new) == 0) {
        return PyArray_Copy(new);
    }

    if (PyArray_NDIM(new) == 0) {
        /* Handle 0-d array by converting it to a 1-d array */
        PyArrayObject *temp;
        PyArray_Dims newdim = {NULL, 1};
        npy_intp shape = 1;

        newdim.ptr = &shape;
        temp = (PyArrayObject *)PyArray_Newshape(new, &newdim, NPY_CORDER);
        if (temp == NULL) {
            goto fail;
        }
        Py_DECREF(new);
        new = temp;
    }

    /* Setup output shape */
    for (i=0; i<PyArray_NDIM(new); i++) {
        outdims[i] = PyArray_DIM(new, i);
    }

    /* Multiply axis dimension by 8 */
    outdims[axis] <<= 3;

    /* Create output array */
    out = (PyArrayObject *)PyArray_New(Py_TYPE(new),
                        PyArray_NDIM(new), outdims, NPY_UBYTE,
                        NULL, NULL, 0, PyArray_ISFORTRAN(new), NULL);
    if (out == NULL) {
        goto fail;
    }
    /* Setup iterators to iterate over all but given axis */
    it = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)new, &axis);
    ot = (PyArrayIterObject *)PyArray_IterAllButAxis((PyObject *)out, &axis);
    if (it == NULL || ot == NULL) {
        Py_XDECREF(it);
        Py_XDECREF(ot);
        goto fail;
    }

    NPY_BEGIN_THREADS_THRESHOLDED(PyArray_DIM(new, axis));

    n_in = PyArray_DIM(new, axis);
    in_stride = PyArray_STRIDE(new, axis);
    out_stride = PyArray_STRIDE(out, axis);

    while (PyArray_ITER_NOTDONE(it)) {
        npy_intp index;
        unsigned const char *inptr = PyArray_ITER_DATA(it);
        char *outptr = PyArray_ITER_DATA(ot);

        for (index = 0; index < n_in; index++) {
            unsigned char mask = 128;

            for (i = 0; i < 8; i++) {
                *outptr = ((mask & (*inptr)) != 0);
                outptr += out_stride;
                mask >>= 1;
            }
            inptr += in_stride;
        }
        PyArray_ITER_NEXT(it);
        PyArray_ITER_NEXT(ot);
    }
    NPY_END_THREADS;

    Py_DECREF(it);
    Py_DECREF(ot);

    Py_DECREF(new);
    return (PyObject *)out;

fail:
    Py_XDECREF(new);
    Py_XDECREF(out);
    return NULL;
}


static PyObject *
io_pack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"in", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&" , kwlist,
                &obj, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return pack_bits(obj, axis);
}

static PyObject *
io_unpack(PyObject *NPY_UNUSED(self), PyObject *args, PyObject *kwds)
{
    PyObject *obj;
    int axis = NPY_MAXDIMS;
    static char *kwlist[] = {"in", "axis", NULL};

    if (!PyArg_ParseTupleAndKeywords( args, kwds, "O|O&" , kwlist,
                &obj, PyArray_AxisConverter, &axis)) {
        return NULL;
    }
    return unpack_bits(obj, axis);
}

/* The docstrings for many of these methods are in add_newdocs.py. */
static struct PyMethodDef methods[] = {
    {"_insert", (PyCFunction)arr_insert,
        METH_VARARGS | METH_KEYWORDS, arr_insert__doc__},
    {"bincount", (PyCFunction)arr_bincount,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"digitize", (PyCFunction)arr_digitize,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"interp", (PyCFunction)arr_interp,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"ravel_multi_index", (PyCFunction)arr_ravel_multi_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unravel_index", (PyCFunction)arr_unravel_index,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"add_docstring", (PyCFunction)arr_add_docstring,
        METH_VARARGS, NULL},
    {"add_newdoc_ufunc", (PyCFunction)add_newdoc_ufunc,
        METH_VARARGS, NULL},
    {"packbits", (PyCFunction)io_pack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"unpackbits", (PyCFunction)io_unpack,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {"mergesorted", (PyCFunction)arr_mergesorted,
        METH_VARARGS | METH_KEYWORDS, NULL},
    {NULL, NULL, 0, NULL}    /* sentinel */
};

static void
define_types(void)
{
    PyObject *tp_dict;
    PyObject *myobj;

    tp_dict = PyArrayDescr_Type.tp_dict;
    /* Get "subdescr" */
    myobj = PyDict_GetItemString(tp_dict, "fields");
    if (myobj == NULL) {
        return;
    }
    PyGetSetDescr_TypePtr = Py_TYPE(myobj);
    myobj = PyDict_GetItemString(tp_dict, "alignment");
    if (myobj == NULL) {
        return;
    }
    PyMemberDescr_TypePtr = Py_TYPE(myobj);
    myobj = PyDict_GetItemString(tp_dict, "newbyteorder");
    if (myobj == NULL) {
        return;
    }
    PyMethodDescr_TypePtr = Py_TYPE(myobj);
    return;
}

#if defined(NPY_PY3K)
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_compiled_base",
        NULL,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};
#endif

#if defined(NPY_PY3K)
#define RETVAL m
PyMODINIT_FUNC PyInit__compiled_base(void)
#else
#define RETVAL
PyMODINIT_FUNC
init_compiled_base(void)
#endif
{
    PyObject *m, *d;

#if defined(NPY_PY3K)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule("_compiled_base", methods);
#endif
    if (!m) {
        return RETVAL;
    }

    /* Import the array objects */
    import_array();
    import_umath();

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);

    /*
     * PyExc_Exception should catch all the standard errors that are
     * now raised instead of the string exception "numpy.lib.error".
     * This is for backward compatibility with existing code.
     */
    PyDict_SetItemString(d, "error", PyExc_Exception);


    /* define PyGetSetDescr_Type and PyMemberDescr_Type */
    define_types();

    return RETVAL;
}
