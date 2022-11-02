/**
* Interative Segmentation based on Dynamic and Iterative Spanning Forest (Python3)
*
* @date september, 2020
* https://docs.python.org/3.5/extending/extending.html
* 
* https://books.google.com.br/books?id=YEoiYr4H2A0C&pg=PA494&lpg=PA494&dq=PyArray_GETPTR2&source=bl&ots=ozjTHGYBiZ&sig=ACfU3U0PI8w3xalCcjC3wj1QmZ7M1u5foA&hl=pt-BR&sa=X&ved=2ahUKEwi7t9O5prDsAhUiHbkGHeVzDCcQ6AEwCHoECAwQAg#v=onepage&q=PyArray_GETPTR2&f=false
* (pag 494)
*/

//=============================================================================
// Includes
//=============================================================================
#include <Python.h>
#include <numpy/arrayobject.h>

#include "Image.h"
#include "iDISF.h"

//=============================================================================
// Prototypes
//=============================================================================
void usage();
PyMODINIT_FUNC PyInit_idisf(void);
static PyObject* iDISF_scribbles(PyObject* self, PyObject* args);

Graph *createGraphFromPyArray(PyObject *pyarr, int ndim, npy_intp *dims, Image **border_img, Image *img);
PyObject *createPyObjectFromGrayImage(Image *img);

//=============================================================================
// Structures
//=============================================================================
static PyMethodDef methods[] = {
    { "iDISF_scribbles", iDISF_scribbles, METH_VARARGS, "Generates segmentation with the iDISF algorithm" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef idisfModule = 
{
    PyModuleDef_HEAD_INIT,
    "iDISF algorithm",
    "Interative Segmentation based on Dynamic and Iterative Spanning Forest",
    -1,
    methods
};

//=============================================================================
// Methods
//=============================================================================
void usage()
{
    printf("Usage: [<a>,<b>] = iDISF_scribbles(<1>,<2>,<3>,..,<13>)\n");
    printf("----------------------------------\n");
    printf("INPUTS:\n");
    printf("<1> - 2D int32 grayscale/RGB numpy array\n" );
    printf("<2> - Initial number of seeds (e.g., N0 = 50)\n");
    printf("<3> - Number of iterations (e.g., it = 3)\n");
    printf("<4> - vector coords of pixels in scribble\n");
    printf("<5> - Vector with the number of pixels in each scribble\n");
    printf("<6> - Number of object scribbles\n");
    printf("<7> - path-cost function 1-5 \n");
    printf("<8> - c1 constant to gradient functions\n");
    printf("<9> - c2 constant to gradient functions\n");
    printf("<10> - Segmentation method 1-3 \n");
    printf("<11> - flag to display tree borders\n");
    printf("OUTPUTS:\n");
    printf("<a> - 2D int32 label numpy array\n" );
    printf("<b> - 2D int32 border numpy array\n");
}

PyMODINIT_FUNC PyInit_idisf(void)
{
    _import_array();
    return PyModule_Create(&idisfModule);
}

static PyObject* iDISF_scribbles(PyObject* self, PyObject* args)
{
    int n_0,ndim, iter, i, j;
    Image *label_img, *border_img;
    Graph *graph;
    PyObject *in_obj, *in_arr, *label_obj, *border_obj;
    PyObject *py_coords, *py_size_scribbles;
    npy_intp *dims;
    int *size_scribbles, n_scribbles;
    int length, n_obj_scribbles, f, segm_method, all_borders;
    double c1, c2;
    NodeCoords **coords;

    /* arguments: image, n0, iterations, coords_x, coords_y, marker_sizes, obj_markers, function, c1, c2, segm_method, all_borders */
    if(!PyArg_ParseTuple(args, "O!iiO!O!iiddii", &PyArray_Type, &in_obj, &n_0, &iter, &PyArray_Type, &py_coords, &PyArray_Type, &py_size_scribbles, &n_obj_scribbles, &f, &c1, &c2, &segm_method, &all_borders))
    {
        usage(); return NULL;
    }

    in_arr = PyArray_FROM_OTF(in_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    if(in_arr == NULL) return PyErr_Format(PyExc_TypeError, "Could not convert the input data to a C contiguous int32 numpy array!");

    if(n_0 < 0) 
        return PyErr_Format(PyExc_ValueError, "N0 must be >= 0!");
    if(iter < 0 || (iter < 1 && segm_method != 3)) 
        return PyErr_Format(PyExc_ValueError, "Iterations/final GRID trees must be >= 1 with segm={1,2}, and >= 0 with segm=3!");
    if(iter > n_0 && segm_method == 3)
        return PyErr_Format(PyExc_ValueError, "Final GRID trees must be <= N0!");
    if(n_obj_scribbles < 1) 
        return PyErr_Format(PyExc_ValueError, "Number of object scribbles must be >= 1!");
    if(f < 1 || f > 5) 
        return PyErr_Format(PyExc_ValueError, "Invalid path-cost function ID: %d!", f);
    if(c1 <= 0.0 || c1 > 1.0)
        return PyErr_Format(PyExc_ValueError, "Invalid c1 value: %f!", c1);
    if(c2 <= 0.0 || c2 > 1.0)
        return PyErr_Format(PyExc_ValueError, "Invalid c2 value: %f!", c2);
    if(all_borders != 0 && all_borders != 1)
        return PyErr_Format(PyExc_ValueError, "Invalid all_borders value: %d!", all_borders);
    

    // -------------------------------------------------------------------------------
    /* Reading scribbles sizes */
    if (PyArray_NDIM(py_size_scribbles) != 1) 
        return PyErr_Format(PyExc_ValueError, "py_size_scribbles array is %d-dimensional or not of type int.",PyArray_NDIM(py_size_scribbles));

    n_scribbles = (int)PyArray_DIM(py_size_scribbles,0);
    size_scribbles = (int *)malloc(sizeof(int) * n_scribbles);
    
    int count = 0, acum[n_scribbles];
    
    for (i = 0; i < n_scribbles; i++)
    {
        size_scribbles[i] = *(int*)PyArray_GETPTR1(py_size_scribbles, i);
        acum[i] = count;
        count+=size_scribbles[i];
    }

    if(n_obj_scribbles > n_scribbles) 
        return PyErr_Format(PyExc_ValueError, "Number of object scribbles must be at most the number of scribbles!");

    // -------------------------------------------------------------------------------
    /* Reading scribbles coords*/
    if (PyArray_NDIM(py_coords) != 2) 
        return PyErr_Format(PyExc_ValueError, "Coords array is %d-dimensional or not of type int. Type=%s.",PyArray_NDIM(py_coords),PyArray_TYPE(py_coords));

    length = (int)PyArray_DIM(py_coords,0);
    if(count != length)
            return PyErr_Format(PyExc_ValueError, "Coords array and size_scribbles don't match.");

    coords = (NodeCoords **)malloc(sizeof(NodeCoords*) * n_scribbles);

#pragma omp parallel for \
    private(i,j) \
    firstprivate(n_scribbles) \
    shared(size_scribbles, coords, py_coords, acum)
    for (i = 0; i < n_scribbles; i++)
    {
        coords[i] = (NodeCoords *)malloc(sizeof(NodeCoords) * size_scribbles[i]);
        for (j = 0; j < size_scribbles[i]; j++)
        {
            coords[i][j].x = *(int*)PyArray_GETPTR2(py_coords, acum[i]+j, 0);
            coords[i][j].y = *(int*)PyArray_GETPTR2(py_coords, acum[i]+j, 1);
        }
    }

    // -------------------------------------------------------------------------------

    ndim = PyArray_NDIM(in_arr);
    dims = (npy_intp *)PyArray_DIMS(in_arr);

    if(ndim < 2 || ndim > 3) return PyErr_Format(PyExc_Exception, "The number of dimensions must be either 2 or 3!");
    if(ndim == 3 && dims[2] != 3) return PyErr_Format(PyExc_Exception, "The image must be RGB-colored (i.e., 3 channels)");

    Image *img;
    int num_channels;
    if(ndim == 2) num_channels = 1;
    else num_channels = 3;
    img = createImage(dims[0], dims[1], num_channels);

    graph = createGraphFromPyArray(in_arr, ndim, dims, &border_img, img);

    //label_img = runiDISF_scribbles(graph, n_0, iter, &border_img, coords, n_scribbles, size_scribbles, n_obj_scribbles, f, c1, c2, segm_method, all_borders);
    if(segm_method == 1){
        freeImage(&img);
        label_img = runiDISF_scribbles_rem(graph, n_0, iter, &border_img, coords, n_scribbles, size_scribbles, f, all_borders, c1, c2, n_obj_scribbles);
    }
    else{
        freeImage(&img);
        label_img = runiDISF(graph, n_0, iter, &border_img, coords, n_scribbles, size_scribbles, f, all_borders, c1, c2, n_obj_scribbles);
    }

    freeGraph(&graph);
    label_obj = createPyObjectFromGrayImage(label_img);
    border_obj = createPyObjectFromGrayImage(border_img);
    
    Py_DECREF(in_arr);
    Py_INCREF(label_obj);
    Py_INCREF(border_obj);

    freeImage(&label_img);
    freeImage(&border_img);

    free(size_scribbles);
    free(coords);

    return Py_BuildValue("OO",label_obj,border_obj);
}

Graph *createGraphFromPyArray(PyObject *pyarr, int ndim, npy_intp *dims, Image **border_img, Image *img)
{
    int num_cols, num_rows, num_channels;
    Graph *graph;
    //Image *img;
    int x, y, f;

    num_rows = dims[0]; num_cols = dims[1];

    if(ndim == 2) num_channels = 1;
    else num_channels = 3;

    //img = createImage(num_rows, num_cols, num_channels);
    (*border_img) = createImage(num_rows, num_cols, 1);

    #pragma omp parallel for \
        private(x,y,f) \
        firstprivate(num_rows,num_cols,num_channels) \
        shared(img,pyarr)
    for(y = 0; y < num_rows; ++y)
        for(x = 0; x < num_cols; ++x)
            for(f = 0; f < num_channels; ++f)
                img->val[x + num_cols * y][f] = *(int*)(PyArray_GETPTR3(pyarr, y, x, f));

    graph = createGraph(img);
    //freeImage(&img);

    return graph;
}

PyObject *createPyObjectFromGrayImage(Image *img)
{
    npy_intp *dims;
    PyObject *pyobj;
    int num_rows, num_cols, x, y, *ptr;

    num_rows = img->num_rows;
    num_cols = img->num_cols;

    dims = (npy_intp *)malloc(2 * sizeof(npy_intp));
    dims[0] = num_rows; dims[1] = num_cols;
    pyobj = (PyObject *)PyArray_SimpleNew(2, dims, PyArray_INT);

    #pragma omp parallel for \
        private(x,y,ptr) \
        firstprivate(num_rows, num_cols) \
        shared(img, pyobj)
    for(y = 0; y < num_rows; y++)
        for(x = 0; x < num_cols; x++)
        {
            ptr = (int*)PyArray_GETPTR2(pyobj, y, x);
            *ptr = img->val[x + y*num_cols][0];
        }

    free(dims);

    return pyobj;
}
