
#include <Python.h>
#include "../include/type_check.h"
#include "../include/mixture/mixture.h"

/**
 * Create components struct
 * @param comp_methods : Component model specifications
 * @param model_methods : Mixture model specifications
 * @param params : Python dict params
 * @param size : Number of data points
 * @param dim : Number of dimensions
 * @param type : Numpy enumerated type
 * @return allocated components_t; initialized empty
 */
struct mixture_model_t *create_mixture(
    ComponentMethods *comp_methods,
    ModelMethods *model_methods,
    PyObject *params,
    uint32_t size, uint32_t dim, int type)
{
    #ifdef SHOW_TRACE
    printf("create_mixture\n");
    #endif

    // Allocate vector
    struct mixture_model_t *mixture = (
        (struct mixture_model_t *) malloc(sizeof(struct mixture_model_t)));
    mixture->mem_size = BASE_VEC_SIZE;
    mixture->num_clusters = 0;
    mixture->clusters = (
        (Component **) malloc(sizeof(Component *) * mixture->mem_size));

    // Bind methods, params, dim
    mixture->comp_methods = comp_methods;
    mixture->model_methods = model_methods;

    mixture->comp_params = comp_methods->params_create(params);
    if(mixture->comp_params == NULL) {
        goto error;
    }

    mixture->model_params = model_methods->create(params);
    if(mixture->model_params == NULL) {
        comp_methods->params_destroy(mixture->comp_params);
        goto error;
    }

    // Size
    mixture->size = size;
    mixture->dim = dim;
    mixture->npy_type = type;
    mixture->stride = type_get_size(type);

    return mixture;

    // Error condition: free all and return NULL
    error:
        free(mixture->clusters);
        free(mixture);
        return NULL;
}


/**
 * Destroy mixture model struct
 * @param model : mixture_model_t struct to destroy
 */
void destroy_mixture(PyObject *model_py)
{
    struct mixture_model_t *model_tc = (
        (struct mixture_model_t *) PyCapsule_GetPointer(
            model_py, MIXTURE_MODEL_API));

    #ifdef SHOW_TRACE
    printf("destroy_mixture [k=%d]\n", model_tc->num_clusters);
    #endif

    // Destroy backwards for models that have a global tracker
    for(int i = model_tc->num_clusters; i > 0; i--) {
        destroy(model_tc, model_tc->clusters[i - 1]);
    }
    free(model_tc);
}

