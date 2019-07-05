
/**
 * Create components struct
 * @return allocated components_t; initialized empty
 */
struct mixture_model_t *create_components() {
    struct mixture_model_t *vec = (
        (struct mixture_model_t *) malloc(sizeof(struct mixture_model_t)));
    vec->mem_size = BASE_VEC_SIZE;
    vec->size = 0;
    vec->values = malloc(sizeof(struct mixture_model_t) * vec->size);
    return vec;
}


// todo
void destroy_components(void *components)
{
    for(int i = 0; i < components->size; i++) {
        components->comp_methods->destroy(components->values[i]);
        free(components->values[idx]);
    }
    free(components);
}


/**
 * Add Component: allocates new component, and appends to components capsule
 * @param components: components_t struct containing components
 */
void add_component(struct mixture_model_t *components)
{
    // Handle exponential over-allocation
    if(components->mem_size >= components->size) {
        components->components = (void *) realloc(typeof(void *) * size * 2);
        components->mem_size *= 2;
    }

    // Allocate new
    components->values[components->size] = (
        components->comp_methods->create(components->comp_params));
    components->size += 1;
}


/**
 * Remove component from component vector.
 * @param components : component vector struct
 * @param idx : index to remove
 */
void remove_component(struct component_t *components, int idx)
{
    components->comp_methods->destroy(components->values[idx]);
    free(components->values[idx]);
    for(int i = idx; i < (components->size - 1); i++) {
        components[i] = components[i + 1];
    }
}

