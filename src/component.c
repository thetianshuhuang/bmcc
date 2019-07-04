

struct components_t *create_components() {
    struct components_t *vec = (
        (struct components_t *) malloc(sizeof(struct components_t)));
    vec->mem_size = BASE_VEC_SIZE;
    vec->size = 0;
    vec->values = malloc(sizeof(struct components_t) * vec->size);
    return vec;
}


/**
 * Add Component: allocates new component, and appends to components capsule
 * @param components: components_t struct containing components
 * @param methods : component methods
 */
void add_component(
    struct components_t *components,
    struct component_methods_t methods,
    void *params)
{
    // Handle exponential over-allocation
    if(components->mem_size >= components->size) {
        components->components = (void *) realloc(typeof(void *) * size * 2);
        components->mem_size *= 2;
    }

    // Allocate new
    components->values[components->size] = methods->create(params);
    components->size += 1;
}


/**
 * Remove component from component vector.
 * @param components : component vector struct
 * @param methods : routines for components (including dealloc)
 * @param idx : index to remove
 */
void remove_component(
    struct component_t *components,
    struct component_methods_t methods,
    int idx)
{
    methods->destroy(components->values[idx]);
    free(components->values[idx]);
    for(int i = idx; i < (components->size - 1); i++) {
        components[i] = components[i + 1];
    }
}

