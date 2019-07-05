
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


/**
 * Destroy mixture model struct
 * @param model : mixture_model_t struct to destroy
 */
void destroy_components(void *model)
{
    struct mixture_model_t *model_tc = (struct mixture_model_t *) model_py;
    for(int i = 0; i < model_tc->num_clusters; i++) {
        model_tc->comp_methods->destroy(model_tc->clusters[i]);
        free(model_tc->clusters[idx]);
    }
    free(model_tc);
}


/**
 * Add Component: allocates new component, and appends to components capsule
 * @param components: components_t struct containing components
 */
void add_component(struct mixture_model_t *model)
{
    // Handle exponential over-allocation
    if(model->mem_size >= model->num_clusters) {
        model->clusters = (void *) realloc(
            typeof(void *) * clusters->mem_size * 2);
        model->mem_size *= 2;
    }

    // Allocate new
    model->values[model->num_clusters] = (
        model->comp_methods->create(model->comp_params));
    model->num_clusters += 1;
}


/**
 * Remove component from component vector.
 * @param components : component vector struct
 * @param idx : index to remove
 */
void remove_component(struct mixture_model_t *model, int idx)
{
    model->comp_methods->destroy(model->clusters[idx]);
    free(model->clusters[idx]);
    for(int i = idx; i < (model->num_clusters - 1); i++) {
        model->clusters[i] = model->clusters[i + 1];
    }
}


/**
 * Remove empty components.
 * @param components components_t vector to remove from
 * @param assignments assignment vector; indices in assignments greater than
 *      the empty vector index are decremented.
 * @return true if a empty component was removed.
 */
bool remove_empty(struct mixture_model_t *model, uint16_t *assignments)
{
    // Search for empty
    for(int i = 0; i < model->num_clusters; i++) {
        if(get_size((model->clusters)[i]) == 0) {
            // Deallocate component; remove component from vector
            remove_component(model, i);
            // Decrement values
            for(int j = 0; j < size; j++) {
                if(assignments[j] > i) { assignments[j] -= 1; }
            }
            model->num_clusters -= 1;
            return true;
        }
    }
    return false;
}

