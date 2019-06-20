

/**
 * Gibbs sampling initialization
 * @param data : input data; stored in row-major order
 *      (i.e. idx, x -> idx * dim + x)
 * @param assignments : array of pointers to set objects. Should start as null
 * @param size : data size
 * @param dim : data dimension
 * @param r : pCRP power penalty
 * @param alpha : CRP clustering hyperparameter
 * @param l_cond : conditional likelihood function
 * @param l_uncond : unconditional likelihood function
 */
std::vector<std::set<int>> crp_init(
    float *data, std::set<int> *assignments, int size, int dim,
    float r, float alpha,
    (float) l_cond(float **, std::set<int>, int, int),
    (float) l_uncond(float **, int, int))
{

    // Initialize clusters
    std::vector<std::set<int>> clusters = new std::vector<std::set<int>>;
    int num_clusters = 0;

    // Apply CRP to data points
    for(int i = 0; i < size; i++) {

        // Initialize weight vector
        float *weights = new float[num_clusters + 1];

        // Iterate over current points in each cluster
        std::vector<std::set<int>>::iterator it;
        for(it = clusters.begin(); it != clusters.end(); ++it) {
            weights[j] = l_cond(data, *it, i, dim) * pow(*it.size(), r);
        }
        // Sample new cluster
        weights[num_clusters + 1] = l_uncond(data, i, dim) * alpha;
        int assign = sample_proportional(weights, num_clusters);

        // Assign to new cluster
        if(assign == nm_clusters) {
            std::set<int> new_cluster = new std::set<int>;
            new_cluster.insert(i);
            clusters.push_back(new_cluster);
            assignments[i] = &new_cluster;
        }
        // Assign to existing cluster
        else {
            clusters[assign].insert(i);
            assignments[i] = &clusters[assign];
        }

        delete weights;
    }

    return clusters;
}


/**
 * Gibbs sampling step
 */
gibbs_sampling(
    float **data,
    std::set<int> *assignments,
    std::vector<std::set<int>> clusters,
    int size, int dim, float r, float alpha,
    (float) l_cond(float **, std::set<int>, int, int),
    (float) l_uncond(float **, int, int))
{
    // Shuffle data points
    int *shuffle = new int[size];
    for(int i = 0; i < size; i++) { shuffle[i] = i; }
    std::random_shuffle(std::begin(shuffle), std::end(shuffle));

    // Iterate on each point assignment
    for(int i = 0; i < size; i++) {
        // Remove from current cluster
        *(assigments[i]).erase(i);

        // Initialize weight vector
        float *weights = new float[num_clusters + 1];

        // Iterate over current points in each cluster
        std::vector<std::set<int>>::iterator it;
        for(it = clusters.begin(); it != clusters.end(); ++it) {
            // Make sure cluster is not empty
            if(*it.size() > 0 && it != ) {
                weights[j] = l_cond(data, *it, i, dim) * pow(*it.size(), r);
            }
        }
        // Sample new cluster
        weights[num_clusters + 1] = l_uncond(data, i, dim) * alpha;
        int assign = sample_proportional(weights, num_clusters);

        // Assign to new cluster
        if(assign == nm_clusters) {
            std::set<int> new_cluster = new std::set<int>;
            new_cluster.insert(i);
            clusters.push_back(new_cluster);
            assignments[i] = &new_cluster;
        }
        // Assign to existing cluster
        else {
            clusters[assign].insert(i);
            assignments[i] = &clusters[assign];
        }

        delete weights;
    }
}








void gibbs_iteration(
    float *data,
    uint16_t *assignments,
    std::vector<std::set<int>> clusters,
    int *size, int *dim, float r, float alpha,
    (float) l_cond(float *, std::set<int>, int, int),
    (float) l_uncond(float *, int, int))
{
    // Shuffle data points
    int *shuffle = new int[size];
    for(int i = 0; i < size; i++) { shuffle[i] = i; }
    std::random_shuffle(std::begin(shuffle), std::end(shuffle));

    // Iterate on point assignment
    for(int i = 0; i < size; i++) {
        // Remove from current cluster
        clusters[assignments[i]].erase(i);

        // Intialize weight vector
        float *weights = new float[clusters.size() + 1];

        // Track first empty index to reuse empty clusters instead of
        // expanding (which causes the vector cluster to expand)
        int empty_idx = -1;

        // Iterate over current points in each cluster
        for(int i = 0; i < clusters.size(); i++) {
            // Add to weight vector
            if(clusters[i].size() > 0) {
                weights[j] = l_cond(data, *it, i, dim) * pow(*it.size(), r);
            }
            // Don't add to weight vector; track empty idx
            else {
                empty_idx = i;
                weights[j] = 0;
            }
        }

        // New cluser probability
        weights[num_clusters + 1] = l_uncond(data, i, dim) * alpha;

        // Sample new assignment
        int assign = sample_proportional(weights, num_clusters);
    
        // Assign to new cluster
        if(assign == clusters.size()) {
            // Reuse already-allocated existing
            if(empty_idx != -1) {
                clusters[empty_idx].insert(i);
                assignments[i] = empty_idx;
            }
            // Add new
            else {
                std::set<int> new_cluster = new std::set<int>;
                new_cluster.insert(i);
                assignments[i] = clusters.size();
                clusters.push_back(new_cluster);
            }
        }
        // Add to existing cluster
        else {
            clusters[assign].insert(i);
            assignments[i] = assign;
        }

        // Clean up
        delete weights;
    }
}