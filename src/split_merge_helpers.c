/**
 * Split Merge Helpers
 */

#include "../include/mixture.h"
#include "../include/mixture_shortcuts.h"

/**
 * Reconstruct proposal probability for merge operation
 * @param c1 first cluster index
 * @param c2 second cluster index
 * @param data data array; row-major
 * @param assignments assignment vector
 * @param components vector containing component structs, stored as void *.
 * @return proposal probability
 */
double merge_propose_prob(
	int c1, int c2,
	double *data, uint16_t *assignments,
	struct mixture_model_t *model)
{
    // Temporary clusters
    void *c1_cpy = model->comp_methods->create(model->comp_params);
    void *c2_cpy = model->comp_methods->create(model->comp_params);

    // Iterate over clusters c1, c2
	double res = 0;
	for(int i = 0; i < model->size; i++) {
        if(assignments[i] == c1 || assignments[i] == c2) {
            double *point = &(data[i * model->dim]);

            // Likelihoods
            double asn1 = marginal_loglik(
            	model, get_cluster(model, c1), point);
            double asn2 = marginal_loglik(
            	model, get_cluster(model, c2), point);
            res += (assignments[i] == c1) ? asn1 : asn2;

            // Normalization Constant
            res -= log(exp(asn1) + exp(asn2));
        }
    }

    // Clean up
    destroy(model, c1_cpy);
    destroy(model, c2_cpy);
    return res;
}
