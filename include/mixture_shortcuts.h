/** 
 * Core Mixture Struct shortcuts
 */

#ifndef MIXTURE_SHORTCUTS_H
#define MIXTURE_SHORTCUTS_H

#include "../include/mixture.h"

// Get marginal log likelihood
double marginal_loglik(
	struct mixture_model_t *model, void *cluster, double *point);

// Get new cluster log likelihood
double new_cluster_loglik(struct mixture_model_t *model, double *point);

// Add point to cluster
void *add_point(struct mixture_model_t *model, void *cluster, double *point);

// Delete cluster (held separately -- not in model)
void destroy(struct mixture_model_t *model, void *cluster);

// Get cluster at index, safely
void *get_cluster(struct mixture_model_t *model, int idx);

#endif
