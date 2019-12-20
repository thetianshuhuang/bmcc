/** 
 * Core Mixture Struct shortcuts
 */

#ifndef MIXTURE_SHORTCUTS_H
#define MIXTURE_SHORTCUTS_H

#include "../include/mixture.h"

// Get marginal log likelihood
double marginal_loglik(
	struct mixture_model_t *model, Component *cluster, void *point);

// Get new cluster log likelihood
double new_cluster_loglik(struct mixture_model_t *model, void *point);

// Add point to cluster
void add_point(
	struct mixture_model_t *model, Component *cluster, void *point);

// Remove point from cluster
void remove_point(
	struct mixture_model_t *model, Component *cluster, void *point);

// Delete cluster (held separately -- not in model)
void destroy(struct mixture_model_t *model, Component *cluster);

// Get cluster at index, safely
Component *get_cluster(struct mixture_model_t *model, int idx);

#endif
