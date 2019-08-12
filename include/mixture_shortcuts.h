/** 
 * Core Mixture Struct shortcuts
 */

#ifndef MIXTURE_SHORTCUTS_H
#define MIXTURE_SHORTCUTS_H

#include "../include/mixture.h"

double marginal_loglik(
	struct mixture_model_t *model, void *cluster, double *point);

double new_cluster_loglik(struct mixture_model_t *model, double *point);

void *add_point(struct mixture_model_t *model, void *cluster, double *point);

void destroy(struct mixture_model_t *model, void *cluster);

#endif
