/**
 * Split Merge Helpers
 */

#ifndef SPLIT_MERGE_HELPERS_H
#define SPLIT_MERGE_HELPERS_H

#include "../include/mixture.h"

double merge_propose_prob(
	int c1, int c2,
	double *data, uint16_t *assignments,
	struct mixture_model_t *model);

#endif
