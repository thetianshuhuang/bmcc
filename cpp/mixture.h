/**
 *
 */

#ifndef MIXTURE_H
#define MIXTURE_H

#include <component.h>


/*
 * FT: Data Type
 * CT: Component Type
 * PT: Hyperparameter Type
 */
template <FT, CT, PT>
class Mixture {

	private:
		// Model hyperparameter
		// (alpha for DPM, gamma for MFM, or something else)
		PT params;
		// Model coefficients
		FT (*model_coef)(int size, PT &params);
		// Model new component coefficient
		FT (*model_new_coef)(int size, PT &params);
		// Update parameters
		void (*update_params)(PT &params, Mixture &mixture);

	public:
		// Constructor
		Mixture(arma::Mat<FT> data, arma::Col<FT> assignments, PT params);
		// Gibbs update
		gibbs_iter();

		// Data
		arma::Mat<FT> data;
		// Assignments
		arma::Col<int> assignments;
		// Components of type CT; pass FT to component type
		std::vector<CT::FT> components;
}

