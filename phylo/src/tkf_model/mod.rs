use std::fmt::Display;

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ModelSearchCost, ParamRange, TreeSearchCost};
use crate::substitution_models::{FreqVector, QMatrix, SubstitutionCost};
use crate::tree::Tree;

pub mod tkf91;
pub use tkf91::*;
pub mod tkf92;
pub use tkf92::*;
pub mod reestimate;
pub mod tkf92_fixed;
pub use reestimate::*;
pub mod tkf_indel;
pub use tkf_indel::*;

#[derive(Clone, Debug)]
pub struct TKFCost<Q: QMatrix + Display, T: TKFModel, AA: AncestralAlignment> {
    // TODO: if we have just the sum of the two costs like this, we need to keep track of the
    // phylo (which is tree and alignment) twice, which might be too big of a downside, since the
    // cost is copied often. Alternatively we could implement the substitution cost inside the
    // tkf92 cost, which would duplicate some code.
    indel_cost: TKFIndelCost<T, AA>,
    subst_cost: SubstitutionCost<Q, AA>,
}

impl<Q: QMatrix, T: TKFModel, AA: AncestralAlignment> Display for TKFCost<Q, T, AA> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} and {}",
            self.indel_cost.model, self.subst_cost.model.qmatrix
        )
    }
}

impl<Q: QMatrix, T: TKFModel, AA: AncestralAlignment> ModelSearchCost for TKFCost<Q, T, AA> {
    fn cost(&self) -> f64 {
        ModelSearchCost::cost(&self.subst_cost) + ModelSearchCost::cost(&self.indel_cost)
    }

    fn param_count(&self) -> usize {
        self.indel_cost.model.params().len() + self.subst_cost.model.qmatrix.params().len()
    }

    fn param(&self, idx: usize) -> f64 {
        let num_params_indel_model = self.indel_cost.param_count();
        if idx < num_params_indel_model {
            return self.indel_cost.param(idx);
        }
        let idx = idx - num_params_indel_model;
        self.subst_cost.param(idx)
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        let num_params_indel_model = self.indel_cost.param_count();
        if idx < num_params_indel_model {
            self.indel_cost.set_param(idx, value);
            return;
        }
        let idx = idx - num_params_indel_model;
        self.subst_cost.set_param(idx, value);
    }

    fn param_range(&self, idx: usize) -> ParamRange {
        let num_params_indel_model = self.indel_cost.param_count();
        if idx < num_params_indel_model {
            return self.indel_cost.param_range(idx);
        }
        let idx = idx - num_params_indel_model;
        self.subst_cost.param_range(idx)
    }

    fn set_freqs(&mut self, freqs: FreqVector) {
        self.subst_cost.set_freqs(freqs);
    }

    fn empirical_freqs(&self) -> FreqVector {
        self.subst_cost.info.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        self.subst_cost.freqs()
    }
}

impl<Q: QMatrix, T: TKFModel, AA: AncestralAlignment> TreeSearchCost for TKFCost<Q, T, AA> {
    fn cost(&self) -> f64 {
        TreeSearchCost::cost(&self.subst_cost) + TreeSearchCost::cost(&self.indel_cost)
    }

    fn update_tree(&mut self, tree: Tree) {
        self.indel_cost.update_tree(tree.clone());
        self.subst_cost.update_tree(tree.clone());
    }

    fn tree(&self) -> &Tree {
        self.indel_cost.tree()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod reestimate_tests;

// TODO: remove this before merge
#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod non_git_tkf;
