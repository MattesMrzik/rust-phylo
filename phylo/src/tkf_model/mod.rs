use std::cell::RefCell;
use std::fmt::Display;

use fixedbitset::FixedBitSet;
use lazy_static::lazy_static;
use nalgebra::{DMatrix, DVector};

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ModelSearchCost, ParamRange};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{FreqVector, QMatrix, SubstitutionCost};
use crate::tree::NodeIdx::{self, Internal, Leaf};

lazy_static! {
    static ref DUMMY_FREQS: DVector<f64> = DVector::<f64>::zeros(0);
}

static DEFAULT_LAMBDA: f64 = 1.0;
static DEFAULT_MU: f64 = 1.1;
static DEFAULT_LAMBDA_MU_RATIO: f64 = 0.9;
static DEFAULT_R: f64 = 0.5;

pub mod tkf91;
pub use tkf91::*;
pub mod tkf92;
pub use tkf92::*;

#[derive(Copy, Clone)]
enum Action {
    Insertion,
    Deletion,
    Homolog,
    Nothing,
}

#[allow(clippy::upper_case_acronyms)]
pub trait TKFModel: Clone + Display {
    // TODO: it might be better for model optimisation to have parameter lambda and scale s = mu/lambda,
    // because of the constraint that mu > lambda.
    fn lambda(&self) -> f64;
    fn mu(&self) -> f64;
    // TKF91 has 2 parameters: lambda and mu, TKF92 has 3 parameters: lambda, mu and r.
    fn params(&self) -> &[f64];
    fn set_param(&mut self, idx: usize, value: f64) -> bool;
    fn param_range(&self, idx: usize) -> ParamRange;
    fn insertion_prob_at_root(&self) -> f64;
    fn insertion_prob_at_non_root(&self, beta: f64) -> f64;
    fn block_prob(&self, x: f64, block_len: usize) -> f64;
    fn get_blocks<AA: AncestralAlignment>(msa: &AA) -> Vec<usize>;
}

// TODO: link our paper once it is published. For now see original TKF92 paper: https://doi.org/10.1007/bf00163848
#[derive(Clone, Debug)]
struct TKFIndelModelInfo {
    /// aggregated_x[(node, block)] = the product of the xs of all the edges in the subtree below
    /// <node> (including the x of <node> itself) for the block with id <block>.
    aggregated_x: DMatrix<f64>,

    /// node_x[(node, block)] = the x value for the edge above <node> for the block with id <block>.
    node_x: DMatrix<f64>,

    /// node_factor_n[(node, block)] = the factor_n value for the edge above <node> for the block
    /// with id <block>.
    node_factor_n: DMatrix<f64>,

    /// factor_ns[(node, block)] = n1/ (n0 * lambda * beta(v.blen)) if there is a node <v> in the
    /// subtree rooted in <node> where the current event is an insertion and the previous one was a deletion.
    factor_ns: DMatrix<f64>,

    /// n0[node] = n0(node.blen)
    n0: Vec<f64>,

    /// h1[node] = h1(node.blen)
    h1: Vec<f64>,

    /// insertion[node], may hold previously computed values for insertion
    /// that can be reused.
    insertion: Vec<f64>,

    /// insertion_value_valid[node] = true if insertion[node] holds a valid value.
    insertion_value_valid: FixedBitSet,

    /// factor_n[node] = n1/ (n0 * lambda * beta(node.blen)), may hold previously computed
    /// values for factor_n that can be reused.
    factor_n: Vec<f64>,

    /// factor_n_value_valid[node] = true if factor_n[node] holds a valid value.
    factor_n_value_valid: FixedBitSet,

    /// beta[node] = beta(node.blen)).
    beta: Vec<f64>,

    /// The right exclusive interval borders of the blocks.
    blocks: Vec<usize>,

    /// The lengths of the blocks.
    block_lengths: Vec<usize>,

    /// last_event_deletion[node] = true if the last event was a deletion for a that <node>.
    previous_action_deletion: Vec<bool>,

    /// valid[node] = true if the intermediate values for that <node> are valid.
    valid: Vec<bool>,
}

impl TKFIndelModelInfo {
    fn new<AA: AncestralAlignment, T: TKFModel>(phylo: &PhyloInfo<AA>) -> TKFIndelModelInfo {
        let blocks = T::get_blocks(&phylo.msa);
        let block_lengths = get_block_lengths(&blocks);
        let n_blocks = blocks.len();
        let n_nodes = phylo.tree.len();
        TKFIndelModelInfo {
            aggregated_x: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            node_x: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            node_factor_n: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            factor_ns: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            n0: vec![0.0; n_nodes],
            h1: vec![0.0; n_nodes],
            insertion: vec![0.0; n_nodes],
            insertion_value_valid: FixedBitSet::with_capacity(n_nodes),
            factor_n: vec![0.0; n_nodes],
            factor_n_value_valid: FixedBitSet::with_capacity(n_nodes),
            beta: vec![0.0; n_nodes],
            blocks,
            block_lengths,
            previous_action_deletion: vec![false; n_nodes],
            valid: vec![false; n_nodes],
        }
    }
}

#[derive(Debug)]
pub struct TKFIndelCost<AA: AncestralAlignment, T: TKFModel> {
    model: T,
    phylo: PhyloInfo<AA>,
    model_info: RefCell<TKFIndelModelInfo>,
}

impl<AA: AncestralAlignment + Clone, T: TKFModel> Clone for TKFIndelCost<AA, T> {
    fn clone(&self) -> Self {
        TKFIndelCost {
            model: self.model.clone(),
            phylo: self.phylo.clone(),
            model_info: RefCell::new(self.model_info.borrow().clone()),
        }
    }
}

impl<AA: AncestralAlignment, T: TKFModel> Display for TKFIndelCost<AA, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

impl<AA: AncestralAlignment, T: TKFModel> TKFIndelCost<AA, T> {
    fn logl(&self) -> f64 {
        for node_idx in self.phylo.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    if self.phylo.tree.root == *node_idx {
                        self.set_root();
                    } else {
                        self.set_non_root(node_idx);
                    }
                }
                Leaf(_) => {
                    self.set_non_root(node_idx);
                }
            };
        }

        let l: f64 = self.model.lambda();
        let m = self.model.mu();
        let root_id = usize::from(self.phylo.tree.root);
        let mut logl = 0.0;
        logl += (1.0 - l / m).ln();
        for node in self.phylo.tree.postorder() {
            if node == &self.phylo.tree.root {
                continue;
            }
            logl += log_i1(l, self.model_info.borrow_mut().beta[usize::from(node)]);
        }
        for block_id in 0..self.model_info.borrow().blocks.len() {
            let block_len = self.model_info.borrow().block_lengths[block_id];
            logl += self.model_info.borrow().factor_ns[(root_id, block_id)];
            let x = self.model_info.borrow().aggregated_x[(root_id, block_id)];
            logl += self.model.block_prob(x, block_len);
        }
        logl
    }

    fn set_root(&self) {
        let root_idx = &self.phylo.tree.root;
        if self.model_info.borrow().valid[usize::from(root_idx)] {
            return;
        }
        self.reset_cached_factors(root_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            let x = self.get_indel_x_for_root(block_id);
            self.set_node_values(root_idx, block_id, x, 0.0);
        }
        self.model_info.borrow_mut().valid[usize::from(root_idx)] = true;
    }

    fn set_non_root(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        if self.model_info.borrow().valid[node_id] {
            return;
        }
        self.reset_cached_factors(node_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            let action = self.get_action(node_idx, block_id);
            let x = self.get_indel_x_for_non_root(node_idx, action);
            let factor_n = self.get_factor_n_for_non_root(node_idx, action);
            self.set_node_values(node_idx, block_id, x, factor_n);
            self.update_previous_event(node_idx, action);
        }

        if let Some(parent_idx) = self.phylo.tree.parent(node_idx) {
            self.model_info.borrow_mut().valid[usize::from(parent_idx)] = false;
        }
        self.model_info.borrow_mut().valid[node_id] = true;
    }

    fn update_previous_event(&self, node_idx: &NodeIdx, action: Action) {
        let node_id = usize::from(node_idx);
        match action {
            Action::Deletion => {
                self.model_info.borrow_mut().previous_action_deletion[node_id] = true;
            }
            Action::Insertion | Action::Homolog => {
                self.model_info.borrow_mut().previous_action_deletion[node_id] = false;
            }
            Action::Nothing => {}
        }
    }

    fn reset_cached_factors(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let blen = self.phylo.tree.node(node_idx).blen;
        let beta = b(lambda, mu, blen);
        self.model_info.borrow_mut().beta[node_id] = beta;
        self.model_info.borrow_mut().n0[node_id] = n0(mu, beta);
        self.model_info.borrow_mut().h1[node_id] = h1(lambda, mu, beta, blen);
        self.model_info.borrow_mut().previous_action_deletion[node_id] = false;
        self.model_info
            .borrow_mut()
            .insertion_value_valid
            .set(node_id, false);
        self.model_info
            .borrow_mut()
            .factor_n_value_valid
            .set(node_id, false);
        self.model_info.borrow_mut().valid[node_id] = false;
    }

    fn set_node_values(&self, node_idx: &NodeIdx, block_id: usize, mut x: f64, mut factor_n: f64) {
        let node_id = usize::from(node_idx);
        self.model_info.borrow_mut().node_x[(node_id, block_id)] = x;
        self.model_info.borrow_mut().node_factor_n[(node_id, block_id)] = factor_n;
        for child in &self.phylo.tree.node(node_idx).children {
            let child_id = usize::from(child);
            x *= self.model_info.borrow().aggregated_x[(child_id, block_id)];
            factor_n += self.model_info.borrow().factor_ns[(child_id, block_id)];
        }
        self.model_info.borrow_mut().factor_ns[(node_id, block_id)] = factor_n;
        self.model_info.borrow_mut().aggregated_x[(node_id, block_id)] = x;
    }

    fn get_indel_x_for_root(&self, block_id: usize) -> f64 {
        let root_idx = &self.phylo.tree.root;
        let site = self.model_info.borrow().blocks[block_id] - 1;
        let char_at_root = self.phylo.msa.ancestral_map(root_idx)[site].is_some();
        if char_at_root {
            if self
                .model_info
                .borrow()
                .insertion_value_valid
                .contains(usize::from(root_idx))
            {
                return self.model_info.borrow().insertion[usize::from(root_idx)];
            } else {
                let insertion_prob = self.model.insertion_prob_at_root();
                self.model_info.borrow_mut().insertion[usize::from(root_idx)] = insertion_prob;
                self.model_info
                    .borrow_mut()
                    .insertion_value_valid
                    .set(usize::from(root_idx), true);
                return insertion_prob;
            }
        }
        1.0
    }

    fn get_action(&self, node_idx: &NodeIdx, block_id: usize) -> Action {
        if block_id == 0 {
            self.model_info.borrow_mut().previous_action_deletion[usize::from(node_idx)] = false;
        }
        let parent_idx = self.phylo.tree.node(node_idx).parent.unwrap();
        let site = self.model_info.borrow().blocks[block_id] - 1;
        let parent_is_gap = match parent_idx {
            Internal(_) => self.phylo.msa.ancestral_map(&parent_idx)[site].is_none(),
            _ => unreachable!("The parent of a node cannot be a leaf."),
        };
        let current_is_gap = match node_idx {
            Internal(_) => self.phylo.msa.ancestral_map(node_idx)[site].is_none(),
            Leaf(_) => self.phylo.msa.leaf_map(node_idx)[site].is_none(),
        };
        if !parent_is_gap && current_is_gap {
            Action::Deletion
        } else if !parent_is_gap && !current_is_gap {
            Action::Homolog
        } else if parent_is_gap && !current_is_gap {
            Action::Insertion
        } else {
            Action::Nothing
        }
    }

    fn get_factor_n_for_non_root(&self, node_idx: &NodeIdx, action: Action) -> f64 {
        if matches!(action, Action::Insertion)
            && self.model_info.borrow().previous_action_deletion[usize::from(node_idx)]
        {
            let node_id = usize::from(node_idx);
            let lambda = self.model.lambda();
            let mu = self.model.mu();
            let beta = self.model_info.borrow().beta[node_id];
            let blen = self.phylo.tree.node(node_idx).blen;

            let n0_option = self.model_info.borrow().n0[node_id];
            if self
                .model_info
                .borrow()
                .factor_n_value_valid
                .contains(node_id)
            {
                return self.model_info.borrow().factor_n[node_id];
            } else {
                let mut factor_n = log_n1(lambda, mu, beta, blen);
                factor_n -= (lambda * beta).ln();
                // since last event was a deletion n0 is not None
                factor_n -= n0_option.ln();
                self.model_info.borrow_mut().factor_n[node_id] = factor_n;
                self.model_info
                    .borrow_mut()
                    .factor_n_value_valid
                    .set(node_id, true);
                factor_n
            }
        } else {
            0.0
        }
    }

    fn get_indel_x_for_non_root(&self, node_idx: &NodeIdx, action: Action) -> f64 {
        let node_id = usize::from(node_idx);
        let beta = self.model_info.borrow().beta[node_id];

        match action {
            Action::Deletion => self.model_info.borrow_mut().n0[node_id],
            Action::Homolog => self.model_info.borrow().h1[node_id],
            Action::Insertion => {
                if self
                    .model_info
                    .borrow()
                    .insertion_value_valid
                    .contains(node_id)
                {
                    self.model_info.borrow().insertion[node_id]
                } else {
                    let insertion_prob = self.model.insertion_prob_at_non_root(beta);
                    self.model_info.borrow_mut().insertion[node_id] = insertion_prob;
                    self.model_info
                        .borrow_mut()
                        .insertion_value_valid
                        .set(node_id, true);
                    insertion_prob
                }
            }
            Action::Nothing => 1.0,
        }
    }
}

impl<AA: AncestralAlignment, T: TKFModel> ModelSearchCost for TKFIndelCost<AA, T> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn param_count(&self) -> usize {
        self.model.params().len()
    }

    fn param(&self, idx: usize) -> f64 {
        self.model.params()[idx]
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        if self.model.set_param(idx, value) {
            self.model_info.borrow_mut().valid.fill(false);
        }
    }

    /// Returns the valid range for a model parameter [min, max], inclusive.
    /// Assumes that current parameter values are valid.
    fn param_range(&self, idx: usize) -> ParamRange {
        self.model.param_range(idx)
    }

    fn set_freqs(&mut self, _: FreqVector) {}

    fn empirical_freqs(&self) -> FreqVector {
        // TODO: At the time of writing this, this method is only used to set the frequencies of
        // the model, but the TKF92IndelCost does not have frequencies.
        self.phylo.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        // TODO: Alternatively, we don't implement costs for just the indel part, or we make the
        // trait def to return an Option<&FreqVector>.
        &DUMMY_FREQS
    }
}

#[derive(Clone, Debug)]
pub struct TKFCost<Q: QMatrix + Display, T: TKFModel, AA: AncestralAlignment> {
    // TODO: if we have just the sum of the two costs like this, we need to keep track of the
    // phylo (which is tree and alignment) twice, which might be too big of a downside, since the
    // cost is copied often. Alternatively we could implement the substitution cost inside the
    // tkf92 cost, which would duplicate some code.
    indel_cost: TKFIndelCost<AA, T>,
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
        self.indel_cost.cost() + self.subst_cost.cost()
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

fn log_i1(lambda: f64, beta: f64) -> f64 {
    (1.0 - lambda * beta).ln()
}

// TKF beta function
fn b(lambda: f64, mu: f64, time: f64) -> f64 {
    let exp_term = ((lambda - mu) * time).exp();
    (1.0 - exp_term) / (mu - lambda * exp_term)
}

fn h1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    (-mu * time).exp() * (1.0 - lambda * beta)
}

#[inline]
fn n0(mu: f64, beta: f64) -> f64 {
    mu * beta
}

fn log_n1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    ((1.0 - (-mu * time).exp() - mu * beta) * (1.0 - lambda * beta)).ln()
}

/// Given the right exclusive block borders, returns the lengths of the blocks.
/// For example, given [3, 5, 8], the block lengths are [3, 2, 3].
fn get_block_lengths(blocks: &[usize]) -> Vec<usize> {
    let mut block_lens = vec![0; blocks.len()];
    for (i, block) in blocks.iter().enumerate() {
        block_lens[i] = if i == 0 {
            *block
        } else {
            block - blocks[i - 1]
        };
    }
    block_lens
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
