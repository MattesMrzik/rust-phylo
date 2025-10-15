use std::cell::RefCell;
use std::fmt::Display;
use std::sync::LazyLock;

use anyhow::bail;
use hashbrown::HashSet;
use nalgebra::{DMatrix, DVector};

use crate::alignment::AncestralAlignment;
use crate::evolutionary_models::EvoModel;
use crate::likelihood::ModelSearchCost;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{
    FreqVector, QMatrix, SubstModel, SubstitutionCost, SubstitutionCostBuilder as SCB,
};
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::Result;

static DUMMY_FREQS: LazyLock<DVector<f64>> = LazyLock::new(|| DVector::<f64>::zeros(0));

#[derive(Clone)]
pub struct TKF92IndelModel {
    params: Vec<f64>,
}

impl TKF92IndelModel {
    fn lambda(&self) -> f64 {
        self.params[0]
    }
    fn mu(&self) -> f64 {
        self.params[1]
    }
    fn r(&self) -> f64 {
        self.params[2]
    }
    fn params(&self) -> &[f64] {
        &self.params
    }
}

/// For a detailed explanation of the TKF92 model and how the probabilities are calculated see:
/// <coming paper>.
#[derive(Clone)]
pub struct TKF92IndelModelInfo {
    /// aggregated_x\[node, block] = the product of the xs of all the edges in the subtree below
    /// <node> (including the x of <node> itself) for the block with id <block>.
    aggregated_x: DMatrix<f64>,

    /// node_x\[node, block] = the x value for the edge above <node> for the block with id <block>.
    // TODO: this could be optimized to only store the values for the dirty nodes. Then, we would
    // have to determine these values not during the cost calculation but during the tree update.
    node_x: DMatrix<f64>,

    /// node_factor_n\[node, block] = the factor_n value for the edge above <node> for the block
    /// with id <block>.
    // TODO: this could be optimized to only store the values for the dirty nodes. Then, we would
    // have to determine these values not during the cost calculation but during the tree update.
    node_factor_n: DMatrix<f64>,

    /// factor_ns\[node, block] = n1/ (n0 * lambda * beta(v.blen)) if there is a node <v> in the subtree
    // where the current event is an insertion and the previous one was a deletion.
    factor_ns: DMatrix<f64>,

    /// n0\[node] = n0(node.blen), may hold already computed values for n0 that can be reused.
    n0: Vec<Option<f64>>,

    /// h1\[node] = h1(node.blen), may hold already computed values for h1 that can be reused.
    h1: Vec<Option<f64>>,

    /// insertion\[node] = l * beta * (1.0 - r) / r, may hold already computed values for insertion
    /// that can be reused.
    insertion: Vec<Option<f64>>,

    /// factor_n\[node] = n1/ (n0 * lambda * beta(node.blen)), may hold already computed values for factor_n that can be
    /// reused.
    factor_n: Vec<Option<f64>>,

    /// beta\[usize::from(node)] = beta(node.blen)).
    beta: Vec<f64>,

    /// The right exclusive interval borders of the blocks.
    blocks: Vec<usize>,

    /// The lengths of the blocks.
    block_lens: Vec<usize>,

    /// last_event_deletion\[usize::from(node)] = true if the last event was a deletion for a that <node>.
    last_event_deletion: Vec<bool>,

    /// valid\[usize::from(node)] = true if the intermediate values for that <node> are valid.
    valid: Vec<bool>,
}

impl TKF92IndelModelInfo {
    pub fn new<AA: AncestralAlignment>(phylo: &PhyloInfo<AA>) -> TKF92IndelModelInfo {
        let blocks = get_blocks(&phylo.msa);
        let block_lens = get_block_lens(&blocks);
        let n_blocks = blocks.len();
        let n_nodes = phylo.tree.len();
        TKF92IndelModelInfo {
            aggregated_x: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            node_x: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            node_factor_n: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            factor_ns: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            n0: vec![None; n_nodes],
            h1: vec![None; n_nodes],
            insertion: vec![None; n_nodes],
            factor_n: vec![None; n_nodes],
            beta: vec![0.0; n_nodes],
            blocks,
            block_lens,
            last_event_deletion: vec![false; n_nodes],
            valid: vec![false; n_nodes],
        }
    }
}

pub struct TKF92IndelCost<AA: AncestralAlignment> {
    model: TKF92IndelModel,
    phylo: PhyloInfo<AA>,
    model_info: RefCell<TKF92IndelModelInfo>,
}

impl<AA: AncestralAlignment + Clone> Clone for TKF92IndelCost<AA> {
    fn clone(&self) -> Self {
        TKF92IndelCost {
            model: self.model.clone(),
            phylo: self.phylo.clone(),
            model_info: RefCell::new(self.model_info.borrow().clone()),
        }
    }
}

#[derive(Clone)]
pub struct TKF92Cost<Q: QMatrix + Display, AA: AncestralAlignment> {
    // TODO: if we have just the sum of the two costs like this, we need to keep track of the
    // phylo (which is tree and alignment) twice, which might be too big of a downside. Since the
    // cost is copied often. Alternatively we could implement the substitution cost inside the
    // tkf92 cost, which would duplicate some code.
    tkf92_cost: TKF92IndelCost<AA>,
    subst_cost: SubstitutionCost<Q, AA>,
    combined_parameters: Vec<f64>,
}

impl<AA: AncestralAlignment> Display for TKF92IndelCost<AA> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 only indels with lambda = {}, mu = {}, r = {}",
            self.model.lambda(),
            self.model.mu(),
            self.model.r(),
        )
    }
}

impl<Q: QMatrix, AA: AncestralAlignment> Display for TKF92Cost<Q, AA> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with lambda = {}, mu = {}, r = {}, Q = {}",
            self.tkf92_cost.model.lambda(),
            self.tkf92_cost.model.mu(),
            self.tkf92_cost.model.r(),
            self.subst_cost.model.qmatrix
        )
    }
}

struct TKF92CostBuilder<Q: QMatrix, AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    r: f64,
    subst_model: SubstModel<Q>,
    phylo: PhyloInfo<AA>,
}

impl<Q: QMatrix, AA: AncestralAlignment> TKF92CostBuilder<Q, AA> {
    pub fn new(
        lambda: f64,
        mu: f64,
        r: f64,
        subst_model: SubstModel<Q>,
        phylo: PhyloInfo<AA>,
    ) -> Self {
        Self {
            lambda,
            mu,
            r,
            subst_model,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKF92Cost<Q, AA>> {
        if self.phylo.msa.alphabet() != self.subst_model.alphabet() {
            bail!("Alphabet mismatch between model and alignment");
        }

        let tkf92_model = TKF92IndelModel {
            params: vec![self.lambda, self.mu, self.r],
        };
        let tkf92_info = TKF92IndelModelInfo::new(&self.phylo);
        let tkf92_cost = TKF92IndelCost {
            model: tkf92_model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(tkf92_info),
        };
        let combined_parameters = [tkf92_cost.model.params(), self.subst_model.params()].concat();
        Ok(TKF92Cost {
            tkf92_cost,
            subst_cost: SCB::new(self.subst_model, self.phylo).build().unwrap(),
            combined_parameters,
        })
    }
}

impl<AA: AncestralAlignment> ModelSearchCost for TKF92IndelCost<AA> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn set_param(&mut self, i: usize, v: f64) {
        self.model.params[i] = v;
        self.model_info.borrow_mut().valid.fill(false);
    }

    fn params(&self) -> &[f64] {
        self.model.params()
    }

    fn set_freqs(&mut self, _: FreqVector) {}

    fn empirical_freqs(&self) -> FreqVector {
        // TODO: At the time of writing this, this method is only used to set the frequencies of
        // the model, but the TKF92IndelCost does not have frequencies. So we could just return
        // a dummy vector here.
        self.phylo.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        // TODO: Alternatively, we don't implement costs for just the indel part, or we make the
        // trait def to return an Option<&FreqVector>.
        &DUMMY_FREQS
    }
}

impl<Q: QMatrix, AA: AncestralAlignment> ModelSearchCost for TKF92Cost<Q, AA> {
    fn cost(&self) -> f64 {
        self.tkf92_cost.cost() + self.subst_cost.cost()
    }

    fn set_param(&mut self, i: usize, v: f64) {
        // TODO: do we want to check that i is in bounds?
        self.combined_parameters[i] = v;
        if i < 3 {
            self.tkf92_cost.set_param(i, v);
            return;
        }
        let i = i - 3;
        self.subst_cost.set_param(i, v);
        self.tkf92_cost.model_info.borrow_mut().valid.fill(false);
    }

    fn params(&self) -> &[f64] {
        &self.combined_parameters
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

impl<AA: AncestralAlignment> TKF92IndelCost<AA> {
    fn logl(&self) -> f64 {
        // println!("logl start in tkf");
        for node_idx in self.phylo.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    if self.phylo.tree.root == *node_idx {
                        self.set_root();
                    } else {
                        self.set_internal(node_idx);
                    }
                }
                Leaf(_) => {
                    self.set_leaf(node_idx);
                }
            };
        }

        let l: f64 = self.model.lambda();
        let m = self.model.mu();
        let r = self.model.r();
        let root_id = usize::from(self.phylo.tree.root);
        let mut logl = 0.0;
        logl += (1.0 - l / m).ln();
        logl += self.phylo.msa.len() as f64 * r.ln();
        for node in self.phylo.tree.postorder() {
            if node == &self.phylo.tree.root {
                continue;
            }
            logl += log_i1(l, self.model_info.borrow_mut().beta[usize::from(node)]);
        }
        for block_id in 0..self.model_info.borrow().blocks.len() {
            let block_len = self.model_info.borrow().block_lens[block_id];
            logl += self.model_info.borrow().factor_ns[(root_id, block_id)];
            let x = self.model_info.borrow().aggregated_x[(root_id, block_id)];
            if x != 1.0 {
                logl += x.ln();
                logl += (block_len as f64 - 1.0) * (1.0 + x).ln();
            }
        }
        logl
    }

    fn set_root(&self) {
        let root_idx = &self.phylo.tree.root;
        if self.model_info.borrow().valid[usize::from(root_idx)] {
            return;
        }
        // this call should not be necessary because at the root we do not use any cached
        // values, but it does not hurt.
        self.reset_cached_factors(root_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            self.set_indel_x_and_factor_n_for_root(block_id);
        }
        self.model_info.borrow_mut().valid[usize::from(root_idx)] = true;
    }

    fn set_internal(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        if self.model_info.borrow().valid[node_id] {
            return;
        }
        self.reset_cached_factors(node_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            self.set_indel_x_and_factor_n_for_non_root(node_idx, block_id);
        }

        if let Some(parent_idx) = self.phylo.tree.parent(node_idx) {
            self.model_info.borrow_mut().valid[usize::from(parent_idx)] = false;
        }
        self.model_info.borrow_mut().valid[node_id] = true;
    }

    fn set_leaf(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        if self.model_info.borrow().valid[node_id] {
            return;
        }
        self.reset_cached_factors(node_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            // println!("block id = {block_id}");
            self.set_indel_x_and_factor_n_for_non_root(node_idx, block_id);
        }

        if let Some(parent_idx) = self.phylo.tree.parent(node_idx) {
            self.model_info.borrow_mut().valid[usize::from(parent_idx)] = false;
        }
        self.model_info.borrow_mut().valid[node_id] = true;
    }

    fn reset_cached_factors(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let t = self.phylo.tree.node(node_idx).blen;
        self.model_info.borrow_mut().beta[node_id] = b(lambda, mu, t);
        self.model_info.borrow_mut().n0[node_id] = None;
        self.model_info.borrow_mut().h1[node_id] = None;
        self.model_info.borrow_mut().insertion[node_id] = None;
        self.model_info.borrow_mut().factor_n[node_id] = None;
    }

    fn set_indel_x_and_factor_n_for_root(&self, block_id: usize) {
        let mut x = self.get_indel_x_for_root(block_id);
        let mut factor_n = 0.0;
        // this is the same as in set_indel_x_and_prob_for_not_root
        let root_idx = self.phylo.tree.root;
        let root_id = usize::from(root_idx);
        self.model_info.borrow_mut().node_x[(root_id, block_id)] = x;
        self.model_info.borrow_mut().node_factor_n[(root_id, block_id)] = 0.0;
        for child in &self.phylo.tree.node(&root_idx).children {
            let child_id = usize::from(child);
            x *= self.model_info.borrow().aggregated_x[(child_id, block_id)];
            factor_n += self.model_info.borrow().factor_ns[(child_id, block_id)];
        }
        self.model_info.borrow_mut().factor_ns[(root_id, block_id)] = factor_n;
        self.model_info.borrow_mut().aggregated_x[(root_id, block_id)] = x;
    }

    fn set_indel_x_and_factor_n_for_non_root(&self, node_idx: &NodeIdx, block_id: usize) {
        let (mut x, mut factor_n) = self.get_indel_x_and_factor_n_for_non_root(node_idx, block_id);
        let node_id = usize::from(node_idx);
        self.model_info.borrow_mut().node_x[(node_id, block_id)] = x;
        self.model_info.borrow_mut().node_factor_n[(node_id, block_id)] = factor_n;
        // this is the same as in set_indel_x_and_prob_for_root
        for child in &self.phylo.tree.node(node_idx).children {
            let child_id = usize::from(child);
            x *= self.model_info.borrow().aggregated_x[(child_id, block_id)];
            factor_n += self.model_info.borrow().factor_ns[(child_id, block_id)];
        }
        self.model_info.borrow_mut().factor_ns[(node_id, block_id)] = factor_n;
        self.model_info.borrow_mut().aggregated_x[(node_id, block_id)] = x;
    }

    fn get_indel_x_for_root(&self, block_id: usize) -> f64 {
        let l = self.model.lambda();
        let m = self.model.mu();
        let r = self.model.r();

        let root_idx = &self.phylo.tree.root;
        if self.phylo.msa.ancestral_map(root_idx)[self.model_info.borrow().blocks[block_id] - 1]
            .is_some()
        {
            return l / m * (1.0 - r) / r;
        }
        1.0
    }

    fn get_indel_x_and_factor_n_for_non_root(
        &self,
        node_idx: &NodeIdx,
        block_id: usize,
    ) -> (f64, f64) {
        let parent_idx = self.phylo.tree.node(node_idx).parent.unwrap();
        let node_id = usize::from(node_idx);
        let mut factor_n = 0.0;
        let mut x: f64 = 1.0;
        let site = self.model_info.borrow().blocks[block_id] - 1;
        // println!("before getting stuff in et_indel_x_and_factor_n_for_not_root");
        let parent_is_gap = match parent_idx {
            Internal(_) => self.phylo.msa.ancestral_map(&parent_idx)[site].is_none(),
            Leaf(_) => self.phylo.msa.leaf_map(&parent_idx)[site].is_none(),
        };
        let current_is_gap = match node_idx {
            Internal(_) => self.phylo.msa.ancestral_map(node_idx)[site].is_none(),
            Leaf(_) => self.phylo.msa.leaf_map(node_idx)[site].is_none(),
        };

        let time = self.phylo.tree.node(node_idx).blen;
        // println!("time in et_indel_x_and_factor_n_for_not_root= {time}");

        if block_id == 0 {
            self.model_info.borrow_mut().last_event_deletion[node_id] = false;
        }

        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let r = self.model.r();
        let beta = self.model_info.borrow().beta[node_id];

        if !parent_is_gap && current_is_gap {
            // deletion
            x *= *self.model_info.borrow_mut().n0[node_id].get_or_insert_with(|| n0(mu, beta));
            self.model_info.borrow_mut().last_event_deletion[node_id] = true;
        } else if !parent_is_gap && !current_is_gap {
            // homolog
            x *= *self.model_info.borrow_mut().h1[node_id]
                .get_or_insert_with(|| h1(lambda, mu, beta, time));
            self.model_info.borrow_mut().last_event_deletion[node_id] = false;
        } else if parent_is_gap && !current_is_gap {
            // insertion
            if self.model_info.borrow().last_event_deletion[node_id] {
                let n0_option = self.model_info.borrow().n0[node_id];
                factor_n +=
                    *self.model_info.borrow_mut().factor_n[node_id].get_or_insert_with(|| {
                        let mut factor_n = log_n1(lambda, mu, beta, time);
                        factor_n -= (lambda * beta).ln();
                        // since last event was deletion n0 must be defined
                        factor_n -= n0_option.unwrap().ln();
                        factor_n
                    });
            }
            x *= *self.model_info.borrow_mut().insertion[node_id]
                .get_or_insert_with(|| lambda * beta * (1.0 - r) / r);
            self.model_info.borrow_mut().last_event_deletion[node_id] = false;
        }
        (x, factor_n)
    }
}

fn log_i1(l: f64, b: f64) -> f64 {
    (1.0 - l * b).ln()
}

fn b(l: f64, m: f64, t: f64) -> f64 {
    (1.0 - ((l - m) * t).exp()) / (m - l * ((l - m) * t).exp())
}

fn h1(l: f64, m: f64, b: f64, t: f64) -> f64 {
    (-m * t).exp() * (1.0 - l * b)
}

fn n0(m: f64, b: f64) -> f64 {
    m * b
}

fn log_n1(l: f64, m: f64, b: f64, t: f64) -> f64 {
    ((1.0 - (-m * t).exp() - m * b) * (1.0 - l * b)).ln()
}
fn get_block_lens(blocks: &[usize]) -> Vec<usize> {
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

fn get_blocks<AA: AncestralAlignment>(msa: &AA) -> Vec<usize> {
    let mut blocks: HashSet<usize> = HashSet::new();
    for map in msa
        .ancestral_maps()
        .values()
        .chain(msa.leaf_maps().values())
    {
        let mut last = map[0].is_some();
        for (i, c) in map.iter().skip(1).enumerate() {
            let current: bool = c.is_some();
            if current != last {
                blocks.insert(i + 1);
            }
            last = current;
        }
        blocks.insert(map.len());
    }
    let mut blocks: Vec<usize> = blocks.iter().copied().collect();
    blocks.sort();
    blocks
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
