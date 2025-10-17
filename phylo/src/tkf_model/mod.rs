use std::cell::RefCell;
use std::fmt::Display;
use std::sync::LazyLock;

use anyhow::bail;
use hashbrown::HashSet;
use log::warn;
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

#[allow(clippy::upper_case_acronyms)]
pub trait TKF: Clone + Display {
    fn lambda(&self) -> f64;
    fn mu(&self) -> f64;
    fn params(&self) -> &[f64];
    fn set_param(&mut self, i: usize, v: f64) -> bool;
    fn insertion_prob_at_root(&self) -> f64;
    fn insertion_prob_at_non_root(&self, beta: f64) -> f64;
    fn block_prob(&self, x: f64, block_len: usize) -> f64;
    fn get_blocks<AA: AncestralAlignment>(msa: &AA) -> Vec<usize>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct TKF91IndelModel {
    params: Vec<f64>,
}

impl Display for TKF91IndelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF91 with lambda = {}, mu = {}",
            self.lambda(),
            self.mu(),
        )
    }
}

impl TKF for TKF91IndelModel {
    fn lambda(&self) -> f64 {
        self.params[0]
    }

    fn mu(&self) -> f64 {
        self.params[1]
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    /// Sets the parameter if it is valid then returns true, otherwise the parameter is not changed
    /// and false is returned.
    /// This assumes that the other parameter is valid
    fn set_param(&mut self, i: usize, v: f64) -> bool {
        if i > 1 {
            return false;
        }
        if i == 0 {
            // lambda
            if v > 0.0 && v < self.mu() {
                self.params[0] = v;
                return true;
            }
        } else if i == 1 {
            // mu
            if v > self.lambda() {
                self.params[1] = v;
                return true;
            }
        }
        false
    }

    fn insertion_prob_at_root(&self) -> f64 {
        self.lambda() / self.mu()
    }

    fn insertion_prob_at_non_root(&self, beta: f64) -> f64 {
        self.lambda() * beta
    }

    fn block_prob(&self, x: f64, block_len: usize) -> f64 {
        if x == 1.0 {
            0.0
        } else {
            (block_len as f64) * x.ln()
        }
    }

    fn get_blocks<AA: AncestralAlignment>(msa: &AA) -> Vec<usize> {
        (0..msa.len()).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TKF92IndelModel {
    // TODO: it might be better for model optimisation to have parameter lambda and scale s = mu/lambda,
    // because of the constraint that mu > lambda.
    params: Vec<f64>,
}

impl TKF92IndelModel {
    fn r(&self) -> f64 {
        self.params[2]
    }
}

impl TKF for TKF92IndelModel {
    fn lambda(&self) -> f64 {
        self.params[0]
    }

    fn mu(&self) -> f64 {
        self.params[1]
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    /// Sets the parameter if it is valid then returns true, otherwise the parameter is not changed
    /// and false is returned.
    /// This assumes that the other parameters are valid
    fn set_param(&mut self, i: usize, v: f64) -> bool {
        if i > 2 {
            return false;
        }
        if i == 0 {
            // lambda
            if v > 0.0 && v < self.mu() {
                self.params[0] = v;
                return true;
            }
        } else if i == 1 {
            // mu
            if v > self.lambda() {
                self.params[1] = v;
                return true;
            }
        } else {
            // r
            if (0.0..1.0).contains(&v) {
                self.params[2] = v;
                return true;
            }
        }
        false
    }

    fn insertion_prob_at_root(&self) -> f64 {
        self.lambda() / self.mu() * (1.0 - self.r()) / self.r()
    }

    fn insertion_prob_at_non_root(&self, beta: f64) -> f64 {
        self.lambda() * beta * (1.0 - self.r()) / self.r()
    }

    fn block_prob(&self, x: f64, block_len: usize) -> f64 {
        if x == 1.0 {
            0.0
        } else {
            x.ln() + (block_len as f64 - 1.0) * (1.0 + x).ln() + (block_len as f64) * self.r().ln()
        }
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
}

impl Display for TKF92IndelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with lambda = {}, mu = {}, r = {}",
            self.lambda(),
            self.mu(),
            self.r(),
        )
    }
}

/// For a detailed explanation of the TKF92 model and how the probabilities are calculated see:
/// <coming paper>.
#[derive(Clone, Debug)]
struct TKFIndelModelInfo {
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
    /// where the current event is an insertion and the previous one was a deletion.
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

impl TKFIndelModelInfo {
    fn new<AA: AncestralAlignment, T: TKF>(phylo: &PhyloInfo<AA>) -> TKFIndelModelInfo {
        let blocks = T::get_blocks(&phylo.msa);
        let block_lens = get_block_lens(&blocks);
        let n_blocks = blocks.len();
        let n_nodes = phylo.tree.len();
        TKFIndelModelInfo {
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
#[derive(Debug)]
pub struct TKFIndelCost<AA: AncestralAlignment, T: TKF> {
    model: T,
    phylo: PhyloInfo<AA>,
    model_info: RefCell<TKFIndelModelInfo>,
}

impl<AA: AncestralAlignment + Clone, T: TKF> Clone for TKFIndelCost<AA, T> {
    fn clone(&self) -> Self {
        TKFIndelCost {
            model: self.model.clone(),
            phylo: self.phylo.clone(),
            model_info: RefCell::new(self.model_info.borrow().clone()),
        }
    }
}

impl<AA: AncestralAlignment, T: TKF> Display for TKFIndelCost<AA, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

fn valid_tkf_indel_parameters(lambda: f64, mu: f64, r: f64) -> Vec<f64> {
    let mut valid_lambda = lambda;
    let mut valid_mu = mu;
    let mut valid_r = r;
    if lambda <= 0.0 && mu <= 0.0 {
        warn!("Both lambda and mu must be positive. Setting lambda to 1 and mu to 1.1");
        valid_lambda = 1.0;
        valid_mu = 1.1;
    }
    if lambda <= 0.0 {
        warn!(
            "Tried to set lambda to invalid value {}. It must be in (0, mu) with mu = {}. Setting lambda to 0.9*mu = {}",
            lambda, mu, 0.9*mu
        );
        valid_lambda = 0.9 * mu;
    }
    if mu <= 0.0 {
        warn!(
            "Tried to set mu to invalid value {}. It must be in (lambda, inf) with lambda = {}. Setting mu to 1.1*lambda = {}",
            mu, lambda, 1.1*lambda
        );
        valid_mu = 1.1 * lambda;
    }
    if lambda >= mu {
        warn!(
            "Tried to set mu to invalid value {}. It must be in (lambda, inf) with lambda = {}. Setting mu to 1.1*lambda = {}",
            mu, lambda, 1.1*lambda
        );
        valid_mu = 1.1 * lambda;
    }
    if !(0.0..1.0).contains(&r) {
        warn!(
            "Tried to set r to invalid value {}. It must be in [0, 1). Setting r to 0.5",
            r
        );
        valid_r = 0.5;
    }
    vec![valid_lambda, valid_mu, valid_r]
}

pub struct TKF91IndelCostBuilder<AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    phylo: PhyloInfo<AA>,
}

impl<AA: AncestralAlignment> TKF91IndelCostBuilder<AA> {
    pub fn new(lambda: f64, mu: f64, phylo: PhyloInfo<AA>) -> Self {
        Self { lambda, mu, phylo }
    }

    pub fn build(self) -> Result<TKFIndelCost<AA, TKF91IndelModel>> {
        let model = TKF91IndelModel {
            params: valid_tkf_indel_parameters(self.lambda, self.mu, 0.0),
        };
        let info = TKFIndelModelInfo::new::<_, TKF91IndelModel>(&self.phylo);
        Ok(TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        })
    }
}

pub struct TKF92IndelCostBuilder<AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    r: f64,
    phylo: PhyloInfo<AA>,
}

impl<AA: AncestralAlignment> TKF92IndelCostBuilder<AA> {
    pub fn new(lambda: f64, mu: f64, r: f64, phylo: PhyloInfo<AA>) -> Self {
        Self {
            lambda,
            mu,
            r,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFIndelCost<AA, TKF92IndelModel>> {
        let model = TKF92IndelModel {
            params: valid_tkf_indel_parameters(self.lambda, self.mu, self.r),
        };
        let info = TKFIndelModelInfo::new::<_, TKF92IndelModel>(&self.phylo);
        Ok(TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        })
    }
}
#[derive(Clone, Debug)]
pub struct TKFCost<Q: QMatrix + Display, T: TKF, AA: AncestralAlignment> {
    // TODO: if we have just the sum of the two costs like this, we need to keep track of the
    // phylo (which is tree and alignment) twice, which might be too big of a downside, since the
    // cost is copied often. Alternatively we could implement the substitution cost inside the
    // tkf92 cost, which would duplicate some code.
    indel_cost: TKFIndelCost<AA, T>,
    subst_cost: SubstitutionCost<Q, AA>,
    combined_parameters: Vec<f64>,
}

impl<Q: QMatrix, T: TKF, AA: AncestralAlignment> Display for TKFCost<Q, T, AA> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} and {}",
            self.indel_cost.model, self.subst_cost.model.qmatrix
        )
    }
}

pub struct TKF91CostBuilder<Q: QMatrix, AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    subst_model: SubstModel<Q>,
    phylo: PhyloInfo<AA>,
}

impl<Q: QMatrix, AA: AncestralAlignment> TKF91CostBuilder<Q, AA> {
    pub fn new(lambda: f64, mu: f64, subst_model: SubstModel<Q>, phylo: PhyloInfo<AA>) -> Self {
        Self {
            lambda,
            mu,
            subst_model,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFCost<Q, TKF91IndelModel, AA>> {
        if self.phylo.msa.alphabet() != self.subst_model.alphabet() {
            bail!("Alphabet mismatch between model and alignment");
        }

        let model = TKF91IndelModel {
            params: valid_tkf_indel_parameters(self.lambda, self.mu, 0.0),
        };
        let info = TKFIndelModelInfo::new::<_, TKF91IndelModel>(&self.phylo);
        let tkf_cost = TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        };
        let combined_parameters = [tkf_cost.model.params(), self.subst_model.params()].concat();
        Ok(TKFCost {
            indel_cost: tkf_cost,
            subst_cost: SCB::new(self.subst_model, self.phylo).build().unwrap(),
            combined_parameters,
        })
    }
}

pub struct TKF92CostBuilder<Q: QMatrix, AA: AncestralAlignment> {
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

    pub fn build(self) -> Result<TKFCost<Q, TKF92IndelModel, AA>> {
        if self.phylo.msa.alphabet() != self.subst_model.alphabet() {
            bail!("Alphabet mismatch between model and alignment");
        }

        let model = TKF92IndelModel {
            params: valid_tkf_indel_parameters(self.lambda, self.mu, self.r),
        };
        let info = TKFIndelModelInfo::new::<_, TKF92IndelModel>(&self.phylo);
        let tkf_cost = TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        };
        let combined_parameters = [tkf_cost.model.params(), self.subst_model.params()].concat();
        Ok(TKFCost {
            indel_cost: tkf_cost,
            subst_cost: SCB::new(self.subst_model, self.phylo).build().unwrap(),
            combined_parameters,
        })
    }
}

impl<AA: AncestralAlignment, T: TKF> ModelSearchCost for TKFIndelCost<AA, T> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn set_param(&mut self, i: usize, v: f64) {
        if self.model.set_param(i, v) {
            self.model_info.borrow_mut().valid.fill(false);
        }
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

impl<Q: QMatrix, T: TKF, AA: AncestralAlignment> ModelSearchCost for TKFCost<Q, T, AA> {
    fn cost(&self) -> f64 {
        self.indel_cost.cost() + self.subst_cost.cost()
    }

    fn set_param(&mut self, i: usize, v: f64) {
        self.combined_parameters[i] = v;
        let num_params_indel_model = self.indel_cost.model.params().len();
        if i < num_params_indel_model {
            self.indel_cost.set_param(i, v);
            return;
        }
        let i = i - num_params_indel_model;
        self.subst_cost.set_param(i, v);
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

impl<AA: AncestralAlignment, T: TKF> TKFIndelCost<AA, T> {
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
            let block_len = self.model_info.borrow().block_lens[block_id];
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
            let (x, factor_n) = self.get_indel_x_and_factor_n_for_non_root(node_idx, block_id);
            self.set_node_values(node_idx, block_id, x, factor_n);
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
        if self.phylo.msa.ancestral_map(root_idx)[self.model_info.borrow().blocks[block_id] - 1]
            .is_some()
        {
            return self.model.insertion_prob_at_root();
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
        let parent_is_gap = match parent_idx {
            Internal(_) => self.phylo.msa.ancestral_map(&parent_idx)[site].is_none(),
            _ => unreachable!("The parent of a node cannot be a leaf."),
        };
        let current_is_gap = match node_idx {
            Internal(_) => self.phylo.msa.ancestral_map(node_idx)[site].is_none(),
            Leaf(_) => self.phylo.msa.leaf_map(node_idx)[site].is_none(),
        };

        let time = self.phylo.tree.node(node_idx).blen;

        if block_id == 0 {
            self.model_info.borrow_mut().last_event_deletion[node_id] = false;
        }

        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let beta = self.model_info.borrow().beta[node_id];

        if !parent_is_gap && current_is_gap {
            // deletion
            x *= *self.model_info.borrow_mut().n0[node_id].get_or_insert_with(|| n0(mu, beta));
            self.model_info.borrow_mut().last_event_deletion[node_id] = true;
        } else if !parent_is_gap && !current_is_gap {
            // homologue
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
                        // since last event was a deletion n0 is not None
                        factor_n -= n0_option.unwrap().ln();
                        factor_n
                    });
            }
            x *= *self.model_info.borrow_mut().insertion[node_id]
                .get_or_insert_with(|| self.model.insertion_prob_at_non_root(beta));
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

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
