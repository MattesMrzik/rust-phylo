use inc_stats::DerefCopy;
use nalgebra::DMatrix;

pub mod reassignment;

use crate::{
    alignment::AncestralAlignment,
    likelihood::TreeSearchCost,
    phylo_info::PhyloInfoAncestors,
    substitution_models::{QMatrix, SubstMatrix},
    tree::{
        NodeIdx::{self, Internal, Leaf},
        Tree,
    },
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Display,
    marker::PhantomData,
};

#[derive(Clone)]
pub struct TKF92Model<Q: QMatrix> {
    q: Q,
    params: Vec<f64>,
}

// TODO: pip_model has also Clone as trait bound for Q, is this also needed here?
impl<Q: QMatrix> TKF92Model<Q> {
    fn lambda(&self) -> f64 {
        self.params[0]
    }
    fn mu(&self) -> f64 {
        self.params[1]
    }
    fn r(&self) -> f64 {
        self.params[2]
    }

    // pip defined these for the trait EvoModel, but i dont think that i need all of this
    // p should be sufficient to calc this once per edge
    // perhaps for some optimizer it is required that i implement the EvoModel trait for my model
    // even if i implement the evomodel trait for my model, do i need to put this on every edge or is it
    // sufficient if i put a substmatrix there
    fn p(&self, time: f64) -> SubstMatrix {
        // if i dont clone here would it take ownership?
        (self.q.q().clone() * time).exp()
    }
}

impl<Q: QMatrix + Display + Clone> Display for TKF92Model<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with [lambda = {:.5}, mu = {:.5}]\n, r = {:.5} and {}",
            self.lambda(),
            self.mu(),
            self.r(),
            self.q
        )
    }
}

pub struct TKF92ModelInfo<Q: QMatrix> {
    phantom: PhantomData<Q>,

    // TODO: the aggregated nature of x and factor_n is not really needed now,
    //       maybe later for realignment.
    // aggregated_x[node, block] = the product of the xs of all the edges in the subtree
    aggregated_x: DMatrix<f64>,

    // factor_n[node, block] = n1/ (n0 * lambda * beta(node.blen)) if there is an edge in the subtree
    // where the current event is an insertion and the last one was a deletion
    factor_ns: DMatrix<f64>,

    // felsenstein_prob[node, block] = contains the felsenstein prob of that column, i
    // if there is an insertion in the subtree
    // TODO: this could be merged with factor_n
    // TODO: can i even decouple the substitution cost from indel probabilities?
    felsenstein_prob: DMatrix<f64>,

    /// felsenstein\[node\]\[site, state\]
    felsenstein: Vec<DMatrix<f64>>,

    // n0[node] = n0(node.blen), I might not need this for every node
    n0: Vec<Option<f64>>,

    // h1[node] = h1(node.blen), I might not need this for every node
    h1: Vec<Option<f64>>,

    // insertion[node] = l * beta * (1.0 - r) / r;, I might not need this for every node
    insertion: Vec<Option<f64>>,

    // log_n1[node] = ln(n1(node.blen)), I might not need this for every node
    factor_n: Vec<Option<f64>>,

    // beta[usize::from(node)] = beta(node.blen)), i need beta for every node anyways
    // since i1 uses them, so these values a precomputed
    beta: Vec<f64>,

    // the right exclusive interval borders of the blocks
    blocks: Vec<usize>,

    // the lengths of the blocks
    block_lens: Vec<usize>,

    // a vector for every edge storing if the edge is time reversed or not,
    // reversed: Vec<bool>,

    // models[usize::from(node)] = Q.exp(node.blen)
    models: Vec<SubstMatrix>,

    // TODO: couldn't AncestralAlignment::get_node_map() be used instead?
    //       If I keep this var, then I should use it not for every site,
    //       but only for every block
    // leaf_sequence_info[node.id][site, state] = the prob of observing this state
    leaf_sequence_info: HashMap<String, DMatrix<f64>>,

    // usize::from(node), this is not the root of the tree but a virtual root for computational ease
    virtual_root: NodeIdx,

    // edge_is_time_reversed[usize::from(node)] = true if the edge is time reversed
    edge_is_time_reversed: Vec<bool>,

    // last_event_deletion[usize::from(node)] = true if the last event was a deletion for a that node
    last_event_deletion: Vec<bool>,

    // last_event_insertion[usize::from(node)] = true if the last event was an insertion for a that node
    last_event_insertion: Vec<bool>,

    // true if the all the intermediate values are correctly set
    valid: bool,
}

impl<Q: QMatrix> TKF92ModelInfo<Q> {
    pub fn new(phylo: &PhyloInfoAncestors, model: &TKF92Model<Q>) -> TKF92ModelInfo<Q> {
        let blocks = Self::get_blocks(&phylo.msa);
        let block_lens = Self::get_block_lens(&blocks);
        let n_blocks = blocks.len();
        let n_states = model.q.n();
        let n_nodes = phylo.tree.len();
        TKF92ModelInfo::<Q> {
            phantom: PhantomData,
            aggregated_x: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            factor_ns: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            felsenstein: vec![DMatrix::from_element(phylo.msa.len(), n_states, 1.0); n_nodes],
            felsenstein_prob: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            n0: vec![None; n_nodes],
            h1: vec![None; n_nodes],
            insertion: vec![None; n_nodes],
            factor_n: vec![None; n_nodes],
            beta: vec![0.0; n_nodes],
            blocks,
            block_lens,
            models: vec![SubstMatrix::zeros(n_states, n_states); n_nodes],
            leaf_sequence_info: Self::get_leaf_seq_info(&phylo, &model.q),
            virtual_root: phylo.tree.root,
            edge_is_time_reversed: vec![false; n_nodes],
            last_event_deletion: vec![false; n_nodes],
            last_event_insertion: vec![false; n_nodes],
            valid: false,
        }
    }

    fn get_leaf_seq_info(phylo: &PhyloInfoAncestors, q: &Q) -> HashMap<String, DMatrix<f64>> {
        let mut leaf_seq_info = HashMap::<String, DMatrix<f64>>::new();
        let leaf_encodings = &phylo.msa.leaf_encoding;
        let n = q.n();
        for node in phylo.tree.leaves() {
            let alignment_map = &phylo.msa.get_node_map()[&node.idx];
            let leaf_encoding = &leaf_encodings[&node.id];
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(n, alignment_map.len());
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    let encoding = &leaf_encoding.column(c);
                    site_info.copy_from(encoding);
                    // TODO: in PIP it was: site_info.scale_mut((1.0) / site_info.sum());
                } else {
                    site_info.fill(1.0);
                }
            }
            leaf_seq_info.insert(node.id.clone(), leaf_seq_w_gaps);
            // println!("insert at node {}", node.id);
        }

        // for node in phylo.tree.leaf_ids() {
        //     println!("looking at node {}", node);
        //     println!("{}", leaf_seq_info[&node]);
        // }

        leaf_seq_info
    }

    fn get_block_lens(blocks: &Vec<usize>) -> Vec<usize> {
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

    fn get_blocks(msa: &AncestralAlignment) -> Vec<usize> {
        let mut blocks: HashSet<usize> = HashSet::new();
        for (_, map) in msa.get_node_map() {
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
pub struct TKF92Cost<Q: QMatrix + Display> {
    // TODO: q was also 'static in pip
    model: TKF92Model<Q>,
    phylo: PhyloInfoAncestors,
    // TODO: maybe refcell is used here to make it mutable in impl fn while keeping the other fields
    // of this struct immutable, i.e. only passing &self and not &mut self
    model_info: RefCell<TKF92ModelInfo<Q>>,
}

impl<Q: QMatrix + Display + Clone + 'static> TreeSearchCost for TKF92Cost<Q> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn update_tree(&mut self, tree: crate::tree::Tree, dirty_nodes: &[NodeIdx]) {
        self.phylo.tree = tree;
        // the update tree can either be a single edge len

        // do i have to update all the nodes up to the root?
        // perhaps i only need to this once before realignment

        // so if i only want to update the root node i cannot use the set_node method

        // if i do NNI without fixing up the tree, i can still to realignment
        // at that node, and then fix the tree values after realignment
        // so either I reroot before, then i dont have to fix the tree afterwards
        // or fix the tree. only considering this one change the two are both O(n)
        // rerooting only might be better if we expect to have multiple changes (NNI & realignment)
        // in similar locations

        // or after an NNI move
        println!("{:?}", dirty_nodes);

        if dirty_nodes.len() == 1 {}
    }

    fn tree(&self) -> &crate::tree::Tree {
        &self.phylo.tree
    }
}

impl<Q: QMatrix + Display> TKF92Cost<Q>
// where
//     TKF92Model<Q>: EvoModel,
{
    fn logl_old(&self) -> f64 {
        let blocks = TKF92ModelInfo::<Q>::get_blocks(&self.phylo.msa);
        let tree = &self.phylo.tree;
        let model = &self.model;
        let node_map = self.phylo.msa.get_node_map();
        let l = model.lambda();
        let m = model.mu();
        let r = model.r();

        // for the root
        let mut prob: f64 = (1.0 - l / m).ln();

        let mut last_event_deletion = vec![false; tree.len()];
        let mut last_event_insertion = vec![false; tree.len()];
        for (i, fragment) in blocks.iter().enumerate() {
            let mut x = 1.0;
            let fragment_len = if i == 0 {
                *fragment
            } else {
                fragment - blocks[i - 1]
            };
            if node_map[&self.model_info.borrow().virtual_root][fragment - 1].is_some() {
                // the eq seq at the root has a fragment
                x *= l / m * (1.0 - r) / r;
                prob += fragment_len as f64 * r.ln();
            }
            for node_idx in tree.postorder() {
                let node_id_value = usize::from(node_idx);

                if node_idx == &tree.root {
                    continue;
                }
                let time = tree.node(node_idx).blen;
                let parent_id = &tree.node(node_idx).parent.unwrap();
                let mut parent_is_gap = node_map[parent_id][fragment - 1].is_none();
                let mut current_is_gap = node_map[node_idx][fragment - 1].is_none();

                if self.model_info.borrow().edge_is_time_reversed[usize::from(node_idx)] {
                    println!("this edge is time reversed {}", node_idx);
                    let h = parent_is_gap;
                    parent_is_gap = current_is_gap;
                    current_is_gap = h;
                }

                let b = Self::b(l, m, time);
                if i == 0 {
                    prob += Self::log_i1(l, b);
                }
                if parent_is_gap && current_is_gap {
                    continue;
                }
                if !parent_is_gap && !current_is_gap {
                    // homolog block
                    x *= Self::h1(l, m, b, time);
                    last_event_deletion[node_id_value] = false;
                    last_event_insertion[node_id_value] = false;
                }
                if !parent_is_gap && current_is_gap {
                    // deletion
                    x *= Self::n0(m, b);
                    if last_event_insertion[node_id_value]
                        && self.model_info.borrow().edge_is_time_reversed[node_id_value]
                    {
                        prob += Self::log_n1(l, m, b, time);
                        prob -= (l * b).ln();
                        prob -= Self::n0(m, b).ln();
                    }
                    last_event_deletion[node_id_value] = true;
                    last_event_insertion[node_id_value] = false;
                }
                if parent_is_gap && !current_is_gap {
                    // insertion
                    if last_event_deletion[node_id_value]
                        && !self.model_info.borrow().edge_is_time_reversed[node_id_value]
                    {
                        prob += Self::log_n1(l, m, b, time);
                        prob -= (l * b).ln();
                        prob -= Self::n0(m, b).ln();
                    }
                    x *= l * b * (1.0 - r) / r;
                    prob += fragment_len as f64 * r.ln();
                    last_event_deletion[node_id_value] = false;
                    last_event_insertion[node_id_value] = true;
                }
            }
            prob += x.ln();
            prob += (fragment_len - 1) as f64 * (1.0 + x).ln();
        }
        prob
    }

    fn logl(&self) -> f64 {
        self.logl_strip(10000, false)
    }

    fn logl_strip(&self, only_until_block: usize, exclude_const: bool) -> f64 {
        // println!("logl");
        let not_valid = !self.model_info.borrow().valid;
        if not_valid {
            self.reset_all_nodes();
        }
        let l: f64 = self.model.lambda();
        let m = self.model.mu();
        let r = self.model.r();
        let root_id = usize::from(self.model_info.borrow().virtual_root);
        let mut logl = 0.0;
        if !exclude_const {
            logl += (1.0 - l / m).ln();
            logl += self.phylo.msa.len() as f64 * r.ln();
            for node in self.phylo.tree.postorder() {
                if node == &self.phylo.tree.root {
                    continue;
                }
                logl += Self::log_i1(l, self.model_info.borrow_mut().beta[usize::from(node)]);
            }
        }
        // println!("calculating loglike, and the const part is {}", logl);
        for block_id in 0..self.model_info.borrow().blocks.len() {
            if block_id == only_until_block {
                break;
            }
            let block_len = self.model_info.borrow().block_lens[block_id];
            logl += self.model_info.borrow().factor_ns[(root_id, block_id)];
            logl += self.model_info.borrow().felsenstein_prob[(root_id, block_id)];

            let x = self.model_info.borrow().aggregated_x[(root_id, block_id)];
            // println!(
            //     "block = {}, x = {:.11}, integrated x = {:.11}, factor_n = {:.11}, felsensteinprob = {:.11}",
            //     block_id,
            //     x,
            //     x.ln() + (block_len as f64 - 1.0) * (1.0 + x).ln(),
            //     self.model_info.borrow().factor_ns[(root_id, block_id)],
            //     self.model_info.borrow().felsenstein_prob[(root_id, block_id)]
            // );

            if x != 1.0 {
                // if x < 0.0 {
                //     let block = self.model_info.borrow().blocks[block_id];
                //     println!(
                //         "x is less than 0.0: x = {}, for block {} = ({}, {})",
                //         x,
                //         block_id,
                //         block - block_len,
                //         block
                //     );
                //     exit(1);
                // }
                logl += x.ln();
                logl += (block_len as f64 - 1.0) * (1.0 + x).ln();
            } else {
                let block = self.model_info.borrow().blocks[block_id];
                let block_len = self.model_info.borrow().block_lens[block_id];
                // println!("an x is 1.0 for block {} = ({},{}), this means there is to action which should not happen", block_id, block-block_len, block);
                // std::process::exit(1);
            }
        }
        // println!("\n\n");
        logl
    }

    fn virtual_reroot_with_id(&self, tree: &Tree, neighbor_node_id: &str) {
        // TODO: this should only be a warning
        println!(
            "rerooting from {} to {}",
            tree.node(&self.model_info.borrow().virtual_root).id,
            neighbor_node_id
        );

        assert!(
            *neighbor_node_id != tree.node(&self.model_info.borrow().virtual_root).id,
            "The new root location is the same as the old"
        );
        let parent_option = tree.node(&self.model_info.borrow().virtual_root).parent;
        match parent_option {
            // reroot to tree parent
            Some(parent) => {
                if tree.node(&parent).id == *neighbor_node_id {
                    let old_root = self.model_info.borrow().virtual_root;
                    self.model_info.borrow_mut().edge_is_time_reversed[usize::from(old_root)] =
                        false;
                    if parent == self.phylo.tree.root {
                        self.model_info.borrow_mut().edge_is_time_reversed[usize::from(parent)] =
                            false;
                    }
                    self.model_info.borrow_mut().virtual_root = parent;
                    self.set_internal(&old_root);
                    self.set_root();
                    return;
                }
            }
            _ => (),
        }

        let children = tree
            .node(&self.model_info.borrow().virtual_root)
            .children
            .clone();

        for child in children {
            // reroot to tree child
            if tree.node(&child).id == *neighbor_node_id {
                assert!(
                    tree.node(&child).children.len() > 0,
                    "you cannot change to a leaf node"
                );

                let old_root = self.model_info.borrow().virtual_root;

                self.model_info.borrow_mut().virtual_root = child;
                self.model_info.borrow_mut().edge_is_time_reversed[usize::from(child)] = true;
                if old_root == self.phylo.tree.root {
                    self.model_info.borrow_mut().edge_is_time_reversed[usize::from(old_root)] =
                        true;
                }

                self.set_internal(&old_root);

                self.set_root();
                return;
            }
        }
        // TODO: this should be a warning
        assert!(false, "no neighbor found with this node_idx")
    }

    // TODO: should this setting not be part of the model info instead of cost?
    fn reset_all_nodes(&self) {
        // println!("resetting all nodes");
        for node_idx in self.phylo.tree.postorder() {
            let time = self.phylo.tree.node(node_idx).blen;
            let l = self.model.lambda();
            let m = self.model.mu();
            let beta = Self::b(l, m, time);
            self.model_info.borrow_mut().beta[usize::from(node_idx)] = beta;
            let virtual_root = self.model_info.borrow().virtual_root;
            match node_idx {
                Internal(_) => {
                    if virtual_root == *node_idx {
                        println!("root");
                        self.set_root();
                    } else {
                        println!("internal");
                        self.set_internal(node_idx);
                    }
                }
                Leaf(_) => {
                    println!("leaf");
                    self.set_leaf(node_idx);
                }
            };
        }
        self.model_info.borrow_mut().valid = true;
    }

    fn set_root(&self) {
        let root_idx = &self.phylo.tree.root;
        let len = self.model_info.borrow().blocks.len();
        for block_id in 0..len {
            self.set_felsenstein_for_internal(root_idx, block_id);
            self.set_felsenstein_prob_for_root(block_id);
            self.set_indel_x_and_factor_n_for_root(block_id);
        }
    }

    fn set_internal(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        self.model_info.borrow_mut().models[node_id] =
            self.model.p(self.phylo.tree.node(node_idx).blen);
        let len = self.model_info.borrow().blocks.len();
        for block_id in 0..len {
            self.set_felsenstein_for_internal(node_idx, block_id);
            self.set_felsenstein_prob_for_non_root(node_idx, block_id);
            self.set_indel_x_and_factor_n_for_not_root(node_idx, block_id);
        }
    }

    fn set_leaf(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        let len = self.model_info.borrow().blocks.len();
        self.model_info.borrow_mut().models[node_id] =
            self.model.p(self.phylo.tree.node(node_idx).blen);
        for block_id in 0..len {
            self.set_felsenstein_for_leaf(node_idx, block_id);
            self.set_felsenstein_prob_for_non_root(node_idx, block_id);
            self.set_indel_x_and_factor_n_for_not_root(node_idx, block_id);
        }
    }

    // TODO: set_felsenstein_prob_for_root and set_felsenstein_prob_for_non_root
    //       only differ in the call to is_insertion_at_root and is_insertion_at_non_root.
    //       To avoid duplication the correct of these fn could be passed as argument.
    fn set_felsenstein_prob_for_root(&self, block_id: usize) {
        let root_idx = self.model_info.borrow().virtual_root;
        let root_id = usize::from(root_idx);
        let mut f_prob = 0.0;
        if self.is_insertion_at_root(block_id) {
            f_prob += self.felsenstein_to_prob(&root_idx, block_id);
        } else {
            for child in self.get_childs_in_time_reversed_tree(&root_idx) {
                let child_id = usize::from(child);
                f_prob += self.model_info.borrow().felsenstein_prob[(child_id, block_id)];
            }
        }
        self.model_info.borrow_mut().felsenstein_prob[(root_id, block_id)] = f_prob;
    }

    fn set_felsenstein_prob_for_non_root(&self, node_idx: &NodeIdx, block_id: usize) {
        let node_id = usize::from(node_idx);
        let mut f_prob = 0.0;
        if self.is_insertion_at_non_root(node_idx, block_id) {
            f_prob += self.felsenstein_to_prob(node_idx, block_id);
        } else {
            for child in self.get_childs_in_time_reversed_tree(node_idx) {
                let child_id = usize::from(child);
                f_prob += self.model_info.borrow().felsenstein_prob[(child_id, block_id)];
            }
        }
        self.model_info.borrow_mut().felsenstein_prob[(node_id, block_id)] = f_prob;
    }

    fn set_indel_x_and_factor_n_for_root(&self, block_id: usize) {
        let mut x = self.get_indel_x_for_root(block_id);
        let mut factor_n = 0.0;
        // this is the same as in set_indel_x_and_prob_for_not_root
        let root_idx = self.model_info.borrow().virtual_root;
        let root_id = usize::from(root_idx);
        for child in self.get_childs_in_time_reversed_tree(&root_idx) {
            let child_id = usize::from(child);
            x *= self.model_info.borrow().aggregated_x[(child_id, block_id)];
            factor_n += self.model_info.borrow().factor_ns[(child_id, block_id)];
        }
        self.model_info.borrow_mut().factor_ns[(root_id, block_id)] = factor_n;
        self.model_info.borrow_mut().aggregated_x[(root_id, block_id)] = x;
    }

    fn set_indel_x_and_factor_n_for_not_root(&self, node_idx: &NodeIdx, block_id: usize) {
        let (mut x, mut factor_n) = self.get_indel_x_and_factor_n_for_not_root(node_idx, block_id);
        // this is the same as in set_indel_x_and_prob_for_root
        let node_id = usize::from(node_idx);
        for child in self.get_childs_in_time_reversed_tree(node_idx) {
            let child_id = usize::from(child);
            x *= self.model_info.borrow().aggregated_x[(child_id, block_id)];
            factor_n += self.model_info.borrow().factor_ns[(child_id, block_id)];
        }
        if x < 0.0 {
            println!(
                "x is smaller than 0 for node {}",
                self.phylo.tree.node(node_idx).id
            );
        }
        self.model_info.borrow_mut().factor_ns[(node_id, block_id)] = factor_n;
        self.model_info.borrow_mut().aggregated_x[(node_id, block_id)] = x;
    }

    fn get_childs_in_time_reversed_tree(&self, node_idx: &NodeIdx) -> Vec<&NodeIdx> {
        // since we are in a binary tree, but at some virtual root location it has 3 children
        let mut childs = Vec::<&NodeIdx>::with_capacity(3);
        let node_id = usize::from(node_idx);
        if self.model_info.borrow().edge_is_time_reversed[node_id] {
            match &self.phylo.tree.node(node_idx).parent {
                Some(parent_idx) => childs.push(parent_idx),
                _ => (),
            }
        }
        for actual_child in &self.phylo.tree.node(node_idx).children {
            if !self.model_info.borrow().edge_is_time_reversed[usize::from(actual_child)] {
                childs.push(actual_child);
            }
        }
        childs
    }

    fn get_indel_x_for_root(&self, block_id: usize) -> f64 {
        let l = self.model.lambda();
        let m = self.model.mu();
        let r = self.model.r();

        let root_idx = &self.model_info.borrow().virtual_root;
        if self.phylo.msa.get_node_map()[root_idx][self.model_info.borrow().blocks[block_id] - 1]
            .is_some()
        {
            return l / m * (1.0 - r) / r;
        }
        1.0
    }

    // ie every node that is not the virtual node, (this may include the actual node)
    fn get_indel_x_and_factor_n_for_not_root(
        &self,
        node_idx: &NodeIdx,
        block_id: usize,
    ) -> (f64, f64) {
        // this assumes that in the case of that the node_idx that also this infinite edge of the true root is reversed
        let parent_idx = self.get_parent_in_time_reversed_tree_for_not_virtual_root(node_idx);
        let node_id = usize::from(node_idx);
        let mut factor_n = 0.0;
        let mut x: f64 = 1.0;
        let site = self.model_info.borrow().blocks[block_id] - 1;
        let parent_is_gap = self.phylo.msa.get_node_map()[&parent_idx][site].is_none();
        let current_is_gap = self.phylo.msa.get_node_map()[node_idx][site].is_none();

        let time = self.phylo.tree.node(node_idx).blen;

        if block_id == 0 {
            self.model_info.borrow_mut().last_event_deletion[node_id] = false;
        }

        let l = self.model.lambda();
        let m = self.model.mu();
        let r = self.model.r();

        let mut action = "";
        if !parent_is_gap && current_is_gap {
            // deletion
            action = "deletion";

            if self.model_info.borrow().last_event_insertion[node_id]
                && self.model_info.borrow().edge_is_time_reversed[node_id]
            {
                let n_option = self.model_info.borrow().factor_n[node_id];
                match n_option {
                    Some(n) => factor_n += n,
                    None => {
                        let b = self.model_info.borrow().beta[node_id];
                        let mut n = Self::log_n1(l, m, b, time);
                        n -= (l * b).ln();
                        let n0_option = self.model_info.borrow().n0[node_id];
                        match n0_option {
                            Some(n0) => n -= n0.ln(),
                            None => {
                                let n0 = Self::n0(m, self.model_info.borrow().beta[node_id]);
                                self.model_info.borrow_mut().n0[node_id] = Some(n0);
                                n -= n0.ln();
                            }
                        }
                        self.model_info.borrow_mut().factor_n[node_id] = Some(n);
                        factor_n += n;
                    }
                }
            }

            let n0_option = self.model_info.borrow().n0[node_id];
            match n0_option {
                Some(n0) => x *= n0,
                None => {
                    let n0 = Self::n0(m, self.model_info.borrow().beta[node_id]);
                    self.model_info.borrow_mut().n0[node_id] = Some(n0);
                    x *= n0;
                }
            }
            self.model_info.borrow_mut().last_event_deletion[node_id] = true;
            self.model_info.borrow_mut().last_event_insertion[node_id] = false;
        }
        if !parent_is_gap && !current_is_gap {
            action = "homolog";
            // homolog
            let h1_option = self.model_info.borrow().h1[node_id];
            match h1_option {
                Some(h1) => x *= h1,
                None => {
                    let h1 = Self::h1(l, m, self.model_info.borrow().beta[node_id], time);
                    self.model_info.borrow_mut().h1[node_id] = Some(h1);
                    x *= h1;
                }
            }
            self.model_info.borrow_mut().last_event_deletion[node_id] = false;
            self.model_info.borrow_mut().last_event_deletion[node_id] = false;
        }
        if parent_is_gap && !current_is_gap {
            // insertion
            action = "insertion";
            if self.model_info.borrow().last_event_deletion[node_id]
                && !self.model_info.borrow().edge_is_time_reversed[node_id]
            {
                let n_option = self.model_info.borrow().factor_n[node_id];
                match n_option {
                    Some(n) => factor_n += n,
                    None => {
                        let b = self.model_info.borrow().beta[node_id];
                        let mut n = Self::log_n1(l, m, b, time);
                        n -= (l * b).ln();
                        let n0_option = self.model_info.borrow().n0[node_id];
                        match n0_option {
                            Some(n0) => n -= n0.ln(),
                            None => {
                                let n0 = Self::n0(m, self.model_info.borrow().beta[node_id]);
                                self.model_info.borrow_mut().n0[node_id] = Some(n0);
                                n -= n0.ln();
                            }
                        }
                        self.model_info.borrow_mut().factor_n[node_id] = Some(n);
                        factor_n += n;
                    }
                }
            }
            let insertion_option = self.model_info.borrow().insertion[node_id];
            match insertion_option {
                Some(insertion) => x *= insertion,
                None => {
                    let insertion = l * self.model_info.borrow().beta[node_id] * (1.0 - r) / r;
                    self.model_info.borrow_mut().insertion[node_id] = Some(insertion);
                    x *= insertion;
                }
            }
            self.model_info.borrow_mut().last_event_deletion[node_id] = false;
            self.model_info.borrow_mut().last_event_insertion[node_id] = true;
        }
        // if block_id == 4 {
        //     println!(
        //         "block = {}, node = {:?}, x = {}, type = {}{}",
        //         block_id,
        //         self.phylo.tree.node(node_idx).id,
        //         x,
        //         parent_is_gap,
        //         current_is_gap
        //     );
        // }
        if x < 0.0 {
            println!(
                "x is less than 0, node = {}, action = {}",
                self.phylo.tree.node(&node_idx).id,
                action
            )
        }
        (x, factor_n)
    }

    fn get_parent_in_time_reversed_tree_for_not_virtual_root(&self, node_idx: &NodeIdx) -> NodeIdx {
        println!(
            "in get_parent_in_time_reversed_tree_for_not_virtual_root the vroot is {}",
            self.phylo.tree.node(node_idx).id
        );
        assert!(*node_idx != self.model_info.borrow().virtual_root);
        let is_time_reversed =
            self.model_info.borrow().edge_is_time_reversed[usize::from(node_idx)];

        if is_time_reversed {
            // the edge is time reversed to one if its children is also time reversed,
            // also we are not at the virtual root
            // so i can safely call unwrap
            let parent = self
                .phylo
                .tree
                .node(node_idx)
                .children
                .iter()
                .find(|child| self.model_info.borrow().edge_is_time_reversed[usize::from(*child)])
                .unwrap()
                .deref_copy();

            parent
        } else {
            println!(
                "node in get_parent_in_time_reversed_tree_for_not_virtual_root {}",
                self.phylo.tree.node(node_idx).id
            );
            self.phylo.tree.node(node_idx).parent.unwrap()
        }
    }

    fn set_felsenstein_for_internal(&self, node_idx: &NodeIdx, block_id: usize) {
        // TODO: this avoids unnecessary computation but breaks reassignment,
        //       since there the get_node_map might disagree with the NNI node assignment
        // let site = self.model_info.borrow().blocks[block_id] - 1;
        // let current_node_is_gap = self.phylo.msa.get_node_map()[node_idx][site].is_none();
        // if current_node_is_gap {
        //     return;
        // }
        let block = self.model_info.borrow().blocks[block_id];
        let block_len = self.model_info.borrow().block_lens[block_id];
        let node_id = usize::from(node_idx);
        // TODO: can this also be written with matrix operations?
        for site in (block - block_len)..block {
            for current_state in 0..self.model.q.n() {
                let mut prod_over_children = 1.0;
                for child_idx in self.get_childs_in_time_reversed_tree(node_idx) {
                    if self.phylo.msa.get_node_map()[child_idx][site].is_none() {
                        continue;
                    }
                    let mut sum_over_children_states = 0.0;
                    for child_state in 0..self.model.q.n() {
                        let prob_of_mutating_to_child = self.model_info.borrow().models
                            [usize::from(child_idx)][(current_state, child_state)];
                        let child_prob = self.model_info.borrow().felsenstein
                            [usize::from(child_idx)][(site, child_state)];
                        sum_over_children_states += (prob_of_mutating_to_child) * (child_prob);
                    }
                    prod_over_children *= sum_over_children_states;
                }
                self.model_info.borrow_mut().felsenstein[node_id][(site, current_state)] =
                    prod_over_children;
            }
        }
    }

    fn set_felsenstein_for_leaf(&self, node_idx: &NodeIdx, block_id: usize) {
        // this assumes that the edge is note time reversed
        // which would mean that the leaf is the virtual root
        let site = self.model_info.borrow().blocks[block_id] - 1;
        let current_node_is_gap = self.phylo.msa.get_node_map()[node_idx][site].is_none();
        if current_node_is_gap {
            return;
        }
        let block = self.model_info.borrow().blocks[block_id];
        let block_len = self.model_info.borrow().block_lens[block_id];
        let node_name = &self.phylo.tree.node(node_idx).id;
        let node_id = usize::from(node_idx);

        // TODO: can this also be written with matrix operations?
        for site in (block - block_len)..block {
            for current_state in 0..self.model.q.n() {
                let leaf_prob =
                    self.model_info.borrow().leaf_sequence_info[node_name][(current_state, site)];
                self.model_info.borrow_mut().felsenstein[node_id][(site, current_state)] =
                    leaf_prob;
            }
        }
    }

    fn is_insertion_at_root(&self, block_id: usize) -> bool {
        let site = self.model_info.borrow().blocks[block_id] - 1;
        self.phylo.msa.get_node_map()[&self.model_info.borrow().virtual_root][site].is_some()
    }

    fn is_insertion_at_non_root(&self, node_idx: &NodeIdx, block_id: usize) -> bool {
        let site = self.model_info.borrow().blocks[block_id] - 1;
        let parent_id = self.get_parent_in_time_reversed_tree_for_not_virtual_root(node_idx);
        let parent_is_gap = self.phylo.msa.get_node_map()[&parent_id][site].is_none();
        let current_is_not_gap = self.phylo.msa.get_node_map()[node_idx][site].is_some();
        parent_is_gap && current_is_not_gap
    }

    fn felsenstein_to_prob(&self, node_idx: &NodeIdx, block_id: usize) -> f64 {
        let block = self.model_info.borrow().blocks[block_id];
        let block_len = self.model_info.borrow().block_lens[block_id];
        let node_id = usize::from(node_idx);
        let mut sum = 0.0;
        for site in (block - block_len)..block {
            let mut sum_for_state = 0.0;
            for state in 0..self.model.q.n() {
                // println!(
                //     "for site ({}), state ({}), the freq is ({}), and felsenstein is ({})",
                //     site,
                //     state,
                //     self.model.q.freqs()[state],
                //     self.model_info.borrow().felsenstein[node_id][(site, state)]
                // );
                sum_for_state += self.model.q.freqs()[state]
                    * self.model_info.borrow().felsenstein[node_id][(site, state)];
            }
            sum += sum_for_state.ln();
        }
        // println!("sum in felsenstein to prob is {}", sum);
        sum
    }

    fn print_felsenstein(&self) {
        for node in self.phylo.tree.preorder() {
            println!("{}", self.phylo.tree.node(node).id);
            if self.phylo.tree.node(node).id == "C2" {
                println!(
                    "c2[0,1]={}",
                    self.model_info.borrow().felsenstein[usize::from(node)][(0, 1)]
                );
            }
            println!(
                "{}",
                self.model_info.borrow().felsenstein[usize::from(node)]
            );
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
}

// impl<Q: QMatrix> TKF92ModelInfo<Q> {
//     // Positions x within a sequence s \in S where s[x] xor s[x+1] is a gap define a necessary
//     // fragment boundary that is present in all sequences in S.

// }

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
