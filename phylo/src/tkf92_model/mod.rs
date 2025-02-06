use nalgebra::DMatrix;

use crate::{
    alignment::AncestralAlignment, likelihood::TreeSearchCost, phylo_info::PhyloInfoAncestors, substitution_models::{QMatrix, SubstMatrix}, tree::NodeIdx::{self, Internal, Leaf}
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt::Display,
    marker::PhantomData,
};

pub struct TKF92Model<Q: QMatrix> {
    q: Q,
    params: Vec<f64>,
    // do i also need the substmatrix and freqvector even though it is part of Q i think
    // what is the index? [usize; 255] 255 size array storing usize
    // do i want some additional vars that maybe save something like the fraction lambda/mu?
}

impl<Q: QMatrix + Clone> TKF92Model<Q> {
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
    // pip had <Q: QMatrix> here, i also need to use this if i implement the felsenstein
    aggregated_x: DMatrix<f64>,
    // this can maybe be also reduced to only save one val per msa col,
    // maybe i dont need a vector solely for N, but maybe i can call this partial_logl where i can also
    // i also want this to be a per node thing since if i reassign the nni moves nodes i might need to remove some N factors
    prob: DMatrix<f64>,
    // add the probs for the Immortal link, N, and the block_len * log(r)
    // factor_N: DMatrix<f64>,
    blocks: Vec<usize>,
    block_lens: Vec<usize>,

    //a vector for every edge storing if the edge is time reversed or not,

    // says if the x and prob per node are calculated or not
    valid: bool,

    // this will hold the p(v.blen) = exp(q*v.blen) for every v in tree.nodes
    models: Vec<SubstMatrix>,

    // felsenstein vars: i need per node and per block for every state in q
    // or do i want something like Array3D
    felsenstein: Vec<DMatrix<f64>>,

    // do i really want string here. cant i also use the usize::from(NodeIdx) as key
    // i think this is because in pip we dont have a seq for every node
    leaf_sequence_info: HashMap<String, DMatrix<f64>>,

    // last action: i may only need one if I don't compute nodes in parallel
    // if i want to compute them in parallel i need to have one bool per edge
    // if i am only using one then i have to reset it for every node
    last_action: bool,
}

impl<Q: QMatrix + Clone> TKF92ModelInfo<Q> {
    // i am passing the model here since i need to know the size of q
    // cant i also use let m = phylo.msa.seqs.alphabet().symbols().len();
    // then i have to assume that the q and the alphabet are matching, or that this is asserted somewhere else

    pub fn new(phylo: &PhyloInfoAncestors, model: &TKF92Model<Q>) -> TKF92ModelInfo<Q> {
        let blocks = Self::get_blocks(&phylo.msa);
        let block_lens = Self::get_block_lens(&blocks);
        let number_of_blocks = blocks.len();
        let n_states_q = model.q.n();
        TKF92ModelInfo::<Q> {
            phantom: PhantomData,
            aggregated_x: DMatrix::<f64>::zeros(phylo.tree.len(), number_of_blocks),
            prob: DMatrix::<f64>::zeros(phylo.tree.len(), number_of_blocks),
            // factor_N: DMatrix::<f64>::zeros(phylo.tree.len(), number_of_blocks),
            blocks,
            block_lens,
            valid: false,
            models: vec![SubstMatrix::zeros(n_states_q, n_states_q); phylo.tree.len()],
            // node x site x state
            felsenstein: vec![DMatrix::zeros(phylo.msa.len(), n_states_q); phylo.tree.len()],
            leaf_sequence_info: Self::get_leaf_seq_info(&phylo, &model.q),
            last_action: false,
        }
    }

    fn get_leaf_seq_info(phylo: &PhyloInfoAncestors, q: &Q) -> HashMap<String, DMatrix<f64>> {
        // TODO: for me here the q is used to get the number of states which should always be the same
        // as the size of the alphabet
        // then i could also move this to the AncestralAlignment
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
                    // site_info.scale_mut((1.0) / site_info.sum());
                }
            }
            leaf_seq_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        leaf_seq_info
    }

    fn get_block_lens(blocks: &Vec<usize>) -> Vec<usize> {
        let mut block_lens = vec![0, blocks.len()];
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
    // q was also 'static in pip
    model: TKF92Model<Q>,
    phylo: PhyloInfoAncestors,
    //maybe refcell was used here to make it mutable in impl fn while keeping the other fields
    // of this struct immutable, i.e. only passing &self and not &mut self
    model_info: RefCell<TKF92ModelInfo<Q>>,

}

impl<Q: QMatrix + Display + Clone + 'static> TreeSearchCost for TKF92Cost<Q>
{
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn update_tree(&mut self, tree: crate::tree::Tree, dirty_nodes: &[NodeIdx]) {
        self.phylo.tree = tree;
        println!("{:?}", dirty_nodes);
    }

    fn tree(&self) -> &crate::tree::Tree {
        &self.phylo.tree
    }
}


impl<Q: QMatrix + Display + Clone> TKF92Cost<Q>
// where
//     TKF92Model<Q>: EvoModel,
{
    fn logl_old(&self) -> f64 {
        // pip uses matrix with first dimension being the len of the msa i think
        // do i also want to use this approach?
        // what about the N? How does multi threading work with matrix or for loop?
        let blocks = TKF92ModelInfo::<Q>::get_blocks(&self.phylo.msa);
        let tree = &self.phylo.tree;
        let model = &self.model;
        let node_map = self.phylo.msa.get_node_map();
        let l = model.lambda();
        let m = model.mu();
        let r = model.r();

        // for the root
        let mut prob: f64 = (1.0 - l / m).ln();

        // TODO: move this vector somewhere such that allocation only happens once
        let mut last_action = vec![false; tree.len()];
        // does it make sense to do sth like r_frac = (1-r)/r, because it is used more than once
        for (i, fragment) in blocks.iter().enumerate() {
            let mut x = 1.0;
            let fragment_len = if i == 0 {
                *fragment
            } else {
                fragment - blocks[i - 1]
            };
            if node_map[&tree.root][i].is_some() {
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
                let parent_is_gap = node_map[parent_id][i].is_none();
                let current_is_gap = node_map[node_idx][i].is_none();

                let b = Self::b(l, m, time);
                // not using time reversed attr since we are not doing any rerooting yet
                if i == 0 {
                    prob += Self::log_i1(l, b);
                }
                if parent_is_gap && current_is_gap {
                    continue;
                }
                if !parent_is_gap && !current_is_gap {
                    // homolog block
                    x *= Self::h1(l, m, b, time);
                    last_action[node_id_value] = false;
                }
                if !parent_is_gap && current_is_gap {
                    // deletion
                    x *= Self::n0(m, b);
                    last_action[node_id_value] = true;
                }
                if parent_is_gap && !current_is_gap {
                    // insertion
                    if last_action[node_id_value] {
                        prob += Self::log_n1(l, m, b, time);
                        prob -= (l * b).ln();
                        prob -= Self::n0(m, b).ln();
                    }
                    x *= l * b * (1.0 - r) / r;
                    prob += fragment_len as f64 * r.ln();
                    last_action[node_id_value] = false
                }
            }
            prob += x.ln();
            prob += (fragment_len - 1) as f64 * (1.0 + x).ln();
        }
        prob
    }

    fn logl(&self) -> f64 {
        if !self.model_info.borrow().valid {
            self.reset_all_nodes();
        }
        // println!("x = {:?}", self.model_info.borrow().aggregated_x);
        // println!("prob = {:?}", self.model_info.borrow().prob);
        let mut logl = 0.0;
        let root_id = usize::from(self.phylo.tree.root);
        for i in 0..self.model_info.borrow().blocks.len() {
            let block_len = self.model_info.borrow().block_lens[i];
            logl += self.model_info.borrow().prob[(root_id, i)];
            // TODO: maybe i also want to move the felsenstein root insertion calculation to here
            // and not have it in set root
            let x = self.model_info.borrow().aggregated_x[(root_id, i)];
            if x != 1.0 {
                logl += x.ln();
                logl += (block_len as f64 - 1.0) * (1.0 + x).ln();
            }
        }
        logl
    }

    fn reset_all_nodes(&self) {
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
        self.model_info.borrow_mut().valid = true;
    }

    fn set_root(&self) {
        self.model_info.borrow_mut().last_action = false;
        let root_idx = &self.phylo.tree.root;
        let root_id = usize::from(root_idx);
        let len = self.model_info.borrow().blocks.len();
        for block_id in 0..len {
            self.set_felsenstein_for_internal(root_idx, block_id);
            self.set_indel_x_and_prob_for_root(block_id);
            if self.is_insertion_at_root(block_id) {
                // println!("trying to add felsenstein prob at the root");
                self.model_info.borrow_mut().prob[(root_id, block_id)] +=
                    self.felsenstein_to_prob(root_idx, block_id);
            }
        }
    }

    fn set_internal(&self, node_idx: &NodeIdx) {
        // self.set_not_root(node_idx);
        self.model_info.borrow_mut().last_action = false;
        let node_id = usize::from(node_idx);
        self.model_info.borrow_mut().models[node_id] = self.model.p(self.phylo.tree.node(node_idx).blen);
        let len = self.model_info.borrow().blocks.len();
        for block_id in 0..len {
            self.set_felsenstein_for_internal(node_idx, block_id);
            self.set_indel_x_and_prob_for_not_root(node_idx, block_id);
            if self.is_insertion_at_non_root(node_idx, block_id) {
                self.model_info.borrow_mut().prob[(node_id, block_id)] +=
                    self.felsenstein_to_prob(node_idx, block_id);
            }
        }
    }

    fn set_leaf(&self, node_idx: &NodeIdx) {
        // self.set_not_root(node_idx);
        self.model_info.borrow_mut().last_action = false;
        let node_id = usize::from(node_idx);
        let len = self.model_info.borrow().blocks.len();
        self.model_info.borrow_mut().models[node_id] = self.model.p(self.phylo.tree.node(node_idx).blen);
        for block_id in 0..len {
            self.set_felsenstein_for_leaf(node_idx, block_id);
            self.set_indel_x_and_prob_for_not_root(node_idx, block_id);
            if self.is_insertion_at_non_root(node_idx, block_id) {
                self.model_info.borrow_mut().prob[(node_id, block_id)] +=
                    self.felsenstein_to_prob(node_idx, block_id);
            }
        }
    }

    fn set_indel_x_and_prob_for_root(&self, block_id: usize) {
        let block_len = self.model_info.borrow().block_lens[block_id];

        let l = self.model.lambda();
        let m = self.model.mu();
        let r = self.model.r();
        let mut x = 1.0;
        let mut prob = 0.0;
        if block_id == 0 {
            prob += (1.0 - l / m).ln();
        }

        let root_idx = &self.phylo.tree.root;
        if self.phylo.msa.get_node_map()[root_idx][self.model_info.borrow().blocks[block_id] - 1]
            .is_some()
        {
            x *= l / m * (1.0 - r) / r;
            prob += block_len as f64 * r.ln();
        }

        // this is the same as in set_indel_x_and_prob_for_not_root
        let node_id = usize::from(root_idx);
        for child in &self.phylo.tree.node(root_idx).children {
            let child_id = usize::from(child);
            x *= self.model_info.borrow().aggregated_x[(child_id, block_id)];
            prob += self.model_info.borrow().prob[(child_id, block_id)];
        }
        self.model_info.borrow_mut().prob[(node_id, block_id)] = prob;
        self.model_info.borrow_mut().aggregated_x[(node_id, block_id)] = x;
    }

    fn set_indel_x_and_prob_for_not_root(&self, node_idx: &NodeIdx, block_id: usize) {
        // or do i want to set the prob and x and not return it
        // for the felsenstein prob i need to add it to the prob
        // so i always have to set it with this method first and then
        // add the felsenstein prob to it
        let parent_id = &self.phylo.tree.node(node_idx).parent.unwrap();
        let mut prob = 0.0;
        let mut x: f64 = 1.0;
        let block_len = self.model_info.borrow().block_lens[block_id];
        let parent_is_gap = self.phylo.msa.get_node_map()[parent_id][block_id].is_none();
        let current_is_gap = self.phylo.msa.get_node_map()[node_idx][block_id].is_none();

        let time = self.phylo.tree.node(node_idx).blen;

        let l = self.model.lambda();
        let m = self.model.mu();
        let r = self.model.r();
        let b = Self::b(l, m, time);

        // not using time reversed attr since we are not doing any rerooting yet
        if block_id == 0 {
            prob += Self::log_i1(l, b);
        }
        if !parent_is_gap && current_is_gap {
            // deletion
            x *= Self::n0(m, b);
            self.model_info.borrow_mut().last_action = true;
        }
        if !parent_is_gap && !current_is_gap {
            // homolog block
            x *= Self::h1(l, m, b, time);
            self.model_info.borrow_mut().last_action = false;
        }
        if parent_is_gap && !current_is_gap {
            // insertion
            if self.model_info.borrow().last_action {
                prob += Self::log_n1(l, m, b, time);
                prob -= (l * b).ln();
                prob -= Self::n0(m, b).ln();
            }
            x *= l * b * (1.0 - r) / r;
            prob += block_len as f64 * r.ln();
            self.model_info.borrow_mut().last_action = false;
        }

        // this is the same as in set_indel_x_and_prob_for_root
        for child in &self.phylo.tree.node(node_idx).children {
            let child_id = usize::from(child);
            x *= self.model_info.borrow().aggregated_x[(child_id, block_id)];
            prob += self.model_info.borrow().prob[(child_id, block_id)];
        }
        let node_id = usize::from(node_idx);
        self.model_info.borrow_mut().prob[(node_id, block_id)] = prob;
        self.model_info.borrow_mut().aggregated_x[(node_id, block_id)] = x;
    }

    fn set_felsenstein_for_internal(&self, node_idx: &NodeIdx, block_id: usize) {
        let current_node_is_gap = self.phylo.msa.get_node_map()[node_idx][block_id].is_none();
        if current_node_is_gap {
            return;
        }
        let block = self.model_info.borrow().blocks[block_id];
        let block_len = self.model_info.borrow().block_lens[block_id];
        let node_id = usize::from(node_idx);
        // TODO: can this also be written with matrix operations?
        // println!("blocklens {:?}", self.model_info.borrow(s).block_lens);
        for site in (block - block_len)..block {
            for current_state in 0..self.model.q.n() {
                let mut prod_over_children = 1.0;
                for child_idx in &self.phylo.tree.node(node_idx).children {
                    if self.phylo.msa.get_node_map()[child_idx][block - 1].is_none() {
                        // println!("skipping node {} child {} block {} site {}", node_id, usize::from(child_idx), block, site);
                        continue;
                    }
                    let mut sum_over_children_states = 0.0;
                    for child_state in 0..self.model.q.n() {
                        let prob_of_mutating_to_child = self.model_info.borrow().models
                            [usize::from(child_idx)][(current_state, child_state)];
                        let child_prob = self.model_info.borrow().felsenstein
                            [usize::from(child_idx)][(site, child_state)];

                        // println!("mutation prob {}, child_prob {}", prob_of_mutating_to_child, child_prob);
                        sum_over_children_states += (prob_of_mutating_to_child) * (child_prob);
                    }
                    prod_over_children *= sum_over_children_states;
                }
                // println!("set felsenstein for internal, prod over children = {}", prod_over_children);
                self.model_info.borrow_mut().felsenstein[node_id][(site, current_state)] =
                    prod_over_children;
            }
        }
    }

    fn set_felsenstein_for_leaf(&self, node_idx: &NodeIdx, block_id: usize) {
        let current_node_is_gap = self.phylo.msa.get_node_map()[node_idx][block_id].is_none();
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
                let leaf_prob = self.model_info.borrow().leaf_sequence_info[node_name]
                [(current_state, block - 1)];
                self.model_info.borrow_mut().felsenstein[node_id][(site, current_state)] = leaf_prob;
                    
            }
        }
    }

    fn is_insertion_at_root(&self, block_id: usize) -> bool {
        self.phylo.msa.get_node_map()[&self.phylo.tree.root][block_id].is_some()
    }

    fn is_insertion_at_non_root(&self, node_idx: &NodeIdx, block_id: usize) -> bool {
        let parent_id = &self.phylo.tree.node(node_idx).parent.unwrap();
        let parent_is_gap = self.phylo.msa.get_node_map()[parent_id][block_id].is_none();
        let current_is_not_gap = self.phylo.msa.get_node_map()[node_idx][block_id].is_some();
        parent_is_gap && current_is_not_gap
    }

    fn felsenstein_to_prob(&self, node_idx: &NodeIdx, block_id: usize) -> f64 {
        let current_node_is_gap = self.phylo.msa.get_node_map()[node_idx][block_id].is_none();
        if current_node_is_gap {
            return 0.0;
        }
        let block = self.model_info.borrow().blocks[block_id];
        let block_len = self.model_info.borrow().block_lens[block_id];
        let node_id = usize::from(node_idx);
        let mut sum = 0.0;

        // sum over all off the eq prob
        // maybe i dont want to change the felsenstein vars to also include the eq probs of the
        // insertion char. but add it straight to the prob. which might make sense
        // if i want to reuse the felsenstein vars without the eq probs
        for site in (block - block_len)..block {
            let mut sum_for_state = 0.0;
            for state in 0..self.model.q.n() {
                sum_for_state += self.model.q.freqs()[state]
                    * self.model_info.borrow().felsenstein[node_id][(site, state)];
            }
            sum += sum_for_state.ln();
        }
        sum
    }

    // create a hash table such that for every node and therefore branch len we have this values precomputed
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
