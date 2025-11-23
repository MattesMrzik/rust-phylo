use fixedbitset::FixedBitSet;
use log::warn;

use crate::alignment::{AncestralAlignment, Mapping};
use crate::tkf_model::{log_i1, Event, TKFIndelCost, TKFModel};
use crate::tree::NodeIdx::{self, Internal, Leaf};

type EdgeAssignment = (bool, bool);
type EdgeAssignmentPossibilities = Vec<EdgeAssignment>;
type NodeSeq = FixedBitSet;
// pub(super) type EdgeSeqs = (NodeSeq, NodeSeq);
type QuartetDelOrNot = [bool; N_EDGES_IN_QUARTET];
type QuartetDelOrNotPossibilities = Vec<QuartetDelOrNot>;
type QuartetEvents = [Event; N_EDGES_IN_QUARTET];

const DP_ASSIGNMENT_AND_EVENTS_SIZE: usize = 128; // 2 (assignments) * 2 (deletion or not) ^ 5 (edges) = 128
const N_EDGES_IN_QUARTET: usize = 5; // v1, v2, t2, t3, t4

/// ```text
///       t1
///       |
///       v1
///      /  \
///     /    \
///    v2    t2
///   / \
///  /   \
/// t3   t4
/// ```
/// The ancestral wild card sequences for v1 and v2 are re-estimated. The assignments in the
/// dynamic programming are for \[v1, v2].
///
#[derive(Clone)]
struct QuartetEdges {
    edges: [NodeIdx; N_EDGES_IN_QUARTET],
    node_mappings: [Mapping; 3],
    t1_mapping: Option<Mapping>,
}

impl QuartetEdges {
    fn default() -> Self {
        QuartetEdges {
            edges: [NodeIdx::Leaf(0); N_EDGES_IN_QUARTET],
            node_mappings: std::array::from_fn(|_| vec![None]),
            t1_mapping: None,
        }
    }

    fn new(
        v1: NodeIdx,
        v2: NodeIdx,
        t2: NodeIdx,
        t3: NodeIdx,
        t4: NodeIdx,
        cost: &TKFIndelCost<impl TKFModel, impl AncestralAlignment>,
    ) -> Self {
        let phylo = &cost.phylo;
        // TODO: or is it better to have references here (then i need life times)?
        let t2_mapping = get_map_from_any_node(&phylo.msa, &t2).clone();
        let t3_mapping = get_map_from_any_node(&phylo.msa, &t3).clone();
        let t4_mapping = get_map_from_any_node(&phylo.msa, &t4).clone();
        let t1_mapping = if v1 != phylo.tree.root {
            let t1_idx = phylo.tree.node(&v1).parent.unwrap();
            Some(get_map_from_any_node(&phylo.msa, &t1_idx).clone())
        } else {
            None
        };

        QuartetEdges {
            edges: [v1, v2, t2, t3, t4],
            node_mappings: [t2_mapping, t3_mapping, t4_mapping],
            t1_mapping,
        }
    }

    fn edges(&self) -> &[NodeIdx; N_EDGES_IN_QUARTET] {
        &self.edges
    }

    fn v1(&self) -> &NodeIdx {
        &self.edges[0]
    }
    fn v2(&self) -> &NodeIdx {
        &self.edges[1]
    }
    fn t2_has_char(&self, site: usize) -> bool {
        self.node_mappings[0][site].is_some()
    }
    fn t3_has_char(&self, site: usize) -> bool {
        self.node_mappings[1][site].is_some()
    }
    fn t4_has_char(&self, site: usize) -> bool {
        self.node_mappings[2][site].is_some()
    }
    fn t1_has_char(&self, site: usize) -> bool {
        match &self.t1_mapping {
            Some(mapping) => mapping[site].is_some(),
            None => false,
        }
    }
}

struct BackTrackingResult {
    v1_bitset: FixedBitSet,
    v2_bitset: FixedBitSet,
    logl: f64,
}

/// Reestimator for ancestral wild card sequences at internal nodes after NNI.
/// Assumes that only the topology has changed, not the branch lengths.
pub struct EdgeSeqsReestimator<'a, T: TKFModel, AA: AncestralAlignment> {
    dp_table: Vec<[f64; DP_ASSIGNMENT_AND_EVENTS_SIZE]>,
    backtracking_table: Vec<[usize; DP_ASSIGNMENT_AND_EVENTS_SIZE]>, // pointers to prev gamma argmax,
    cost: &'a mut TKFIndelCost<T, AA>,
    quartet_edges: QuartetEdges,
    // TODO: add randomiser to choose between argmaxes or do stochastic backtracking
}

impl<'a, T: TKFModel, AA: AncestralAlignment> EdgeSeqsReestimator<'a, T, AA> {
    // TODO: before merge revert to pub(super)
    pub fn new(cost: &'a mut TKFIndelCost<T, AA>) -> Self {
        let num_blocks = cost.model_info.borrow().blocks.len();
        EdgeSeqsReestimator {
            dp_table: vec![[f64::NEG_INFINITY; DP_ASSIGNMENT_AND_EVENTS_SIZE]; num_blocks],
            backtracking_table: vec![[0; DP_ASSIGNMENT_AND_EVENTS_SIZE]; num_blocks],
            cost,
            quartet_edges: QuartetEdges::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn get_phylo(&self) -> &crate::phylo_info::PhyloInfo<AA> {
        &self.cost.phylo
    }

    // instead i want to have a cfg test helper that gets the logl from the dp
    pub fn reestimate(&mut self, v2_idx: &NodeIdx) -> f64 {
        let v2_id = self.cost.phylo.tree.node(v2_idx).id.clone();
        if v2_idx == &self.cost.phylo.tree.root {
            warn!(
                "Reestimation can only be performed on non root internal nodes. Skipping root {v2_id}"
            );
            return self.cost.logl();
        }
        if let Leaf(_) = v2_idx {
            warn!(
                "Reestimation can only be performed on non root internal nodes. Skipping {v2_id}"
            );
            return self.cost.logl();
        }
        for node in self.cost.phylo.tree.postorder() {
            if !self.cost.model_info.borrow().valid_for_reestimation[usize::from(*node)] {
                warn!("Reestimation can only be performed on a cost with valid_for_reestimation internal nodes tmp values. Making them valid now.");
                println!("Reestimation can only be performed on a cost with valid_for_reestimation internal nodes tmp values. Making them valid now.");
                self.cost.logl();
                break;
            }
        }
        self.prepare_for_dp(v2_idx);
        self.fill_dp_table();
        let backtrack_res = self.backtrack();
        self.update_mappings(&backtrack_res);
        self.valid_false();
        self.make_valid_for_further_reestimate_calls();
        backtrack_res.logl
    }

    // fn compare_to_logl_from_root(&self, backtrack: f64) {
    //     let lambda = self.cost.model.lambda();
    //     let mu = self.cost.model.mu();
    //     let root_id = usize::from(self.cost.phylo.tree.root);
    //     let mut logl = 0.0;
    //     logl += (1.0 - lambda / mu).ln();
    //     let model_info = self.cost.model_info.borrow();
    //     for node in self.cost.phylo.tree.postorder() {
    //         if node == &self.cost.phylo.tree.root {
    //             continue;
    //         }
    //         logl += log_i1(lambda, model_info.beta[usize::from(node)]);
    //     }
    //     for block_id in 0..model_info.blocks.len() {
    //         let block_len = model_info.block_lengths[block_id];
    //         logl += model_info.subtree_eta[(root_id, block_id)];
    //         let tree_event_prob = model_info.subtree_event_prob[(root_id, block_id)];
    //         logl += self.cost.model.block_prob(tree_event_prob, block_len);
    //     }
    //     println!(
    //         "in reestimate logl from root aggregation: {logl}, logl from backtrack: {backtrack}"
    //     );
    // }

    fn valid_false(&mut self) {
        for edge in self.quartet_edges.edges() {
            self.cost
                .model_info
                .borrow_mut()
                .valid
                .set(usize::from(*edge), false);
        }
    }

    pub(super) fn prepare_for_dp(&mut self, v2_idx: &NodeIdx) {
        let num_blocks = self.cost.model_info.borrow().blocks.len();
        self.dp_table = vec![[f64::NEG_INFINITY; DP_ASSIGNMENT_AND_EVENTS_SIZE]; num_blocks];
        //self.backtracking_table = vec![[0; DP_ASSIGNMENT_AND_EVENTS_SIZE]; num_blocks];
        self.quartet_edges = self.get_quartet(v2_idx);
        for block_id in 0..num_blocks {
            self.remove_old_quartet_event_prob_from_root(block_id);
            self.remove_old_quartet_eta_from_root(block_id);
        }
    }

    // updates only the nodes vals and the root aggregated but not other internal aggregated vals
    // can be called if reestimate without tree change
    // assumes that new mappings were already set in the msa
    fn make_valid_for_further_reestimate_calls(&mut self) {
        let num_blocks = self.cost.model_info.borrow().blocks.len();
        for block_id in 0..num_blocks {
            let root_id = usize::from(self.cost.phylo.tree.root);
            for edge in self.quartet_edges.edges() {
                let event = self.cost.determine_event(edge, block_id);
                // even though the edge can be the root, i thinks its still fine to call these
                let node_event_prob = self.cost.event_prob_for_non_root(edge, event);
                let node_eta = self.cost.eta_for_non_root(edge, event);
                self.cost.update_previous_event(edge, event);
                let mut model_info = self.cost.model_info.borrow_mut();
                model_info.node_event_prob[(usize::from(edge), block_id)] = node_event_prob;
                model_info.node_eta[(usize::from(edge), block_id)] = node_eta;
                model_info.subtree_event_prob[(root_id, block_id)] *= node_event_prob;
                model_info.subtree_eta[(root_id, block_id)] += node_eta;
            }
            // println!(
            //     "update new subtree_event_prob at root for block {block_id}: {}",
            //     self.cost.model_info.borrow().subtree_event_prob[(root_id, block_id)]
            // );
            // println!(
            //     "update new subtree_eta at root for block {block_id}: {}",
            //     self.cost.model_info.borrow().subtree_eta[(root_id, block_id)]
            // );
        }
        for edge in self.quartet_edges.edges() {
            self.cost
                .model_info
                .borrow_mut()
                .valid_for_reestimation
                .set(usize::from(edge), true);
        }
    }

    fn remove_old_quartet_eta_from_root(&self, block_id: usize) {
        let root_id = usize::from(self.cost.phylo.tree.root);
        self.cost.model_info.borrow_mut().subtree_eta[(root_id, block_id)] -=
            self.quartet_eta_pre_reestimation_or_move(block_id);
    }

    fn remove_old_quartet_event_prob_from_root(&self, block_id: usize) {
        let root_id = usize::from(self.cost.phylo.tree.root);
        self.cost.model_info.borrow_mut().subtree_event_prob[(root_id, block_id)] /=
            self.quartet_event_prob_pre_reestimation_or_move(block_id);
    }

    fn quartet_eta_pre_reestimation_or_move(&self, block_id: usize) -> f64 {
        let mut eta = 0.0;
        for node in self.quartet_edges.edges() {
            eta += self.cost.model_info.borrow().node_eta[(usize::from(*node), block_id)];
        }
        eta
    }

    /// Computes the product of x values for the nodes in the quartet before reestimation
    fn quartet_event_prob_pre_reestimation_or_move(&self, block_id: usize) -> f64 {
        let mut x = 1.0;
        for node in self.quartet_edges.edges() {
            x *= self.cost.model_info.borrow().node_event_prob[(usize::from(*node), block_id)];
        }
        x
    }

    fn update_mappings(&mut self, backtrack_res: &BackTrackingResult) {
        let block_lengths = &self.cost.model_info.borrow().block_lengths;
        let seq_len = self.cost.phylo.msa.len();
        let v1_mapping = mapping_from_node_seq(&backtrack_res.v1_bitset, block_lengths, seq_len);
        let v2_mapping = mapping_from_node_seq(&backtrack_res.v2_bitset, block_lengths, seq_len);
        self.cost
            .phylo
            .msa
            .update_ancestral_map(self.quartet_edges.v1(), v1_mapping);
        self.cost
            .phylo
            .msa
            .update_ancestral_map(self.quartet_edges.v2(), v2_mapping);
    }

    fn fill_dp_table(&mut self) {
        // isnt this already checked before calling reestimate?
        let root = &self.cost.phylo.tree.root;
        if self.quartet_edges.v2() == root {
            return;
        }
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            for assignment in self.possible_assignments_of_nni_edge(block_id) {
                // println!("assignment at block {block_id}: {:?}", assignment);
                let actions = self.event_for_assignment(assignment, block_id);
                let x_prob = self.integrated_root_event_prob(&actions, block_id);
                for q_del_or_not in self.possible_events(&actions, block_id == 0) {
                    // println!("quartet del or not at block {block_id}: {:?}", q_del_or_not);
                    let dp_index = bools_to_index(&assignment, &q_del_or_not);
                    // println!("{:?}", index_to_bools(dp_index));
                    if block_id == 0 {
                        self.dp_table[block_id][dp_index] = x_prob;
                        continue;
                    }
                    let (max_prev_gamma, argmax) =
                        self.max_over_previous(&q_del_or_not, &actions, block_id);
                    self.backtracking_table[block_id][dp_index] = argmax;
                    let root_id = usize::from(self.cost.phylo.tree.root);
                    let eta_for_block =
                        self.cost.model_info.borrow().subtree_eta[(root_id, block_id)];
                    self.dp_table[block_id][dp_index] = max_prev_gamma + eta_for_block + x_prob;
                }
            }
        }
    }

    /// Finds the max over previous assignments and events that lead to the provided events
    fn max_over_previous(
        &self,
        current_events: &QuartetDelOrNot,
        current_actions: &QuartetEvents,
        block_id: usize,
    ) -> (f64, usize) {
        let mut max = f64::NEG_INFINITY;
        let mut argmax = 0;

        // instead of recalculating this it could be reused from the previous block
        for prev_assignment in self.possible_assignments_of_nni_edge(block_id - 1) {
            for prev_event in self.prev_compatible_events(current_events, current_actions) {
                let prev_dp_index = bools_to_index(&prev_assignment, &prev_event);
                let prev_gamma = self.dp_table[block_id - 1][prev_dp_index];
                if prev_gamma == f64::NEG_INFINITY {
                    // because the compatibility of the previous event with the previous
                    // assignment is not checked
                    continue;
                }
                let current = prev_gamma + self.quartet_eta(current_actions, &prev_event);
                if current > max {
                    max = current;
                    argmax = prev_dp_index;
                }
            }
        }
        (max, argmax)
    }

    fn quartet_eta(&self, actions: &QuartetEvents, prev_events: &[bool]) -> f64 {
        for i in 0..N_EDGES_IN_QUARTET {
            if actions[i] == Event::Insertion && prev_events[i] {
                let edge = &self.quartet_edges.edges()[i];
                return self.cost.model_info.borrow_mut().eta[usize::from(edge)];
            }
        }
        0.0
    }

    /// Based on the current events and actions finds all compatible previous events.
    // TODO: these are not ensured to be compatible with the previous assignments. Would it be
    // worth to filter them here?
    fn prev_compatible_events(
        &self,
        current_events: &QuartetDelOrNot,
        current_actions: &[Event; N_EDGES_IN_QUARTET],
    ) -> QuartetDelOrNotPossibilities {
        // let mut choices: Vec<Vec<bool>> = vec![vec![]; N_EDGES_IN_QUARTET];
        let mut no_choices = [false; N_EDGES_IN_QUARTET];
        let mut choices = Vec::with_capacity(N_EDGES_IN_QUARTET);
        for i in 0..N_EDGES_IN_QUARTET {
            match current_actions[i] {
                // we have a gap col, so we have no action here but pass through the previous one
                Event::Nothing => no_choices[i] = current_events[i],
                // we have an action here, so we can choose any previous action
                _ => choices.push(i),
            };
        }
        possibilities_for_choices(&choices, &no_choices)
    }

    /// Finds all possible combinations of deletion or not for each edge in the quartet
    /// given the (current) actions taken on those edges.
    fn possible_events(
        &self,
        actions: &[Event; N_EDGES_IN_QUARTET],
        is_first_block: bool,
    ) -> QuartetDelOrNotPossibilities {
        let mut no_choices = [false; N_EDGES_IN_QUARTET];
        let mut choices = Vec::with_capacity(N_EDGES_IN_QUARTET);

        for (i, edge) in self.quartet_edges.edges().iter().enumerate() {
            match actions[i] {
                Event::Deletion => no_choices[i] = true,
                Event::Insertion | Event::Homolog => no_choices[i] = false,
                Event::Nothing => {
                    if is_first_block || edge == &self.cost.phylo.tree.root {
                        no_choices[i] = false;
                    } else {
                        choices.push(i);
                    }
                }
            };
        }
        // TODO also replace this with get_possibilities_for_choices?
        possibilities_for_choices(&choices, &no_choices)
    }

    fn get_quartet(&self, v2_idx: &NodeIdx) -> QuartetEdges {
        let tree = &self.cost.phylo.tree;
        let v1_idx = tree.node(v2_idx).parent.unwrap();
        let t2_idx = tree.sibling(v2_idx).unwrap();
        let children_of_v2 = &tree.node(v2_idx).children;
        let t3_idx = children_of_v2[0];
        let t4_idx = children_of_v2[1];
        QuartetEdges::new(v1_idx, *v2_idx, t2_idx, t3_idx, t4_idx, self.cost)
    }

    /// Computes the integrated x probability for the quartet given the actions
    fn integrated_root_event_prob(&self, actions: &QuartetEvents, block_id: usize) -> f64 {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let block_len = self.cost.model_info.borrow().block_lengths[block_id];

        let mut x = self.cost.model_info.borrow().subtree_event_prob[(root_id, block_id)];
        // TODO: does it make sense to pre-compute this functions return values?
        x *= self.quartet_event_prob(actions);
        self.cost.model.block_prob(x, block_len)
    }

    /// Computes the product of x values for the nodes in the quartet for the provided actions
    /// which correspond to an assignment of characters at v1 and v2 that is currently considered
    /// in the dynamic programming.
    fn quartet_event_prob(&self, events: &QuartetEvents) -> f64 {
        let mut quartet_event_prob = 1.0;
        let model_info = self.cost.model_info.borrow();
        // Here it is assumed that the blens have not changed, only the topology.
        for (i, node) in self.quartet_edges.edges().iter().enumerate() {
            let node_id = usize::from(*node);

            quartet_event_prob *= match events[i] {
                Event::Insertion => model_info.insertion[node_id],
                Event::Deletion => model_info.n0[node_id],
                Event::Homolog => model_info.h1[node_id],
                Event::Nothing => 1.0,
            };
        }
        quartet_event_prob
    }

    /// Based on whether there are chars at the "leaves" of the quartet finds
    /// all possible [assignment for v1, assignment for v2] combinations that
    /// follow Dollo's principle.
    pub fn possible_assignments_of_nni_edge(&self, block_id: usize) -> EdgeAssignmentPossibilities {
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;

        let t1_is_char = match &self.quartet_edges.t1_mapping {
            Some(mapping) => mapping[site].is_some(),
            None => false,
        };
        let t2_is_char = self.quartet_edges.t2_has_char(site);
        let t3_is_char = self.quartet_edges.t3_has_char(site);
        let t4_is_char = self.quartet_edges.t4_has_char(site);
        let left_is_some = t1_is_char || t2_is_char;
        let right_is_some = t3_is_char || t4_is_char;
        let both_left_are_some = t1_is_char && t2_is_char;
        let both_right_are_some = t3_is_char && t4_is_char;
        if left_is_some && !right_is_some {
            if both_left_are_some {
                vec![(true, true), (true, false)]
            } else {
                vec![(true, true), (true, false), (false, false)]
            }
        } else if !left_is_some && right_is_some {
            if both_right_are_some {
                vec![(true, true), (false, true)]
            } else {
                vec![(true, true), (false, true), (false, false)]
            }
        } else if !left_is_some && !right_is_some {
            vec![(false, false)]
        } else {
            vec![(true, true)]
        }
    }

    fn event_for_assignment(&self, assignment: EdgeAssignment, block_id: usize) -> QuartetEvents {
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;
        let mut actions = [Event::Nothing; N_EDGES_IN_QUARTET];
        let v1_has_char = assignment.0;
        let v2_has_char = assignment.1;
        // edge (t1 = pa(v1) -> v1)
        actions[0] = event_for_edge(v1_has_char, self.quartet_edges.t1_has_char(site));
        // edge (v1 = pa(v2) -> v2)
        actions[1] = event_for_edge(v2_has_char, v1_has_char);
        // edge (v1 = pa(t2) -> t2)
        actions[2] = event_for_edge(self.quartet_edges.t2_has_char(site), v1_has_char);
        // edge (v2 = pa(t3) -> t3)
        actions[3] = event_for_edge(self.quartet_edges.t3_has_char(site), v2_has_char);
        // edge (v2 = pa(t4) -> t4)
        actions[4] = event_for_edge(self.quartet_edges.t4_has_char(site), v2_has_char);
        actions
    }

    fn backtrack(&self) -> BackTrackingResult {
        // prepare
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        let mut v1_bitset = FixedBitSet::with_capacity(n_blocks);
        let mut v2_bitset = FixedBitSet::with_capacity(n_blocks);

        // start from the last block
        let (last_max, last_argmax) = self.max_of_last_col();
        let (assignment, _quartet_del_or_not) = index_to_bools(last_argmax);
        v1_bitset.set(n_blocks - 1, assignment[0]);
        v2_bitset.set(n_blocks - 1, assignment[1]);
        let mut came_from = self.backtracking_table[n_blocks - 1][last_argmax];
        // go back the path
        for block_id in (0..(n_blocks - 1)).rev() {
            let (assignment, _) = index_to_bools(came_from);
            v1_bitset.set(block_id, assignment[0]);
            v2_bitset.set(block_id, assignment[1]);
            if block_id > 0 {
                came_from = self.backtracking_table[block_id][came_from];
            }
        }
        BackTrackingResult {
            v1_bitset,
            v2_bitset,
            logl: last_max + self.const_per_alignment(),
        }
    }

    fn max_of_last_col(&self) -> (f64, usize) {
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        let mut max = f64::NEG_INFINITY;
        let mut max_index = 0;
        for (index, &value) in self.dp_table[n_blocks - 1].iter().enumerate() {
            if value > max {
                max = value;
                max_index = index;
            }
        }
        (max, max_index)
    }

    // fn same_as_before(&self, v1_bitset: &FixedBitSet, v2_bitset: &FixedBitSet) -> bool {
    //     let old_v1_mapping = self.cost.phylo.msa.ancestral_map(self.quartet_edges.v1());
    //     let old_v2_mapping = self.cost.phylo.msa.ancestral_map(self.quartet_edges.v2());
    //     for i in 0..v1_bitset.len() {
    //         let v1_has_char = v1_bitset.contains(i);
    //         let v2_has_char = v2_bitset.contains(i);
    //         let site = self.cost.model_info.borrow().blocks[i] - 1;
    //         let old_v1_has_char = old_v1_mapping[site].is_some();
    //         let old_v2_has_char = old_v2_mapping[site].is_some();
    //         if v1_has_char != old_v1_has_char || v2_has_char != old_v2_has_char {
    //             return false;
    //         }
    //     }
    //     true
    // }

    fn const_per_alignment(&self) -> f64 {
        let l = self.cost.model.lambda();
        let m = self.cost.model.mu();
        let mut const_per_alignment: f64 = (1.0 - l / m).ln();
        for node in self.cost.phylo.tree.postorder() {
            if node == &self.cost.phylo.tree.root {
                continue;
            }
            const_per_alignment +=
                log_i1(l, self.cost.model_info.borrow_mut().beta[usize::from(node)]);
        }
        const_per_alignment
    }

    pub fn print_dp_table(&self) {
        println!("dp table");
        for block_id in 0..self.cost.model_info.borrow().blocks.len() {
            for (index, &value) in self.dp_table[block_id].iter().enumerate() {
                if value != f64::NEG_INFINITY {
                    println!(
                        "block {}, assignment & events {:?}, {}",
                        block_id,
                        index_to_bools(index),
                        value
                    );
                }
            }
        }
    }
}

pub fn mapping_from_node_seq(node_seq: &NodeSeq, block_lens: &[usize], seq_len: usize) -> Mapping {
    let mut mapping = Vec::with_capacity(seq_len);
    let mut count = 0;
    for (i, &block_len) in block_lens.iter().enumerate() {
        for _ in 0..block_len {
            if node_seq.contains(i) {
                mapping.push(Some(count));
                count += 1;
            } else {
                mapping.push(None);
            }
        }
    }
    mapping
}

pub(super) fn possibilities_for_choices(
    choices: &[usize],
    no_choices: &[bool; N_EDGES_IN_QUARTET],
) -> QuartetDelOrNotPossibilities {
    let num_combinations = 1 << choices.len();
    let mut all_possibilities = Vec::with_capacity(num_combinations);
    for possibility_idx in 0..num_combinations {
        let mut possibility = *no_choices;
        for (j, &edge_index) in choices.iter().enumerate() {
            let bit = (possibility_idx >> j) & 1;
            possibility[edge_index] = bit != 0;
        }
        all_possibilities.push(possibility);
    }
    all_possibilities
}

fn bools_to_index(assignment: &EdgeAssignment, q_del_or_not: &QuartetDelOrNot) -> usize {
    [assignment.0, assignment.1]
        .iter()
        .chain(q_del_or_not.iter())
        .fold(0, |index, &b| (index << 1) | (b as usize))
}

// TODO: theoretically only the returned assignment is needed
fn index_to_bools(index: usize) -> ([bool; 2], Vec<bool>) {
    let mut bits = index;
    let mut event = vec![false; N_EDGES_IN_QUARTET];

    for i in (0..N_EDGES_IN_QUARTET).rev() {
        event[i] = (bits & 1) != 0;
        bits >>= 1;
    }

    let assignment = [(bits & 2) != 0, (bits & 1) != 0];

    (assignment, event)
}

fn event_for_edge(node_has_char: bool, parent_has_char: bool) -> Event {
    match (node_has_char, parent_has_char) {
        (true, true) => Event::Homolog,
        (true, false) => Event::Insertion,
        (false, true) => Event::Deletion,
        (false, false) => Event::Nothing,
    }
}

pub(super) fn get_map_from_any_node<'a, AA: AncestralAlignment>(
    msa: &'a AA,
    node: &'a NodeIdx,
) -> &'a Mapping {
    match node {
        Internal(_) => msa.ancestral_map(node),
        Leaf(_) => msa.leaf_map(node),
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use crate::alphabets::dna_alphabet;
    use crate::tkf_model::tests::setup_test_phylo;
    use crate::tkf_model::{EdgeSeqsReestimator, TKF92IndelCostBuilder};

    #[test]
    fn tkf_remove_and_add_back_quartet() {
        let phylo = setup_test_phylo(dna_alphabet());
        let mut cost = TKF92IndelCostBuilder::new(0.4, 0.5, 0.8, phylo)
            .build()
            .unwrap();
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost);
        let original_logl = reestimator.cost.logl();
        assert_eq!(original_logl, reestimator.cost.logl_from_root_model_info());
        let dummy_v2_idx = reestimator.cost.phylo.tree.by_id("I3").idx;
        reestimator.prepare_for_dp(&dummy_v2_idx);
        assert_ne!(reestimator.cost.logl_from_root_model_info(), original_logl);
        reestimator.make_valid_for_further_reestimate_calls();
        assert_eq!(reestimator.cost.logl_from_root_model_info(), original_logl);
    }
}
