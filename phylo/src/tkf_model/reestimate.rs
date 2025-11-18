use anyhow::bail;
use itertools::Itertools;

use crate::alignment::{AncestralAlignment, Mapping};
use crate::tkf_model::{log_i1, TKFIndelCost, TKFModel};
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::Result;

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

#[derive(PartialEq, Debug, Clone, Copy)]
enum Action {
    Insertion,
    Deletion,
    Homolog,
    Nothing,
}

/// Reestimatator for ancestral wild card sequences at internal nodes after NNI.
/// Assumes that only the topology has changed, not the branch lengths.
#[derive(Clone)]
pub struct Reestimator<'a, T: TKFModel, AA: AncestralAlignment> {
    dp_table: Vec<[f64; DP_ASSIGNMENT_AND_EVENTS_SIZE]>,
    backtracking_table: Vec<[usize; DP_ASSIGNMENT_AND_EVENTS_SIZE]>, // pointers to prev gamma argmax,
    cost: &'a TKFIndelCost<T, AA>,
    quartet_edges: QuartetEdges,
    // TODO: add randomizer to choose between argmaxes or do stochastic backtracking
}

impl<'a, T: TKFModel, AA: AncestralAlignment> Reestimator<'a, T, AA> {
    pub(crate) fn new(cost: &'a TKFIndelCost<T, AA>) -> Self {
        let num_blocks = cost.model_info.borrow().blocks.len();
        Reestimator {
            dp_table: vec![[f64::NEG_INFINITY; DP_ASSIGNMENT_AND_EVENTS_SIZE]; num_blocks],
            backtracking_table: vec![[0; DP_ASSIGNMENT_AND_EVENTS_SIZE]; num_blocks],
            cost,
            quartet_edges: QuartetEdges::default(),
        }
    }

    pub(crate) fn reestimate(
        &mut self,
        v2_idx: &NodeIdx,
    ) -> Result<(Vec<(NodeIdx, Mapping)>, f64)> {
        if v2_idx == &self.cost.phylo.tree.root {
            bail!("Reestimation can only be performed on internal nodes.");
        }
        if let Leaf(_) = v2_idx {
            bail!("Reestimation can only be performed on internal nodes.");
        }
        for node in self.cost.phylo.tree.postorder() {
            if !self.cost.model_info.borrow().valid[usize::from(*node)] {
                // TODO: i might want another bool that indicates that root aggregated_x and node_x
                // are valid. despite perhaps the internal aggregated_x not being valid, because
                // during reestimation we dont care for those
                bail!("Reestimation can only be performed on a cost where internal nodes tmp values are valid");
            }
        }
        self.quartet_edges = self.get_quartet(v2_idx);
        self.fill_dp_table();
        Ok(self.max_mappings())
        // after calling update maps not in tree search (but in the repeated asr), i can also implement and call a new method that udpates
        // the node_x and root aggregated values for the changed quartet.
        // then i could set valid_(tree move, realingment) to false
        // but valid for blen and reestimation to true
    }

    fn fill_dp_table(&mut self) {
        let root = &self.cost.phylo.tree.root;
        if self.quartet_edges.v2() == root {
            return;
        }
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            for assignment in self.possible_assignments_of_nni_edge(block_id) {
                let actions = self.actions_for_assignment(assignment, block_id);
                let x_prob = self.integrated_x(&actions, block_id);
                for event in self.possible_events(&actions, block_id == 0) {
                    let dp_index = bools_to_index(&assignment, &event);
                    if block_id == 0 {
                        self.dp_table[block_id][dp_index] = x_prob;
                        continue;
                    }
                    let (max_prev_gamma, argmax) =
                        self.max_over_previous(&event, &actions, block_id);
                    self.backtracking_table[block_id][dp_index] = argmax;
                    self.dp_table[block_id][dp_index] =
                        max_prev_gamma + self.factor_n_for_block(block_id) + x_prob;
                }
            }
        }
    }

    /// Finds the max over previous assignments and events that lead to the provided events
    fn max_over_previous(
        &self,
        current_events: &[bool],
        current_actions: &[Action; N_EDGES_IN_QUARTET],
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
                let current = prev_gamma + self.quartet_factor_n(current_actions, &prev_event);
                if current > max {
                    max = current;
                    argmax = prev_dp_index;
                }
            }
        }
        (max, argmax)
    }

    fn quartet_factor_n(
        &self,
        actions: &[Action; N_EDGES_IN_QUARTET],
        prev_events: &[bool],
    ) -> f64 {
        for i in 0..N_EDGES_IN_QUARTET {
            if actions[i] == Action::Insertion && prev_events[i] {
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
        current_events: &[bool],
        current_actions: &[Action; N_EDGES_IN_QUARTET],
    ) -> Vec<Vec<bool>> {
        let mut choices: Vec<Vec<bool>> = vec![vec![]; N_EDGES_IN_QUARTET];
        for i in 0..N_EDGES_IN_QUARTET {
            match current_actions[i] {
                // we have a gap col, so we have no action here but pass through the previous one
                Action::Nothing => choices[i] = vec![current_events[i]],
                // we have an action here, so we can choose any previous action
                _ => choices[i] = vec![false, true],
            };
        }
        choices.into_iter().multi_cartesian_product().collect()
    }

    /// Finds all possible combinations of deletion or not for each edge in the quartet
    /// given the (current) actions taken on those edges.
    fn possible_events(
        &self,
        actions: &[Action; N_EDGES_IN_QUARTET],
        is_first_block: bool,
    ) -> Vec<Vec<bool>> {
        let mut choices = vec![vec![]; N_EDGES_IN_QUARTET];
        for (i, edge) in self.quartet_edges.edges().iter().enumerate() {
            match actions[i] {
                Action::Deletion => choices[i] = vec![true],
                Action::Insertion => choices[i] = vec![false],
                Action::Homolog => choices[i] = vec![false],
                // If the current action is nothing, then we can pass through the previous event
                Action::Nothing => {
                    if is_first_block || edge == &self.cost.phylo.tree.root {
                        choices[i] = vec![false];
                    } else {
                        choices[i] = vec![false, true];
                    }
                }
            }
        }
        choices.into_iter().multi_cartesian_product().collect()
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

    fn factor_n_for_block(&self, block_id: usize) -> f64 {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let mut factor_n = self.cost.model_info.borrow().subtree_eta[(root_id, block_id)];
        factor_n -= self.quartet_factor_n_pre_reestimation(block_id);
        // the factor n of the new quartet is added during dp filling in max_over_previous
        factor_n
    }

    fn quartet_factor_n_pre_reestimation(&self, block_id: usize) -> f64 {
        let mut factor_n = 0.0;
        for node in self.quartet_edges.edges() {
            factor_n += self.cost.model_info.borrow().node_eta[(usize::from(*node), block_id)];
        }
        factor_n
    }

    /// Computes the integrated x probability for the quartet given the actions
    fn integrated_x(&self, actions: &[Action; N_EDGES_IN_QUARTET], block_id: usize) -> f64 {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let block_len = self.cost.model_info.borrow().block_lengths[block_id];

        let mut x = self.cost.model_info.borrow().subtree_event_prob[(root_id, block_id)];
        x /= self.quartet_x_pre_reestimation(block_id);
        x *= self.quartet_x(actions);
        self.cost.model.block_prob(x, block_len)
    }

    /// Computes the product of x values for the nodes in the quartet before reestimation
    fn quartet_x_pre_reestimation(&self, block_id: usize) -> f64 {
        let mut x = 1.0;
        for node in self.quartet_edges.edges() {
            x *= self.cost.model_info.borrow().node_event_prob[(usize::from(*node), block_id)];
        }
        x
    }

    /// Computes the product of x values for the nodes in the quartet for the provided actions
    /// which correspond to an assignment of characters at v1 and v2 that is currently considered
    /// in the dynamic programming.
    fn quartet_x(&self, actions: &[Action; N_EDGES_IN_QUARTET]) -> f64 {
        let mut x = 1.0;
        let model_info = self.cost.model_info.borrow();
        // Here it is assumed that the blens have not changed, only the topology.
        for (i, node) in self.quartet_edges.edges().iter().enumerate() {
            let node_id = usize::from(*node);

            x *= match actions[i] {
                Action::Insertion => model_info.insertion[node_id],
                Action::Deletion => model_info.n0[node_id],
                Action::Homolog => model_info.h1[node_id],
                Action::Nothing => 1.0,
            };
        }
        x
    }

    /// Based on whether there are chars at the "leaves" of the quartet finds
    /// all possible [assignment for v1, assignment for v2] combinations that
    /// follow Dollo's principle.
    fn possible_assignments_of_nni_edge(&self, block_id: usize) -> Vec<[bool; 2]> {
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
                vec![[true, true], [true, false]]
            } else {
                vec![[true, true], [true, false], [false, false]]
            }
        } else if !left_is_some && right_is_some {
            if both_right_are_some {
                vec![[true, true], [false, true]]
            } else {
                vec![[true, true], [false, true], [false, false]]
            }
        } else if !left_is_some && !right_is_some {
            vec![[false, false]]
        } else {
            vec![[true, true]]
        }
    }

    fn actions_for_assignment(
        &self,
        assignment: [bool; 2],
        block_id: usize,
    ) -> [Action; N_EDGES_IN_QUARTET] {
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;
        let mut actions = [Action::Nothing; N_EDGES_IN_QUARTET];
        let v1_has_char = assignment[0];
        let v2_has_char = assignment[1];
        // edge (t1 = pa(v1) -> v1)
        actions[0] = action_for_edge(v1_has_char, self.quartet_edges.t1_has_char(site));
        // edge (v1 = pa(v2) -> v2)
        actions[1] = action_for_edge(v2_has_char, v1_has_char);
        // edge (v1 = pa(t2) -> t2)
        actions[2] = action_for_edge(self.quartet_edges.t2_has_char(site), v1_has_char);
        // edge (v2 = pa(t3) -> t3)
        actions[3] = action_for_edge(self.quartet_edges.t3_has_char(site), v2_has_char);
        // edge (v2 = pa(t4) -> t4)
        actions[4] = action_for_edge(self.quartet_edges.t4_has_char(site), v2_has_char);
        actions
    }

    fn backtrack(&self) -> (Vec<[bool; 2]>, f64) {
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        let mut assignments = vec![[false, false]; n_blocks];
        let mut max = f64::NEG_INFINITY;
        let mut max_index: Option<usize> = None;
        for (index, &value) in self.dp_table[n_blocks - 1].iter().enumerate() {
            if value > max {
                max = value;
                max_index = Some(index);
            }
        }
        let last_argmax = max_index.unwrap();
        let last_max = max;

        let (assignment, _) = index_to_bools(last_argmax);
        let mut came_from = self.backtracking_table[n_blocks - 1][last_argmax];
        assignments[n_blocks - 1] = assignment;
        // go back the path
        for block_id in (0..(n_blocks - 1)).rev() {
            let (assignment, _) = index_to_bools(came_from);
            assignments[block_id] = assignment;
            if block_id > 0 {
                came_from = self.backtracking_table[block_id][came_from];
            }
        }
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

        (assignments, last_max + const_per_alignment)
    }

    fn mapping_from_vec(&self, assignments: &[[bool; 2]]) -> Vec<(NodeIdx, Mapping)> {
        let mut mappings = vec![vec![]; 2];
        let mut counts = [0; 2];
        let block_lens = &self.cost.model_info.borrow().block_lengths;
        // for i in 0..self.cost.model_info.borrow().blocks.len() {
        for (i, assignment) in assignments.iter().enumerate() {
            for node in 0..2 {
                if assignment[node] {
                    for _ in 0..block_lens[i] {
                        mappings[node].push(Some(counts[node]));
                        counts[node] += 1;
                    }
                } else {
                    for _ in 0..block_lens[i] {
                        mappings[node].push(None);
                    }
                }
            }
        }
        vec![
            (*self.quartet_edges.v1(), mappings.remove(0)),
            (*self.quartet_edges.v2(), mappings.remove(0)),
        ]
    }

    fn max_mappings(&self) -> (Vec<(NodeIdx, Mapping)>, f64) {
        let backtrack = self.backtrack();
        (self.mapping_from_vec(&backtrack.0), backtrack.1)
    }

    pub fn print_dp_table(&self) {
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

fn bools_to_index(assignment: &[bool; 2], event: &[bool]) -> usize {
    assignment
        .iter()
        .chain(event.iter())
        .fold(0, |index, &b| (index << 1) | (b as usize))
}

// TODO: theoretically only the assignment is needed
pub fn index_to_bools(index: usize) -> ([bool; 2], Vec<bool>) {
    let mut bits = index;
    let mut event = vec![false; N_EDGES_IN_QUARTET];

    for i in (0..N_EDGES_IN_QUARTET).rev() {
        event[i] = (bits & 1) != 0;
        bits >>= 1;
    }

    let assignment = [(bits & 2) != 0, (bits & 1) != 0];

    (assignment, event)
}

fn action_for_edge(node_has_char: bool, parent_has_char: bool) -> Action {
    match (node_has_char, parent_has_char) {
        (true, true) => Action::Homolog,
        (true, false) => Action::Insertion,
        (false, true) => Action::Deletion,
        (false, false) => Action::Nothing,
    }
}

fn get_map_from_any_node<'a, AA: AncestralAlignment>(
    msa: &'a AA,
    node: &'a NodeIdx,
) -> &'a Mapping {
    match node {
        Internal(_) => msa.ancestral_map(node),
        Leaf(_) => msa.leaf_map(node),
    }
}

