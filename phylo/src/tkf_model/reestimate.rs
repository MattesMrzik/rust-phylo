use anyhow::bail;
use fixedbitset::FixedBitSet;
use log::info;
use rand::{Rng, SeedableRng};

use crate::alignment::{AncestralAlignment, Mapping};
use crate::phylo_info::PhyloInfo;
use crate::random::RandomGenerator;
use crate::tkf_model::{log_i1, Event, TKFIndelCost, TKFModel};
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::Result;

/// Size of the dynamic programming column: 2 (assignments) * 2 (deletion or not) ^ 5 (edges) = 128, see [`QuartetEdges`].
const DP_COL_SIZE: usize = 128;
const BACKTRACKING_INVALID: usize = DP_COL_SIZE + 1;
/// #{v1, v2, t2, t3, t4}, see [`QuartetEdges`].
const N_EDGES_IN_QUARTET: usize = 5;
/// Given the presence/absence of chars at `t1`, `t2`, `t3`, and `t4`, provides
/// all possible assignments of chars at `v1` and `v2` that are compatible with
/// Dollo's principle. See [`QuartetEdges`].
const DOLLO_ASSIGNMENTS: [&[EdgeAssignment]; 16] = [
    /* 0000 */ &[(false, false)],
    /* 0001 */ &[(true, true), (false, true), (false, false)],
    /* 0010 */ &[(true, true), (false, true), (false, false)],
    /* 0011 */ &[(true, true), (false, true)],
    /* 0100 */ &[(true, true), (true, false), (false, false)],
    /* 0101 */ &[(true, true)],
    /* 0110 */ &[(true, true)],
    /* 0111 */ &[(true, true)],
    /* 1000 */ &[(true, true), (true, false), (false, false)],
    /* 1001 */ &[(true, true)],
    /* 1010 */ &[(true, true)],
    /* 1011 */ &[(true, true)],
    /* 1100 */ &[(true, true), (true, false)],
    /* 1101 */ &[(true, true)],
    /* 1110 */ &[(true, true)],
    /* 1111 */ &[(true, true)],
];

/// Assignment of chars present/absent at `v1` and `v2` (see [`QuartetEdges`])
/// and at current [block](`super::TKFModel::get_blocks`).
type EdgeAssignment = (bool, bool);
// type EdgeAssignmentPossibilities = Vec<EdgeAssignment>;
pub(super) type EdgeAssignmentPossibilities = &'static [EdgeAssignment];
/// Represents whether chars are present or absent for every [block](`super::TKFModel::get_blocks`)
/// for a given [node](`crate::tree::Node`).
type NodeSeq = FixedBitSet;
/// Represent whether the previous event on each edge in the [quartet](`QuartetEdges`) was a deletion or not.
type QuartetDelOrNot = [bool; N_EDGES_IN_QUARTET];
type QuartetDelOrNotPossibilities = Vec<QuartetDelOrNot>;
type QuartetEvents = [Event; N_EDGES_IN_QUARTET];

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
/// The ancestral wild card sequences for `v1` and `v2` are re-estimated. The assignments in the
/// dynamic programming are for `(v1, v2)`.
/// The edges in this sketch are directed downwards.
#[derive(Clone)]
struct QuartetEdges {
    edges: [NodeIdx; N_EDGES_IN_QUARTET],
    /// `t1_mapping` is `None` if `v1` is the root since then `t1` does not exist
    t1_mapping: Option<Mapping>,
    t2_mapping: Mapping,
    t3_mapping: Mapping,
    t4_mapping: Mapping,
}

impl QuartetEdges {
    fn default() -> Self {
        QuartetEdges {
            edges: [NodeIdx::Leaf(0); N_EDGES_IN_QUARTET],
            t1_mapping: None,
            t2_mapping: vec![],
            t3_mapping: vec![],
            t4_mapping: vec![],
        }
    }

    /// Panics if `v2` is the root or has no sibling.
    fn new(v2: &NodeIdx, cost: &TKFIndelCost<impl TKFModel, impl AncestralAlignment>) -> Self {
        let phylo = &cost.phylo;
        let tree = &cost.phylo.tree;
        let v1 = tree.node(v2).parent.unwrap();
        let t2 = tree.sibling(v2).unwrap();
        let children_of_v2 = &tree.node(v2).children;
        let t3 = children_of_v2[0];
        let t4 = children_of_v2[1];
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
            edges: [v1, *v2, t2, t3, t4],
            t2_mapping,
            t3_mapping,
            t4_mapping,
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

    fn t1_has_char(&self, site: usize) -> bool {
        // in case the t1_mapping is None, v1 is the root and t1 does not exist
        // therefore it cannot have a character
        self.t1_mapping
            .as_ref()
            .is_some_and(|mapping| mapping[site].is_some())
    }

    fn t2_has_char(&self, site: usize) -> bool {
        self.t2_mapping[site].is_some()
    }

    fn t3_has_char(&self, site: usize) -> bool {
        self.t3_mapping[site].is_some()
    }

    fn t4_has_char(&self, site: usize) -> bool {
        self.t4_mapping[site].is_some()
    }
}

#[derive(Debug, PartialEq)]
struct BackTrackingResult {
    v1_bitset: FixedBitSet,
    v2_bitset: FixedBitSet,
    logl: f64,
}

#[cfg(doc)]
use crate::likelihood::TreeSearchCost;
/// Reestimator for indel points in the ancestral alignment at an internal neighbouring node pair.
/// Calling [`EdgeSeqsReestimator::reestimate`] will re-estimate the ancestral sequences
/// of the node that is passed as argument and its parent node under the [`TKFModel`]
/// maximum likelihood criterion. The re-estimation of indel points can remove characters or add new
/// characters ([wild cards](`crate::alphabets::AMB_CHAR`)) to the ancestral sequences.
/// Can be used as an ASR refinement method if repeatedly called on all internal nodes (i.e. edges),
/// see the [example](#example) below.
/// It is also used after an NNI move was applied during tree inference to fix the
/// ancestral sequences of the affected nodes, see
/// [`crate::tkf_model::TKFIndelCost::update_tree`].
///
/// # Example
/// ```rust
/// # fn main() -> std::result::Result<(), anyhow::Error> {
/// use phylo::phylo_info::PhyloInfoBuilder;
/// use phylo::random::DefaultGenerator;
/// use phylo::tkf_model::{EdgeSeqsReestimator, TKF92IndelCostBuilder};
/// use phylo::tree::NodeIdx::Internal;
/// // Re-estimation for ASR refinement.
/// // The alignment below includes ancestral sequences for which the indel points
/// // will be refined.
/// let msa = "data/tkf/reestimate/masa.fasta";
/// let phylo = PhyloInfoBuilder::new(msa).build_with_ancestors()?;
/// let internal_nodes = phylo
///     .tree
///     .postorder()
///     .iter()
///     .cloned()
///     .filter(|n| matches!(n, Internal(_)))
///     .collect::<Vec<_>>();
/// let lambda = 0.9;
/// let mu = 1.0;
/// let r = 0.5;
/// let mut tkf92_indel_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
///     .build()?;
/// let mut rng = DefaultGenerator::default();
/// let mut reestimator = EdgeSeqsReestimator::new(&mut tkf92_indel_cost, &mut rng);
/// for node in internal_nodes {
///     if let Internal(_) = node {
///         let new_cost = reestimator.reestimate(&node)?;
///         println!("Re-estimated sequences at node {node}, cost after re-estimation: {new_cost}",);
///     }
///     break; // stopping early for doc test, remove in real usage
/// }
/// println!(
///     "The re-estimated ancestral MSA is {}",
///     reestimator.phylo().msa
/// );
/// println!("Re-estimation complete.");
/// # Ok(()) }
/// ```
pub struct EdgeSeqsReestimator<'a, T: TKFModel, AA: AncestralAlignment, R: Rng + SeedableRng> {
    dp_table: Vec<[f64; DP_COL_SIZE]>,
    backtracking_table: Vec<[usize; DP_COL_SIZE]>,
    pub(super) cost: &'a mut TKFIndelCost<T, AA>,
    quartet_edges: QuartetEdges,
    rng: &'a mut RandomGenerator<R>,
}

impl<'a, T, AA, R> EdgeSeqsReestimator<'a, T, AA, R>
where
    T: TKFModel,
    AA: AncestralAlignment,
    R: Rng + SeedableRng,
{
    /// Creates a new [`EdgeSeqsReestimator`] for the provided [`TKFIndelCost`].
    /// The reestimator can then be repeatedly used to [re-estimate](`EdgeSeqsReestimator::reestimate`)
    /// ancestral wild card sequences for different internal node pairs.
    pub fn new(
        cost: &'a mut TKFIndelCost<T, AA>,
        rng: &'a mut RandomGenerator<R>,
    ) -> EdgeSeqsReestimator<'a, T, AA, R> {
        let num_blocks = cost.model_info.borrow().blocks.len();
        EdgeSeqsReestimator {
            dp_table: vec![[f64::NEG_INFINITY; DP_COL_SIZE]; num_blocks],
            backtracking_table: vec![[BACKTRACKING_INVALID; DP_COL_SIZE]; num_blocks],
            cost,
            quartet_edges: QuartetEdges::default(),
            rng,
        }
    }

    pub fn phylo(&self) -> &PhyloInfo<AA> {
        &self.cost.phylo
    }

    /// Reestimate ancestral wildcard sequences at `v2_idx` and its parent.
    ///
    /// This method reestimates under the maximum [TKF](`TKFModel`) likelihood criterion the ancestral wildcard sequences
    /// associated with the given internal node `v2_idx` and its parent,
    /// while keeping all other sequences and tree fixed. See also
    /// [`EdgeSeqsReestimator::reestimate_unchecked`].
    ///
    /// # Errors
    /// Reestimation is only defined for non-root internal nodes that have
    /// a sibling. Accordingly, this method will return an error if this is not the case.
    ///
    /// # Returns
    /// On success, returns the resulting log likelihood of the MASA given the tree after reestimation.
    pub fn reestimate(&mut self, v2_idx: &NodeIdx) -> Result<f64> {
        let v2_id = self.cost.phylo.tree.node(v2_idx).id.clone();
        if v2_idx == &self.cost.phylo.tree.root {
            bail!("Reestimation can't be performed for the root '{v2_id}'.");
        }
        if let Leaf(_) = v2_idx {
            bail!("Reestimation can't be performed for leaf node '{v2_id}'.");
        }
        Ok(self.reestimate_unchecked(v2_idx))
    }

    /// Reestimate ancestral wildcard sequences at `v2_idx` and its parent.
    ///
    /// This method reestimates under the maximum TKF likelihood criterion the ancestral wildcard sequences
    /// associated with the given internal node `v2_idx` and its parent,
    /// while keeping all other sequences and the tree fixed. In contrast to
    /// [`EdgeSeqsReestimator::reestimate`], this method does not perform any
    /// validity checks on `v2_idx`.
    ///
    /// # Panics
    /// Reestimation is only defined for non-root internal nodes that have
    /// a sibling. If this condition is violated, this method panics.
    ///
    /// # Returns
    /// Returns the resulting log likelihood of the MASA given the tree after reestimation.
    pub fn reestimate_unchecked(&mut self, v2_idx: &NodeIdx) -> f64 {
        if !self
            .cost
            .model_info
            .borrow()
            .valid_for_reestimation
            .is_full()
        {
            info!("Reestimation can only be performed on a cost with valid_for_reestimation internal nodes tmp values. Making them valid now.");
            self.cost.logl();
        }

        // When re-estimating ancestral wild card sequences the tmp values
        // of the model info for all nodes in the quartet, but also for all nodes along the
        // path to the root are invalidated. However, not all tmp values are needed for
        // re-estimation if the tree does not change between re-estimation calls.
        // Therefore, we recompute the tmp values for only the quartet nodes and the root,
        // making them usable for further re-estimation calls. Node flags are still set to
        // false, such that tmp values are properly recomputed when the logl is called.
        self.prepare_for_dp(v2_idx);
        self.fill_dp_table();
        let backtrack_res = self.backtrack();
        self.set_invalid();
        self.update_mappings(&backtrack_res);
        debug_assert!(self.cost.phylo.check_dollos_constraint().is_ok());
        self.make_valid_for_further_reestimate_calls();
        backtrack_res.logl
    }

    /// Resets the DP and backtracking tables. Initialises the [`QuartetEdges`]. Removes the old
    /// quartet contributions from the root aggregated values.
    ///
    /// # Panics
    /// Panics if `v2_idx` is the root or has no sibling.
    fn prepare_for_dp(&mut self, v2_idx: &NodeIdx) {
        let num_blocks = self.cost.model_info.borrow().blocks.len();
        for row in &mut self.dp_table {
            row.fill(f64::NEG_INFINITY);
        }
        for row in &mut self.backtracking_table {
            row.fill(BACKTRACKING_INVALID);
        }
        self.quartet_edges = QuartetEdges::new(v2_idx, self.cost);
        for edge in self.quartet_edges.edges() {
            self.cost.reset_cache(edge);
        }
        for block_id in 0..num_blocks {
            self.remove_old_quartet_event_factor_from_root(block_id);
            self.remove_old_quartet_eta_from_root(block_id);
        }
    }

    fn remove_old_quartet_event_factor_from_root(&self, block_id: usize) {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let mut model_info = self.cost.model_info.borrow_mut();
        for node in self.quartet_edges.edges() {
            let x = model_info.node_event_factor[(usize::from(*node), block_id)];
            model_info.subtree_event_factor[(root_id, block_id)] /= x;
        }
    }

    fn remove_old_quartet_eta_from_root(&self, block_id: usize) {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let mut model_info = self.cost.model_info.borrow_mut();
        for node in self.quartet_edges.edges() {
            let eta = model_info.node_eta[(usize::from(*node), block_id)];
            model_info.subtree_eta[(root_id, block_id)] -= eta;
        }
    }

    /// Sets the valid flags of the model info to `false` for all edges (i.e. nodes) in the
    /// [quartet](`QuartetEdges`). This ensures that the next time the logl is computed,
    /// the tmp values for these nodes are recomputed.
    fn set_invalid(&mut self) {
        let mut model_info = self.cost.model_info.borrow_mut();
        for edge in self.quartet_edges.edges() {
            model_info.valid.set(usize::from(*edge), false);
        }
    }

    fn update_mappings(&mut self, backtrack_res: &BackTrackingResult) {
        let block_lengths = &self.cost.model_info.borrow().block_lengths;
        let seq_len = self.cost.phylo.msa.len();
        let v1_mapping = mapping_from_node_seq(&backtrack_res.v1_bitset, block_lengths, seq_len);
        let v2_mapping = mapping_from_node_seq(&backtrack_res.v2_bitset, block_lengths, seq_len);
        let msa = &mut self.cost.phylo.msa;
        assert!(v1_mapping.len() == msa.len());
        assert!(v2_mapping.len() == msa.len());
        msa.update_ancestral_map(self.quartet_edges.v1(), v1_mapping);
        msa.update_ancestral_map(self.quartet_edges.v2(), v2_mapping);
    }

    /// Updates the tmp values of the model info such that there are valid for further
    /// [re-estimation](`EdgeSeqsReestimator::reestimate`) calls.
    /// Assumes that the new mappings were already updated in the msa, see
    /// [`AncestralAlignment::update_ancestral_map`].
    fn make_valid_for_further_reestimate_calls(&mut self) {
        let num_blocks = self.cost.model_info.borrow().blocks.len();
        self.cost
            .model_info
            .borrow_mut()
            .previous_event_deletion
            .clear();
        let root_id = usize::from(self.cost.phylo.tree.root);
        for block_id in 0..num_blocks {
            for edge in self.quartet_edges.edges() {
                let event = self.cost.determine_event(edge, block_id);
                let node_event_factor = self.cost.event_factor(edge, event);
                let node_eta = self.cost.eta_for_non_root(edge, event);
                // self.cost.update_previous_event(edge, event);
                let mut model_info = self.cost.model_info.borrow_mut();
                if let Some(val) = self.cost.updated_previous_is_deletion(event) {
                    model_info
                        .previous_event_deletion
                        .set(usize::from(*edge), val);
                }
                model_info.node_event_factor[(usize::from(edge), block_id)] = node_event_factor;
                model_info.node_eta[(usize::from(edge), block_id)] = node_eta;
                model_info.subtree_event_factor[(root_id, block_id)] *= node_event_factor;
                model_info.subtree_eta[(root_id, block_id)] += node_eta;
            }
        }
        let mut model_info = self.cost.model_info.borrow_mut();
        for edge in self.quartet_edges.edges() {
            model_info
                .valid_for_reestimation
                .set(usize::from(edge), true);
        }
    }

    fn fill_dp_table(&mut self) {
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            let mut found_at_least_one = false;
            for assignment in self.possible_assignments(block_id) {
                let events = self.event_for_assignment(assignment, block_id);
                let event_prob = self.integrated_root_event_prob(&events, block_id);
                let is_first_block = block_id == 0;
                for q_del_or_not in self.possible_del_or_not_for_event(&events, is_first_block) {
                    let dp_index = bools_to_index(assignment, &q_del_or_not);
                    if block_id == 0 {
                        self.dp_table[block_id][dp_index] = event_prob;
                        found_at_least_one = true;
                        // Since we are at the first position, the `del_or_not` does not have a
                        // meaning, so we can just skip all other `del_or_not` combinations.
                        continue;
                    }
                    let Some((max_prev, argmax)) =
                        self.max_over_previous(&q_del_or_not, &events, block_id)
                    else {
                        continue;
                    };

                    self.backtracking_table[block_id][dp_index] = argmax;
                    let root_id = usize::from(self.cost.phylo.tree.root);
                    // collect eta that corresponds to nodes outside of the quartet
                    let eta_for_block =
                        self.cost.model_info.borrow().subtree_eta[(root_id, block_id)];
                    self.dp_table[block_id][dp_index] = max_prev + eta_for_block + event_prob;
                    found_at_least_one = true;
                }
            }
            // TODO: perhaps instead return any valid assignment that is compatible with Dollo's
            // constraint, and return -infinity in the reassignment method. If we have -infinity here,
            // then any valid assignment is -infinity.
            // See issue #153 https://github.com/acg-team/rust-phylo/issues/153
            assert!(
                found_at_least_one,
                "No valid assignments found for block_id = {block_id}, due to -inf logl"
            );
        }
    }

    /// Finds the max over previous `assignments` and `del_or_not` that lead to the provided
    /// `del_or_not`. Since if we have [`Event::Nothing`] we have to pass through the previous `del_or_not`.
    /// May return [`None`] since we consider all possible `del_or_not` that are
    /// compatible with the `current_events` even though some of these might
    /// not be reached since the previous possible assignment might not produce
    /// these `del_or_not` scenarios. See the implementation of [`EdgeSeqsReestimator::fill_dp_table`].
    ///
    /// # Arguments
    /// Takes a mutable reference to self to be able to use the random generator to break ties.
    // TODO: More sophisticated filtering could be done, but might add more complexity and is perhaps not worth it.
    // See issue #151 https://github.com/acg-team/rust-phylo/issues/151
    fn max_over_previous(
        &mut self,
        current_del_or_not: &QuartetDelOrNot,
        current_events: &QuartetEvents,
        block_id: usize,
    ) -> Option<(f64, usize)> {
        let mut max = f64::NEG_INFINITY;
        let mut argmaxes = Vec::new();

        // TODO: instead of recalculating the possible assignments it could be reused from the previous block
        // See issue #151 https://github.com/acg-team/rust-phylo/issues/151
        for prev_assignment in self.possible_assignments(block_id - 1) {
            // TODO: here it is not checked whether the `prev_del_or_not` matches the `prev_assignment`
            // which will lead to -inf which is then skipped.
            // See issue #151 https://github.com/acg-team/rust-phylo/issues/151
            for prev_del_or_not in
                self.prev_compatible_del_or_not(current_del_or_not, current_events)
            {
                let prev_dp_index = bools_to_index(prev_assignment, &prev_del_or_not);
                let prev_gamma = self.dp_table[block_id - 1][prev_dp_index];
                if prev_gamma == f64::NEG_INFINITY {
                    continue;
                }
                let current = prev_gamma + self.quartet_eta(current_events, &prev_del_or_not);
                if current > max {
                    max = current;
                    argmaxes.clear();
                    argmaxes.push(prev_dp_index);
                }
                if current == max {
                    argmaxes.push(prev_dp_index);
                }
            }
        }
        if argmaxes.is_empty() {
            debug_assert!(max == f64::NEG_INFINITY);
            None
        } else {
            let argmax = argmaxes[self.rng.random_range(0..argmaxes.len())];
            Some((max, argmax))
        }
    }

    fn quartet_eta(&self, events: &QuartetEvents, prev_events: &[bool]) -> f64 {
        for i in 0..N_EDGES_IN_QUARTET {
            if events[i] == Event::Insertion && prev_events[i] {
                let edge = &self.quartet_edges.edges()[i];
                return self.cost.model_info.borrow().eta[usize::from(edge)];
            }
        }
        0.0
    }

    /// Based on the `current_events` and `del_or_not` finds all compatible previous `del_or_not`.
    // TODO: these are not ensured to be compatible with the previous assignments. Would it be
    // worth to filter them here?
    // See issue #151 https://github.com/acg-team/rust-phylo/issues/151
    fn prev_compatible_del_or_not(
        &self,
        current_del_or_not: &QuartetDelOrNot,
        current_events: &QuartetEvents,
    ) -> QuartetDelOrNotPossibilities {
        let mut base = [false; N_EDGES_IN_QUARTET];
        let mut positions_to_vary = Vec::with_capacity(N_EDGES_IN_QUARTET);
        for i in 0..N_EDGES_IN_QUARTET {
            match current_events[i] {
                // we have a gap col, so we have no event here but pass through the previous one
                Event::Nothing => base[i] = current_del_or_not[i],
                // we have an event here, so we can choose any previous `del_or_not`
                _ => positions_to_vary.push(i),
            };
        }
        del_or_not_combinations(&positions_to_vary, &base)
    }

    /// Returns all possible combinations of `deletion or not` for each edge in the quartet
    /// given the (current) events taken on those edges.
    fn possible_del_or_not_for_event(
        &self,
        events: &QuartetEvents,
        is_first_block: bool,
    ) -> QuartetDelOrNotPossibilities {
        let mut base = [false; N_EDGES_IN_QUARTET];
        let mut position_to_vary = Vec::with_capacity(N_EDGES_IN_QUARTET);

        // collect for each edge whether we have a choice (whether last event was deletion or not) or not
        for (i, edge) in self.quartet_edges.edges().iter().enumerate() {
            let can_be_varied = matches!(events[i], Event::Nothing) // we can't vary if there is an event
                && !is_first_block // we can't vary at the first block, since there is no event that can be passed through
                && edge != &self.cost.phylo.tree.root; // we can't vary since deletions cannot happen above the root
            if can_be_varied {
                position_to_vary.push(i);
            } else {
                // determine the fixed del or not
                base[i] = matches!(events[i], Event::Deletion);
            }
        }
        del_or_not_combinations(&position_to_vary, &base)
    }

    /// Computes the integrated event probability for the quartet given the events
    fn integrated_root_event_prob(&self, events: &QuartetEvents, block_id: usize) -> f64 {
        let root_id = usize::from(self.cost.phylo.tree.root);
        let model_info = self.cost.model_info.borrow();
        let block_len = model_info.block_lengths[block_id];
        let mut x = model_info.subtree_event_factor[(root_id, block_id)];
        x *= self.quartet_event_factor(events);
        self.cost.model.block_prob(x, block_len)
    }

    /// Computes the product of event factor values for the nodes in the quartet for the provided events
    /// which correspond to an assignment of characters at `v1` and `v2` that is currently considered
    /// in the dynamic programming.
    fn quartet_event_factor(&self, events: &QuartetEvents) -> f64 {
        let mut quartet_event_factor = 1.0;
        let model_info = self.cost.model_info.borrow();
        // Here it is assumed that the cache is already updated for all nodes in the quartet,
        // see `EdgeSeqsReestimator::prepare_for_dp`.
        for (i, node) in self.quartet_edges.edges().iter().enumerate() {
            let node_id = usize::from(*node);
            quartet_event_factor *= match events[i] {
                Event::Insertion => model_info.insertion[node_id],
                Event::Deletion => model_info.n0[node_id],
                Event::Homolog => model_info.h1[node_id],
                Event::Nothing => 1.0,
            };
        }
        quartet_event_factor
    }

    /// Based on whether there are chars at the "leaves" of the quartet finds
    /// all possible [assignment for v1, assignment for v2] combinations that
    /// follow Dollo's principle.
    fn possible_assignments(&self, block_id: usize) -> EdgeAssignmentPossibilities {
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;
        let t1_has_char = self.quartet_edges.t1_has_char(site);
        let t2_has_char = self.quartet_edges.t2_has_char(site);
        let t3_has_char = self.quartet_edges.t3_has_char(site);
        let t4_has_char = self.quartet_edges.t4_has_char(site);
        possible_assignments_of_edge(t1_has_char, t2_has_char, t3_has_char, t4_has_char)
    }

    fn event_for_assignment(&self, assignment: &EdgeAssignment, block_id: usize) -> QuartetEvents {
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;
        let mut events = [Event::Nothing; N_EDGES_IN_QUARTET];
        let v1_has_char = assignment.0;
        let v2_has_char = assignment.1;
        // edge (t1 = pa(v1) -> v1)
        events[0] = event_for_edge(v1_has_char, self.quartet_edges.t1_has_char(site));
        // edge (v1 = pa(v2) -> v2)
        events[1] = event_for_edge(v2_has_char, v1_has_char);
        // edge (v1 = pa(t2) -> t2)
        events[2] = event_for_edge(self.quartet_edges.t2_has_char(site), v1_has_char);
        // edge (v2 = pa(t3) -> t3)
        events[3] = event_for_edge(self.quartet_edges.t3_has_char(site), v2_has_char);
        // edge (v2 = pa(t4) -> t4)
        events[4] = event_for_edge(self.quartet_edges.t4_has_char(site), v2_has_char);
        events
    }

    fn backtrack(&mut self) -> BackTrackingResult {
        // prepare
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        let mut v1_bitset = FixedBitSet::with_capacity(n_blocks);
        let mut v2_bitset = FixedBitSet::with_capacity(n_blocks);

        // start from the last block
        let (last_max, last_argmax) = self.max_of_last_col();
        let (assignment, _quartet_del_or_not) = index_to_bools(last_argmax);
        v1_bitset.set(n_blocks - 1, assignment.0);
        v2_bitset.set(n_blocks - 1, assignment.1);
        let mut came_from = self.backtracking_table[n_blocks - 1][last_argmax];
        // go back the path
        for block_id in (0..(n_blocks - 1)).rev() {
            if came_from == BACKTRACKING_INVALID {
                unreachable!("Backtracking table contains invalid value at block_id = {block_id}");
            }
            let (assignment, _) = index_to_bools(came_from);
            v1_bitset.set(block_id, assignment.0);
            v2_bitset.set(block_id, assignment.1);
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

    /// Finds the maximum value in the last column of the DP table and its index.
    /// If there are multiple maxima, one is chosen at random.
    /// This is used to start the backtracking.
    ///
    /// # Arguments
    /// Takes a mutable reference to self to be able to use the random generator to break ties.
    fn max_of_last_col(&mut self) -> (f64, usize) {
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        let mut max = f64::NEG_INFINITY;
        let mut max_indices = Vec::new();
        for (index, &value) in self.dp_table[n_blocks - 1].iter().enumerate() {
            if value > max {
                max = value;
                max_indices.clear();
                max_indices.push(index);
            } else if value == max {
                max_indices.push(index);
            }
        }
        let max_index = max_indices[self.rng.random_range(0..max_indices.len())];
        (max, max_index)
    }

    /// Computes the constant part of the log likelihood that is independent of the alignment and
    /// only depends on the tree and model parameters.
    fn const_per_alignment(&self) -> f64 {
        let l = self.cost.model.lambda();
        let m = self.cost.model.mu();
        let mut const_per_alignment: f64 = (1.0 - l / m).ln();
        let nodes = self.cost.phylo.tree.preorder().iter().skip(1); // skip root
        let model_info = self.cost.model_info.borrow();
        for node in nodes {
            const_per_alignment += log_i1(l, model_info.beta[usize::from(node)]);
        }
        const_per_alignment
    }
}

#[inline]
pub(super) fn possible_assignments_of_edge(
    t1_has_char: bool,
    t2_has_char: bool,
    t3_has_char: bool,
    t4_has_char: bool,
) -> EdgeAssignmentPossibilities {
    let idx = (t1_has_char as usize) << 3
        | (t2_has_char as usize) << 2
        | (t3_has_char as usize) << 1
        | (t4_has_char as usize);

    DOLLO_ASSIGNMENTS[idx]
}

#[inline]
pub(super) fn mapping_from_node_seq(
    node_seq: &NodeSeq,
    block_lens: &[usize],
    seq_len: usize,
) -> Mapping {
    debug_assert!(
        block_lens.iter().sum::<usize>() == seq_len,
        "Block lengths do not sum up to the sequence length."
    );
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

/// Generates all possible combinations of deletion-or-not for the provided choices,
/// while keeping the no_choices fixed.
///
/// # Example
/// ```rust
/// let base = [true, false, false, true, false];
/// let positions_to_vary = vec![1, 3];
/// ```
/// Keeps all the boolean values in `no_choices` fixed except for indices 1 and 3,
/// which are varied over all possible combinations, i.e., the result will be:
/// ```rust
/// let result = [[true, false, false, false, false],
///               [true, false, false, true, false],
///               [true, true, false, false, false],
///               [true, true, false, true, false]];
/// ```
fn del_or_not_combinations(
    positions_to_vary: &[usize],
    base: &QuartetDelOrNot,
) -> QuartetDelOrNotPossibilities {
    let num_combinations = 1 << positions_to_vary.len();
    let mut all_possibilities = Vec::with_capacity(num_combinations);
    for possibility_idx in 0..num_combinations {
        let mut possibility = *base;
        for (j, &edge_index) in positions_to_vary.iter().enumerate() {
            let bit = (possibility_idx >> j) & 1;
            possibility[edge_index] = bit != 0;
        }
        all_possibilities.push(possibility);
    }
    all_possibilities
}

/// Converts the provided assignment and quartet del_or_not combination
/// into a unique index for the DP table. The DP algorithm calculates probabilities
/// for such an assignment and quartet del_or_not combination. To store these     
/// results in a flat array, we need to convert the combination of booleans
/// into a unique index.
/// Is the inverse of [`index_to_bools`].
fn bools_to_index(assignment: &EdgeAssignment, q_del_or_not: &QuartetDelOrNot) -> usize {
    // Iterate over all booleans (i.e., assignment and del_or_not concatenated):
    // first shift the index to the left by 1 (multiply by 2)
    // then add 1 if the boolean is true, or 0 if it is false
    [assignment.0, assignment.1]
        .iter()
        .chain(q_del_or_not.iter())
        .fold(0, |index, &b| (index << 1) | (b as usize))
}

/// Converts the provided index as used by the DP table into the corresponding `assignment` and
/// quartet `del_or_not` combination.
/// Is the inverse of [`bools_to_index`].
/// During backtracking we only need to know the assignment.
fn index_to_bools(index: usize) -> (EdgeAssignment, QuartetDelOrNot) {
    let mut bits = index;
    let mut q_del_or_not = [false; N_EDGES_IN_QUARTET];

    // extract del_or_not booleans first
    for i in (0..N_EDGES_IN_QUARTET).rev() {
        q_del_or_not[i] = (bits & 1) != 0;
        bits >>= 1;
    }
    // then extract assignment booleans
    let assignment = ((bits & 2) != 0, (bits & 1) != 0);

    (assignment, q_del_or_not)
}

#[inline]
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
    use std::path::Path;

    use rstest::rstest;

    use crate::alignment::{Alignment, Sequences, MASA};
    use crate::alphabets::Alphabet;
    use crate::phylo_info::PhyloInfoBuilder;
    use crate::random::{DefaultGenerator, FakeGenerator, FakeRng};
    use crate::tkf_model::{tests::setup_test_phylo, EdgeSeqsReestimator, TKF92IndelCostBuilder};
    use crate::{record_wo_desc as record, tree};

    use super::*;

    #[test]
    fn tkf_index_to_bools_and_back() {
        for i in 0..DP_COL_SIZE {
            let (assignment, del_or_not) = index_to_bools(i);
            let j = bools_to_index(&assignment, &del_or_not);
            assert_eq!(i, j);
        }
    }

    #[test]
    fn tkf_reestimate_possibilities_for_choices_no_choice() {
        let no_choices = [true, false, false, true, false];
        let choices = vec![];
        let possibilities = del_or_not_combinations(&choices, &no_choices);
        let expected = vec![[true, false, false, true, false]];
        assert_eq!(possibilities, expected);
    }

    #[test]
    fn tkf_reestimate_possibilities_for_choices_one() {
        let no_choices = [true, false, false, true, false];
        let choices = vec![0];
        let mut possibilities = del_or_not_combinations(&choices, &no_choices);
        let mut expected = vec![
            [true, false, false, true, false],
            [false, false, false, true, false],
        ];
        possibilities.sort();
        expected.sort();
        assert_eq!(possibilities, expected);
    }

    #[test]
    fn tkf_reestimate_possibilities_for_choices() {
        let no_choices = [true, false, false, true, false];
        let choices = vec![1, 3];
        let mut possibilities = del_or_not_combinations(&choices, &no_choices);
        let mut expected = vec![
            [true, false, false, false, false],
            [true, false, false, true, false],
            [true, true, false, false, false],
            [true, true, false, true, false],
        ];
        possibilities.sort();
        expected.sort();
        assert_eq!(possibilities, expected);
    }

    #[test]
    fn tkf_reestimate_possibilities_for_choices_all() {
        let no_choices = [true, false, false, true, false];
        let choices = vec![0, 1, 2, 3, 4];
        let possibilities = del_or_not_combinations(&choices, &no_choices);
        let expected = (0..32)
            .map(|i| {
                [
                    (i & 0b00001) != 0,
                    (i & 0b00010) != 0,
                    (i & 0b00100) != 0,
                    (i & 0b01000) != 0,
                    (i & 0b10000) != 0,
                ]
            })
            .collect::<Vec<[bool; 5]>>();

        assert_eq!(possibilities, expected);
    }

    #[rstest]
    #[case(0, 0, 0, 0, vec![(0, 0)])]
    #[case(0, 0, 0, 1, vec![(0, 0), (0, 1), (1, 1)])]
    #[case(0, 0, 1, 0, vec![(0, 0), (0, 1), (1, 1)])]
    #[case(0, 0, 1, 1, vec![(0, 1), (1, 1)])]
    #[case(0, 1, 0, 0, vec![(0, 0), (1, 0), (1, 1)])]
    #[case(0, 1, 0, 1, vec![(1, 1)])]
    #[case(0, 1, 1, 0, vec![(1, 1)])]
    #[case(0, 1, 1, 1, vec![(1, 1)])]
    #[case(1, 0, 0, 0, vec![(0, 0), (1, 0), (1, 1)])]
    #[case(1, 0, 0, 1, vec![(1, 1)])]
    #[case(1, 0, 1, 0, vec![(1, 1)])]
    #[case(1, 0, 1, 1, vec![(1, 1)])]
    #[case(1, 1, 0, 0, vec![(1, 1), (1, 0)])]
    #[case(1, 1, 0, 1, vec![(1, 1)])]
    #[case(1, 1, 1, 0, vec![(1, 1)])]
    #[case(1, 1, 1, 1, vec![(1, 1)])]
    fn tkf_possible_assignments_of_edge(
        #[case] t1_has_char: u8,
        #[case] t2_has_char: u8,
        #[case] t3_has_char: u8,
        #[case] t4_has_char: u8,
        #[case] expected: Vec<(u8, u8)>,
    ) {
        // convert expected to bools and sort
        let mut expected = expected
            .into_iter()
            .map(|(a, b)| (a != 0, b != 0))
            .collect::<Vec<(bool, bool)>>();
        expected.sort();

        let mut result = possible_assignments_of_edge(
            t1_has_char != 0,
            t2_has_char != 0,
            t3_has_char != 0,
            t4_has_char != 0,
        )
        .to_vec();
        result.sort();

        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(true, true, Event::Homolog)]
    #[case(true, false, Event::Insertion)]
    #[case(false, true, Event::Deletion)]
    #[case(false, false, Event::Nothing)]
    fn tkf_event_for_edge(
        #[case] node_has_char: bool,
        #[case] parent_has_char: bool,
        #[case] expected: Event,
    ) {
        let result = event_for_edge(node_has_char, parent_has_char);
        assert_eq!(result, expected);
    }

    #[cfg(test)]
    fn confirm_quartet_edges_for_setup_test_phylo<T, AA, R>(
        reestimator: &EdgeSeqsReestimator<T, AA, R>,
    ) where
        T: TKFModel,
        AA: AncestralAlignment,
        R: Rng + SeedableRng,
    {
        assert_eq!(
            reestimator.quartet_edges.edges()[0],
            reestimator.phylo().tree.by_id("R5").idx
        ); // v1
        assert_eq!(
            reestimator.quartet_edges.edges()[1],
            reestimator.phylo().tree.by_id("I3").idx
        ); // v2
        assert_eq!(
            reestimator.quartet_edges.edges()[2],
            reestimator.phylo().tree.by_id("C4").idx
        ); // t2
        assert_eq!(
            reestimator.quartet_edges.edges()[3],
            reestimator.phylo().tree.by_id("A1").idx
        ); // t3
        assert_eq!(
            reestimator.quartet_edges.edges()[4],
            reestimator.phylo().tree.by_id("B2").idx
        ); // t4
    }

    #[rstest]
    // for every block test one of the four possible assignments. In these tests we don't care about Dollo's principle.
    #[case::first_block(0, (false, false), [Event::Nothing, Event::Nothing, Event::Insertion, Event::Nothing, Event::Nothing])]
    #[case::second_block(1, (true, false), [Event::Insertion, Event::Deletion, Event::Homolog, Event::Insertion, Event::Nothing])]
    #[case::thrid_block(2, (true, true), [Event::Insertion, Event::Homolog, Event::Deletion, Event::Homolog, Event::Deletion])]
    #[case::forth_block(3, (false, true), [Event::Nothing, Event::Insertion, Event::Nothing, Event::Deletion, Event::Homolog])]
    fn tkf_event_for_assignment(
        #[case] block_id: usize,
        #[case] assignment: EdgeAssignment,
        #[case] expected_events: QuartetEvents,
    ) {
        let phylo = setup_test_phylo(Alphabet::dna());
        let mut cost = TKF92IndelCostBuilder::new(0.4, 0.5, 0.8, phylo)
            .build()
            .unwrap();
        let rng = &mut FakeGenerator::default();
        let v2_idx = cost.phylo.tree.by_id("I3").idx;
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        reestimator.prepare_for_dp(&v2_idx);
        confirm_quartet_edges_for_setup_test_phylo(&reestimator);
        let events = reestimator.event_for_assignment(&assignment, block_id);
        assert_eq!(events, expected_events);
    }

    #[test]
    fn tkf_get_map_from_any_node() {
        let phylo = setup_test_phylo(Alphabet::dna());
        let msa = &phylo.msa;
        let leaf_node = phylo.tree.by_id("A1").idx;
        let internal_node = phylo.tree.by_id("I3").idx;

        let leaf_map = get_map_from_any_node(msa, &leaf_node);
        let expected_leaf_map = msa.leaf_map(&leaf_node);
        assert_eq!(leaf_map, expected_leaf_map);

        let internal_map = get_map_from_any_node(msa, &internal_node);
        let expected_internal_map = msa.ancestral_map(&internal_node);
        assert_eq!(internal_map, expected_internal_map);
    }

    #[test]
    fn tkf_mapping_from_node_seq() {
        let mut node_seq = FixedBitSet::with_capacity(5);
        node_seq.insert(0);
        node_seq.insert(2);
        node_seq.insert(4);
        let block_lens = [2, 3, 1, 4, 1];
        let seq_len: usize = block_lens.iter().sum();
        let expected_mapping = [
            Some(0),
            Some(1), // first block finished
            None,
            None,
            None,    // second block finished
            Some(2), // third block finished
            None,
            None,
            None,
            None,    // fourth block finished
            Some(3), // fifth block finished
        ];
        let mapping = mapping_from_node_seq(&node_seq, &block_lens, seq_len);
        assert_eq!(mapping, expected_mapping);
    }

    #[test]
    fn tkf_backtrack() {
        let phylo = setup_test_phylo(Alphabet::dna());
        // the parameters here do not matter for the backtracking test
        let mut cost = TKF92IndelCostBuilder::new(0.4, 0.5, 0.8, phylo)
            .build()
            .unwrap();
        // FakeRng such that we can test tie-breaking (max value in last column) in backtracking
        let rng = &mut FakeGenerator::from_rng(FakeRng::from_f64_values(vec![0.1, 0.2]));

        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        // backtracking_table dimensions num_blocks = 4, DP_COL_SIZE = 128
        reestimator.backtracking_table[1][32] = 13;
        // a path that splits in the middle
        reestimator.backtracking_table[2][110] = 32;
        reestimator.backtracking_table[2][20] = 32;
        reestimator.backtracking_table[3][85] = 110;
        reestimator.backtracking_table[3][42] = 20;
        // to select the argmax at the end of backtracking
        let dp_last_col_logl = -5.0;
        reestimator.dp_table[3][85] = dp_last_col_logl;
        reestimator.dp_table[3][42] = dp_last_col_logl;

        // the 85 is selected here
        let backtrack_res = reestimator.backtrack();
        let indices = [13, 32, 110, 85];
        let mut expected_v1_bitset = FixedBitSet::with_capacity(4);
        let mut expected_v2_bitset = FixedBitSet::with_capacity(4);
        for (i, &idx) in indices.iter().enumerate() {
            let (first, second) = index_to_bools(idx).0;
            if first {
                expected_v1_bitset.insert(i);
            }
            if second {
                expected_v2_bitset.insert(i);
            }
        }
        assert_eq!(backtrack_res.v1_bitset, expected_v1_bitset);
        assert_eq!(backtrack_res.v2_bitset, expected_v2_bitset);
        assert_eq!(
            backtrack_res.logl,
            reestimator.const_per_alignment() + dp_last_col_logl
        );

        // the 42 is selected here
        let backtrack_res = reestimator.backtrack();
        let indices = [13, 32, 20, 42];
        let mut expected_v1_bitset = FixedBitSet::with_capacity(4);
        let mut expected_v2_bitset = FixedBitSet::with_capacity(4);
        for (i, &idx) in indices.iter().enumerate() {
            let (first, second) = index_to_bools(idx).0;
            if first {
                expected_v1_bitset.insert(i);
            }
            if second {
                expected_v2_bitset.insert(i);
            }
        }
        assert_eq!(backtrack_res.v1_bitset, expected_v1_bitset);
        assert_eq!(backtrack_res.v2_bitset, expected_v2_bitset);
        assert_eq!(
            backtrack_res.logl,
            reestimator.const_per_alignment() + dp_last_col_logl
        );
    }

    #[test]
    fn tkf_const_per_alignment() {
        let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
        let msa = MASA::from_aligned_with_ancestral(
            Sequences::new(vec![
                record!("A1", b""),
                record!("B2", b""),
                record!("I3", b""),
                record!("C4", b""),
                record!("R5", b""),
            ]),
            &tree,
        )
        .unwrap();
        let phylo = PhyloInfo { msa, tree };
        let mut cost = TKF92IndelCostBuilder::new(0.4, 0.5, 0.8, phylo)
            .build()
            .unwrap();
        let logl = cost.logl(); // must be called to initialize the model_info, which is
                                // needed for const_per_alignment().
        let rng = &mut DefaultGenerator::default();
        let reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        assert_eq!(reestimator.const_per_alignment(), logl);
    }

    #[test]
    fn tkf_remove_and_add_back_quartet() {
        let phylo = setup_test_phylo(Alphabet::dna());
        let mut cost = TKF92IndelCostBuilder::new(0.4, 0.5, 0.8, phylo)
            .build()
            .unwrap();

        let rng = &mut DefaultGenerator::default();
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        let original_logl = reestimator.cost.logl();
        assert_eq!(original_logl, reestimator.cost.logl_from_root_model_info());
        let dummy_v2_idx = reestimator.cost.phylo.tree.by_id("I3").idx;
        reestimator.prepare_for_dp(&dummy_v2_idx);
        assert_ne!(reestimator.cost.logl_from_root_model_info(), original_logl);
        reestimator.make_valid_for_further_reestimate_calls();
        assert_eq!(reestimator.cost.logl_from_root_model_info(), original_logl);
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn tkf_remove_and_add_back_quartet_large_tree() {
        let dir = Path::new("data/tkf/reestimate/");
        let msa = dir.join("masa.fasta");
        let tree = dir.join("tree.newick");
        let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
            .build_with_ancestors()
            .unwrap();

        let mut cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, phylo)
            .build()
            .unwrap();
        let rng = &mut DefaultGenerator::default();
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        let original_logl = reestimator.cost.logl();
        assert_eq!(
            original_logl,
            reestimator.cost.logl_from_root_model_info(),
            "before removing quartet"
        );
        let v2_idx = reestimator.cost.phylo.tree.by_id("N312").idx;
        reestimator.prepare_for_dp(&v2_idx);
        assert_ne!(reestimator.cost.logl_from_root_model_info(), original_logl);
        reestimator.make_valid_for_further_reestimate_calls();
        assert_eq!(
            reestimator.cost.logl_from_root_model_info(),
            original_logl,
            "after adding back quartet"
        );
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn tkf_remove_and_add_back_quartet_large_tree_child_of_root() {
        let dir = Path::new("data/tkf/reestimate/");
        let msa = dir.join("masa.fasta");
        let tree = dir.join("tree.newick");
        let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
            .build_with_ancestors()
            .unwrap();

        let mut cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, phylo)
            .build()
            .unwrap();
        let rng = &mut DefaultGenerator::default();
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        let original_logl = reestimator.cost.logl();
        assert_eq!(
            original_logl,
            reestimator.cost.logl_from_root_model_info(),
            "before removing quartet"
        );
        let v2_idx = reestimator.cost.phylo.tree.by_id("N380").idx;
        reestimator.prepare_for_dp(&v2_idx);
        assert_ne!(reestimator.cost.logl_from_root_model_info(), original_logl);
        reestimator.make_valid_for_further_reestimate_calls();
        assert_eq!(
            reestimator.cost.logl_from_root_model_info(),
            original_logl,
            "after adding back quartet"
        );
    }
}
