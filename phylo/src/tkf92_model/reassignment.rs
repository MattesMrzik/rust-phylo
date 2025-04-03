use itertools::Itertools;
use nalgebra::DMatrix;
use std::process::exit;
use std::{collections::HashMap, fmt::Display};

use crate::alignment::LeafMapping;
use crate::substitution_models::{dna_models::JC69, QMatrix, SubstMatrix};
use crate::tkf92_model::TKF92Cost;
use crate::tree::NodeIdx;

const DP_ASSIGNMENT_AND_EVENTS_SIZE: usize = 128;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy)]
enum DirtyTreeEdge {
    // the names are the child of the edge
    V1,
    V2,
    // does not contain T1 since the corresponding edge is not in the dirty tree
    T2,
    T3,
    T4,
}
#[derive(PartialEq, Debug)]
enum Action {
    Insertion,
    Deletion,
    Homolog,
    Nothing,
}

pub struct ReassignEdge<Q: QMatrix + Display> {
    // TODO: maybe also put this in a refcell such that, pass non mutable
    //       self in the fill_dp
    dp_table: Vec<[f64; DP_ASSIGNMENT_AND_EVENTS_SIZE]>,
    backtracking_table: Vec<[usize; DP_ASSIGNMENT_AND_EVENTS_SIZE]>, // pointers to prev gamma argmax,
    pub cost: TKF92Cost<Q>,
}

impl<Q: QMatrix + Display> ReassignEdge<Q> {
    pub fn new(cost: TKF92Cost<Q>) -> ReassignEdge<Q> {
        let number_of_blocks = cost.model_info.borrow().blocks.len();
        ReassignEdge {
            dp_table: vec![[f64::NEG_INFINITY; DP_ASSIGNMENT_AND_EVENTS_SIZE]; number_of_blocks],
            backtracking_table: vec![[0; DP_ASSIGNMENT_AND_EVENTS_SIZE]; number_of_blocks],
            cost,
        }
    }

    // TODO: to get the actual logl we still need to add the immortal links, ...
    //       see tkf_model::logl()
    // TODO: i dont really need the x from the root, removing the old dirty tree and
    //       adding the new one, since the rest of the tree stays the same,
    pub fn fill_dp(&mut self, v2_idx: &NodeIdx) {
        println!("filling dp");
        if !self.cost.model_info.borrow().valid {
            self.cost.reset_all_nodes();
        }
        // Question: here i thought that without the mut self, and therefore self.cost would not change
        // but self.cost.model_info.borrow_mut() can change self
        let root = &self.cost.phylo.tree.root;
        if v2_idx == root {
            return;
        }
        let n_blocks = self.cost.model_info.borrow().blocks.len();
        let (models, which_model, which_node) =
            self.get_felsenstein_substitution_models_up_to_insertion_point(v2_idx);

        self.remove_edges_from_factor_n_and_x_and_felsenstein(
            v2_idx,
            &models,
            &which_model,
            &which_node,
        );

        // TODO: currently i am using the order of this hashmap to convert between edges and their index in the
        // events bool vector, maybe it is saver and more explicit to write some conversion somewhere
        let dirty_edges = self.get_dirty_edges(v2_idx);

        let v1_idx = &self.cost.phylo.tree.node(v2_idx).parent.unwrap();
        for block_id in 0..n_blocks {
            let block_len = self.cost.model_info.borrow().block_lens[block_id];
            let (t1, t2, t3, t4) = self.are_chars_at_leafs(v2_idx, block_id);
            for assignment in Self::get_allowed_assignments(t1, t2, t3, t4) {
                let block = self.cost.model_info.borrow().blocks[block_id];
                println!(
                    "\nblock_id {} = ({},{}), assignment {:?} follows dollo",
                    block_id,
                    block - block_len,
                    block,
                    assignment
                );
                let actions = self.get_actions(block_id, &dirty_edges, &assignment);
                // indel x
                let dirty_tree_x = self.get_x(&dirty_edges, &actions);
                let x: f64 = self.cost.model_info.borrow().aggregated_x
                    [(usize::from(root), block_id)]
                    * dirty_tree_x;

                println!(
                    "dirty_tree_x = {}, root_x = {}, product of these = {}",
                    dirty_tree_x,
                    self.cost.model_info.borrow().aggregated_x[(usize::from(root), block_id)],
                    x
                );
                let integrated_x_prop = x.ln() + (block_len as f64 - 1.0) * (1.0 + x).ln();
                // println!("integrated_x_prob {}", integrated_x_prop);

                // substitution process
                self.cost.set_felsenstein_for_internal(v2_idx, block_id);
                self.cost.set_felsenstein_for_internal(v1_idx, block_id);

                // if block_id == 0 {
                //     println!(
                //         "v2 felsenstein {}",
                //         self.cost.model_info.borrow().felsenstein[usize::from(v2_idx)].row(0)
                //     );
                //     println!(
                //         "v1 felsenstein {}",
                //         self.cost.model_info.borrow().felsenstein[usize::from(v1_idx)].row(0)
                //     );
                // }

                // if insertion happened on the dirty tree, then the prob is captured here
                let felsenstein_prob = self.get_felsenstein_prob_for_insertion_on_dirty_tree(
                    &dirty_edges,
                    &actions,
                    block_id,
                );
                println!("felsenstein_prob {}", felsenstein_prob);
                // if insertion happened further up the tree, then the prob is captured here
                let felsenstein_prob_for_up_the_tree = self
                    .get_felsenstein_prob_for_insertion_higher_than_on_dirty_tree(
                        v1_idx,
                        block_id,
                        &models,
                        &which_model,
                        &which_node,
                    );
                println!(
                    "felsenstein_prob_for_up_the_tree {}",
                    felsenstein_prob_for_up_the_tree
                );
                let mut felsenstein_prob_in_different_subtree = 0.0;
                if felsenstein_prob + felsenstein_prob_for_up_the_tree == 0.0 {
                    felsenstein_prob_in_different_subtree =
                        self.cost.model_info.borrow().felsenstein_prob
                            [(usize::from(root), block_id)];
                }
                println!(
                    "felsenstein in different subtree {}",
                    felsenstein_prob_in_different_subtree
                );

                let mut was_in_loop = false;
                for events in
                    self.get_possible_events_for_assignment(&actions, &dirty_edges, block_id)
                {
                    println!("current events in fill_dp() = {:?}", events);
                    was_in_loop = true;
                    let dp_index = Self::bools_to_index(&assignment, &events);
                    // TODO: in case v1 is the root, then i dont need to consider the event on that edge since
                    //       there cant have been a deletion, does my implementation care?
                    //       Then i think its not worth to implement some special case got speed up
                    if block_id == 0 {
                        // initial case
                        // no previous gamma and no factor_n since there is no last event
                        self.dp_table[block_id][dp_index] = integrated_x_prop
                            + felsenstein_prob
                            + felsenstein_prob_for_up_the_tree
                            + felsenstein_prob_in_different_subtree;
                        continue;
                    }

                    // max gamma with corresponding factor ns on the dirty tree
                    let (max, argmax) = self.get_max(&events, &actions, &dirty_edges, block_id);
                    self.backtracking_table[block_id][dp_index] = argmax;

                    // if there was a factor n due to an insertion somewhere but the dirty tree
                    let root_factor_n =
                        self.cost.model_info.borrow().factor_ns[(usize::from(root), block_id)];

                    // if max == f64::NEG_INFINITY {
                    //     println!("got neg_inf");
                    // }

                    // println!("max = {}, root_factor_n = {}", max, root_factor_n);

                    self.dp_table[block_id][dp_index] = integrated_x_prop
                        + felsenstein_prob
                        + felsenstein_prob_for_up_the_tree
                        + felsenstein_prob_in_different_subtree
                        + max
                        + root_factor_n;
                }
                if !was_in_loop {
                    println!("wasnt in loop");
                }
            }
            println!("\n");
        }
        self.print_dp_table();
        self.print_backtracking_table();
    }

    pub fn backtracking(&self) -> Vec<[bool; 2]> {
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
        if max_index.is_none() {
            println!("no max in dp table last col found");
        }
        let last_argmax = max_index.unwrap();

        // println!("last argmax {:?}", Self::index_to_bools(last_argmax));
        let (assignment, _) = Self::index_to_bools(last_argmax);
        let mut came_from = self.backtracking_table[n_blocks - 1][last_argmax];
        // println!("came from {:?}", Self::index_to_bools(came_from));
        assignments[n_blocks - 1] = assignment;
        // go back the path
        for block_id in (0..(n_blocks - 1)).rev() {
            // println!("backtracing block {}", block_id);
            let (assignment, _) = Self::index_to_bools(came_from);
            assignments[block_id] = assignment;
            if block_id > 0 {
                came_from = self.backtracking_table[block_id][came_from];
            }
        }
        assignments
    }

    pub fn print_backtracking_assignment(&self, v2_idx: &NodeIdx) {
        let v1_idx = self.cost.phylo.tree.node(&v2_idx).parent.unwrap();
        for (i, assignment) in self.backtracking().iter().enumerate() {
            let original = vec![
                self.cost.phylo.msa.get_node_map()[&v1_idx]
                    [self.cost.model_info.borrow().blocks[i] - 1]
                    .is_some(),
                self.cost.phylo.msa.get_node_map()[&v2_idx]
                    [self.cost.model_info.borrow().blocks[i] - 1]
                    .is_some(),
            ];
            let are_equal = assignment[0] == original[0] && assignment[1] == original[1];
            println!(
                "backtracking = {:?},\t original = {:?}{}",
                assignment,
                original,
                if are_equal { "" } else { " are not the same" }
            );
        }
    }

    pub fn get_mapping_from_vec(
        &self,
        v2_idx: &NodeIdx,
        assignments: &Vec<[bool; 2]>,
    ) -> LeafMapping {
        let mut mappings = vec![vec![]; 2];
        let mut counts = [0; 2];
        let v1_idx = &self.cost.phylo.tree.node(v2_idx).parent.unwrap();
        for i in 0..self.cost.model_info.borrow().blocks.len() {
            // update mapping for id in the ancestral alignment
            // or a method that copies and sets the new mappings for also the id and mappings
            // that i pass
            for node in 0..2 {
                if assignments[i][node] {
                    for _ in 0..self.cost.model_info.borrow().block_lens[i] {
                        mappings[node].push(Some(counts[node]));
                        counts[node] += 1;
                    }
                } else {
                    for _ in 0..self.cost.model_info.borrow().block_lens[i] {
                        mappings[node].push(None);
                    }
                }
            }
        }
        LeafMapping::from([
            (v1_idx.clone(), mappings.remove(0)),
            (v2_idx.clone(), mappings.remove(0)),
        ])
    }

    pub fn get_mapping_from_backtracking(&self, v2_idx: &NodeIdx) -> LeafMapping {
        // TODO: or do i also want to update the felsenstein / felsenstein prob / indel x
        // TODO: only update the node_mapping is sufficient for likelihood calculation
        //       but leaves the seqs in a broken state
        let backtrack = self.backtracking();
        self.get_mapping_from_vec(v2_idx, &backtrack)
    }

    fn get_max(
        &self,
        events: &Vec<bool>,
        actions: &HashMap<DirtyTreeEdge, Action>,
        dirty_edges: &HashMap<DirtyTreeEdge, (NodeIdx, usize)>,
        block_id: usize,
    ) -> (f64, usize) {
        // TODO: if some prevmax are tied, then make a random choice
        let mut max: Option<f64> = None;
        let mut argmax: Option<usize> = None;
        let l = self.cost.model.lambda();
        let m = self.cost.model.mu();
        let mut factor_n_from_max = -1.1111;
        let (t1, t2, t3, t4) = self.are_chars_at_leafs(
            &dirty_edges.get(&DirtyTreeEdge::V2).unwrap().0,
            block_id - 1,
        );
        for previous_assignment in Self::get_allowed_assignments(t1, t2, t3, t4) {
            // println!("prev assignment {:?}", previous_assignment);
            for previous_event in self.get_previous_allowed_events(events, actions, dirty_edges) {
                // TODO: checking if the previous assignment matches the previous events.
                //       for now i think this can be skipped since if they dont match then this previous gamma is -inf.
                //       but it might be slightly faster to get only the events that match both the current events
                //       as well as the previous assignment
                let dp_index = Self::bools_to_index(&previous_assignment, &previous_event);
                let last_gamma = self.dp_table[block_id - 1][dp_index];
                if last_gamma == f64::NEG_INFINITY {
                    // TODO: this seems to happen quite of, check if this can be avoided
                    // println!(
                    //     "last_gamma is neg infinity for prev events {:?}, continuing",
                    //     previous_event
                    // );
                    continue;
                }
                println!(
                    "using last entry {:?}{:?} = {}",
                    previous_assignment, previous_event, last_gamma
                );
                // depending on the previous events and the actions now
                // return the prob of gamma and factor Ns
                let mut factor_n = 0.0;
                for (typ, (edge, edge_id)) in dirty_edges {
                    if edge == &self.cost.phylo.tree.root {
                        continue;
                    }
                    if actions.get(typ).unwrap() == &Action::Insertion && previous_event[*edge_id] {
                        let time = self.cost.phylo.tree.node(edge).blen;
                        let b = TKF92Cost::<JC69>::b(l, m, time);
                        // println!(
                        //     "factor n added: edge = {}, time = {}, beta = {}",
                        //     self.cost.phylo.tree.node(edge).id,
                        //     time,
                        //     b
                        // );
                        // TODO: can i already update the saved values for these terms and reuse them
                        //       at upcoming sites?
                        factor_n += TKF92Cost::<JC69>::log_n1(l, m, b, time);
                        // println!("log_n1 = {}", TKF92Cost::<JC69>::log_n1(l, m, b, time));
                        factor_n -= (l * b).ln();
                        // println!("lb = {}", (l * b).ln());
                        factor_n -= TKF92Cost::<JC69>::n0(m, b).ln();
                        // println!("n0 = {}", TKF92Cost::<JC69>::n0(m, b).ln());
                    }
                }
                // println!(
                //     "prev assignment = {:?}, prev event = {:?}, last gamma = {}, factor_n = {}",
                //     previous_assignment, previous_event, last_gamma, factor_n
                // );
                let current = last_gamma + factor_n;
                if let Some(ref mut maximum) = max {
                    if current > *maximum {
                        // println!("updating max to {}", current);
                        *maximum = current;
                        argmax = Some(dp_index);
                        factor_n_from_max = factor_n;
                    }
                } else {
                    // println!("init max to {}", current);
                    max = Some(current);
                    argmax = Some(dp_index);
                    factor_n_from_max = factor_n;
                }
            }
        }
        if max.is_none() {
            (f64::NEG_INFINITY, 0)
        } else {
            if max.unwrap().is_nan() {
                println!("max is NaN    : this should never happen");
                self.print_dp_table();
                (0.0, 0);
                exit(1);
            }
            println!("factor_n from max = {}", factor_n_from_max);
            (max.unwrap(), argmax.unwrap())
        }
    }

    fn get_previous_allowed_events(
        &self,
        events: &Vec<bool>,
        actions: &HashMap<DirtyTreeEdge, Action>,
        dirty_edges: &HashMap<DirtyTreeEdge, (NodeIdx, usize)>,
    ) -> Vec<Vec<bool>> {
        let mut choices: Vec<Vec<bool>> = vec![vec![]; 5];
        for (typ, (_, edge_id)) in dirty_edges {
            match actions.get(typ).unwrap() {
                // we have a gap col, so we have no action here but pass the previous one
                Action::Nothing => choices[*edge_id] = vec![events[*edge_id]],
                // we have an action here, so we can choose any previous action
                _ => choices[*edge_id] = vec![false, true],
            };
        }
        // println!("previous allowed events = {:?}", choices);
        choices.into_iter().multi_cartesian_product().collect()
    }

    // that corresponds to C_a^i
    fn get_possible_events_for_assignment(
        &self,
        actions: &HashMap<DirtyTreeEdge, Action>,
        dirty_edges: &HashMap<DirtyTreeEdge, (NodeIdx, usize)>,
        block_id: usize,
    ) -> Vec<Vec<bool>> {
        // println!("in get_possible_events_for_assignment");
        let mut choices: Vec<Vec<bool>> = vec![vec![]; 5];
        for (typ, (edge, edge_id)) in dirty_edges {
            match actions.get(typ).unwrap() {
                Action::Deletion => choices[*edge_id] = vec![true],
                Action::Nothing => {
                    if edge == &self.cost.phylo.tree.root {
                        choices[*edge_id] = vec![false];
                    } else {
                        if block_id == 0 {
                            choices[*edge_id] = vec![false];
                        } else {
                            choices[*edge_id] = vec![false, true];
                        }
                    }
                }
                _ => choices[*edge_id] = vec![false],
            };
        }

        println!("choises for events = {:?}", choices);

        choices.into_iter().multi_cartesian_product().collect()
    }

    /// after the dirty tree felsenstein got removed from the insertion point
    /// felsenstein we now multiply back the felsenstein of v_1 * all the models
    /// from v_1 to the insertion point
    fn get_felsenstein_prob_for_insertion_higher_than_on_dirty_tree(
        &self,
        v1_idx: &NodeIdx,
        block_id: usize,
        models: &HashMap<usize, SubstMatrix>,
        which_model: &Vec<usize>,
        which_node: &Vec<NodeIdx>,
    ) -> f64 {
        // TODO: I could always take the model up until the root.
        //       that might be a bit slower since right now i dont have to move up
        //       to the root if all insertions happened before, but if we are doing
        //       no rerooting, then the most insertions are at the root anyways.
        //       but this might make the code easier
        // only if v1 - t1 is homolog
        // if actions.get(&DirtyTreeEdge::V1).unwrap() != &Action::Homolog {
        //     return 0.0;
        // }
        // If insertion happened anywhere but the current dirty tree.
        // Then then it either happened below, up, or in a different subtree
        // if it happened below, then

        // TODO: no, i always have to do that if i did not use felsenstein to prob
        //       somewhere below in the tree already
        let block = self.cost.model_info.borrow().blocks[block_id];
        let block_len = self.cost.model_info.borrow().block_lens[block_id];
        let insertion_node = which_node[block_id];
        // if block_id == 8 {
        //     println!(
        //         "insertion node = {}",
        //         self.cost.phylo.tree.node(&insertion_node).id
        //     );
        // }
        let number_of_states = self.cost.model.q.n();
        let p = &models[&which_model[block_id]];

        // TODO: this modifies the felsenstein of the insertion node
        //       and this again has to be undone like it was in remove_edges.
        //       instead i could modify the felsenstein to prob to not take a node
        //       but a felsenstein vector.
        for site in (block - block_len)..block {
            let felsenstein_of_v1: Vec<f64> = self.cost.model_info.borrow().felsenstein
                [usize::from(v1_idx)]
            .row(site)
            .iter()
            .copied()
            .collect();
            for state_i in 0..number_of_states {
                let mut sum_over_j = 0.0;
                for state_j in 0..number_of_states {
                    sum_over_j += p[(state_i, state_j)] * felsenstein_of_v1[state_j];
                }
                self.cost.model_info.borrow_mut().felsenstein[usize::from(insertion_node)]
                    [(site, state_i)] *= sum_over_j;
            }
        }
        // if block_id == 0 {
        //     println!(
        //         "felsenstein up, p {:?}, return {}",
        //         p,
        //         self.cost.felsenstein_to_prob(&insertion_node, block_id)
        //     )
        // }
        let prob = self.cost.felsenstein_to_prob(&insertion_node, block_id);
        for site in (block - block_len)..block {
            let felsenstein_of_v1: Vec<f64> = self.cost.model_info.borrow().felsenstein
                [usize::from(v1_idx)]
            .row(site)
            .iter()
            .copied()
            .collect();
            for state_i in 0..number_of_states {
                let mut sum_over_j = 0.0;
                for state_j in 0..number_of_states {
                    sum_over_j += p[(state_i, state_j)] * felsenstein_of_v1[state_j];
                }
                self.cost.model_info.borrow_mut().felsenstein[usize::from(insertion_node)]
                    [(site, state_i)] /= sum_over_j;
            }
        }
        prob
    }

    fn get_felsenstein_prob_for_insertion_on_dirty_tree(
        &self,
        dirty_edges: &HashMap<DirtyTreeEdge, (NodeIdx, usize)>,
        actions: &HashMap<DirtyTreeEdge, Action>,
        block_id: usize,
    ) -> f64 {
        for (typ, (node, _)) in dirty_edges {
            // if node == &self.cost.phylo.tree.root {
            //     println!("action for root {:?}", actions.get(typ).unwrap());
            // }
            if actions.get(typ).unwrap() == &Action::Insertion {
                return self.cost.felsenstein_to_prob(node, block_id);
            }
        }
        0.0
    }

    fn get_x(
        &self,
        dirty_edges: &HashMap<DirtyTreeEdge, (NodeIdx, usize)>,
        actions: &HashMap<DirtyTreeEdge, Action>,
    ) -> f64 {
        let l = self.cost.model.lambda();
        let m = self.cost.model.mu();
        let r = self.cost.model.r();

        let mut x = 1.0;

        for (typ, (node, _)) in dirty_edges {
            let time = self.cost.phylo.tree.node(node).blen;
            if node == &self.cost.phylo.tree.root {
                if actions.get(typ).unwrap() == &Action::Insertion {
                    x *= l / m * (1.0 - r) / r;
                }
            } else {
                let b = TKF92Cost::<JC69>::b(l, m, time);
                x *= match actions.get(typ).unwrap() {
                    Action::Insertion => l * b * (1.0 - r) / r,
                    Action::Deletion => TKF92Cost::<JC69>::n0(m, b),
                    Action::Homolog => TKF92Cost::<JC69>::h1(l, m, b, time),
                    _ => 1.0,
                }
            }
        }
        x
    }

    // TODO: i could reuse are_chars_at_leafs or rework this to have the ordering of the dirty edges hardcoded
    fn get_actions(
        &self,
        block_id: usize,
        dirty_edges: &HashMap<DirtyTreeEdge, (NodeIdx, usize)>,
        assignment: &[bool; 2],
    ) -> HashMap<DirtyTreeEdge, Action> {
        let mut actions = HashMap::<DirtyTreeEdge, Action>::with_capacity(5);
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;
        let node_map = self.cost.phylo.msa.get_node_map();
        for (typ, (node, _)) in dirty_edges {
            let (parent_is_char, current_is_char) = match typ {
                DirtyTreeEdge::T3 | DirtyTreeEdge::T4 => {
                    let parent_is_char = assignment[1];
                    let current_is_char = node_map[node][site].is_some();
                    (parent_is_char, current_is_char)
                }
                DirtyTreeEdge::T2 => {
                    let parent_is_char = assignment[0];
                    let current_is_char = node_map[node][site].is_some();
                    (parent_is_char, current_is_char)
                }
                DirtyTreeEdge::V2 => {
                    let parent_is_char = assignment[0];
                    let current_is_char = assignment[1];
                    (parent_is_char, current_is_char)
                }
                DirtyTreeEdge::V1 => {
                    if *node == self.cost.phylo.tree.root {
                        (false, assignment[0])
                    } else {
                        let parent_idx = &self.cost.phylo.tree.node(node).parent.unwrap();
                        let parent_is_char = node_map[parent_idx][site].is_some();
                        let current_is_char = assignment[0];
                        (parent_is_char, current_is_char)
                    }
                }
            };
            if parent_is_char && current_is_char {
                actions.insert(*typ, Action::Homolog);
            }
            if parent_is_char && !current_is_char {
                actions.insert(*typ, Action::Deletion);
            }
            if !parent_is_char && current_is_char {
                actions.insert(*typ, Action::Insertion);
            }
            if !parent_is_char && !current_is_char {
                actions.insert(*typ, Action::Nothing);
            }
        }
        actions
    }

    fn bools_to_index(bools1: &[bool; 2], bools2: &Vec<bool>) -> usize {
        bools1
            .iter()
            .chain(bools2.iter())
            .fold(0, |index, &b| (index << 1) | (b as usize))
    }

    // only made this pub to test, can i avoid it? or use pub(crate)
    pub fn index_to_bools(index: usize) -> ([bool; 2], Vec<bool>) {
        let mut bits = index;
        let bools2_len = 5;
        let mut bools2 = vec![false; bools2_len];

        // Extract bools2 from the least significant bits
        for i in (0..bools2_len).rev() {
            bools2[i] = (bits & 1) != 0;
            bits >>= 1;
        }

        // Extract the fixed `[bool; 2]`
        let bools1 = [
            (bits & 2) != 0, // Extract second bit
            (bits & 1) != 0, // Extract first bit
        ];

        (bools1, bools2)
    }

    // update model_info
    // updates the path from t1 to the root with set_node
    // this also needs to update the last action on the edges of the dirty tree
    // ^ or do i only have to to that once dp is done
    // return the accumulated p from t1 to the root
    fn remove_edges_from_factor_n_and_x_and_felsenstein(
        &self,
        v2_idx: &NodeIdx,
        acc_models: &HashMap<usize, SubstMatrix>,
        which_model: &Vec<usize>,
        which_node: &Vec<NodeIdx>,
    ) {
        // TODO: If we only have one aggregated x per block, then i only need to update this
        //       single value. However since felsenstein needs to be updated anyways, also
        //       updating indel values doesnt change time complexity.

        // TODO: this needs to be updated when NNI moves are integrated
        let old_tree = &self.cost.phylo.tree;
        let edges_to_remove = self.get_dirty_edges(v2_idx);
        let root_id = usize::from(old_tree.root);
        let (v1_idx, _) = edges_to_remove.get(&DirtyTreeEdge::V1).unwrap();
        let len = self.cost.model_info.borrow().blocks.len();
        for block_id in 0..len {
            let mut total_removed_x = 1.0;
            let mut total_removed_factor_n = 0.0;
            let mut total_removed_felsenstein_prob = 0.0;
            for (edge, _) in edges_to_remove.values() {
                // removing the x and factor_n that came from this edges indel process
                let (x, factor_n) = if edge == &old_tree.root {
                    (self.cost.get_indel_x_for_root(block_id), 0.0)
                } else {
                    // TODO: since we iterate over blocks and then over nodes the self.model_info.borrow_mut().last_action = false;
                    // is messed up
                    self.cost
                        .get_indel_x_and_factor_n_for_not_root(&edge, block_id)
                };
                total_removed_x *= x;
                total_removed_factor_n += factor_n;
                // println!(
                //     "removing x {}, from root x {}",
                //     x,
                //     self.cost.model_info.borrow_mut().aggregated_x[(root_id, block_id)]
                // );
                self.cost.model_info.borrow_mut().aggregated_x[(root_id, block_id)] /= x;
                // println!(
                //     "and it is now {}",
                //     self.cost.model_info.borrow_mut().aggregated_x[(root_id, block_id)]
                // );
                self.cost.model_info.borrow_mut().factor_ns[(root_id, block_id)] -= factor_n;
                // removing the prob that came from this edges substitution process
                // this will only remove the felsenstein if there is an insertion in the dirty tree
                if edge == &old_tree.root {
                    if self.cost.is_insertion_at_root(block_id) {
                        self.cost.model_info.borrow_mut().felsenstein_prob[(root_id, block_id)] -=
                            self.cost.felsenstein_to_prob(&old_tree.root, block_id);
                        total_removed_felsenstein_prob +=
                            self.cost.felsenstein_to_prob(&edge, block_id);
                    }
                } else {
                    // println!(
                    //     "block {}, edge {}, is_insertion {}",
                    //     block_id,
                    //     self.cost.phylo.tree.node(&edge).id,
                    //     self.cost.is_insertion_at_non_root(&edge, block_id),
                    // );
                    if self.cost.is_insertion_at_non_root(&edge, block_id) {
                        // TODO: this is the same code as above
                        self.cost.model_info.borrow_mut().felsenstein_prob[(root_id, block_id)] -=
                            self.cost.felsenstein_to_prob(&old_tree.root, block_id);
                        total_removed_felsenstein_prob +=
                            self.cost.felsenstein_to_prob(&edge, block_id);
                    }
                }
            }
            // removing the felsenstein if the insertion happened further up the tree
            let t1_is_char = if v1_idx == &old_tree.root {
                false
            } else {
                let t1 = self.cost.phylo.tree.node(v1_idx).parent.unwrap();
                let site = self.cost.model_info.borrow().blocks[block_id] - 1;
                self.cost.phylo.msa.get_node_map()[&t1][site].is_some()
            };
            // if there is no char (or even no node) then we dont have to fix anything up the tree
            if t1_is_char {
                assert!(total_removed_felsenstein_prob == 0.0);
                self.cost.model_info.borrow_mut().felsenstein_prob[(root_id, block_id)] -= self
                    .cost
                    .felsenstein_to_prob(&which_node[block_id], block_id);
                total_removed_felsenstein_prob += self
                    .cost
                    .felsenstein_to_prob(&which_node[block_id], block_id);

                let block = self.cost.model_info.borrow().blocks[block_id];
                let block_len = self.cost.model_info.borrow().block_lens[block_id];
                for site in (block - block_len)..block {
                    for state_i in 0..self.cost.model.q.n() {
                        let mut sum_over_j = 0.0;
                        for state_j in 0..self.cost.model.q.n() {
                            let prob_of_mutating_to_child =
                                acc_models[&which_model[block_id]][(state_i, state_j)];
                            let v1_felsenstein = self.cost.model_info.borrow().felsenstein
                                [usize::from(v1_idx)][(site, state_j)];
                            sum_over_j += (prob_of_mutating_to_child) * (v1_felsenstein);
                        }
                        self.cost.model_info.borrow_mut().felsenstein
                            [usize::from(which_node[block_id])][(site, state_i)] /= sum_over_j;
                    }
                }
            }
            // println!(
            //     "removing: block = {block_id}, x = {:.11}, factor_n = {:.11}, felsensteinprob = {:.11}",
            //     total_removed_x, total_removed_factor_n, total_removed_felsenstein_prob
            // );
        }
    }

    fn get_dirty_edges(&self, v2_idx: &NodeIdx) -> HashMap<DirtyTreeEdge, (NodeIdx, usize)> {
        let old_tree = &self.cost.phylo.tree;
        let mut edges_to_remove = HashMap::<DirtyTreeEdge, (NodeIdx, usize)>::with_capacity(5);
        edges_to_remove.insert(DirtyTreeEdge::V2, (v2_idx.clone(), 0));
        let t3_and_t4 = &old_tree.node(v2_idx).children;
        edges_to_remove.insert(DirtyTreeEdge::T3, (t3_and_t4[0].clone(), 1));
        edges_to_remove.insert(DirtyTreeEdge::T4, (t3_and_t4[1].clone(), 2));
        let v1_idx = &old_tree.node(v2_idx).parent.unwrap();
        edges_to_remove.insert(DirtyTreeEdge::V1, (v1_idx.clone(), 3));
        let sibling = &old_tree.sibling(v2_idx).unwrap();
        edges_to_remove.insert(DirtyTreeEdge::T2, (sibling.clone(), 4));
        edges_to_remove
    }

    /// also includes the model from the edge (t_1, v_1)
    fn get_felsenstein_substitution_models_up_to_insertion_point(
        &self,
        v2_idx: &NodeIdx,
    ) -> (HashMap<usize, SubstMatrix>, Vec<usize>, Vec<NodeIdx>) {
        let mut models = HashMap::<usize, SubstMatrix>::new();
        let mut which_model = Vec::<usize>::new();
        let mut which_node = Vec::<NodeIdx>::new();
        let n = self.cost.model.q.n();
        models.insert(0, DMatrix::<f64>::identity(n, n));
        let len = self.cost.model_info.borrow().blocks.len();
        for block_id in 0..len {
            // starts with v1
            let mut current = self.cost.phylo.tree.node(v2_idx).parent.unwrap();
            let site = self.cost.model_info.borrow().blocks[block_id] - 1;
            let mut parent_is_char = if current == self.cost.phylo.tree.root {
                false
            } else {
                let parent_idx = &self.cost.phylo.tree.node(&current).parent.unwrap();
                self.cost.phylo.msa.get_node_map()[parent_idx][site].is_some()
            };
            let mut count = 0;
            while current != self.cost.phylo.tree.root && parent_is_char {
                count += 1;
                if !models.contains_key(&count) {
                    // here i am multiplying the matrices but it should be
                    // p(t1) * p(t2) = p(t1+t2)
                    // but i think multiplying the matrices is cheaper than
                    // doing another matrix exponentiation
                    models.insert(
                        count,
                        &self.cost.model_info.borrow().models[usize::from(current)]
                            * &models[&(count - 1)],
                    );
                }
                current = self.cost.phylo.tree.node(&current).parent.unwrap();
                parent_is_char = if current == self.cost.phylo.tree.root {
                    false
                } else {
                    let parent_id = &self.cost.phylo.tree.node(&current).parent.unwrap();
                    self.cost.phylo.msa.get_node_map()[parent_id][site].is_some()
                };
            }
            which_model.push(count);
            which_node.push(current.clone());
        }
        (models, which_model, which_node)
    }

    pub fn get_allowed_assignments(
        t1_is_char: bool,
        t2_is_char: bool,
        t3_is_char: bool,
        t4_is_char: bool,
    ) -> Vec<[bool; 2]> {
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

    // checking if the assignment follows dollo's principle
    // pub fn assignment_follows_dollo(
    //     &self,
    //     assignment: &[bool; 2],
    //     block_id: usize,
    //     v2_idx: &NodeIdx,
    // ) -> bool {
    //     let v1_assignment = assignment[0];
    //     let v2_assignment = assignment[1];
    //     // TODO: this only needs to be recomputed per block and not every time this
    //     //       fn is called
    //     let (t1_is_char, t2_is_char, t3_is_char, t4_is_char) =
    //         self.are_chars_at_leafs(v2_idx, block_id);
    //     let left_is_some = t1_is_char || t2_is_char;
    //     let right_is_some = t3_is_char || t4_is_char;

    //     // this if must come first
    //     if left_is_some && right_is_some {
    //         return v1_assignment && v2_assignment;
    //     }
    //     if left_is_some && v2_assignment {
    //         return v1_assignment;
    //     }
    //     if right_is_some && v1_assignment {
    //         return v2_assignment;
    //     }
    //     if t1_is_char && t2_is_char {
    //         return v1_assignment;
    //     }
    //     if t3_is_char && t4_is_char {
    //         return v2_assignment;
    //     }
    //     // TODO: i should also test if there is a insertion in one of the subtrees
    //     //       bc it could also have been that the old edge contained the only char
    //     //       or does an internal char exist without any leaf also containing a char?
    //     //       so is it even possible that all chars in a col are removed if
    //     //       an internal node char is removed?
    //     //       But if there is no char anywhere else, then it is i think always suboptimal
    //     //       to have and insertion and subsequent deletion all on the tree.
    //     if !t1_is_char && !t2_is_char && !t3_is_char && !t4_is_char {
    //         return !v1_assignment && !v2_assignment;
    //     }
    //     true
    // }

    // not actually at the leafs but also at the red subtree
    pub fn are_chars_at_leafs(
        &self,
        v2_idx: &NodeIdx,
        block_id: usize,
    ) -> (bool, bool, bool, bool) {
        let v1_idx = &self.cost.phylo.tree.node(v2_idx).parent.unwrap();
        let v1_is_root = *v1_idx == self.cost.phylo.tree.root;
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;

        let t1_is_char = if v1_is_root {
            false
        } else {
            let t1_idx = &self.cost.phylo.tree.node(v1_idx).parent.unwrap();
            self.cost.phylo.msa.get_node_map()[t1_idx][site].is_some()
        };
        let t2_idx = &self.cost.phylo.tree.sibling(v2_idx).unwrap();
        let t2_is_char = self.cost.phylo.msa.get_node_map()[t2_idx][site].is_some();
        let children_of_v2 = &self.cost.phylo.tree.node(v2_idx).children;
        let t3_idx = &children_of_v2[0];
        let t4_idx = &children_of_v2[1];
        let t3_is_char = self.cost.phylo.msa.get_node_map()[t3_idx][site].is_some();
        let t4_is_char = self.cost.phylo.msa.get_node_map()[t4_idx][site].is_some();
        // println!(
        //     "ts are chars{:?}",
        //     (t1_is_char, t2_is_char, t3_is_char, t4_is_char)
        // );
        (t1_is_char, t2_is_char, t3_is_char, t4_is_char)
    }

    pub fn print_dp_table(&self) {
        println!("dp table");
        for block_id in 0..self.cost.model_info.borrow().blocks.len() {
            for (index, &value) in self.dp_table[block_id].iter().enumerate() {
                if value != f64::NEG_INFINITY {
                    println!(
                        "block {}, assignment & events {:?}, {}",
                        block_id,
                        Self::index_to_bools(index),
                        value
                    );
                }
            }
        }
    }

    fn print_backtracking_table(&self) {
        println!("backtracking table");
        for block_id in 0..self.cost.model_info.borrow().blocks.len() {
            for (index, value) in self.backtracking_table[block_id].iter().enumerate() {
                if self.dp_table[block_id][index] != f64::NEG_INFINITY {
                    println!(
                        "block {}, assignment & events {:?}, {:?}",
                        block_id,
                        Self::index_to_bools(index),
                        Self::index_to_bools(*value)
                    );
                }
            }
        }
    }
}

macro_rules! bools_to_index {
    ($($b:expr),*) => {{
        let mut index = 0;
        $(
            index = (index << 1) | ($b as usize);
        )*
        index
    }};
}
