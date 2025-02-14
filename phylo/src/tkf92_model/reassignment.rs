use itertools::{iproduct, Itertools};
use nalgebra::DMatrix;
use std::{collections::HashMap, fmt::Display};

use crate::substitution_models::{dna_models::JC69, QMatrix, SubstMatrix};
use crate::tkf92_model::TKF92Cost;
use crate::tree::NodeIdx;

static NUMBER_OF_BOOLS: usize = 7;

static ASSIGNMENT_COMBINATIONS: [[bool; 2]; 4] =
    [[false, false], [false, true], [true, false], [true, true]];

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
#[derive(PartialEq)]
enum Action {
    Insertion,
    Deletion,
    Homolog,
    Nothing,
}

struct ReassignEdge<Q: QMatrix + Display> {
    dp: Vec<[f64; 32]>, // maybe also put this in a refcell such that
    cost: TKF92Cost<Q>,
}

impl<Q: QMatrix + Display> ReassignEdge<Q> {
    fn new(cost: TKF92Cost<Q>) -> ReassignEdge<Q> {
        let number_of_blocks = cost.model_info.borrow().blocks.len();
        ReassignEdge {
            dp: Vec::<[f64; 32]>::with_capacity(number_of_blocks),
            cost,
        }
        // i could also save the node here and also its parent
        // then i can ensure uppon creation of that struct that v2 is not the root
        // and that v1 exists
    }

    fn fill_dp(&mut self, v2_idx: &NodeIdx) {
        // here i thought that without the mut self, and therefore self.cost would not change
        // but self.cost.model_info.borrow_mut() can change self

        // the topo optimiser gives the new tree and also the two dirty nodes
        // then the update tree can subtract from the prob of the root the probs that happened
        // on the edges of the dirty subtree of the old tree
        // also divide the aggregated x of the root the x that happened on the subtree of the
        // old tree
        // i can go from t1 to the root and multiply all ps together, then i can divide the
        // felsenstein by the old felsenstein of the dirty tree of the old tree multiplied
        // by the aggregated p and

        // then i can redo the assignment and calculate the prob and x and felsenstein
        // without going up the root (and updating all the values)
        // however after i found a optimal assignment i need to update the intermediate
        // values on the path to the root
        // or can i change the way the intermediate values are stored on the tree to
        // avoid this updating?

        // i dont really need to save some aggregated values
        // at every single node, at least for now

        // perhaps for realignment this would make sense

        // i only need to a single aggregated x per block, then i also only need to
        // update this single value
        // however i still need to update the felsenstein intermediate values up to the root
        // so also updating aggregated x and prob wouldnt change time complexity

        // i could also use this to only do ASR by using the update_tree() and passing the same
        // tree and as dirty tree the edge that is reestimated
        let root = &self.cost.phylo.tree.root;
        if v2_idx == root {
            return;
        }
        let bool_options = vec![vec![false, true]; 2];
        let model_info = self.cost.model_info.borrow();
        let number_of_blocks = model_info.blocks.len();

        // i dont need the for the case of v1 is the root
        // do i need it of t1 is the root?
        let (models, which_model, which_node) =
            self.get_felsenstein_substitution_models_up_to_insertion_point(v2_idx);

        self.remove_edges_from_prob_and_x_and_felsenstein(
            v2_idx,
            &models,
            &which_model,
            &which_node,
        );

        // currently i am using the order of this hashmap to convert between edges and their index in the
        // events bool vector, maybe it is saver and more explicit to write some conversion somewhere
        let dirty_edges = self.get_dirty_edges(v2_idx);

        let v1_idx = &self.cost.phylo.tree.node(v2_idx).parent.unwrap();

        for block_id in 0..number_of_blocks {
            let block_len = model_info.block_lens[block_id];
            let block = model_info.blocks[block_id]; // instead of going over all possibilities for the assignments i can look at
                                                     // the children and see what assignments are even possible

            // iterate through assignments and events separately
            for assignment in ASSIGNMENT_COMBINATIONS {
                if !self.assignment_follows_dollo(&assignment, block_id, v2_idx) {
                    continue;
                }
                let actions = self.get_actions(block_id, &dirty_edges, &assignment);
                // indel x
                let dirty_tree_x = self.get_x(&dirty_edges, &actions);
                let x = model_info.aggregated_x[(usize::from(root), 0)] * dirty_tree_x;
                let integrated_x_prop = x.ln() + (block_len as f64 - 1.0) * (1.0 + x).ln();

                // substitution process
                self.cost.set_felsenstein_for_internal(v2_idx, block_id);
                self.cost.set_felsenstein_for_internal(v1_idx, block_id);
                // do i also want to set the felsenstein for t1?
                // if insertion happened on the dirty tree
                let felsenstein_prob = self.get_felsenstein_prob_for_insertion_on_dirty_tree(
                    &dirty_edges,
                    &actions,
                    block_id,
                );
                // if insertion happened further up the tree
                let felsenstein_prob_for_up_the_tree = self
                    .get_felsenstein_prob_for_insertion_higher_than_on_dirty_tree(
                        &actions,
                        v1_idx,
                        block_id,
                        &models,
                        &which_model,
                        &which_node,
                    );

                // if we dont have an insertion in the dirty tree,
                // then update the dirty tree and v1_felsenstein
                // multiply the v1_felsenstein with the accumulated p (use which_model) and to point wise multiplication
                // with the felsenstein of the which_node insertion site and call
                // felsenstein_to_prob for this node and add it to the dp prob

                // i can take the insertion points and take the felsenstein with the removed v1 subtree
                // and * the new v1 felsenstein to the insertion point and then use add to to the prob

                // here i want to call the method

                for events in self.get_possible_events_for_assignment(&actions, &dirty_edges) {
                    // in case v1 is the root, then i dont need to consider the event on that edge since
                    // there cant have been a deletion

                    // i can take the x from the root (where the dirty x and probs got removed)
                    // and add the probs and multiply the x from the dirty tree

                    // initial case
                    // go over all edges of the dirty tree and get their new x
                    // indel process
                    // i can move

                    // this depends on the events since i might have to add a factor N
                    // but i could also get the prob for everything else
                    // and in this loop only add the factor Ns and set the prob to the gamma

                    // instead of adding it already here, i could also calculate the max prev gamma and then
                    // add everything at once to the dp table
                    
                    if block_id == 0 {
                        continue;
                    }
                    
                    // recursion and factor N added to the dp here
                    // this does not depend on the current events, but the past events and current action
                    // in latex tkf92 it means we are using e' and not e to determine the factor N
                    
                    // find max of previous gamma including the factor N and add it to the dp entry
                    

                    // this is i think included in the max
                    // let dirty_tree_prob = self.get_dirty_tree_prob(&dirty_edges, &actions, &events);
                    
                    let max = self.get_max(&events, &actions, &dirty_edges, block_id);
                    
                    let root_prob = model_info.prob[(usize::from(root), block_id)];
                    
                    // here i must take the probs and x of the root, add the ones from above
                    // and do something similar to the logl fn
                    self.dp[block_id][Self::bools_to_index(&assignment, &events)] = 
                        integrated_x_prop
                        + felsenstein_prob
                        + felsenstein_prob_for_up_the_tree
                        + max 
                        + root_prob;
                }
            }
        }
    }

    fn get_max(
        &self,
        events: &Vec<bool>,
        actions: &HashMap<DirtyTreeEdge, Action>,
        dirty_edges: &HashMap<DirtyTreeEdge, NodeIdx>,
        block_id: usize,
    ) -> f64 {
        // get the previous events, for every gap col edge the previous must be the same as the current event
        let mut max = None;
        let l = self.cost.model.lambda();
        let m = self.cost.model.mu();
        for previous_event in self.get_previous_allowed_events(events, actions, dirty_edges) {
            for previous_assignment in ASSIGNMENT_COMBINATIONS {
                if !self.assignment_follows_dollo(
                    &previous_assignment,
                    block_id - 1,
                    dirty_edges.get(&DirtyTreeEdge::V2).unwrap(),
                ) {
                    continue;
                }
                // get the last gamma
                let last_gamma = self.dp[block_id - 1]
                    [Self::bools_to_index(&previous_assignment, &previous_event)];
                // depending on the previous events and the actions now
                // return the prob of gamma and factor Ns
                let mut factorN = 0.0;
                for (i, (typ, edge)) in dirty_edges.iter().enumerate() {
                    if actions.get(typ).unwrap() == &Action::Insertion && previous_assignment[i] {
                        let time = self.cost.phylo.tree.node(edge).blen;
                        let b = TKF92Cost::<JC69>::b(l, m, time);
                        factorN += TKF92Cost::<JC69>::log_n1(l, m, b, time);
                        factorN -= (l * b).ln();
                        factorN -= TKF92Cost::<JC69>::n0(m, b).ln();
                    }
                }
                let current = last_gamma + factorN;
                if let Some(ref mut m) = max {
                    if current > max.unwrap() {
                        *m = current;
                    }
                } else {
                    max = Some(current);
                }
            }
        }
        if max.is_none() {
            println!("this should never happen")
            0.0
        } else {
            return max.unwrap();
        }
    }

    fn get_previous_allowed_events(
        &self,
        events: &Vec<bool>,
        actions: &HashMap<DirtyTreeEdge, Action>,
        dirty_edges: &HashMap<DirtyTreeEdge, NodeIdx>,
    ) -> Vec<Vec<bool>> {
        let mut choices: Vec<Vec<bool>> = Vec::new();
        for (i, (typ, _)) in dirty_edges.iter().enumerate() {
            match actions.get(typ).unwrap() {
                Action::Nothing => choices.push(vec![events[i]]),
                _ => choices.push(vec![false, true]),
            };
        }
        choices.into_iter().multi_cartesian_product().collect()
    }

    // that corresponds to C_a^i
    fn get_possible_events_for_assignment(
        &self,
        actions: &HashMap<DirtyTreeEdge, Action>,
        dirty_edges: &HashMap<DirtyTreeEdge, NodeIdx>,
    ) -> Vec<Vec<bool>> {
        let mut choices: Vec<Vec<bool>> = Vec::new();
        for (typ, _) in dirty_edges {
            match actions.get(typ).unwrap() {
                Action::Deletion => choices.push(vec![true]),
                Action::Nothing => choices.push(vec![false, true]),
                _ => choices.push(vec![false]),
            };
        }
        choices.into_iter().multi_cartesian_product().collect()
    }

    // fn get_dirty_tree_prob(
    //     &self,
    //     dirty_edges: &HashMap<DirtyTreeEdge, NodeIdx>,
    //     actions: &HashMap<DirtyTreeEdge, Action>,
    //     events: &Vec<bool>,
    // ) -> f64 {
    //     let mut prob = 0.0;
    //     for (i, (typ, edge)) in dirty_edges.iter().enumerate() {
    //         // this assumes that the ordering of the edges when looping over them states the same from site to site
    //         if edge == &self.cost.phylo.tree.root {
    //             continue;
    //         }
    //         let is_insertion = actions.get(typ).unwrap() == &Action::Insertion;
    //         let last_was_deletion = events[i];
    //         if is_insertion && last_was_deletion {
    //             let l = self.cost.model.lambda();
    //             let m = self.cost.model.mu();
    //             let time = self.cost.phylo.tree.node(edge).blen;
    //             let b = TKF92Cost::<JC69>::b(l, m, time);
    //             // TODO: i can a precalculated factor for every edge that needs this
    //             // since this factor is maybe not so common, i can calculate it on demand
    //             // and save the result in a hashmap
    //             prob += TKF92Cost::<JC69>::log_n1(l, m, b, time);
    //             prob -= (l * b).ln();
    //             prob -= TKF92Cost::<JC69>::n0(m, b).ln();
    //             break; // bc there can only be one insertion
    //         }
    //     }
    //     prob
    // }

    // fn dirty_edge_to_id(&self, typ: &DirtyTreeEdge) -> usize {
    //     // can this be done more efficiently?
    //     // like i could change
    //     match typ {
    //         DirtyTreeEdge::V1 => 1,
    //         DirtyTreeEdge::T2 => 2,
    //         DirtyTreeEdge::T3 => 3,
    //         DirtyTreeEdge::T4 => 4,
    //         DirtyTreeEdge::V2 => 5,
    //     }
    // }

    fn get_felsenstein_prob_for_insertion_higher_than_on_dirty_tree(
        &self,
        actions: &HashMap<DirtyTreeEdge, Action>,
        v1_idx: &NodeIdx,
        block_id: usize,
        models: &HashMap<usize, SubstMatrix>,
        which_model: &Vec<usize>,
        which_node: &Vec<NodeIdx>,
    ) -> f64 {
        // only if v1 - t1 is homolog
        if actions.get(&DirtyTreeEdge::V1).unwrap() != &Action::Homolog {
            return 0.0;
        }
        let block = self.cost.model_info.borrow().blocks[block_id];
        let block_len = self.cost.model_info.borrow().block_lens[block_id];
        let insertion_node = which_node[block_id];
        let number_of_states = self.cost.model.q.n();
        let p = &models[&which_model[block_id]];
        for site in (block - block_len)..block {
            // do i have already set the felstenstein for t1 if present?
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
                    [(site, state_i)] = sum_over_j;
            }
        }
        self.cost.felsenstein_to_prob(&insertion_node, block_id)
    }

    fn get_felsenstein_prob_for_insertion_on_dirty_tree(
        &self,
        dirty_edges: &HashMap<DirtyTreeEdge, NodeIdx>,
        actions: &HashMap<DirtyTreeEdge, Action>,
        block_id: usize,
    ) -> f64 {
        for (typ, node) in dirty_edges {
            if actions.get(typ).unwrap() == &Action::Insertion {
                return self.cost.felsenstein_to_prob(node, block_id);
            }
        }
        0.0
    }

    fn get_x(
        &self,
        dirty_edges: &HashMap<DirtyTreeEdge, NodeIdx>,
        actions: &HashMap<DirtyTreeEdge, Action>,
    ) -> f64 {
        let l = self.cost.model.lambda();
        let m = self.cost.model.mu();
        let r = self.cost.model.r();

        let mut x = 1.0;

        for (typ, node) in dirty_edges {
            let time = self.cost.phylo.tree.node(node).blen;
            if node == &self.cost.phylo.tree.root {
                if actions.get(typ).unwrap() == &Action::Insertion {
                    x *= l / m * (1.0 - r) / r;
                }
                continue;
            }
            let b = TKF92Cost::<JC69>::b(l, m, time);
            x *= match actions.get(typ).unwrap() {
                Action::Insertion => l * b * (1.0 - r) / r,
                Action::Deletion => TKF92Cost::<JC69>::n0(m, b),
                Action::Homolog => TKF92Cost::<JC69>::h1(l, m, b, time),
                _ => 1.0,
            }
        }
        x
    }

    fn get_actions(
        &self,
        block_id: usize,
        dirty_edges: &HashMap<DirtyTreeEdge, NodeIdx>,
        assignment: &[bool; 2],
    ) -> HashMap<DirtyTreeEdge, Action> {
        let mut actions = HashMap::<DirtyTreeEdge, Action>::with_capacity(5);
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;
        let node_map = self.cost.phylo.msa.get_node_map();
        for (typ, node) in dirty_edges {
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
                        (assignment[0], false)
                    } else {
                        let parent_idx = &self.cost.phylo.tree.node(node).parent.unwrap();
                        let parent_is_char = node_map[parent_idx][site].is_some();
                        let current_is_char = assignment[1];
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

    // this is deprecated since i now have a vec with the actions for the dirty tree
    fn is_insertion(
        &self,
        assignment: &[&bool],
        typ: &DirtyTreeEdge,
        node: &NodeIdx,
        block_id: usize,
    ) -> bool {
        // instead of using a match, i could also rely on a ordering of the
        // dirty edges and say parent_of_t4_is_gap t4_is_not_gap,
        // and the same for the other edges, by doing so i can avoid the
        // overhead of the match at the cost of readability
        let site = self.cost.model_info.borrow().blocks[block_id] - 1;
        match typ {
            DirtyTreeEdge::T3 | DirtyTreeEdge::T4 => {
                let parent_is_gap = *assignment[1];
                let current_is_not_gap = self.cost.phylo.msa.get_node_map()[node][site].is_some();
                parent_is_gap && current_is_not_gap
            }
            DirtyTreeEdge::T2 => {
                let parent_is_gap = *assignment[0];
                let current_is_not_gap = self.cost.phylo.msa.get_node_map()[node][site].is_some();
                parent_is_gap && current_is_not_gap
            }
            DirtyTreeEdge::V2 => {
                let parent_is_gap = !*assignment[0];
                let current_is_not_gap = *assignment[1];
                parent_is_gap && current_is_not_gap
            }
            DirtyTreeEdge::V1 => {
                if *node == self.cost.phylo.tree.root {
                    *assignment[0]
                } else {
                    let parent_is_gap = self.cost.phylo.msa.get_node_map()
                        [&self.cost.phylo.tree.node(node).parent.unwrap()][site]
                        .is_some();
                    let current_is_not_gap = *assignment[1];
                    parent_is_gap && current_is_not_gap
                }
            }
        }
    }

    fn bools_to_index(bools1: &[bool; 2], bools2: &Vec<bool>) -> usize {
        bools1
            .iter()
            .chain(bools2.iter())
            .fold(0, |index, &b| (index << 1) | (b as usize))
    }

    // update model_info
    // updates the path from t1 to the root with set_node
    // this also needs to update the last action on the edges of the dirty tree
    // ^ or do i only have to to that once dp is done
    // return the accumulated p from t1 to the root
    fn remove_edges_from_prob_and_x_and_felsenstein(
        &self,
        v2_idx: &NodeIdx,
        acc_models: &HashMap<usize, SubstMatrix>,
        which_model: &Vec<usize>,
        which_node: &Vec<NodeIdx>,
    ) {
        let old_tree = &self.cost.phylo.tree;
        let edges_to_remove = self.get_dirty_edges(v2_idx);
        let len = self.cost.model_info.borrow().blocks.len();
        let root_id = usize::from(old_tree.root);
        let v1_idx = &old_tree.node(v2_idx).parent.unwrap();
        for block_id in 0..len {
            for &edge in edges_to_remove.values() {
                // removing the x and prob that came from this edges indel process
                let (x, prob) = if edge == old_tree.root {
                    (self.cost.get_indel_x_for_root(block_id), 0.0)
                } else {
                    self.cost
                        .get_indel_x_and_prob_for_not_root(v2_idx, block_id)
                };
                self.cost.model_info.borrow_mut().aggregated_x[(root_id, block_id)] /= x;
                self.cost.model_info.borrow_mut().prob[(root_id, block_id)] -= prob;
                // removing the prob that came from this edges substitution process
                // this will only remove the felsenstein if there is an insertion in the dirty tree
                if edge == old_tree.root {
                    if self.cost.is_insertion_at_root(block_id) {
                        self.cost.model_info.borrow_mut().prob[(root_id, block_id)] -=
                            self.cost.felsenstein_to_prob(&old_tree.root, block_id)
                    }
                } else {
                    if self.cost.is_insertion_at_non_root(&edge, block_id) {
                        self.cost.model_info.borrow_mut().prob[(root_id, block_id)] -=
                            self.cost.felsenstein_to_prob(&old_tree.root, block_id)
                    }
                }
            }
            // removing the whole subtree rooted in t1 from the felsenstein vars of the insertion node

            // if we have a char at t1 then we have to pass update the felsenstein
            // so we could put this in a different for loop and skip this loop if
            // v1 is the root. then we also dont have to calculate the acc_models, which_model and which_node
            let t1_is_char = if v1_idx == &old_tree.root {
                false
            } else {
                let t1 = self.cost.phylo.tree.node(v1_idx).parent.unwrap();
                let site = self.cost.model_info.borrow().blocks[block_id] - 1;
                self.cost.phylo.msa.get_node_map()[&t1][site].is_some()
            };
            // if there is no char (or even no node) then we dont have to fix anything up the tree
            if t1_is_char {
                self.cost.model_info.borrow_mut().prob[(root_id, block_id)] -= self
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
                            // i thinks it must be v1 since
                            let v1_felsenstein = self.cost.model_info.borrow().felsenstein
                                [usize::from(v1_idx)][(site, state_j)];
                            sum_over_j += (prob_of_mutating_to_child) * (v1_felsenstein);
                        }
                        self.cost.model_info.borrow_mut().felsenstein
                            [usize::from(which_node[block_id])][(site, state_i)] /= sum_over_j;
                    }
                }
            }

            // so we either have the insertion below the dirty tree,
            // then we dont have to update the felsenstein
            // we have it in the dirty tree, see above
            // we have it higher up the tree, then we need to update it only
            // if there is a char at t1, bc if there is a gap, there is no felsenstein to pass up
        }
    }

    fn get_dirty_edges(&self, v2_idx: &NodeIdx) -> HashMap<DirtyTreeEdge, NodeIdx> {
        let old_tree = &self.cost.phylo.tree;
        let mut edges_to_remove = HashMap::<DirtyTreeEdge, NodeIdx>::with_capacity(5);
        edges_to_remove.insert(DirtyTreeEdge::V2, v2_idx.clone());
        let t3_and_t4 = &old_tree.node(v2_idx).children;
        edges_to_remove.insert(DirtyTreeEdge::T3, t3_and_t4[0].clone());
        edges_to_remove.insert(DirtyTreeEdge::T4, t3_and_t4[1].clone());
        let v1_idx = &old_tree.node(v2_idx).parent.unwrap();
        edges_to_remove.insert(DirtyTreeEdge::V1, v1_idx.clone());
        let sibling = &old_tree.sibling(v2_idx).unwrap();
        edges_to_remove.insert(DirtyTreeEdge::T2, sibling.clone());
        edges_to_remove
    }

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
            let mut parent_is_char = if current == self.cost.phylo.tree.root {
                false
            } else {
                let parent_id = &self.cost.phylo.tree.node(&current).parent.unwrap();
                self.cost.phylo.msa.get_node_map()[parent_id][block_id].is_some()
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
                    self.cost.phylo.msa.get_node_map()[parent_id][block_id].is_some()
                };
            }
            which_model.push(count);
            which_node.push(current.clone());
        }
        (models, which_model, which_node)
    }

    fn assignment_matches_events(
        &self,
        v2_idx: &NodeIdx,
        block_id: usize,
        assignment: &[&bool],
        events: &[&bool],
    ) -> bool {
        let v1_assignment = *assignment[0];
        let v2_assignment = *assignment[1];
        let event_on_edge_1 = *events[0];
        let event_on_edge_2 = *events[1];
        let event_on_edge_3 = *events[2];
        let event_on_edge_4 = *events[3];
        let event_on_edge_5 = *events[4];
        let (t1_is_char, t2_is_char, t3_is_char, t4_is_char) =
            self.are_chars_at_leafs(v2_idx, block_id);
        let v1_idx = &self.cost.phylo.tree.node(v2_idx).parent.unwrap();
        let v1_is_root = *v1_idx == self.cost.phylo.tree.root;
        // e1
        if !v1_is_root {
            // if v1 is the root there is no event on edge e1 -> i can skip checking this edge
            if event_on_edge_1 {
                let gap_col = !v1_assignment && !t1_is_char;
                let deletion = t1_is_char && !v1_assignment;
                if !gap_col && !deletion {
                    return false;
                }
            }
        }
        // e2
        if event_on_edge_2 {
            let gap_col = !v1_assignment && !t2_is_char;
            let deletion = v1_assignment && !t2_is_char;
            if !gap_col && !deletion {
                return false;
            }
        }
        //e5
        if event_on_edge_5 {
            let gap_col = !v1_assignment && !v2_assignment;
            let deletion = v1_assignment && !v2_assignment;
            if !gap_col && !deletion {
                return false;
            }
        }
        //e3
        if event_on_edge_3 {
            let gap_col = !v2_assignment && !t3_is_char;
            let deletion = v2_assignment && !t3_is_char;
            if !gap_col && !deletion {
                return false;
            }
        }
        //e4
        if event_on_edge_4 {
            let gap_col = !v2_assignment && !t4_is_char;
            let deletion = v2_assignment && !t4_is_char;
            if !gap_col && !deletion {
                return false;
            }
        }
        true
    }

    // not actually at the leafs but also at the red subtree
    fn are_chars_at_leafs(&self, v2_idx: &NodeIdx, block_id: usize) -> (bool, bool, bool, bool) {
        let v1_idx = &self.cost.phylo.tree.node(v2_idx).parent.unwrap();
        let v1_is_root = *v1_idx == self.cost.phylo.tree.root;
        let t1_idx = &self.cost.phylo.tree.node(v1_idx).parent.unwrap();

        let site = self.cost.model_info.borrow().blocks[block_id] - 1;

        let t1_is_char = if v1_is_root {
            false
        } else {
            self.cost.phylo.msa.get_node_map()[t1_idx][site].is_some()
        };
        let t2_idx = &self.cost.phylo.tree.sibling(v2_idx).unwrap();
        let t2_is_char = self.cost.phylo.msa.get_node_map()[t2_idx][site].is_some();
        let children_of_v2 = &self.cost.phylo.tree.node(v2_idx).children;
        let t3_idx = &children_of_v2[0];
        let t4_idx = &children_of_v2[1];
        let t3_is_char = self.cost.phylo.msa.get_node_map()[t3_idx][site].is_some();
        let t4_is_char = self.cost.phylo.msa.get_node_map()[t4_idx][site].is_some();
        (t1_is_char, t2_is_char, t3_is_char, t4_is_char)
    }

    // checking if the assignment follows dollo's principle
    fn assignment_follows_dollo(
        &self,
        assignment: &[bool; 2],
        block_id: usize,
        v2_idx: &NodeIdx,
    ) -> bool {
        let v1_assignment = assignment[0];
        let v2_assignment = assignment[1];
        let (t1_is_char, t2_is_char, t3_is_char, t4_is_char) =
            self.are_chars_at_leafs(v2_idx, block_id);
        let left_is_some = t1_is_char || t2_is_char;
        let right_is_some = t3_is_char || t4_is_char;
        // this if must come first
        if left_is_some && right_is_some {
            return v1_assignment && v2_assignment;
        }
        if left_is_some && v2_assignment {
            return v1_assignment;
        }
        if right_is_some && v1_assignment {
            return v2_assignment;
        }
        if t1_is_char && t2_is_char {
            return v1_assignment;
        }
        if t3_is_char && t4_is_char {
            return v2_assignment;
        }
        // i should also test if there is a insertion in one of the subtrees
        // bc it could also have been that the old edge contained the only char
        // or does an internal char exist without any leaf also containing a char?
        // so is it even possible that all chars in a col are removed if
        // an internal node char is removed?
        if !t1_is_char && !t2_is_char && !t3_is_char && !t4_is_char {
            return !v1_assignment && !v2_assignment;
        }
        true
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
