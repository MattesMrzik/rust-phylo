use std::path::Path;

use approx::assert_relative_eq;
use fixedbitset::FixedBitSet;

use itertools::Itertools;
#[cfg(feature = "multi-thread")]
use rayon::prelude::*;

use crate::alignment::{Alignment, AncestralAlignment, MASA};
use crate::likelihood::ModelSearchCost;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use crate::random::DefaultGenerator;
use crate::tkf_model::{
    mapping_from_node_seq, reestimate_tests::assert_masa_is_dollo, EdgeSeqsReestimator,
    TKF92IndelCostBuilder, TKFIndelCost, TKFModel,
};
use crate::tree::NodeIdx;

/// Given all possible edge assignment for each block in the alignment returns a specific edge
/// assignment corresponding to the given index.
///
/// # Arguments
///
/// * `idx` - index to decode
/// * `possible_edge_assignments` - a vector of possible edge assignment for each block in the
///   alignment, the outer vector is over blocks, the inner vector is over possible assignments for
///   the edge.
///
/// # Examples
///  ```
///  let possibilities= vec![vec![(true, true), (true, false)], vec![(false, false)]];
///  let first = edge_seqs(0, &possibilities); // returns vec![(true, true), (false, false)]
///  let second = edge_seqs(1, &possibilities); // returns vec![(true, false), (false, false)]
///  ```
#[cfg(test)]
fn edge_seqs(
    mut idx: usize,
    possible_edge_assignments: &Vec<Vec<(bool, bool)>>,
) -> Vec<(bool, bool)> {
    let mut tuple = Vec::with_capacity(possible_edge_assignments.len());
    for poss in possible_edge_assignments {
        let choice = idx % poss.len();
        tuple.push((poss[choice].0, poss[choice].1));
        idx /= poss.len();
    }
    tuple
}

#[test]
fn test_decode_index() {
    let possible_edge_assignments = vec![
        vec![(true, true), (true, false), (false, true), (false, false)],
        vec![(true, true), (false, false)],
        vec![(true, true), (true, false), (false, true)],
        vec![(true, true), (false, false)],
        vec![(true, true), (true, false), (false, true), (false, false)],
        vec![(true, true)],
        vec![(true, true), (true, false)],
    ];
    let number_of_possibilities: usize = possible_edge_assignments
        .iter()
        .map(|poss| poss.len())
        .product();
    assert_eq!(number_of_possibilities, 384);
    let mut all_possibilities = Vec::with_capacity(number_of_possibilities);
    for i in 0..number_of_possibilities {
        all_possibilities.push(edge_seqs(i, &possible_edge_assignments));
    }
    let unique: Vec<_> = all_possibilities.iter().unique().collect();
    assert_eq!(unique.len(), number_of_possibilities);
    for possibility in possible_edge_assignments
        .into_iter()
        .multi_cartesian_product()
    {
        assert!(all_possibilities.contains(&possibility));
    }
}

#[cfg(test)]
fn print_progress(i: usize, number_of_possibilities: usize) {
    if (i + 1) % 10000 == 0 {
        let percent = i as f64 / number_of_possibilities as f64 * 100.0;
        println!("Brute force progress for current edge {percent:.4}% ");
    }
}

#[cfg(test)]
fn number_of_possibilities(possible_edge_assignments: &[Vec<(bool, bool)>]) -> usize {
    possible_edge_assignments
        .iter()
        .map(|poss| poss.len())
        .product()
}

#[cfg(test)]
fn edge_seqs_to_mappings(
    edge_seqs: &[(bool, bool)],
    block_lens: &[usize],
    seq_len: usize,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut v2_fixed_bit_set = FixedBitSet::with_capacity(block_lens.len());
    let mut v1_fixed_bit_set = FixedBitSet::with_capacity(block_lens.len());
    for (site, edge_assignment) in edge_seqs.iter().enumerate() {
        if edge_assignment.0 {
            v1_fixed_bit_set.insert(site);
        }
        if edge_assignment.1 {
            v2_fixed_bit_set.insert(site);
        }
    }
    let new_v1_mapping = mapping_from_node_seq(&v1_fixed_bit_set, block_lens, seq_len);
    let new_v2_mapping = mapping_from_node_seq(&v2_fixed_bit_set, block_lens, seq_len);
    (new_v1_mapping, new_v2_mapping)
}

fn make_nodes_dirty<T: TKFModel>(cost: &mut TKFIndelCost<T, MASA>, v2_idx: &NodeIdx) {
    for child in &cost.phylo.tree.node(v2_idx).children {
        cost.model_info
            .borrow_mut()
            .valid
            .set(usize::from(child), false);
    }
    cost.model_info
        .borrow_mut()
        .valid
        .set(usize::from(cost.phylo.tree.sibling(v2_idx).unwrap()), false);
}

#[cfg(test)]
fn cost_for_edge_seqs<T: TKFModel>(
    v2_idx: &NodeIdx,
    cost: &mut TKFIndelCost<T, MASA>,
    edge_seqs: &[(bool, bool)],
) -> f64 {
    let v1_idx = &cost.phylo.tree.node(v2_idx).parent.unwrap();
    let block_lens = cost.model_info.borrow().block_lengths.clone();
    let seq_len = cost.phylo.msa.len();

    let (new_v1_mapping, new_v2_mapping) = edge_seqs_to_mappings(edge_seqs, &block_lens, seq_len);
    cost.phylo.msa.update_ancestral_map(v1_idx, new_v1_mapping);
    cost.phylo.msa.update_ancestral_map(v2_idx, new_v2_mapping);

    make_nodes_dirty(cost, v2_idx);

    cost.logl()
}

#[cfg(test)]
fn brute_force_max_for_possibilities_single_thread<T: TKFModel>(
    possible_edge_assignments: Vec<Vec<(bool, bool)>>,
    number_of_possibilities: usize,
    cost: &mut TKFIndelCost<T, MASA>,
    v2_idx: &NodeIdx,
) -> f64 {
    let mut max: f64 = f64::MIN;
    for i in 0..number_of_possibilities {
        print_progress(i, number_of_possibilities);
        let possibility = edge_seqs(i, &possible_edge_assignments);
        let current = cost_for_edge_seqs(v2_idx, cost, &possibility);
        max = max.max(current);
    }
    max
}

#[cfg(all(test, feature = "multi-thread"))]
fn brute_force_max_for_possibilities_multi_thread(
    possible_edge_assignments: Vec<Vec<(bool, bool)>>,
    number_of_possibilities: usize,
    cost: &TKFIndelCost<impl TKFModel + Send, MASA>,
    v2_idx: &NodeIdx,
    num_threads: usize,
) -> f64 {
    let chunk_size = number_of_possibilities.div_ceil(num_threads);
    let cost_clones = vec![cost.clone(); num_threads];
    let chunk_maxes: Vec<f64> = (0..number_of_possibilities)
        .into_par_iter()
        .chunks(chunk_size)
        .enumerate()
        .zip(cost_clones)
        .into_par_iter()
        .map(move |((chunk_id, chunk), mut thread_cost)| {
            let mut max: f64 = f64::MIN;
            for i in chunk {
                if chunk_id == 0 {
                    print_progress(i, number_of_possibilities / num_threads);
                }
                let possibility = edge_seqs(i, &possible_edge_assignments);
                let current = cost_for_edge_seqs(v2_idx, &mut thread_cost, &possibility);
                max = max.max(current);
            }
            max
        })
        .collect();
    chunk_maxes.into_iter().fold(f64::MIN, |a, b| a.max(b))
}

#[cfg(test)]
fn possible_assignments_of_edge(
    t1_is_char: bool,
    t2_is_char: bool,
    t3_is_char: bool,
    t4_is_char: bool,
) -> Vec<(bool, bool)> {
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

#[cfg(test)]
fn get_edge_assignment_possibilities(
    cost: &TKFIndelCost<impl TKFModel, MASA>,
    v2_idx: &NodeIdx,
) -> Vec<Vec<(bool, bool)>> {
    use crate::tkf_model::tests::get_mapping_for_any_node;

    let blocks = &cost.model_info.borrow().blocks;
    let mut possible_edge_assignments = vec![vec![]; blocks.len()];

    let tree = &cost.phylo.tree;
    let msa = &cost.phylo.msa;
    let v1_idx = &tree.node(v2_idx).parent.unwrap();
    let t1_mapping = if tree.parent(v1_idx).is_some() {
        Some(msa.ancestral_map(&cost.phylo.tree.parent(v1_idx).unwrap()))
    } else {
        None
    };

    let sibling = tree.sibling(v2_idx).unwrap();
    let t2_mapping = get_mapping_for_any_node(&cost.phylo.msa, &sibling);
    let t3_mapping = get_mapping_for_any_node(&cost.phylo.msa, &tree.children(v2_idx)[0]);
    let t4_mapping = get_mapping_for_any_node(&cost.phylo.msa, &tree.children(v2_idx)[1]);

    for (block_id, possible_edge_assignment) in possible_edge_assignments.iter_mut().enumerate() {
        let site = blocks[block_id] - 1;
        let t1_is_char = if let Some(t1_map) = &t1_mapping {
            t1_map[site].is_some()
        } else {
            false
        };
        let t2_is_char = t2_mapping[site].is_some();
        let t3_is_char = t3_mapping[site].is_some();
        let t4_is_char = t4_mapping[site].is_some();
        *possible_edge_assignment =
            possible_assignments_of_edge(t1_is_char, t2_is_char, t3_is_char, t4_is_char);
    }
    possible_edge_assignments
}

// single thread calculation
#[cfg(test)]
#[cfg(not(feature = "multi-thread"))]
fn brute_force_max<T: TKFModel + Send>(mut cost: TKFIndelCost<T, MASA>, v2_idx: &NodeIdx) -> f64 {
    let possible_edge_assignments = get_edge_assignment_possibilities(&cost, v2_idx);
    let number_of_possibilities = number_of_possibilities(&possible_edge_assignments);
    brute_force_max_for_possibilities_single_thread(
        possible_edge_assignments,
        number_of_possibilities,
        &mut cost,
        v2_idx,
    )
}

// multi thread  calculation
#[cfg(test)]
#[cfg(feature = "multi-thread")]
fn brute_force_max<T: TKFModel + Send>(mut cost: TKFIndelCost<T, MASA>, v2_idx: &NodeIdx) -> f64 {
    let possible_edge_assignments = get_edge_assignment_possibilities(&cost, v2_idx);
    let number_of_possibilities = number_of_possibilities(&possible_edge_assignments);
    let num_threads = rayon::current_num_threads();
    if number_of_possibilities < num_threads * 2 {
        brute_force_max_for_possibilities_single_thread(
            possible_edge_assignments,
            number_of_possibilities,
            &mut cost,
            v2_idx,
        )
    } else {
        brute_force_max_for_possibilities_multi_thread(
            possible_edge_assignments,
            number_of_possibilities,
            &cost,
            v2_idx,
            num_threads,
        )
    }
}

#[cfg(test)]
fn get_max_dp_reestimated<T: TKFModel>(mut cost: TKFIndelCost<T, MASA>, node_idx: &NodeIdx) -> f64 {
    let mut reassign = EdgeSeqsReestimator::new(&mut cost);
    // let factor_ns_before_reestimate = reassign.count_factor_ns_on_dirty_tree(node_idx);
    let backtracking_prob = reassign.reestimate(node_idx);

    // TODO: for statistic count
    //  - how many etas were changed
    //  - how many times did the msa not change
    cost.model_info.borrow_mut().valid.clear();
    let reestimated_cost = cost.logl();

    assert_relative_eq!(reestimated_cost, backtracking_prob, epsilon = 1e-9);
    reestimated_cost
}

// #[test]
// fn tkf92_compare_dp_vs_brute_force_for_file() {
//     // let msa = "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/outputname_TRUE_strip250.fasta";
//     let msa = "/Users/mrzi/Documents/develop/115-tkf_tree_search/rust-phylo/phylo/data/tkf_masas/fails.fasta";
//     let tree =
//         "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/tree_of_life.newick";
//     let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
//         .build_with_ancestors()
//         .unwrap();
//     assert!(masa_is_dollo(&phylo));
//     let lambda = 1.0;
//     let mu = 2.0;
//     let r = 0.3;
//
//     let tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
//         .build()
//         .unwrap();
//
//     // TODO: do i really need this call?
//     let _ = ModelSearchCost::cost(&tkf_cost);
//     compare_dp_vs_brute_force_for_every_internal_node(tkf_cost);
// }

// #[test]
// fn tkf92_compare_dp_vs_brute_force_for_file_and_node() {
//     // let msa = "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/outputname_TRUE_strip250.fasta";
//     let msa = "/Users/mrzi/Documents/develop/115-tkf_tree_search/rust-phylo/phylo/data/tkf_masas/fails.fasta";
//     let tree =
//         "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/tree_of_life.newick";
//     let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
//         .build_with_ancestors()
//         .unwrap();
//     assert!(masa_is_dollo(&phylo));
//     let lambda = 1.0;
//     let mu = 2.0;
//     let r = 0.3;
//
//     let tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
//         .build()
//         .unwrap();
//     let _ = ModelSearchCost::cost(&tkf_cost);
//     let node = &tkf_cost.phylo.tree.by_id("N343").idx;
//     let max_dp = get_max_dp_reestimated(tkf_cost.clone(), node);
//     let max_bf = find_brute_force_max(tkf_cost.clone(), node);
//
//     assert_relative_eq!(max_dp, max_bf, epsilon = 1e-12);
// }

#[cfg(test)]
fn log_iteration(iteration: usize, node_id: &str, prev_logl: f64, new_logl: f64) {
    println!(
        "Iteration {iteration}, node {node_id}, previous_logl = {prev_logl}, new_logl = {new_logl}, difference = {}",
        new_logl - prev_logl
    );
}

#[cfg(test)]
fn log_msa_unchanged(prev_phylo: &PhyloInfo<MASA>, new_phylo: &PhyloInfo<MASA>) {
    for check_node in new_phylo.tree.postorder() {
        if new_phylo.tree.node(check_node).children.is_empty() || check_node == &new_phylo.tree.root
        {
            continue;
        }
        let prev_v2_mapping = prev_phylo.msa.ancestral_map(check_node).clone();
        let new_v2_mapping = new_phylo.msa.ancestral_map(check_node).clone();
        let v1_idx = new_phylo.tree.node(check_node).parent.unwrap();
        let prev_v1_mapping = prev_phylo.msa.ancestral_map(&v1_idx).clone();
        let new_v1_mapping = new_phylo.msa.ancestral_map(&v1_idx).clone();
        if prev_v2_mapping != new_v2_mapping || prev_v1_mapping != new_v1_mapping {
            println!(" node {} changed msa", new_phylo.tree.node(check_node).id);
        }
    }
}

#[cfg(test)]
fn delete_existing_brute_force_max_file(file_path: &Path) {
    use std::fs;
    use std::path::Path;
    if Path::new(file_path).exists() {
        fs::remove_file(file_path).expect("Could not delete existing brute force max file");
    }
}

#[cfg(test)]
struct IterationInfo {
    iteration: usize,
    node_id: String,
    logl: f64,
}

#[cfg(test)]
fn load_precalculated_brute_force_maxes(file_path: &Path, iteration_info: &mut Vec<IterationInfo>) {
    use std::fs;
    let contents = fs::read_to_string(file_path);
    if contents.is_err() {
        println!("Precalculated brute force maxes file not found at {file_path:?}. Not loading any precalculated values.");
        return;
    }
    let rerun_hint = "Consider rerunning with recompute-brute-force-max-ancestral-seqs feature";
    for line in contents.unwrap().lines() {
        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() != 3 {
            panic!("Invalid line in precalculated brute force maxes file: {line}. It should contain 3 ',' separated columns. {rerun_hint}");
        }
        let iteration = parts[0]
            .parse::<usize>()
            .expect("Could not parse iteration number. {rerun_hint}");
        let node_id = parts[1].to_string();
        let logl = parts[2]
            .parse::<f64>()
            .expect("Could not parse logl value. {rerun_hint}");
        iteration_info.push(IterationInfo {
            iteration,
            node_id,
            logl,
        });
    }
}

#[cfg(test)]
pub fn append_result(path: &str, iter: usize, node: String, logl: f64) -> std::io::Result<()> {
    use std::{fs::OpenOptions, io::BufWriter, io::Write};
    let file = OpenOptions::new().create(true).append(true).open(path)?;

    let mut writer = BufWriter::new(file);

    writeln!(writer, "{iter},{node},{logl}")?;

    Ok(())
}

#[cfg(test)]
fn calc_or_load_brute_force_max(
    iteration: usize,
    iteration_info: &[IterationInfo],
    precalculated_file: &Path,
    tkf_cost: &TKFIndelCost<impl TKFModel + Send, MASA>,
    node: &NodeIdx,
) -> f64 {
    if iteration < iteration_info.len() {
        let saved = &iteration_info[iteration];
        let hint = "Consider rerunning with recompute-brute-force-max-ancestral-seqs feature";
        assert_eq!(
            saved.iteration, iteration,
            "Mismatched iteration number in save file {:?} at iteration {iteration}. {hint}",
            precalculated_file
        );
        assert_eq!(
            saved.node_id,
            tkf_cost.phylo.tree.node(node).id,
            "Mismatched node id in save file {:?} at iteration {iteration}. {hint}",
            precalculated_file
        );
        saved.logl
    } else {
        let max = brute_force_max(tkf_cost.clone(), node);
        append_result(
            precalculated_file.to_str().unwrap(),
            iteration,
            tkf_cost.phylo.tree.node(node).id.clone(),
            max,
        )
        .expect("Could not append brute force max to file");
        max
    }
}

#[test]
fn tkf92_reestimate_large_tree_for_file_iterative() {
    let dir = Path::new("data/tkf_brute_force_ancestors/simulation_01");
    let msa = dir.join("masa.fasta");
    let tree = dir.join("tree.newick");
    let precalculated_file = dir.join("precalculated_brute_force_maxes.csv");

    let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
        .build_with_ancestors()
        .unwrap();
    assert_masa_is_dollo(&phylo);
    let lambda = 1.0;
    let mu = 2.0;
    let r = 0.3;

    if cfg!(feature = "recompute-brute-force-max-ancestral-seqs") {
        delete_existing_brute_force_max_file(&precalculated_file);
    }
    let mut iteration_info = Vec::<IterationInfo>::new();
    load_precalculated_brute_force_maxes(&precalculated_file, &mut iteration_info);

    let mut tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo.clone())
        .build()
        .unwrap();
    let mut prev_logl = tkf_cost.clone().cost();

    // iterating over nodes in random order multiple times
    let mut rng = DefaultGenerator::new(41);
    let repeat = 5;
    let mut random_nodes = phylo
        .tree
        .postorder()
        .iter()
        .collect::<Vec<_>>()
        .repeat(repeat);
    rng.shuffle(&mut random_nodes);

    let mut reestimator = EdgeSeqsReestimator::new(&mut tkf_cost);
    let mut n_skipped = 0;
    for (iteration, node) in random_nodes.into_iter().enumerate() {
        if node == &phylo.tree.root || phylo.tree.node(node).children.is_empty() {
            n_skipped += 1;
            continue;
        }
        let previous_phylo = reestimator.get_phylo().clone();
        log_msa_unchanged(&previous_phylo, reestimator.get_phylo());

        let max_dp = reestimator.reestimate(node);
        assert!(max_dp >= prev_logl);
        let tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, reestimator.get_phylo().clone())
            .build()
            .unwrap();
        assert_masa_is_dollo(&tkf_cost.phylo);
        let new_logl = tkf_cost.cost();
        log_iteration(iteration, &phylo.tree.node(node).id, prev_logl, new_logl);
        assert_relative_eq!(max_dp, new_logl, epsilon = 1e-10);

        let max_brute_force = calc_or_load_brute_force_max(
            iteration - n_skipped,
            &iteration_info,
            &precalculated_file,
            &tkf_cost,
            node,
        );
        assert_relative_eq!(max_dp, max_brute_force, epsilon = 1e-10);
        iteration_info.push(IterationInfo {
            iteration,
            node_id: phylo.tree.node(node).id.clone(),
            logl: max_brute_force,
        });
        prev_logl = new_logl;
    }
}
