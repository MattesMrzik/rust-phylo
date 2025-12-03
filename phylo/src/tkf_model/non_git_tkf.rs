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
    mapping_from_node_seq, reestimate_tests::masa_is_dollo, EdgeSeqsReestimator,
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
/// alignment, the outer vector is over blocks, the inner vector is over possible assignments for
/// the edge.
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
fn cost_for_edge_seqs<T: TKFModel>(
    v2_idx: &NodeIdx,
    cost: &mut TKFIndelCost<T, MASA>,
    edge_seqs: &[(bool, bool)],
) -> f64 {
    let v1_idx = cost.phylo.tree.node(v2_idx).parent.unwrap();
    let block_lens = cost.model_info.borrow().block_lengths.clone();
    let seq_len = cost.phylo.msa.len();

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

    let new_v1_mapping = mapping_from_node_seq(&v1_fixed_bit_set, &block_lens, seq_len);
    let new_v2_mapping = mapping_from_node_seq(&v2_fixed_bit_set, &block_lens, seq_len);
    cost.phylo.msa.update_ancestral_map(&v1_idx, new_v1_mapping);
    cost.phylo.msa.update_ancestral_map(v2_idx, new_v2_mapping);

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
fn brute_force_max_from_calculation_or_file<T: TKFModel + Send>(
    cost: &TKFIndelCost<T, MASA>,
    v2_idx: &NodeIdx,
    dir: &str,
    iteration: usize,
) -> f64 {
    use std::fs;
    use std::path::Path;

    let file_path = format!(
        "{}/brute_force_max_{}_{}.txt",
        dir,
        iteration,
        cost.phylo.tree.node(v2_idx).id
    );
    if Path::new(&file_path).exists() {
        let contents =
            fs::read_to_string(file_path).expect("Something went wrong reading the file");
        contents
            .trim()
            .parse::<f64>()
            .expect("Could not parse brute force max from file")
    } else {
        let max = brute_force_max(cost.clone(), v2_idx);
        let error_msg = format!("Unable to create directory {dir}, or file {file_path}");
        let error_msg = error_msg.as_str();
        fs::create_dir_all(dir).expect(error_msg);
        fs::write(file_path.clone(), max.to_string()).expect(error_msg);
        max
    }
}

#[cfg(test)]
fn get_max_dp_reestimated<T: TKFModel>(mut cost: TKFIndelCost<T, MASA>, node_idx: &NodeIdx) -> f64 {
    let mut reassign = EdgeSeqsReestimator::new(&mut cost);
    // let factor_ns_before_reestimate = reassign.count_factor_ns_on_dirty_tree(node_idx);
    let backtracking_prob = reassign.reestimate(node_idx);
    // let msa = reassign.get_phylo().msa.clone();
    // println!(
    //     "mapping v2 {}",
    //     msa.ancestral_map(node_idx)
    //         .iter()
    //         .map(|s| if s.is_some() { '1' } else { '0' })
    //         .collect::<String>()
    // );
    // println!("mapping v1 {}", {
    //     let v1_idx = cost.phylo.tree.node(node_idx).parent.unwrap();
    //     msa.ancestral_map(&v1_idx)
    //         .iter()
    //         .map(|s| if s.is_some() { '1' } else { '0' })
    //         .collect::<String>()
    // });

    // for statistic count the number of times the dp max is different to the original mapping
    // let mut reassigned_same_as_ori = true;
    cost.model_info.borrow_mut().valid.clear();
    let reestimated_cost = cost.logl();
    // let factor_ns_after_reestimate = reassign.count_factor_ns_on_dirty_tree(node_idx);

    // println!("checking dp table vs dp armgax cost");
    assert_relative_eq!(reestimated_cost, backtracking_prob, epsilon = 1e-9);
    // println!("factor_ns_after_reestimate = {factor_ns_after_reestimate}");
    // println!("factor_ns_before_reestimate = {factor_ns_before_reestimate}");
    reestimated_cost
}

// #[cfg(test)]
// fn compare_dp_vs_brute_force_for_every_internal_node<T: TKFModel + Send>(
//     cost: TKFIndelCost<T, MASA>,
// ) {
//     for v2_idx in cost.phylo.tree.postorder() {
//         // only re-estimate non root internal nodes
//         if v2_idx == &cost.phylo.tree.root || cost.phylo.tree.node(v2_idx).children.is_empty() {
//             continue;
//         }
//         let max_dp = get_max_dp_reestimated(cost.clone(), v2_idx);
//         let max_bf = find_brute_force_max(cost.clone(), v2_idx);
//         assert_relative_eq!(max_dp, max_bf, epsilon = 1e-12);
//     }
// }

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

#[test]
fn tkf92_reestimate_large_tree_for_file_iterative() {
    let msa = "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/outputname_TRUE_strip250.fasta";
    let tree =
        "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/tree_of_life.newick";
    let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
        .build_with_ancestors()
        .unwrap();
    let lambda = 1.0;
    let mu = 2.0;
    let r = 0.3;

    let repeat = 5;
    let dir_path = "/Users/mrzi/Documents/develop/115-tkf_tree_search/rust-phylo/phylo/data/tkf_masas/brute_force_maxes";

    if cfg!(feature = "recompute-brute-force-max-ancestral-seqs") {
        // delete existing brute force max files
        use std::fs;
        use std::path::Path;
        if Path::new(dir_path).exists() {
            fs::remove_dir_all(dir_path).expect("Could not remove existing brute force max dir");
        }
    }

    assert!(masa_is_dollo(&phylo));
    let mut tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo.clone())
        .build()
        .unwrap();
    let mut prev_logl = tkf_cost.clone().cost();
    let mut rng = DefaultGenerator::new(41);
    let mut random_nodes = phylo
        .tree
        .postorder()
        .iter()
        .collect::<Vec<_>>()
        .repeat(repeat);
    rng.shuffle(&mut random_nodes);

    let mut reestimator = EdgeSeqsReestimator::new(&mut tkf_cost);
    let mut previous_phylo = reestimator.get_phylo().clone();
    for (iteration, node) in random_nodes.into_iter().enumerate() {
        if node == &phylo.tree.root || phylo.tree.node(node).children.is_empty() {
            continue;
        }
        println!("\nreestimating node {}", phylo.tree.node(node).id);
        let backtrack_logl = reestimator.reestimate(node);
        assert!(masa_is_dollo(&reestimator.get_phylo().clone()));
        let mut stayed_same = true;
        for check_node in phylo.tree.postorder() {
            if phylo.tree.node(check_node).children.is_empty() || check_node == &phylo.tree.root {
                continue;
            }
            let prev_v2_mapping = previous_phylo.msa.ancestral_map(check_node).clone();
            let new_v2_mapping = reestimator
                .get_phylo()
                .msa
                .ancestral_map(check_node)
                .clone();
            let v1_idx = phylo.tree.node(check_node).parent.unwrap();
            let prev_v1_mapping = previous_phylo.msa.ancestral_map(&v1_idx).clone();
            let new_v1_mapping = reestimator.get_phylo().msa.ancestral_map(&v1_idx).clone();
            if prev_v2_mapping != new_v2_mapping || prev_v1_mapping != new_v1_mapping {
                stayed_same = false;
            }
        }
        if stayed_same {
            println!(" no change in msa");
        }
        previous_phylo = reestimator.get_phylo().clone();
        let tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, reestimator.get_phylo().clone())
            .build()
            .unwrap();
        let new_logl = tkf_cost.cost();

        let max_brute_force =
            brute_force_max_from_calculation_or_file(&tkf_cost, node, dir_path, iteration);
        println!(
            "backtrack_logl = {backtrack_logl}, new_logl = {new_logl}, max_brute_force = {max_brute_force}"
        );
        let non_stripped_phylo = strip_masa(reestimator.get_phylo(), 0, tkf_cost.phylo.msa.len());
        let non_stripped_tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, non_stripped_phylo)
            .build()
            .unwrap();
        assert_eq!(
            non_stripped_tkf_cost.cost(),
            tkf_cost.cost(),
            "costs differ after stripping msa"
        );
        if (backtrack_logl - new_logl).abs() > 1e-10 {
            println!("warning: backtrack_logl ({backtrack_logl}) != new_logl ({new_logl})");
            // println!("msa = {}", tkf_cost.phylo.msa);
            // also write this msa to a file
            use std::fs::File;
            use std::io::Write;
            let mut file = File::create("/Users/mrzi/Documents/develop/115-tkf_tree_search/rust-phylo/phylo/data/tkf_masas/fails.fasta").unwrap();
            write!(file, "{}", tkf_cost.phylo.msa).unwrap();
        }
        assert_relative_eq!(new_logl, max_brute_force, epsilon = 1e-10);
        assert!(masa_is_dollo(&tkf_cost.phylo));
        assert_relative_eq!(backtrack_logl, new_logl, epsilon = 1e-10);
        assert!(new_logl >= prev_logl);
        prev_logl = new_logl;
    }
}

// extract the phylo where it fails
// strip the msa and to see if it still fails
// first impl a helper that strips the msa
#[cfg(test)]
fn strip_masa<AA: AncestralAlignment>(
    phylo: &PhyloInfo<AA>,
    start: usize,
    end: usize,
) -> PhyloInfo<AA> {
    let mut records = Vec::new();
    use crate::{alignment::Sequences, record_wo_desc};
    for (node, leaf_map) in phylo.msa.leaf_maps() {
        let new_map = &leaf_map[start..end]
            .iter()
            .map(|s| if s.is_none() { "-" } else { "N" })
            .collect::<String>();
        records.push(record_wo_desc!(
            phylo.tree.node(node).id.as_str(),
            new_map.as_bytes()
        ));
    }
    for (node, anc_map) in phylo.msa.ancestral_maps() {
        let new_map = &anc_map[start..end]
            .iter()
            .map(|s| if s.is_none() { "-" } else { "N" })
            .collect::<String>();
        records.push(record_wo_desc!(
            phylo.tree.node(node).id.as_str(),
            new_map.as_bytes()
        ));
    }
    let seqs = Sequences::new(records);
    let msa = AA::from_aligned_with_ancestral(seqs, &phylo.tree).unwrap();
    PhyloInfo {
        msa,
        tree: phylo.tree.clone(),
    }
}
