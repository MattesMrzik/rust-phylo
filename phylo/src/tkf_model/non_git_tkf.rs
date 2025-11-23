use anyhow::{bail, Error};
use approx::assert_relative_eq;
use fixedbitset::FixedBitSet;
use itertools::Itertools;

#[cfg(feature = "par-regraft")]
use rayon::prelude::*;

type Result<T> = std::result::Result<T, Error>;

use crate::alignment::{AncestralAlignment, MASA};
use crate::alphabets::dna_alphabet;
use crate::likelihood::ModelSearchCost;
#[cfg(test)]
use crate::phylo_info::PhyloInfo;
use crate::phylo_info::PhyloInfoBuilder;
use crate::random::DefaultGenerator;
use crate::tkf_model::reestimate_tests::masa_is_dollo;
use crate::tkf_model::tests::setup_test_phylo;
use crate::tkf_model::EdgeSeqsReestimator;
use crate::tkf_model::{TKF92IndelCostBuilder, TKFIndelCost, TKFModel};
use crate::tree::NodeIdx;
use crate::{alignment::Alignment, tkf_model::mapping_from_node_seq};

#[cfg(test)]
fn decode_index(
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
        vec![(true, true), (false, false)],
    ];
    let number_of_possibilities: usize = possible_edge_assignments
        .iter()
        .map(|poss| poss.len())
        .product();
    assert_eq!(number_of_possibilities, 384);
    let mut all_possibilities = Vec::with_capacity(number_of_possibilities);
    for i in 0..number_of_possibilities {
        all_possibilities.push(decode_index(i, &possible_edge_assignments));
    }
    let unique: Vec<_> = all_possibilities.iter().unique().collect();
    assert_eq!(unique.len(), number_of_possibilities);
}

fn print_progress(i: usize, number_of_possibilities: usize) {
    if (i + 1) % 10000 == 0 {
        let percent = i as f64 / number_of_possibilities as f64 * 100.0;
        println!("calculating {i} of {number_of_possibilities}, which is {percent:.4}% ");
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
fn too_many_possibilities(number_of_possibilities: usize) -> Result<()> {
    if number_of_possibilities > 10000000 {
        println!("too many possibilities to brute force: {number_of_possibilities}");
        bail!("too many possibilities to brute force: {number_of_possibilities}",);
    } else {
        println!("calculation of {number_of_possibilities} possibilities");
        Ok(())
    }
}

#[cfg(test)]
fn cost_for_edge_seqs<T: TKFModel>(
    v2_idx: &NodeIdx,
    cost: &mut TKFIndelCost<T, MASA>,
    edge_seqs: &[(bool, bool)],
) -> f64 {
    use crate::likelihood::TreeSearchCost;

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

    // cost.model_info
    //     .borrow_mut()
    //     .valid
    //     .set(usize::from(v2_idx), false);
    // more save than above but slower
    // cost.model_info.borrow_mut().valid.clear();
    for child in &cost.phylo.tree.node(v2_idx).children {
        cost.model_info
            .borrow_mut()
            .valid
            .set(usize::from(child), false);
    }
    cost.model_info
        .borrow_mut()
        .valid
        .set(usize::from(cost.tree().sibling(v2_idx).unwrap()), false);
    cost.logl()
}

fn single_thread_dp_max<T: TKFModel>(
    possible_edge_assignments: std::iter::Enumerate<
        itertools::MultiProduct<std::vec::IntoIter<(bool, bool)>>,
    >,
    number_of_possibilities: usize,
    cost: &mut TKFIndelCost<T, MASA>,
    v2_idx: &NodeIdx,
) -> Result<f64> {
    let mut max: Option<f64> = None;

    for (i, possibility) in possible_edge_assignments {
        print_progress(i, number_of_possibilities);

        let current = cost_for_edge_seqs(v2_idx, cost, &possibility);

        if let Some(ref mut m) = max {
            if current > *m {
                *m = current;
                // arg_max = Some(possibility);
            }
        } else {
            max = Some(current);
            // arg_max = Some(possibility);
        }
    }
    println!("single thread calculation done");
    if let Some(m) = max {
        Ok(m)
    } else {
        bail!("no max found");
    }
}

#[cfg(feature = "par-regraft")]
fn multi_thread_dp_max(
    possible_edge_assignments: Vec<Vec<(bool, bool)>>,
    number_of_possibilities: usize,
    cost: &TKFIndelCost<impl TKFModel + Send, MASA>,
    v2_idx: &NodeIdx,
    num_threads: usize,
) -> Result<f64> {
    let chunk_size = number_of_possibilities.div_ceil(num_threads);
    let cost_clones = vec![cost.clone(); num_threads];
    let chunk_maxes: Vec<f64> = (0..number_of_possibilities)
        .into_par_iter()
        .chunks(chunk_size)
        .enumerate()
        .zip(cost_clones)
        .into_par_iter()
        .map(move |((chunk_id, chunk), mut thread_cost)| {
            let mut local_max: Option<f64> = None;

            for i in chunk {
                let possibility = decode_index(i, &possible_edge_assignments);
                // let new_mapping = reassign.mapping_from_vec(&possibility);

                let current = cost_for_edge_seqs(v2_idx, &mut thread_cost, &possibility);

                local_max = Some(match local_max {
                    Some(m) => m.max(current),
                    None => current,
                });
                if chunk_id == 0 {
                    print_progress(i, number_of_possibilities / num_threads);
                }
            }

            local_max.unwrap()
        })
        .collect();

    let global_max = chunk_maxes.into_iter().fold(f64::MIN, |a, b| a.max(b));
    Ok(global_max)
}

#[cfg(test)]
#[cfg(feature = "par-regraft")]
fn find_brute_force_max<T: TKFModel + Send>(
    mut cost: TKFIndelCost<T, MASA>,
    v2_idx: &NodeIdx,
) -> Result<f64> {
    let n_blocks = cost.model_info.borrow().blocks.len();
    let mut reassign = EdgeSeqsReestimator::new(&mut cost);
    reassign.prepare_for_dp(v2_idx);
    let mut possible_edge_assignments = vec![vec![]; n_blocks];

    for (block_id, possible_edge_assignment) in possible_edge_assignments.iter_mut().enumerate() {
        *possible_edge_assignment = reassign.possible_assignments_of_nni_edge(block_id);
    }

    let number_of_possibilities = number_of_possibilities(&possible_edge_assignments);
    too_many_possibilities(number_of_possibilities)?;

    let num_threads = rayon::current_num_threads();
    if number_of_possibilities < num_threads * 2 {
        let possible_edge_seqs = possible_edge_assignments
            .into_iter()
            .multi_cartesian_product()
            .enumerate();
        single_thread_dp_max(
            possible_edge_seqs,
            number_of_possibilities,
            &mut cost,
            v2_idx,
        )
    } else {
        // let num_threads = 10;
        println!("using {num_threads} threads");
        multi_thread_dp_max(
            possible_edge_assignments,
            number_of_possibilities,
            &cost,
            v2_idx,
            num_threads,
        )
    }
}

#[cfg(test)]
#[cfg(not(feature = "par-regraft"))]
fn find_brute_force_max<T: TKFModel + Send>(
    mut cost: TKFIndelCost<T, MASA>,
    v2_idx: &NodeIdx,
) -> Result<f64> {
    let n_blocks = cost.model_info.borrow().blocks.len();
    let mut reassign = EdgeSeqsReestimator::new(&mut cost);
    reassign.prepare_for_dp(v2_idx);
    let mut possible_edge_assignments = vec![vec![]; n_blocks];

    for (block_id, possible_edge_assignment) in possible_edge_assignments.iter_mut().enumerate() {
        *possible_edge_assignment = reassign.possible_assignments_of_nni_edge(block_id);
    }

    let number_of_possibilities = number_of_possibilities(&possible_edge_assignments);
    too_many_possibilities(number_of_possibilities)?;
    let possible_edge_seqs = possible_edge_assignments
        .into_iter()
        .multi_cartesian_product()
        .enumerate();
    single_thread_dp_max(
        possible_edge_seqs,
        number_of_possibilities,
        &mut cost,
        v2_idx,
    )
}

#[cfg(test)]
fn get_max_dp_reestimated<T: TKFModel>(
    mut cost: TKFIndelCost<T, MASA>,
    node_idx: &NodeIdx,
    assert_check: bool,
) -> f64 {
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
    if assert_check {
        assert_relative_eq!(reestimated_cost, backtracking_prob, epsilon = 1e-9);
    }
    // println!("factor_ns_after_reestimate = {factor_ns_after_reestimate}");
    // println!("factor_ns_before_reestimate = {factor_ns_before_reestimate}");
    reestimated_cost
}

#[cfg(test)]
fn compare_dp_vs_brute_force_for_every_internal_node<T: TKFModel + Send>(
    cost: TKFIndelCost<T, MASA>,
) {
    for v2_idx in cost.phylo.tree.postorder() {
        // only re-estimate non root internal nodes
        if v2_idx == &cost.phylo.tree.root || cost.phylo.tree.node(v2_idx).children.is_empty() {
            continue;
        }
        let max_dp = get_max_dp_reestimated(cost.clone(), v2_idx, true);
        let max_brute_force = find_brute_force_max(cost.clone(), v2_idx);
        match max_brute_force {
            Ok(max_bf) => {
                // println!(
                //     "node {}: dp max = {}, brute force max = {}",
                //     cost.phylo.tree.node(v2_idx).id,
                //     max_dp,
                //     max_bf
                // );
                // if (max_dp - max_bf).abs() > 1e-12 {
                println!(
                    "node {}: dp max = {}, brute force max = {}",
                    cost.phylo.tree.node(v2_idx).id,
                    max_dp,
                    max_bf
                );
                // }
                assert_relative_eq!(max_dp, max_bf, epsilon = 1e-12);
            }
            Err(e) => {
                println!(
                    "skipping brute force for node {} due to error: {}",
                    cost.phylo.tree.node(v2_idx).id,
                    e
                );
            }
        }
    }
}

#[test]
fn tkf92_compare_dp_vs_brute_force() {
    let phylo = setup_test_phylo(dna_alphabet());
    let tkf_cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.5, phylo)
        .build()
        .unwrap();
    let _ = ModelSearchCost::cost(&tkf_cost);
    compare_dp_vs_brute_force_for_every_internal_node(tkf_cost);
}

#[test]
fn tkf92_compare_dp_vs_brute_force_for_file() {
    // let msa = "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/outputname_TRUE_strip250.fasta";
    let msa = "/Users/mrzi/Documents/develop/115-tkf_tree_search/rust-phylo/phylo/data/tkf_masas/fails.fasta";
    let tree =
        "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/tree_of_life.newick";
    let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
        .build_with_ancestors()
        .unwrap();
    assert!(masa_is_dollo(&phylo));
    let lambda = 1.0;
    let mu = 2.0;
    let r = 0.3;

    let tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
        .build()
        .unwrap();

    // TODO: do i really need this call?
    let _ = ModelSearchCost::cost(&tkf_cost);
    compare_dp_vs_brute_force_for_every_internal_node(tkf_cost);
}

#[test]
fn tkf92_compare_dp_vs_brute_force_for_file_and_node() {
    // let msa = "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/outputname_TRUE_strip250.fasta";
    let msa = "/Users/mrzi/Documents/develop/115-tkf_tree_search/rust-phylo/phylo/data/tkf_masas/fails.fasta";
    let tree =
        "/Users/mrzi/Documents/develop/58-TKF92/rust-phylo/phylo/data/runtime/tree_of_life.newick";
    let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
        .build_with_ancestors()
        .unwrap();
    assert!(masa_is_dollo(&phylo));
    let lambda = 1.0;
    let mu = 2.0;
    let r = 0.3;

    let tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
        .build()
        .unwrap();
    let _ = ModelSearchCost::cost(&tkf_cost);
    let node = &tkf_cost.phylo.tree.by_id("N343").idx;
    let max_dp = get_max_dp_reestimated(tkf_cost.clone(), node, true);
    let max_brute_force = find_brute_force_max(tkf_cost.clone(), node);
    match max_brute_force {
        Ok(max_bf) => {
            // println!(
            //     "node {}: dp max = {}, brute force max = {}",
            //     cost.phylo.tree.node(v2_idx).id,
            //     max_dp,
            //     max_bf
            // );
            // if (max_dp - max_bf).abs() > 1e-12 {
            println!(
                "node {}: dp max = {}, brute force max = {}",
                tkf_cost.phylo.tree.node(node).id,
                max_dp,
                max_bf
            );
            // }
            assert_relative_eq!(max_dp, max_bf, epsilon = 1e-12);
        }
        Err(e) => {
            println!(
                "skipping brute force for node {} due to error: {}",
                tkf_cost.phylo.tree.node(node).id,
                e
            );
        }
    }
}

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

    assert!(masa_is_dollo(&phylo));
    let mut tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo.clone())
        .build()
        .unwrap();
    let mut prev_logl = tkf_cost.clone().cost();
    let mut rng = DefaultGenerator::new(41);
    let mut random_nodes = phylo.tree.postorder().iter().collect::<Vec<_>>().repeat(repeat);
    rng.shuffle(&mut random_nodes);

    let mut reestimator = EdgeSeqsReestimator::new(&mut tkf_cost);
    let mut previous_phylo = reestimator.get_phylo().clone();
    for node in random_nodes {
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
            println!("🔴 no change in msa");
        }
        previous_phylo = reestimator.get_phylo().clone();
        let tkf_cost = TKF92IndelCostBuilder::new(lambda, mu, r, reestimator.get_phylo().clone())
            .build()
            .unwrap();
        let new_logl = tkf_cost.cost();
        let max_brute_force = find_brute_force_max(tkf_cost.clone(), node).unwrap();
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
