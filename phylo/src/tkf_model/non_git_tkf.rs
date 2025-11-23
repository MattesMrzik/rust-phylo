use anyhow::{bail, Error};
use approx::assert_relative_eq;
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use rayon::prelude::*;

type Result<T> = std::result::Result<T, Error>;

use crate::alignment::{AncestralAlignment, MASA};
use crate::alphabets::dna_alphabet;
use crate::likelihood::ModelSearchCost;
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
    let v1_idx = cost.phylo.tree.node(v2_idx).parent.unwrap();
    let block_lens = cost.model_info.borrow().block_lengths.clone();
    let seq_len = cost.phylo.msa.len();

    // this probably does not work !!!

    let v1_fixed_bit_set = FixedBitSet::from_iter(edge_seqs.iter().map(|e| e.0 as usize));
    let v2_fixed_bit_set = FixedBitSet::from_iter(edge_seqs.iter().map(|e| e.1 as usize));

    let new_v1_mapping = mapping_from_node_seq(&v1_fixed_bit_set, &block_lens, seq_len);
    let new_v2_mapping = mapping_from_node_seq(&v2_fixed_bit_set, &block_lens, seq_len);
    cost.phylo.msa.update_ancestral_map(&v1_idx, new_v1_mapping);
    cost.phylo.msa.update_ancestral_map(v2_idx, new_v2_mapping);

    cost.model_info
        .borrow_mut()
        .valid
        .set(usize::from(v2_idx), false);
    cost.logl()
}

type IterOverPossibilities =
    std::iter::Enumerate<itertools::MultiProduct<std::vec::IntoIter<(bool, bool)>>>;

fn single_thread_dp_max<T: TKFModel>(
    possible_edge_assignments: IterOverPossibilities,
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
    println!("done");
    if let Some(m) = max {
        Ok(m)
    } else {
        bail!("no max found");
    }
}

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
        .map(move |((chunk_id, chunk), thread_cost)| {
            let mut local_max: Option<f64> = None;

            for i in chunk {
                let possibility = decode_index(i, &possible_edge_assignments);
                // let new_mapping = reassign.mapping_from_vec(&possibility);

                let current = cost_for_edge_seqs(v2_idx, &mut thread_cost.clone(), &possibility);

                local_max = Some(match local_max {
                    Some(m) => m.max(current),
                    None => current,
                });
                if chunk_id == 0 && (i + 1) % 10000 == 0 {
                    let percent = (i + 1) as f64 / chunk_size as f64 * 100.0;
                    // print!("calculating {i} of {number_of_possibilities}, which is {percent:.4}% \r");
                    println!(
                        "calculating {} of {number_of_possibilities}, which is {percent:.4}%",
                        i + 1
                    );
                    // let _ = io::stdout().flush();
                }
            }

            local_max.unwrap()
        })
        .collect();

    let global_max = chunk_maxes.into_iter().fold(f64::MIN, |a, b| a.max(b));
    Ok(global_max)
}

#[cfg(test)]
fn find_brute_force_max<T: TKFModel + Send>(
    mut cost: TKFIndelCost<T, MASA>,
    v2_idx: &NodeIdx,
    multi_threading: bool,
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

    if !multi_threading {
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
        let num_threads = rayon::current_num_threads();
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
fn get_max_dp_reestimated<T: TKFModel>(
    mut cost: TKFIndelCost<T, MASA>,
    node_idx: &NodeIdx,
    assert_check: bool,
) -> f64 {
    let mut reassign = EdgeSeqsReestimator::new(&mut cost);
    // let factor_ns_before_reestimate = reassign.count_factor_ns_on_dirty_tree(node_idx);
    let backtracking_prob = reassign.reestimate(node_idx);

    // for statistic count the number of times the dp max is different to the original mapping
    // let mut reassigned_same_as_ori = true;
    cost.model_info
        .borrow_mut()
        .valid
        .set(usize::from(node_idx), false);
    let reestimated_cost = cost.logl();
    // let factor_ns_after_reestimate = reassign.count_factor_ns_on_dirty_tree(node_idx);

    // println!("checking dp table vs dp armgax cost");
    if assert_check {
        assert_relative_eq!(reestimated_cost, backtracking_prob, epsilon = 1e-12);
    }
    // println!("factor_ns_after_reestimate = {factor_ns_after_reestimate}");
    // println!("factor_ns_before_reestimate = {factor_ns_before_reestimate}");
    reestimated_cost
}

#[cfg(test)]
fn compare_dp_vs_brute_force_for_every_internal_node<T: TKFModel + Send>(
    cost: TKFIndelCost<T, MASA>,
    multi_threading: bool,
) {
    for v2_idx in cost.phylo.tree.postorder() {
        // only re-estimate non root internal nodes
        if v2_idx == &cost.phylo.tree.root || cost.phylo.tree.node(v2_idx).children.is_empty() {
            continue;
        }
        let max_dp = get_max_dp_reestimated(cost.clone(), v2_idx, true);
        let max_brute_force = find_brute_force_max(cost.clone(), v2_idx, multi_threading);
        match max_brute_force {
            Ok(max_bf) => {
                // println!(
                //     "node {}: dp max = {}, brute force max = {}",
                //     cost.phylo.tree.node(v2_idx).id,
                //     max_dp,
                //     max_bf
                // );
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
fn tkf91_compare_dp_vs_brute_force() {
    let phylo = setup_test_phylo(dna_alphabet());
    let tkf_cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.5, phylo)
        .build()
        .unwrap();
    let _ = ModelSearchCost::cost(&tkf_cost);
    compare_dp_vs_brute_force_for_every_internal_node(tkf_cost, true);
}
