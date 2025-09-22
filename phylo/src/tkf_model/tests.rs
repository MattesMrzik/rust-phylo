use std::cell::RefCell;
use std::path::Path;

use anyhow::bail;
use approx::assert_relative_eq;
use itertools::Itertools;
// use this a a optional in the cargo.toml if the flag par_chunk is passed
use rayon::prelude::*;

use crate::alignment::{Alignment, Sequences, MASA};
use crate::optimisers::{NniOptimiser, TopologyOptimiser};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use crate::random::DefaultGenerator;
use crate::substitution_models::GTR;
use crate::substitution_models::{QMatrixMaker, JC69};
use crate::tkf_model::{
    b, h1, log_i1, log_n1, n0,
    reassignment::{
        get_allowed_assignments, get_map_from_any_node, get_mapping_from_vec, ReassignEdge,
    },
};
use crate::tkf_model::{get_blocks, TKF92Cost, TKF92Model, TKF92ModelInfo};
use crate::tree::{NodeIdx, Tree};
use crate::{alignment::AncestralAlignment, substitution_models::QMatrix};
use crate::{record_wo_desc as record, tree, Result};

#[cfg(test)]
fn logl_without_node_values_without_felsenstein<Q: QMatrix, AA: AncestralAlignment>(
    cost: &TKF92Cost<Q, AA>,
) -> f64 {
    let blocks = get_blocks(&cost.phylo.msa);
    let tree = &cost.phylo.tree;
    let model = &cost.model;
    let virtual_root = &cost.model_info.borrow().virtual_root;
    let l = model.lambda();
    let m = model.mu();
    let r = model.r();

    // for the root
    let mut prob: f64 = (1.0 - l / m).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    let mut last_event_insertion = vec![false; tree.len()];
    for (i, fragment) in blocks.iter().enumerate() {
        let mut x = 1.0;
        let fragment_len = if i == 0 {
            *fragment
        } else {
            fragment - blocks[i - 1]
        };
        if get_map_from_any_node(&cost.phylo.msa, virtual_root)[fragment - 1].is_some() {
            // the eq seq at the root has a fragment
            x *= l / m * (1.0 - r) / r;
            prob += fragment_len as f64 * r.ln();
        }
        for node_idx in tree.postorder() {
            // skipping the actual root of the tree bc it has no parent and therefore also no
            // mutations probabilities
            if node_idx == &tree.root {
                continue;
            }
            let node_id_value = usize::from(node_idx);

            let time = tree.node(node_idx).blen;
            let parent_id = &tree.node(node_idx).parent.unwrap();
            let mut parent_is_gap =
                get_map_from_any_node(&cost.phylo.msa, parent_id)[fragment - 1].is_none();
            let mut current_is_gap =
                get_map_from_any_node(&cost.phylo.msa, node_idx)[fragment - 1].is_none();

            if cost.model_info.borrow().edge_is_time_reversed[usize::from(node_idx)] {
                // println!("this edge is time reversed {node_idx}");
                std::mem::swap(&mut parent_is_gap, &mut current_is_gap);
            }

            let b = b(l, m, time);
            if i == 0 {
                prob += log_i1(l, b);
            }
            if parent_is_gap && current_is_gap {
                continue;
            }
            if !parent_is_gap && !current_is_gap {
                // homolog block
                x *= h1(l, m, b, time);
                last_event_deletion[node_id_value] = false;
                last_event_insertion[node_id_value] = false;
            }
            if !parent_is_gap && current_is_gap {
                // deletion
                x *= n0(m, b);
                if last_event_insertion[node_id_value]
                    && cost.model_info.borrow().edge_is_time_reversed[node_id_value]
                {
                    prob += log_n1(l, m, b, time);
                    prob -= (l * b).ln();
                    prob -= n0(m, b).ln();
                }
                last_event_deletion[node_id_value] = true;
                last_event_insertion[node_id_value] = false;
            }
            if parent_is_gap && !current_is_gap {
                // insertion
                if last_event_deletion[node_id_value]
                    && !cost.model_info.borrow().edge_is_time_reversed[node_id_value]
                {
                    prob += log_n1(l, m, b, time);
                    prob -= (l * b).ln();
                    prob -= n0(m, b).ln();
                }
                x *= l * b * (1.0 - r) / r;
                prob += fragment_len as f64 * r.ln();
                last_event_deletion[node_id_value] = false;
                last_event_insertion[node_id_value] = true;
            }
        }
        prob += x.ln();
        prob += (fragment_len - 1) as f64 * (1.0 + x).ln();
    }
    prob
}

#[test]
fn tkf_beta() {
    assert_relative_eq!(b(0.3, 0.5, 0.7), 0.5461782813185221)
}

#[test]
fn tkf_log_i1() {
    // arrange
    let l = 2.0;
    let m = 3.0;
    let time = 1.0;
    let b = b(l, m, time);
    // act & assert
    assert_relative_eq!(log_i1(l, b), -0.8172396554020775) // log((1-2(1-e^(-1))/(3-2*e^(-1)))
}

#[test]
fn tkf_log_n1() {
    // arrange
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = b(l, m, time);
    // act & assert
    assert_relative_eq!(log_n1(l, m, b, time), -2.732135332549935)
    // log((1-e^(-1.5) - 3(1-e^(-.5))/(3-2*e^(-.5)) )* (1-2(1-e^(-.5))/(3-2*e^(-.5)))   (2(1-e^(-1))/(3-2*e^(-1)))^0)
}

#[test]
fn tkf_n0() {
    // arrange
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = b(l, m, time);
    // act & assert
    assert_relative_eq!(n0(m, b), 0.6605755607027574) // (3(1-e^(-.5))/(3-2*e^(-.5)))
}

#[test]
fn tkf_h1() {
    // arrange
    let l = 2.0;
    let m = 3.0;
    let time = 1.5;
    let b = b(l, m, time);
    // act & assert
    assert_relative_eq!(h1(l, m, b, time), 0.004350089645603061)
    // e^(-4.5) * (1-2(1-e^(-1.5))/(3-2*e^(-1.5)))
}

#[test]
fn tkf_get_blocks() {
    // arrange
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new(vec![
        record!("A0", b"AAB-D"),
        record!("B1", b"-ARAW"),
        record!("I1", b"AAA-A"),
    ]);

    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    // act & assert
    assert_eq!(get_blocks(&msa), vec![1, 3, 4, 5]);
}

#[test]
fn tkf_logl_without_substitution() {
    // arrange
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let seqs = Sequences::new(vec![
        record!("A1", b"--NNNNN---"),
        record!("B2", b"-------NNN"),
        record!("I3", b"--N-------"),
        record!("C4", b"NNN-------"),
        record!("R5", b"--N-------"),
    ]);
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    let m = msa.len() as f64;
    let pyhlo = PhyloInfo {
        msa,
        tree: tree.clone(),
    };
    let q = JC69::create(&[], &[]);
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf_model = TKF92Model {
        q,
        params: vec![lambda, mu, r],
    };
    let model_info = RefCell::new(TKF92ModelInfo::new(&pyhlo, &tkf_model));
    let tkf_cost = TKF92Cost {
        model: tkf_model,
        phylo: pyhlo,
        model_info,
    };

    // act
    let logl = tkf_cost.logl();
    let half_manual = logl_without_node_values_without_felsenstein(&tkf_cost);
    let mut manual_calculation = 0.0;
    manual_calculation += (1.0 - lambda / mu).ln();
    manual_calculation += m * r.ln();
    // immortal links
    manual_calculation += log_i1(lambda, b(lambda, mu, tree.by_id("A1").blen));
    manual_calculation += log_i1(lambda, b(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += log_i1(lambda, b(lambda, mu, tree.by_id("I3").blen));
    manual_calculation += log_i1(lambda, b(lambda, mu, tree.by_id("C4").blen));
    // first block
    let x = lambda * b(lambda, mu, tree.by_id("C4").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 1.0 * (1.0 + x).ln();
    // second block
    let mut x = lambda / mu * (1.0 - r) / r;
    x *= h1(
        lambda,
        mu,
        b(lambda, mu, tree.by_id("C4").blen),
        tree.by_id("C4").blen,
    );
    x *= h1(
        lambda,
        mu,
        b(lambda, mu, tree.by_id("A1").blen),
        tree.by_id("A1").blen,
    );
    x *= h1(
        lambda,
        mu,
        b(lambda, mu, tree.by_id("I3").blen),
        tree.by_id("I3").blen,
    );
    x *= n0(mu, b(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += x.ln();
    // third block
    let x = lambda * b(lambda, mu, tree.by_id("C4").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 3.0 * (1.0 + x).ln();
    // fourth block
    let x = lambda * b(lambda, mu, tree.by_id("B2").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 2.0 * (1.0 + x).ln();
    manual_calculation += log_n1(
        lambda,
        mu,
        b(lambda, mu, tree.by_id("B2").blen),
        tree.by_id("B2").blen,
    );
    manual_calculation -= n0(mu, b(lambda, mu, tree.by_id("B2").blen)).ln();
    manual_calculation -= (lambda * b(lambda, mu, tree.by_id("B2").blen)).ln();

    // assert
    assert_relative_eq!(logl, manual_calculation);
    assert_relative_eq!(logl, half_manual);
}

#[test]
fn tkf_logl_with_substitution() {
    // do i even need the felsenstein in my cost or could i simply add in to my cost in the topo
    // opti
    // I will first test my current impl agaist the cft test logl and if
    // all is good then make a commit such that i have this working version
    // and then i can impl a indel tkf and make tkf cost just the indel cost
    // plus the substitution cost and also just call update tree on both
}

#[test]
fn tkf() {
    let _ = env_logger::builder().is_test(true).try_init();
    let fldr = Path::new("./data/");
    let phylo = PhyloInfoBuilder::with_attrs(
        fldr.join("sequences_DNA1.fasta"),
        fldr.join("tree_multiple.newick"),
    )
    .build_with_ancestors()
    .unwrap();
    let q = JC69::create(&[], &[]);
    let tkf_model = TKF92Model {
        q,
        params: [0.1, 0.2, 0.3].to_vec(),
    };
    let model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));

    let tkf_cost = TKF92Cost {
        model: tkf_model,
        phylo,
        model_info,
    };
    let move_opti = NniOptimiser {};
    let rng = &DefaultGenerator::default();
    let topo_opti = TopologyOptimiser::new(tkf_cost, move_opti, rng);
    let result = topo_opti.run().unwrap();
    println!("test print final cost {}", result.final_cost);
    println!("test print msa = {}", result.cost.phylo.msa)
}

fn get_gtr_tkf_cost(tree: Tree, seqs: Sequences) -> TKF92Cost<GTR, MASA> {
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    let phylo = PhyloInfo { msa, tree };
    get_gtr_tkf_cost_from_phylo(phylo)
}
#[cfg(test)]
fn get_gtr_tkf_cost_from_phylo(phylo: PhyloInfo<MASA>) -> TKF92Cost<GTR, MASA> {
    let q = GTR::create(&[0.3; 4], &[0.5; 5]);
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf_model = TKF92Model {
        q,
        params: vec![lambda, mu, r],
    };
    let model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));
    TKF92Cost {
        model: tkf_model,
        phylo,
        model_info,
    }
}

#[cfg(test)]
fn get_tkf_only_felsenstein(seqs: Sequences) -> f64 {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let cost = get_gtr_tkf_cost(tree, seqs);

    let logl = cost.logl();
    let half_manual = logl_without_node_values_without_felsenstein(&cost);
    logl - half_manual
}

#[test]
fn tkf_indel_history_doesnt_change_felsenstein() {
    // julijas impl does by nature not depend on it since it works with msa not masa
    let seqs = Sequences::new(vec![
        record!("A1", b"--GTGTA---"),
        record!("B2", b"-------AGT"),
        record!("I3", b"--N-------"),
        record!("C4", b"GTA-------"),
        record!("R5", b"--N-------"),
    ]);
    let felsenstein_logl_1 = get_tkf_only_felsenstein(seqs);
    let seqs2 = Sequences::new(vec![
        record!("A1", b"--GTGTA---"),
        record!("B2", b"-------AGT"),
        record!("I3", b"--NNNNNNNN"),
        record!("C4", b"GTA-------"),
        record!("R5", b"--NNNNN---"),
    ]);
    let felsenstein_logl_2 = get_tkf_only_felsenstein(seqs2);
    assert_relative_eq!(felsenstein_logl_1, felsenstein_logl_2, epsilon = 1e-12);
}

#[cfg(test)]
fn decode_index(mut idx: usize, possible_edge_assignments: &Vec<Vec<[bool; 2]>>) -> Vec<[bool; 2]> {
    let mut tuple = Vec::with_capacity(possible_edge_assignments.len());
    for poss in possible_edge_assignments {
        let choice = idx % poss.len();
        tuple.push(poss[choice]);
        idx /= poss.len();
    }
    tuple
}

#[test]
fn test_decode_index() {
    let possible_edge_assignments = vec![
        vec![[true, true], [true, false], [false, true], [false, false]], // 4
        vec![[true, true], [false, false]],                               // 2
        vec![[true, true], [true, false], [false, true]],                 // 3
    ];
    let number_of_possibilities: usize = possible_edge_assignments
        .iter()
        .map(|poss| poss.len())
        .product();
    assert_eq!(number_of_possibilities, 24);
    let mut all_possibilities = Vec::with_capacity(number_of_possibilities);
    for i in 0..number_of_possibilities {
        all_possibilities.push(decode_index(i, &possible_edge_assignments));
    }
    let unique: Vec<_> = all_possibilities.iter().unique().collect();
    assert_eq!(unique.len(), number_of_possibilities);
}

#[cfg(test)]
fn find_brute_force_max<Q: QMatrix + Send>(
    cost: TKF92Cost<Q, MASA>,
    v2_idx: &NodeIdx,
) -> Result<f64> {
    let mut reassign = ReassignEdge::new(cost);
    let mut possible_edge_assignments: Vec<Vec<[bool; 2]>> =
        vec![vec![]; reassign.cost.model_info.borrow().blocks.len()];

    for block_id in 0..reassign.cost.model_info.borrow().blocks.len() {
        let (t1, t2, t3, t4) = reassign.are_chars_at_leafs(v2_idx, block_id);
        possible_edge_assignments[block_id] = get_allowed_assignments(t1, t2, t3, t4);
    }

    let number_of_possibilities: usize = possible_edge_assignments
        .iter()
        .map(|poss| poss.len())
        .product();

    if number_of_possibilities > 1000000 {
        bail!("too many possibilities to brute force: {number_of_possibilities}",);
    } else {
        println!("calculation of {number_of_possibilities} possibilities");
    }

    // this is the not parallel version
    let mut max: Option<f64> = None;
    // let mut arg_max: Option<Vec<[bol; 2]>> = None;
    let v1_idx = &reassign.cost.phylo.tree.node(v2_idx).parent.unwrap();
    let block_lens = &reassign.cost.model_info.borrow().block_lens;
    for (i, possibility) in possible_edge_assignments
        .into_iter()
        .multi_cartesian_product()
        .enumerate()
    {
        // print!("calculating {} of {}\r", i, possibilities.len());
        if (i + 1) % 10000 == 0 {
            let percent = i as f64 / number_of_possibilities as f64 * 100.0;
            // print!("calculating {i} of {number_of_possibilities}, which is {percent:.4}% \r");
            print!("calculating {i} of {number_of_possibilities}, which is {percent:.4}% ");
            // let _ = io::stdout().flush();
        }

        let new_mapping = get_mapping_from_vec(v2_idx, v1_idx, &possibility, block_lens);
        for (node_idx, map) in new_mapping {
            reassign.cost.phylo.msa.update_ancestral_map(&node_idx, map);
        }
        reassign.cost.model_info.borrow_mut().valid = false;
        let current = reassign.cost.logl();
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

    // let num_threads = rayon::current_num_threads();
    // println!("using {num_threads} threads");
    // let chunk_size = number_of_possibilities.div_ceil(num_threads);
    // let cost_clones = vec![reassign.cost.clone(); num_threads];
    // let block_lens = &reassign.cost.model_info.borrow().block_lens;
    // let v1_idx = &reassign.cost.phylo.tree.node(v2_idx).parent.unwrap();
    // let chunk_maxes: Vec<f64> = (0..number_of_possibilities)
    //     .into_par_iter()
    //     .chunks(chunk_size)
    //     .zip(cost_clones.into_par_iter())
    //     .map(move |(chunk, mut thread_cost)| {
    //         let mut local_max: Option<f64> = None;
    //
    //         for i in chunk {
    //             let possibility = decode_index(i, &possible_edge_assignments);
    //             let new_mapping = get_mapping_from_vec(v2_idx, v1_idx, &possibility, block_lens);
    //
    //             for (node_idx, map) in new_mapping {
    //                 thread_cost.phylo.msa.update_ancestral_map(&node_idx, map);
    //             }
    //             thread_cost.model_info.borrow_mut().valid = false;
    //             let current = thread_cost.logl();
    //
    //             local_max = Some(match local_max {
    //                 Some(m) => m.max(current),
    //                 None => current,
    //             });
    //         }
    //
    //         local_max.unwrap()
    //     })
    //     .collect();
    //
    // let global_max = chunk_maxes.into_iter().fold(f64::MIN, |a, b| a.max(b));
    // println!("the brute force max = {}", max.unwrap());
    // println!("and the argmax is:");
    // for max_assignment in &arg_max.unwrap() {
    //     println!("{max_assignment:?}");
    // }
    // Ok(global_max)
}

#[cfg(test)]
fn get_max_reestimated<Q: QMatrix>(
    cost: TKF92Cost<Q, MASA>,
    node_idx: &NodeIdx,
) -> (f64, bool, i32) {
    let mut reassign = ReassignEdge::new(cost.clone());
    let factor_ns_before_reestimate = reassign.count_factor_ns_on_dirty_tree(node_idx);
    reassign.fill_dp(node_idx);
    let (new_mapping, backtracking_prob) = reassign.get_mapping_from_backtracking(node_idx);
    // for statistic count the number of times the dp max is different to the original mapping
    let mut reassigned_same_as_ori = true;
    for (node, map) in new_mapping {
        if map != *cost.phylo.msa.ancestral_map(&node) {
            reassigned_same_as_ori = false;
        }
        reassign.cost.phylo.msa.update_ancestral_map(&node, map);
    }
    reassign.cost.model_info.borrow_mut().valid = false;
    let reestimated_cost = reassign.cost.logl();
    let factor_ns_after_reestimate = reassign.count_factor_ns_on_dirty_tree(node_idx);

    // println!("checking dp table vs dp armgax cost");
    assert_relative_eq!(reestimated_cost, backtracking_prob, epsilon = 1e-12);
    // println!("factor_ns_after_reestimate = {factor_ns_after_reestimate}");
    // println!("factor_ns_before_reestimate = {factor_ns_before_reestimate}");
    let diff_in_number_of_factor_ns = factor_ns_after_reestimate - factor_ns_before_reestimate;
    (
        reestimated_cost,
        reassigned_same_as_ori,
        diff_in_number_of_factor_ns,
    )
}

#[test]
fn tkf_reassignment() {
    let fldr = Path::new("./data");
    let phylo = PhyloInfoBuilder::with_attrs(
        fldr.join("outputname_TRUE.fasta"),
        fldr.join("tkf_tree.newick"),
    )
    .build_with_ancestors()
    .unwrap();

    let cost = get_gtr_tkf_cost_from_phylo(phylo);
    /*let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let seqs = Sequences::new(vec![
        record!("A1", b"A"),
        record!("B2", b"A"),
        record!("I3", b"A"),
        record!("C4", b"A"),
        record!("R5", b"A"),
    ]);
    let cost = get_gtr_tkf_cost(tree, seqs);*/
    // println!("original cost = {}", cost.logl());

    let postorder = cost.phylo.tree.postorder().clone();
    // (pa(v2_idx), v2_idx) is the edge we want to re-estimate

    let mut number_of_dp_correct = 0;
    let mut numer_of_same_as_ori = 0;
    let mut number_of_diff_factor_ns = 0;
    for v2_idx in &postorder {
        // only re-estimate non root internal nodes
        if v2_idx == &cost.phylo.tree.root || cost.phylo.tree.node(v2_idx).children.is_empty() {
            continue;
        }
        let node_id = cost.phylo.tree.node(v2_idx).id.clone();
        // if node_id != "N20" {
        //     continue;
        // }
        // println!("\n\n\ndoing re-estimation at node {node_id}");
        let (max_dp, same_as_ori, diff_in_number_of_factor_ns) =
            get_max_reestimated(cost.clone(), v2_idx);
        // println!(
        //     "node factor ns = {:?}",
        //     cost.model_info.borrow().node_factor_n
        // );
        numer_of_same_as_ori += same_as_ori as usize;
        number_of_diff_factor_ns += diff_in_number_of_factor_ns.abs();
        let max_brute_force_opt = find_brute_force_max(cost.clone(), v2_idx);
        match max_brute_force_opt {
            Ok(max_brute_force) => {
                assert_relative_eq!(max_dp, max_brute_force);
                number_of_dp_correct += 1;
            }
            Err(_) => continue,
        }
    }
    println!("number of dp correct = {number_of_dp_correct}");
    println!(
        "skipped brute force on = {}",
        postorder.len() - 1 - cost.phylo.tree.n - number_of_dp_correct
    );
    println!("number of same as ori = {numer_of_same_as_ori}");
    println!("diff in number of factor ns = {number_of_diff_factor_ns}");
}

// TODO: use precomputed

#[test]
fn todo() {
    // i could check the dp table by returning all valid paths in it
    // by starting in the back and doing some recursive calling of the
    // backtracking pointers and collecting all the paths ie mappings
    // and calculate their prob an compare against the last col
    // although since i only save the max pointer i wont get no more paths
    // then non infty values in the last col
    // so instead i could (if probs are tied) save more than one max back pointer
}
