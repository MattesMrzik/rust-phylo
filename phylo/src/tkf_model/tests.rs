use std::cell::RefCell;
use std::path::Path;

use approx::assert_relative_eq;

use crate::alignment::{Alignment, Sequences, MASA};
use crate::optimisers::{NniOptimiser, TopologyOptimiser};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use crate::random::DefaultGenerator;
use crate::substitution_models::{QMatrixMaker, JC69};
use crate::tkf_model::reassignment::get_map_from_any_node;
use crate::tkf_model::{b, h1, log_i1, log_n1, n0};
use crate::{alignment::AncestralAlignment, substitution_models::QMatrix};
use crate::{record_wo_desc as record, tree};

use super::TKF92Cost;
use super::{get_blocks, TKF92Model, TKF92ModelInfo};

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

#[cfg(test)]
fn get_tkf_only_felsenstein(seqs: Sequences) -> f64 {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
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

    let logl = tkf_cost.logl();
    let half_manual = logl_without_node_values_without_felsenstein(&tkf_cost);
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
