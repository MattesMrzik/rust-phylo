use std::cell::RefCell;

use approx::assert_relative_eq;

use crate::alignment::{Alignment, AncestralAlignment, Mapping, Sequences, MASA};
use crate::likelihood::ModelSearchCost;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{SubstModel, SubstitutionCostBuilder as SCB, GTR};
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::{record_wo_desc as record, tree};

use super::*;

#[cfg(test)]
pub(crate) fn get_mapping_for_any_node<'a, AA: AncestralAlignment>(
    msa: &'a AA,
    node: &'a NodeIdx,
) -> &'a Mapping {
    match node {
        Internal(_) => msa.ancestral_map(node),
        Leaf(_) => msa.leaf_map(node),
    }
}

#[cfg(test)]
fn tkf_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF92IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let blocks = get_blocks(&phylo.msa);
    let tree = &phylo.tree;
    let l = model.lambda();
    let m = model.mu();
    let r = model.r();

    // for the root
    let mut prob: f64 = (1.0 - l / m).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    for (i, fragment) in blocks.iter().enumerate() {
        let mut x = 1.0;
        let fragment_len = if i == 0 {
            *fragment
        } else {
            fragment - blocks[i - 1]
        };
        if get_mapping_for_any_node(&phylo.msa, &phylo.tree.root)[fragment - 1].is_some() {
            // the eq seq at the root has a fragment
            x *= l / m * (1.0 - r) / r;
            prob += fragment_len as f64 * r.ln();
        }
        for node_idx in tree.postorder() {
            // skipping the root of the tree because it has no parent and therefore also no
            // mutations probabilities
            if node_idx == &tree.root {
                continue;
            }
            let node_id_value = usize::from(node_idx);

            let time = tree.node(node_idx).blen;
            let parent_id = &tree.node(node_idx).parent.unwrap();
            let parent_is_gap =
                get_mapping_for_any_node(&phylo.msa, parent_id)[fragment - 1].is_none();
            let current_is_gap =
                get_mapping_for_any_node(&phylo.msa, node_idx)[fragment - 1].is_none();

            let b = b(l, m, time);
            if i == 0 {
                prob += log_i1(l, b);
            }
            if parent_is_gap && current_is_gap {
                continue;
            } else if !parent_is_gap && !current_is_gap {
                // homolog block
                x *= h1(l, m, b, time);
                last_event_deletion[node_id_value] = false;
            } else if !parent_is_gap && current_is_gap {
                // deletion
                x *= n0(m, b);
                last_event_deletion[node_id_value] = true;
            } else if parent_is_gap && !current_is_gap {
                // insertion
                if last_event_deletion[node_id_value] {
                    prob += log_n1(l, m, b, time);
                    prob -= (l * b).ln();
                    prob -= n0(m, b).ln();
                }
                x *= l * b * (1.0 - r) / r;
                prob += fragment_len as f64 * r.ln();
                last_event_deletion[node_id_value] = false;
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

    // act
    let blocks = get_blocks(&msa);
    let block_lens = get_block_lens(&blocks);
    assert_eq!(blocks, vec![1, 3, 4, 5]);
    assert_eq!(block_lens, vec![1, 2, 1, 1]);
}

#[test]
fn tkf_indel_logl_() {
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
    let phylo = PhyloInfo {
        msa,
        tree: tree.clone(),
    };
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf_model = TKF92IndelModel {
        params: vec![lambda, mu, r],
    };
    let model_info = RefCell::new(TKF92IndelModelInfo::new(&phylo));
    let tkf_cost = TKF92IndelCost {
        model: tkf_model,
        phylo,
        model_info,
    };

    // act
    let logl = tkf_cost.logl();
    let half_manual = tkf_indel_logl_without_aggregation(&tkf_cost.model, &tkf_cost.phylo);
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
    // arrange
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let seqs = Sequences::new(vec![
        record!("A1", b"--GTGGA---"),
        record!("B2", b"-------NNA"),
        record!("I3", b"--T-------"),
        record!("C4", b"AGG-------"),
        record!("R5", b"--A-------"),
    ]);
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    let phylo = PhyloInfo {
        msa,
        tree: tree.clone(),
    };
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]);
    let subst_cost = SCB::new(subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf_cost = TKF92CostBuilder::new(lambda, mu, r, subst_model, phylo)
        .build()
        .unwrap();

    // act
    let logl = tkf_cost.cost();
    let subst_logl = subst_cost.cost();
    let tkf_logl =
        tkf_indel_logl_without_aggregation(&tkf_cost.tkf92_cost.model, &tkf_cost.tkf92_cost.phylo);

    // assert
    assert_relative_eq!(logl, subst_logl + tkf_logl, epsilon = 1e-12);
}

#[test]
fn tkf_indel_history_doesnt_change_felsenstein() {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let seqs = Sequences::new(vec![
        record!("A1", b"--GTGTA---"),
        record!("B2", b"-------AGT"),
        record!("I3", b"--N-------"),
        record!("C4", b"GTA-------"),
        record!("R5", b"--N-------"),
    ]);
    let seqs2 = Sequences::new(vec![
        record!("A1", b"--GTGTA---"),
        record!("B2", b"-------AGT"),
        record!("I3", b"--NNNNNNNN"),
        record!("C4", b"GTA-------"),
        record!("R5", b"--NNNNN---"),
    ]);
    let msa1 = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    let msa2 = MASA::from_aligned_with_ancestral(seqs2, &tree).unwrap();
    let phylo1 = PhyloInfo {
        msa: msa1,
        tree: tree.clone(),
    };
    let phylo2 = PhyloInfo {
        msa: msa2,
        tree: tree.clone(),
    };
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]);
    let tkf_cost1 = TKF92CostBuilder::new(lambda, mu, r, subst_model.clone(), phylo1)
        .build()
        .unwrap();

    let tkf_cost2 = TKF92CostBuilder::new(lambda, mu, r, subst_model, phylo2)
        .build()
        .unwrap();

    // act
    let tkf_log_1 = tkf_cost1.cost();
    let tkf_log_2 = tkf_cost2.cost();
    let tkf_indel_cost_1 = tkf_cost1.tkf92_cost.clone().cost();
    let tkf_indel_cost_without_agg_1 = tkf_indel_logl_without_aggregation(
        &tkf_cost1.tkf92_cost.model,
        &tkf_cost1.tkf92_cost.phylo,
    );
    let tkf_indel_cost_2 = tkf_cost2.tkf92_cost.clone().cost();
    let tkf_indel_cost_without_agg_2 = tkf_indel_logl_without_aggregation(
        &tkf_cost2.tkf92_cost.model,
        &tkf_cost2.tkf92_cost.phylo,
    );

    // assert
    assert_relative_eq!(tkf_indel_cost_1, tkf_indel_cost_without_agg_1);
    assert_relative_eq!(tkf_indel_cost_2, tkf_indel_cost_without_agg_2);
    assert_relative_eq!(tkf_log_1 - tkf_indel_cost_1, tkf_log_2 - tkf_indel_cost_2);
}
