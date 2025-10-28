use std::fs::{self};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use approx::assert_relative_eq;

use crate::alignment::{Alignment, AncestralAlignment, Mapping, Sequences, MASA};
use crate::alphabets::{dna_alphabet, protein_alphabet, Alphabet};
use crate::likelihood::ModelSearchCost;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use crate::substitution_models::{QMatrixMaker, SubstModel, SubstitutionCostBuilder as SCB};
use crate::substitution_models::{BLOSUM, GTR, HIVB, HKY, JC69, K80, TN93, WAG};
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::{frequencies, record_wo_desc as record, tree};

use super::*;

#[test]
fn tkf_dummy_freqs() {
    assert_eq!(&*DUMMY_FREQS, &DVector::<f64>::zeros(0));
}

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
fn tkf91_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF91IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let tree = &phylo.tree;
    let l = model.lambda();
    let m = model.mu();

    // for the root
    let mut prob: f64 = (1.0 - l / m).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    for i in 0..phylo.msa.len() {
        let mut x = 1.0;
        if get_mapping_for_any_node(&phylo.msa, &phylo.tree.root)[i].is_some() {
            // the eq seq at the root has a fragment
            x *= l / m;
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
            let parent_is_gap = get_mapping_for_any_node(&phylo.msa, parent_id)[i].is_none();
            let current_is_gap = get_mapping_for_any_node(&phylo.msa, node_idx)[i].is_none();

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
                x *= l * b;
                last_event_deletion[node_id_value] = false;
            }
        }
        prob += x.ln();
    }
    prob
}

#[cfg(test)]
fn tkf92_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF92IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let blocks = TKF92IndelModel::get_blocks(&phylo.msa);
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
fn tkf91_get_blocks() {
    // arrange
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new(vec![
        record!("A0", b"AAAB-D"),
        record!("B1", b"--ARAW"),
        record!("I1", b"AAAA-A"),
    ]);
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    // act
    let blocks = TKF91IndelModel::get_blocks(&msa);
    let block_lens = get_block_lens(&blocks);
    assert_eq!(blocks, (1..msa.len() + 1).collect::<Vec<usize>>());
    assert_eq!(block_lens, vec![1; 6]);
}

#[test]
fn tkf92_get_blocks() {
    // arrange
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new(vec![
        record!("A0", b"AAB-D"),
        record!("B1", b"-ARAW"),
        record!("I1", b"AAA-A"),
    ]);

    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    // act
    let blocks = TKF92IndelModel::get_blocks(&msa);
    let block_lens = get_block_lens(&blocks);
    assert_eq!(blocks, vec![1, 3, 4, 5]);
    assert_eq!(block_lens, vec![1, 2, 1, 1]);
}

#[cfg(test)]
fn tkf_set_lambda_and_mu<T: TKF>(model: &mut T) {
    let initial_lambda = model.lambda();
    let initial_mu = model.mu();
    // lambda
    assert!(!model.set_param(0, -1.0));
    assert_eq!(model.lambda(), initial_lambda);
    assert!(!model.set_param(0, 0.0));
    assert_eq!(model.lambda(), initial_lambda);
    assert!(model.set_param(0, initial_mu - 0.001));
    assert_eq!(model.lambda(), initial_mu - 0.001);
    assert!(!model.set_param(0, initial_mu));
    assert_eq!(model.lambda(), initial_mu - 0.001);
    assert!(!model.set_param(0, 2.1));
    assert_eq!(model.lambda(), initial_mu - 0.001);
    // mu
    model.set_param(0, initial_lambda); // reset lambda
    assert!(!model.set_param(1, -0.1));
    assert_eq!(model.mu(), initial_mu);
    assert!(!model.set_param(1, 0.0));
    assert_eq!(model.mu(), initial_mu);
    assert!(!model.set_param(1, initial_lambda));
    assert_eq!(model.mu(), initial_mu);
    assert!(model.set_param(1, initial_lambda + 0.001));
    assert_eq!(model.mu(), initial_lambda + 0.001);
    assert!(model.set_param(1, initial_lambda + 100.0));
    assert_eq!(model.mu(), initial_lambda + 100.0);
}

#[cfg(test)]
fn tkf_set_r(model: &mut TKF92IndelModel) {
    let initial_r = model.r();
    assert!(!model.set_param(2, -0.1));
    assert_eq!(model.r(), initial_r);
    assert!(!model.set_param(2, 0.0));
    assert_eq!(model.r(), initial_r);
    assert!(model.set_param(2, 0.5));
    assert_eq!(model.r(), 0.5);
    assert!(model.set_param(2, 0.9999));
    assert_eq!(model.r(), 0.9999);
    assert!(!model.set_param(2, 1.0));
    assert_eq!(model.r(), 0.9999);
    assert!(!model.set_param(2, 2.0));
    assert_eq!(model.r(), 0.9999);
}
#[test]
fn tkf91_indel_set_param() {
    // arrange
    let mut tkf91_indel_model = TKF91IndelModel {
        params: vec![1.0, 2.0],
    };
    // act & assert
    tkf_set_lambda_and_mu(&mut tkf91_indel_model);
    assert!(!tkf91_indel_model.set_param(2, 0.5)); // no r parameter
}

#[test]
fn tkf92_indel_set_param() {
    // arrange
    let mut tkf92_indel_model = TKF92IndelModel {
        params: vec![1.0, 2.0, 0.3],
    };
    // act & assert
    tkf_set_lambda_and_mu(&mut tkf92_indel_model);
    tkf_set_r(&mut tkf92_indel_model);
}

#[test]
fn tkf_indel_get_and_set_params_and_freqs() {
    let mut tkf_indel_cost =
        TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, setup_test_phylo(dna_alphabet()))
            .build()
            .unwrap();
    // params
    assert_eq!(tkf_indel_cost.params(), vec![1.0, 2.0, 0.3]);
    assert_eq!(tkf_indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_indel_cost.model.r(), 0.3);
    tkf_indel_cost.set_param(2, 0.33);
    assert_eq!(tkf_indel_cost.params(), vec![1.0, 2.0, 0.33]);
    // freqs
    assert_eq!(tkf_indel_cost.freqs(), &*DUMMY_FREQS);
    assert_eq!(
        tkf_indel_cost.empirical_freqs(),
        setup_test_phylo(dna_alphabet()).freqs()
    );
}

#[test]
fn tkf_get_and_set_params() {
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8, 0.9]);
    let mut tkf_cost =
        TKF92CostBuilder::new(1.0, 2.0, 0.3, subst_model, setup_test_phylo(dna_alphabet()))
            .build()
            .unwrap();
    assert_eq!(
        tkf_cost.params(),
        vec![1.0, 2.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    );
    assert_eq!(tkf_cost.indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_cost.indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_cost.indel_cost.model.r(), 0.3);
    tkf_cost.set_param(2, 0.33);
    tkf_cost.set_param(5, 0.77);
    tkf_cost.set_param(0, -5.0); // invalid, should not change
    tkf_cost.set_param(1, 0.1); // invalid, should not change
    tkf_cost.set_param(2, 10.0); // invalid, should not change
    assert_eq!(
        tkf_cost.params(),
        vec![1.0, 2.0, 0.33, 0.5, 0.6, 0.77, 0.8, 0.9]
    );
    assert_eq!(
        tkf_cost.empirical_freqs(),
        setup_test_phylo(dna_alphabet()).freqs()
    );
}

#[cfg(test)]
fn validate_lambda_mu(l: f64, m: f64, l_expected: f64, m_expected: f64) {
    let cost = TKF91IndelCostBuilder::new(l, m, setup_test_phylo(dna_alphabet()))
        .build()
        .unwrap();
    assert_eq!(cost.model.lambda(), l_expected);
    assert_eq!(cost.model.mu(), m_expected);
    let cost = TKF92IndelCostBuilder::new(l, m, 0.1, setup_test_phylo(dna_alphabet()))
        .build()
        .unwrap();
    assert_eq!(cost.model.lambda(), l_expected);
    assert_eq!(cost.model.mu(), m_expected);
}

#[cfg(test)]
fn validate_r(r: f64, r_expected: f64) {
    let cost = TKF92IndelCostBuilder::new(1.0, 2.0, r, setup_test_phylo(dna_alphabet()))
        .build()
        .unwrap();
    assert_eq!(cost.model.r(), r_expected);
}

#[test]
fn tkf_validate_params_for_builder() {
    validate_lambda_mu(-1.0, -2.0, DEFAULT_LAMBDA, DEFAULT_MU);
    validate_lambda_mu(0.0, 2.0, DEFAULT_LAMBDA_MU_RATIO * 2.0, 2.0);
    validate_lambda_mu(2.0, -0.1, 2.0, 2.0 / DEFAULT_LAMBDA_MU_RATIO);
    validate_lambda_mu(2.0, 1.9999, 2.0, 2.0 / DEFAULT_LAMBDA_MU_RATIO);
    validate_lambda_mu(1.2, 1.21, 1.2, 1.21);
    validate_r(-0.5, DEFAULT_R);
    validate_r(0.0, DEFAULT_R);
    validate_r(1.0, DEFAULT_R);
    validate_r(1.5, DEFAULT_R);
    validate_r(0.1, 0.1);
}

#[test]
fn tkf91_model_fmt() {
    // arrange
    let tkf_indel_model = TKF91IndelModel {
        params: vec![1.1, 2.0, 3.0],
    };

    // act
    let fmt = format!("{}", tkf_indel_model);

    // assert
    assert_eq!(fmt, "TKF91 with lambda = 1.1, mu = 2");
}

#[test]
fn tkf91_indel_cost_fmt() {
    // arrange
    let tkf_indel_cost = TKF91IndelCostBuilder::new(1.0, 2.0, setup_test_phylo(dna_alphabet()))
        .build()
        .unwrap();

    // act
    let fmt = format!("{}", tkf_indel_cost);

    // assert
    assert_eq!(fmt, "TKF91 with lambda = 1, mu = 2");
}

#[test]
fn tkf91_cost_fmt() {
    // arrange
    let subst_model = SubstModel::<JC69>::new(&[], &[]);
    let tkf_cost = TKF91CostBuilder::new(1.0, 2.0, subst_model, setup_test_phylo(dna_alphabet()))
        .build()
        .unwrap();

    // act
    let fmt = format!("{}", tkf_cost);

    // assert
    assert_eq!(fmt, "TKF91 with lambda = 1, mu = 2 and JC69");
}

#[test]
fn tkf92_indel_cost_fmt() {
    // arrange
    let tkf_indel_cost =
        TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, setup_test_phylo(dna_alphabet()))
            .build()
            .unwrap();

    // act
    let fmt = format!("{}", tkf_indel_cost);

    // assert
    assert_eq!(fmt, "TKF92 with lambda = 1, mu = 2, r = 0.3");
}

#[test]
fn tkf92_model_fmt() {
    // arrange
    let tkf_indel_model = TKF92IndelModel {
        params: vec![1.1, 2.0, 3.0],
    };

    // act
    let fmt = format!("{}", tkf_indel_model);

    // assert
    assert_eq!(fmt, "TKF92 with lambda = 1.1, mu = 2, r = 3");
}

#[test]
fn tkf92_cost_fmt() {
    // arrange
    let subst_model = SubstModel::<JC69>::new(&[], &[]);
    let tkf_cost =
        TKF92CostBuilder::new(1.0, 2.0, 0.3, subst_model, setup_test_phylo(dna_alphabet()))
            .build()
            .unwrap();

    // act
    let fmt = format!("{}", tkf_cost);

    // assert
    assert_eq!(fmt, "TKF92 with lambda = 1, mu = 2, r = 0.3 and JC69");
}

#[test]
fn tkf_get_and_set_freqs() {
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8, 0.9]);
    let mut tkf_cost =
        TKF92CostBuilder::new(1.0, 2.0, 0.3, subst_model, setup_test_phylo(dna_alphabet()))
            .build()
            .unwrap();
    assert_eq!(tkf_cost.freqs().as_slice(), &[0.1, 0.2, 0.3, 0.4]);
    tkf_cost.set_freqs(frequencies!(&[0.4, 0.3, 0.2, 0.1]));
    assert_eq!(tkf_cost.freqs().as_slice(), &[0.4, 0.3, 0.2, 0.1]);
}

#[test]
fn tkf91_indel_logl_() {
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
    let phylo = PhyloInfo {
        msa,
        tree: tree.clone(),
    };
    let lambda = 0.1;
    let mu = 0.2;
    let tkf91_cost = TKF91IndelCostBuilder::new(lambda, mu, phylo)
        .build()
        .unwrap();

    // act
    let logl = tkf91_cost.logl();
    let half_manual = tkf91_indel_logl_without_aggregation(&tkf91_cost.model, &tkf91_cost.phylo);
    let mut manual_calculation = 0.0;
    manual_calculation += (1.0 - lambda / mu).ln();
    // immortal links
    manual_calculation += log_i1(lambda, b(lambda, mu, tree.by_id("A1").blen));
    manual_calculation += log_i1(lambda, b(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += log_i1(lambda, b(lambda, mu, tree.by_id("I3").blen));
    manual_calculation += log_i1(lambda, b(lambda, mu, tree.by_id("C4").blen));
    // first block
    let x = lambda * b(lambda, mu, tree.by_id("C4").blen);
    manual_calculation += x.ln() * 2.0;
    // second block
    let mut x = lambda / mu;
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
    let x = lambda * b(lambda, mu, tree.by_id("C4").blen);
    manual_calculation += x.ln() * 4.0;
    // fourth block
    let x = lambda * b(lambda, mu, tree.by_id("B2").blen);
    manual_calculation += x.ln() * 3.0;
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
fn tkf92_indel_logl_() {
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
    let tkf92_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
        .build()
        .unwrap();

    // act
    let logl = tkf92_cost.logl();
    let half_manual = tkf92_indel_logl_without_aggregation(&tkf92_cost.model, &tkf92_cost.phylo);
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
fn tkf_cost_builder_fails() {
    // arrange
    let phylo = setup_test_phylo(protein_alphabet());
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]);

    // act
    let tkf91_cost = TKF91CostBuilder::new(0.1, 0.2, subst_model.clone(), phylo.clone())
        .build()
        .unwrap_err()
        .to_string();
    let tkf92_cost = TKF92CostBuilder::new(0.1, 0.2, 0.3, subst_model, phylo)
        .build()
        .unwrap_err()
        .to_string();

    // assert
    assert_eq!(tkf91_cost, "Alphabet mismatch between model and alignment");
    assert_eq!(tkf92_cost, "Alphabet mismatch between model and alignment");
}

#[test]
fn tkf91_logl_with_substitution() {
    // arrange
    let phylo = setup_test_phylo(dna_alphabet());
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]);
    let subst_cost = SCB::new(subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let lambda = 0.1;
    let mu = 0.2;
    let tkf_cost = TKF91CostBuilder::new(lambda, mu, subst_model, phylo)
        .build()
        .unwrap();

    // act
    let logl = tkf_cost.cost();
    let subst_logl = subst_cost.cost();
    let tkf_logl = tkf91_indel_logl_without_aggregation(
        &tkf_cost.indel_cost.model,
        &tkf_cost.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(logl, subst_logl + tkf_logl, epsilon = 1e-12);
}

#[test]
fn tkf92_logl_with_substitution() {
    // arrange
    let phylo = setup_test_phylo(dna_alphabet());
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
    let tkf_logl = tkf92_indel_logl_without_aggregation(
        &tkf_cost.indel_cost.model,
        &tkf_cost.indel_cost.phylo,
    );

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
    let tkf_indel_cost_1 = tkf_cost1.indel_cost.clone().cost();
    let tkf_indel_cost_without_agg_1 = tkf92_indel_logl_without_aggregation(
        &tkf_cost1.indel_cost.model,
        &tkf_cost1.indel_cost.phylo,
    );
    let tkf_indel_cost_2 = tkf_cost2.indel_cost.clone().cost();
    let tkf_indel_cost_without_agg_2 = tkf92_indel_logl_without_aggregation(
        &tkf_cost2.indel_cost.model,
        &tkf_cost2.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(tkf_indel_cost_1, tkf_indel_cost_without_agg_1);
    assert_relative_eq!(tkf_indel_cost_2, tkf_indel_cost_without_agg_2);
    assert_relative_eq!(tkf_log_1 - tkf_indel_cost_1, tkf_log_2 - tkf_indel_cost_2);
}

#[cfg(test)]
fn setup_test_phylo(alphabet: Alphabet) -> PhyloInfo<MASA> {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::with_alphabet(
            vec![
                record!("A1", b"--GTGGA---"),
                record!("B2", b"-------NNA"),
                record!("I3", b"--T-------"),
                record!("C4", b"AGG-------"),
                record!("R5", b"--A-------"),
            ],
            alphabet,
        ),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[cfg(test)]
enum Submodel {
    TKFIndel,
    SubstitutionModel,
}

#[cfg(test)]
fn modify_model_params_costs_match_template<Q: QMatrix + QMatrixMaker>(
    alphabet: Alphabet,
    model_to_change: Submodel,
) {
    // arrange
    let phylo = setup_test_phylo(alphabet);
    let subst_original_param = 1.0;
    let subst_changed_param = 0.5;
    let subst_model = SubstModel::<Q>::new(&[], &[subst_original_param]);
    let tkf_original_mu = 0.2;
    let tkf_changed_mu = 0.25;
    let mut tkf_cost = TKF92CostBuilder::new(0.1, tkf_original_mu, 0.3, subst_model, phylo.clone())
        .build()
        .unwrap();

    // act & assert
    // sanity check
    let logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl, ModelSearchCost::cost(&tkf_cost));

    // The likelihood should change if we change model parameters
    match model_to_change {
        Submodel::TKFIndel => tkf_cost.set_param(1, tkf_changed_mu),
        Submodel::SubstitutionModel => tkf_cost.set_param(3, subst_changed_param),
    }

    let logl2 = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl2, ModelSearchCost::cost(&tkf_cost));
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let (new_subst_param, new_tkf_mu) = match model_to_change {
        Submodel::TKFIndel => (subst_original_param, tkf_changed_mu),
        Submodel::SubstitutionModel => (subst_changed_param, tkf_original_mu),
    };
    let subst_model = SubstModel::<Q>::new(&[], &[new_subst_param]);
    let tkf_cost = TKF92CostBuilder::new(0.1, new_tkf_mu, 0.3, subst_model, phylo)
        .build()
        .unwrap();
    let new_logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(new_logl, ModelSearchCost::cost(&tkf_cost));
    assert_eq!(logl2, new_logl);
}

#[test]
fn tkf_modify_model_params_costs_match() {
    // For these models the substitution model has parameters
    // We test changing both the substitution model and the TKF92 parameters
    modify_model_params_costs_match_template::<K80>(dna_alphabet(), Submodel::SubstitutionModel);
    modify_model_params_costs_match_template::<HKY>(dna_alphabet(), Submodel::SubstitutionModel);
    modify_model_params_costs_match_template::<TN93>(dna_alphabet(), Submodel::SubstitutionModel);
    modify_model_params_costs_match_template::<GTR>(dna_alphabet(), Submodel::SubstitutionModel);

    modify_model_params_costs_match_template::<K80>(dna_alphabet(), Submodel::TKFIndel);
    modify_model_params_costs_match_template::<HKY>(dna_alphabet(), Submodel::TKFIndel);
    modify_model_params_costs_match_template::<TN93>(dna_alphabet(), Submodel::TKFIndel);
    modify_model_params_costs_match_template::<GTR>(dna_alphabet(), Submodel::TKFIndel);

    // For these models the substitution model has no parameters
    // We only test changing the TKF92 parameters
    modify_model_params_costs_match_template::<JC69>(dna_alphabet(), Submodel::TKFIndel);
    modify_model_params_costs_match_template::<WAG>(protein_alphabet(), Submodel::TKFIndel);
    modify_model_params_costs_match_template::<BLOSUM>(protein_alphabet(), Submodel::TKFIndel);
    modify_model_params_costs_match_template::<HIVB>(protein_alphabet(), Submodel::TKFIndel);
}

fn find_fasta_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry_result in fs::read_dir(dir).unwrap() {
        let entry = entry_result.unwrap();
        let path = entry.path();
        if path.is_dir() {
            // recurse into subdirectory
            find_fasta_files(&path, files);
        } else if path
            .extension()
            .map(|ext| ext == "fasta" || ext == "aln")
            .unwrap_or(false)
        {
            files.push(path);
        }
    }
}

#[test]
fn tkf_plot_r() {
    let base_dir = Path::new("./data/benchmark-datasets/");
    let n = 1000;
    let output_dir = Path::new("r_out");
    fs::create_dir_all(output_dir).unwrap();

    let mut fasta_files = Vec::new();
    find_fasta_files(base_dir, &mut fasta_files);

    for path in fasta_files {
        let file_name = path.file_stem().unwrap().to_string_lossy(); // file name without extension

        // i want the two folder names above the file
        let folder_one_up_name = path
            .parent()
            .and_then(|p| p.file_name())
            .unwrap()
            .to_str()
            .unwrap();
        let folder_two_up_name = path
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.file_name())
            .unwrap()
            .to_str()
            .unwrap();
        println!("  Processing file: {:?}", file_name);

        let out_file_name = format!(
            "{}_{}_{}.csv",
            folder_two_up_name, folder_one_up_name, file_name
        );
        println!("    Output file: {}", out_file_name);
        let out_file = output_dir.join(out_file_name);
        let file = fs::File::create(&out_file).unwrap();
        let mut writer = BufWriter::new(file);

        // CSV header
        let _ = writeln!(writer, "r,logl");

        let phylo = PhyloInfoBuilder::new(&path).build_with_ancestors().unwrap();

        for i in 1..n {
            let r = (i as f64) / (n as f64);
            let tkf_indel_cost = TKF92IndelCostBuilder::new(0.1, 0.2, r, phylo.clone())
                .build()
                .unwrap();
            let logl = tkf_indel_cost.logl();

            let _ = writeln!(writer, "{},{}", r, logl);
        }

        let _ = writer.flush();
    }
}

#[test]
fn tkf_test_single_benchmark_file() {
    let path = Path::new("./data/benchmark-datasets/dna/easy/wickd3a_7771.processed.fasta");
    let phylo = PhyloInfoBuilder::new(path)
        .build_with_ancestors_strip(Some((824, 825)))
        .unwrap();
    let tkf_indel_cost = TKF92IndelCostBuilder::new(0.1, 0.2, 0.3, phylo)
        .build()
        .unwrap();
    let logl = tkf_indel_cost.logl();
    println!("Log-likelihood: {}", logl);
}
