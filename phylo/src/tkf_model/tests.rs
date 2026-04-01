use approx::assert_relative_eq;
use assert_matches::assert_matches;
use nalgebra::DVector;
use rstest::rstest;

use crate::alignment::{Alignment, AncestralAlignment, Mapping, Sequences, MASA};
use crate::alphabets::Alphabet;
use crate::likelihood::{
    ModelSearchCost, PARAM_RANGE_POSITIVE, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE,
};
use crate::optimisers::rooted_nni;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{QMatrixMaker, SubstModel, SubstitutionCostBuilder as SCB};
use crate::substitution_models::{BLOSUM, GTR, HIVB, HKY, JC69, K80, TN93, WAG};
use crate::tkf_model::tkf92::TKF92IndelModel;
use crate::tkf_model::tkf92_fixed_fragmentation::TKF92FixedIndelModel;
use crate::tkf_model::tkf_indel::DUMMY_FREQS;
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::{frequencies, record_wo_desc as record, tree, Error};

use super::*;

#[test]
fn tkf_dummy_freqs() {
    assert_eq!(&*DUMMY_FREQS, &DVector::<f64>::zeros(0));
}

#[cfg(test)]
pub(super) fn get_mapping_for_any_node<'a, AA: AncestralAlignment>(
    msa: &'a AA,
    node: &'a NodeIdx,
) -> &'a Mapping {
    match node {
        Internal(_) => msa.ancestral_map(node),
        Leaf(_) => msa.leaf_map(node),
    }
}

// This is a direct implementation of the TKF91 log likelihood calculation without any
// aggregation over subtrees and without substitutions. This direct calculation is sufficient
// if one is only interested in the indel likelihood for a fixed alignment and tree.
// Used for testing purposes only, i.e., to validate the aggregated implementation.
#[cfg(test)]
fn tkf91_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF91IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let tree = &phylo.tree;
    let lambda = model.lambda();
    let mu = model.mu();

    // for the root
    let mut prob: f64 = (1.0 - lambda / mu).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    for i in 0..phylo.msa.len() {
        let mut event_prob = 1.0;
        if get_mapping_for_any_node(&phylo.msa, &phylo.tree.root)[i].is_some() {
            // the eq seq at the root has a fragment
            event_prob *= lambda / mu;
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

            let beta = naive_beta(lambda, mu, time);
            if i == 0 {
                prob += ln_i1(lambda, beta.ln());
            }
            if parent_is_gap && current_is_gap {
                continue;
            } else if !parent_is_gap && !current_is_gap {
                // homolog block
                event_prob *= naive_h1(lambda, mu, beta, time);
                last_event_deletion[node_id_value] = false;
            } else if !parent_is_gap && current_is_gap {
                // deletion
                event_prob *= n0(mu, beta);
                last_event_deletion[node_id_value] = true;
            } else if parent_is_gap && !current_is_gap {
                // insertion
                if last_event_deletion[node_id_value] {
                    prob += naive_log_n1(lambda, mu, beta, time);
                    prob -= (lambda * beta).ln();
                    prob -= n0(mu, beta).ln();
                }
                event_prob *= lambda * beta;
                last_event_deletion[node_id_value] = false;
            }
        }
        prob += event_prob.ln();
    }
    prob
}

// This is a direct implementation of the TKF92 log likelihood calculation without any
// aggregation over subtrees and without substitutions. This direct calculation is sufficient
// if one is only interested in the indel likelihood for a fixed alignment and tree.
// Used for testing purposes only, i.e., to validate the aggregated implementation.
#[cfg(test)]
fn tkf92_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF92IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let blocks = model.get_blocks(&phylo.msa);
    let tree = &phylo.tree;
    let lambda = model.lambda();
    let mu = model.mu();
    let r = model.params()[2];

    // for the root
    let mut prob: f64 = (1.0 - lambda / mu).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    for (i, fragment) in blocks.iter().enumerate() {
        let mut event_prob = 1.0;
        let fragment_len = if i == 0 {
            *fragment
        } else {
            fragment - blocks[i - 1]
        };
        if get_mapping_for_any_node(&phylo.msa, &phylo.tree.root)[fragment - 1].is_some() {
            // the eq seq at the root has a fragment
            event_prob *= lambda / mu * (1.0 - r) / r;
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

            let beta = naive_beta(lambda, mu, time);
            if i == 0 {
                prob += ln_i1(lambda, beta.ln());
            }
            if parent_is_gap && current_is_gap {
                continue;
            } else if !parent_is_gap && !current_is_gap {
                // homolog block
                event_prob *= naive_h1(lambda, mu, beta, time);
                last_event_deletion[node_id_value] = false;
            } else if !parent_is_gap && current_is_gap {
                // deletion
                event_prob *= n0(mu, beta);
                last_event_deletion[node_id_value] = true;
            } else if parent_is_gap && !current_is_gap {
                // insertion
                if last_event_deletion[node_id_value] {
                    prob += naive_log_n1(lambda, mu, beta, time);
                    prob -= (lambda * beta).ln();
                    prob -= n0(mu, beta).ln();
                }
                event_prob *= lambda * beta * (1.0 - r) / r;
                prob += fragment_len as f64 * r.ln();
                last_event_deletion[node_id_value] = false;
            }
        }
        prob += event_prob.ln();
        prob += (fragment_len - 1) as f64 * (1.0 + event_prob).ln();
    }
    prob
}

// ====> beta <====

#[cfg(test)]
fn naive_beta(lambda: f64, mu: f64, time: f64) -> f64 {
    let exp_term = ((lambda - mu) * time).exp();
    (1.0 - exp_term) / (mu - lambda * exp_term)
}

#[test]
fn tkf_beta_calculated_by_hand() {
    assert_relative_eq!(naive_beta(0.3, 0.5, 0.7), 0.5461782813185221);
    assert_relative_eq!(ln_beta(0.3, 0.5, 0.7), 0.5461782813185221f64.ln());
}

#[rstest]
#[case::short_t_small_l_close_m(1.0e-16, 0.0100000, 0.0100001, -36.84136148790473)]
#[case::medium_t_small_l_close_m(1.0, 0.0100000, 0.0100001, -0.00995038035812)]
#[case::long_t_small_l_close_m(100.0, 0.0100000, 0.0100001, 3.91202050542710)]
#[case::short_t_m_gt_l(1.0e-16, 0.0100000, 5.0000000, -36.84136148790473)]
#[case::medium_t_m_gt_l(1.0, 0.0100000, 5.0000000, -1.61625322965137)]
#[case::long_t_m_gt_l(100.0, 0.0100000, 5.0000000, -1.60943791243410)]
#[case::short_t_m_close_l(1.0e-16, 4.9999000, 5.0000000, -36.84136148790473)]
#[case::medium_t_m_close_l(1.0, 4.9999000, 5.0000000, -1.79175113599889)]
#[case::long_t_m_close_l(100.0, 4.9999000, 5.0000000, -1.61142595164059)]
#[case::short_t_large_m_diff_l(1.0e-16, 4.9990000, 100.00000, -36.84136148790474)]
#[case::medium_t_large_m_diff_l(1.0, 4.9990000, 100.00000, -4.60517018598809)]
#[case::long_t_large_m_diff_l(100.0, 4.9990000, 100.00000, -4.60517018598809)]
#[case::short_t_large_l_close_m(1.0e-16, 99.999000, 100.00000, -36.84136148790474)]
#[case::medium_t_large_l_close_m(1.0, 99.999000, 100.00000, -4.61511556715904)]
#[case::long_t_large_l_close_m(100.0, 99.999000, 100.00000, -4.60526526478741)]
// see https://github.com/MattesMrzik/tkf_mathematica
fn tkf_ln_beta_mathematica(
    #[case] time: f64,
    #[case] lambda: f64,
    #[case] mu: f64,
    #[case] expected: f64,
) {
    assert_relative_eq!(ln_beta(lambda, mu, time), expected, epsilon = 1e-10);
}

// ====> n0 <====

#[cfg(test)]
fn n0(mu: f64, beta: f64) -> f64 {
    mu * beta
}

#[test]
fn tkf_ln_n0_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = naive_beta(l, m, time);
    // (3(1-e^(-.5))/(3-2*e^(-.5)))
    assert_relative_eq!(n0(m, b), 0.6605755607027574);
    assert_relative_eq!(ln_n0(m, b.ln()), 0.6605755607027574f64.ln());
}

#[rstest]
#[case::short_t_small_l_close_m(1.0e-16, 0.0100000, 0.0100001, -41.44652167394282)]
#[case::medium_t_small_l_close_m(1.0, 0.0100000, 0.0100001, -4.61511056639621)]
#[case::long_t_small_l_close_m(100.0, 0.0100000, 0.0100001, -0.69313968061099)]
#[case::short_t_m_gt_l(1.0e-16, 0.0100000, 5.0000000, -35.23192357547063)]
#[case::medium_t_m_gt_l(1.0, 0.0100000, 5.0000000, -0.00681531721727)]
#[case::long_t_m_gt_l(100.0, 0.0100000, 5.0000000, 0.0)]
#[case::short_t_m_close_l(1.0e-16, 4.9999000, 5.0000000, -35.23192357547063)]
#[case::medium_t_m_close_l(1.0, 4.9999000, 5.0000000, -0.18231322356479)]
#[case::long_t_m_close_l(100.0, 4.9999000, 5.0000000, -0.00198803920649)]
#[case::short_t_large_m_diff_l(1.0e-16, 4.9990000, 100.00000, -32.23619130191664)]
#[case::medium_t_large_m_diff_l(1.0, 4.9990000, 100.00000, 0.0)]
#[case::long_t_large_m_diff_l(100.0, 4.9990000, 100.00000, 0.0)]
#[case::short_t_large_l_close_m(1.0e-16, 99.999000, 100.00000, -32.23619130191665)]
#[case::medium_t_large_l_close_m(1.0, 99.999000, 100.00000, -0.00994538117095)]
#[case::long_t_large_l_close_m(100.0, 99.999000, 100.00000, -9.50787993145852e-5)]
// see https://github.com/MattesMrzik/tkf_mathematica
fn tkf_ln_n0_mathematica(
    #[case] time: f64,
    #[case] lambda: f64,
    #[case] mu: f64,
    #[case] expected: f64,
) {
    let ln_beta = ln_beta(lambda, mu, time);
    assert_relative_eq!(ln_n0(mu, ln_beta), expected, epsilon = 1e-10);
}

// ====> h1 <====

#[cfg(test)]
fn naive_h1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    (-mu * time).exp() * (1.0 - lambda * beta)
}

#[test]
fn tkf_ln_h1_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.5;
    let b = naive_beta(l, m, time);
    // e^(-4.5) * (1-2(1-e^(-1.5))/(3-2*e^(-1.5)))
    assert_relative_eq!(naive_h1(l, m, b, time), 0.004350089645603061);
    assert_relative_eq!(ln_h1(l, m, b.ln(), time), 0.004350089645603061f64.ln());
}

#[rstest]
#[case::short_t_small_l_close_m(1.0e-16, 0.0100000, 0.0100001, -2.00001000000000e-18)]
#[case::medium_t_small_l_close_m(1.0, 0.0100000, 0.0100001, -0.01995043035812)]
#[case::long_t_small_l_close_m(100.0, 0.0100000, 0.0100001, -1.69315468056515)]
#[case::short_t_m_gt_l(1.0e-16, 0.0100000, 5.0000000, -5.01000000000000e-16)]
#[case::medium_t_m_gt_l(1.0, 0.0100000, 5.0000000, -5.00198839124905)]
#[case::long_t_m_gt_l(100.0, 0.0100000, 5.0000000, -500.0020020026707)]
#[case::short_t_m_close_l(1.0e-16, 4.9999000, 5.0000000, -9.99990000000000e-16)]
#[case::medium_t_m_close_l(1.0, 4.9999000, 5.0000000, -6.79170113641556)]
#[case::long_t_m_close_l(100.0, 4.9999000, 5.0000000, -506.2116003042919)]
#[case::short_t_large_m_diff_l(1.0e-16, 4.9990000, 100.00000, -1.04999000000000e-14)]
#[case::medium_t_large_m_diff_l(1.0, 4.9990000, 100.00000, -100.05128276812716)]
#[case::long_t_large_m_diff_l(100.0, 4.9990000, 100.00000, -10000.051282768127)]
#[case::short_t_large_l_close_m(1.0e-16, 99.999000, 100.00000, -1.99999000000000e-14)]
#[case::medium_t_large_l_close_m(1.0, 99.999000, 100.00000, -104.61461560882571)]
#[case::long_t_large_l_close_m(100.0, 99.999000, 100.00000, -10009.160852082725)]
// see https://github.com/MattesMrzik/tkf_mathematica
fn tkf_ln_h1_mathematica(
    #[case] time: f64,
    #[case] lambda: f64,
    #[case] mu: f64,
    #[case] expected: f64,
) {
    let ln_beta = ln_beta(lambda, mu, time);
    assert_relative_eq!(ln_h1(lambda, mu, ln_beta, time), expected, epsilon = 1e-10);
}

// ====> i1 <====

#[cfg(test)]
fn naive_log_i1(lambda: f64, beta: f64) -> f64 {
    (1.0 - lambda * beta).ln()
}

#[test]
fn tkf_ln_i1_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.0;
    let b = naive_beta(l, m, time);
    // log((1-2(1-e^(-1))/(3-2*e^(-1)))
    assert_relative_eq!(naive_log_i1(l, b), -0.8172396554020775);
    assert_relative_eq!(ln_i1(l, b.ln()), -0.8172396554020775);
}

#[rstest]
#[case::short_t_equal_lm(1.0e-16, 0.0100000, 0.0100001, -1.00000000000000e-18)]
#[case::medium_t_equal_lm(1.0, 0.0100000, 0.0100001, -0.00995033035812)]
#[case::long_t_equal_lm(100.0, 0.0100000, 0.0100001, -0.69314468056515)]
#[case::short_t_m_gt_l(1.0e-16, 0.0100000, 5.0000000, -1.00000000000000e-18)]
#[case::medium_t_m_gt_l(1.0, 0.0100000, 5.0000000, -0.00198839124905)]
#[case::long_t_m_gt_l(100.0, 0.0100000, 5.0000000, -0.00200200267067)]
#[case::short_t_m_close_l(1.0e-16, 4.9999000, 5.0000000, -4.99990000000000e-16)]
#[case::medium_t_m_close_l(1.0, 4.9999000, 5.0000000, -1.79170113641556)]
#[case::long_t_m_close_l(100.0, 4.9999000, 5.0000000, -6.21160030429188)]
#[case::short_t_large_m_diff_l(1.0e-16, 4.9990000, 100.00000, -4.99899999999998e-16)]
#[case::medium_t_large_m_diff_l(1.0, 4.9990000, 100.00000, -0.05128276812716)]
#[case::long_t_large_m_diff_l(100.0, 4.9990000, 100.00000, -0.05128276812716)]
#[case::short_t_large_l_close_m(1.0e-16, 99.999000, 100.00000, -9.99989999999995e-15)]
#[case::medium_t_large_l_close_m(1.0, 99.999000, 100.00000, -4.61461560882571)]
#[case::long_t_large_l_close_m(100.0, 99.999000, 100.00000, -9.16085208272545)]
// see https://github.com/MattesMrzik/tkf_mathematica
fn tkf_ln_i1_mathematica(
    #[case] time: f64,
    #[case] lambda: f64,
    #[case] mu: f64,
    #[case] expected: f64,
) {
    let ln_beta = ln_beta(lambda, mu, time);
    assert_relative_eq!(ln_i1(lambda, ln_beta), expected, epsilon = 1e-10);
}

// ====> n1 <====

#[cfg(test)]
fn naive_log_n1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let term1 = 1.0 - (-mu * time).exp() - mu * beta;
    let term2 = 1.0 - lambda * beta;
    (term1 * term2).ln()
}

#[test]
fn tkf_ln_n1_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = naive_beta(l, m, time);
    // log((1-e^(-1.5) - 3(1-e^(-.5))/(3-2*e^(-.5)) )* (1-2(1-e^(-.5))/(3-2*e^(-.5)))   (2(1-e^(-1))/(3-2*e^(-1)))^0)
    assert_relative_eq!(
        naive_log_n1(l, m, b, time),
        -2.732135332549935,
        epsilon = 1e-14
    );
}

// ====> eta <====

#[cfg(test)]
fn naive_eta(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let mut e = naive_log_n1(lambda, mu, beta, time);
    e -= lambda.ln() + beta.ln();
    e -= (mu * beta).ln();
    e
}

#[test]
fn tkf_eta_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.5;
    let b = naive_beta(l, m, time);
    // math.log( (1 - math.exp(-3*1.5) - 3*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    // * (1 - 2*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5)))))
    // - math.log(2*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    // - math.log(3*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    assert_relative_eq!(
        naive_eta(l, m, b, time),
        -2.922778333826742,
        epsilon = 1e-14
    );
    assert_relative_eq!(eta(l, m, b.ln(), time), -2.922778333826742, epsilon = 1e-14);
}

#[rstest]
#[case::short_t_equal_lm(1.0e-16, 0.0100000, 0.0100001, -std::f64::consts::LN_2)]
#[case::medium_t_equal_lm(1.0, 0.0100000, 0.0100001, -0.69981110152058)]
#[case::long_t_equal_lm(100.0, 0.0100000, 0.0100001, -1.33089630715386)]
#[case::short_t_m_gt_l(1.0e-16, 0.0100000, 5.0000000, -std::f64::consts::LN_2)]
#[case::medium_t_m_gt_l(1.0, 0.0100000, 5.0000000, -3.59660487486015)]
#[case::long_t_m_gt_l(100.0, 0.0100000, 5.0000000, -493.2492380189326)]
#[case::short_t_m_close_l(1.0e-16, 4.9999000, 5.0000000, -0.69314718055993)]
#[case::medium_t_m_close_l(1.0, 4.9999000, 5.0000000, -3.26012517688075)]
#[case::long_t_m_close_l(100.0, 4.9999000, 5.0000000, -12.42920452997077)]
#[case::short_t_large_m_diff_l(1.0e-16, 4.9990000, 100.00000, -std::f64::consts::LN_2)]
#[case::medium_t_large_m_diff_l(1.0, 4.9990000, 100.00000, -92.114758161939)]
#[case::long_t_large_m_diff_l(100.0, 4.9990000, 100.00000, -9497.2066332427)]
#[case::short_t_large_l_close_m(1.0e-16, 99.999000, 100.00000, -std::f64::consts::LN_2)]
#[case::medium_t_large_l_close_m(1.0, 99.999000, 100.00000, -9.21033045525952)]
#[case::long_t_large_l_close_m(100.0, 99.999000, 100.00000, -18.42150400780228)]
// see https://github.com/MattesMrzik/tkf_mathematica
fn tkf_eta_mathematica(
    #[case] time: f64,
    #[case] lambda: f64,
    #[case] mu: f64,
    #[case] expected: f64,
) {
    let ln_beta = ln_beta(lambda, mu, time);
    assert_relative_eq!(eta(lambda, mu, ln_beta, time), expected, epsilon = 1e-7);
}

#[test]
fn tkf91_get_blocks() {
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new(vec![
        record!("A0", b"AAAB-D"),
        record!("B1", b"--ARAW"),
        record!("I1", b"AAAA-A"),
    ]);
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    let blocks = TKF91IndelModel::default().get_blocks(&msa);
    let block_lens = get_block_lengths(&blocks);

    assert_eq!(blocks, (1..msa.len() + 1).collect::<Vec<usize>>());
    assert_eq!(block_lens, vec![1; 6]);
}

#[test]
fn tkf92_get_blocks() {
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new(vec![
        record!("A0", b"AAB-D"),
        record!("B1", b"-ARAW"),
        record!("I1", b"AAA-A"),
    ]);

    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    let blocks = TKF92IndelModel::default().get_blocks(&msa);
    let block_lens = get_block_lengths(&blocks);

    assert_eq!(blocks, vec![1, 3, 4, 5]);
    assert_eq!(block_lens, vec![1, 2, 1, 1]);
}

#[test]
fn tkf92_fixed_get_blocks() {
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new(vec![
        record!("A0", b"AAAAAAB-D"),
        record!("B1", b"---AAARAW"),
        record!("I1", b"AAAAAAA-A"),
    ]);

    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    let fragmentation = vec![1, 2, 7];
    let model = TKF92FixedIndelModel {
        params: vec![0.5, 1.5, 0.2],
        log_r: 0.2_f64.ln(),
        fragmentation,
    };
    let blocks = model.get_blocks(&msa);
    let block_lens = get_block_lengths(&blocks);

    assert_eq!(blocks, vec![1, 2, 3, 7, 8, 9]);
    assert_eq!(block_lens, vec![1, 1, 1, 4, 1, 1]);
}

#[cfg(test)]
pub(super) fn setup_test_phylo(alphabet: &'static Alphabet) -> PhyloInfo<MASA> {
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

#[test]
fn tkf_indel_get_and_set_params_and_freqs() {
    let mut tkf_indel_cost =
        TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
    // params
    assert_eq!(tkf_indel_cost.param_count(), 3);
    assert_eq!(tkf_indel_cost.param(0), 1.0);
    assert_eq!(tkf_indel_cost.param(1), 2.0);
    assert_eq!(tkf_indel_cost.param(2), 0.3);
    assert_eq!(tkf_indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_indel_cost.model.r(), 0.3);
    tkf_indel_cost.set_param(2, 0.33);
    assert_eq!(tkf_indel_cost.param_count(), 3);
    assert_eq!(tkf_indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_indel_cost.model.r(), 0.33);
    // freqs
    assert_eq!(tkf_indel_cost.freqs(), &*DUMMY_FREQS);
    assert_eq!(
        tkf_indel_cost.empirical_freqs(),
        setup_test_phylo(Alphabet::dna()).freqs()
    );
}

#[test]
fn tkf_get_and_set_params() {
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8, 0.9]);
    let mut tkf_cost = TKF92CostBuilder::new(
        1.0,
        2.0,
        0.3,
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    assert_eq!(tkf_cost.param_count(), 8);
    assert_eq!(tkf_cost.param(0), 1.0);
    assert_eq!(tkf_cost.param(1), 2.0);
    assert_eq!(tkf_cost.param(2), 0.3);
    assert_eq!(tkf_cost.param(3), 0.5);
    assert_eq!(tkf_cost.param(4), 0.6);
    assert_eq!(tkf_cost.param(5), 0.7);
    assert_eq!(tkf_cost.param(6), 0.8);
    assert_eq!(tkf_cost.param(7), 0.9);
    assert_eq!(tkf_cost.indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_cost.indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_cost.indel_cost.model.r(), 0.3);
    tkf_cost.set_param(2, 0.33);
    tkf_cost.set_param(5, 0.77);
    assert_eq!(tkf_cost.param_count(), 8);
    assert_eq!(tkf_cost.param(0), 1.0);
    assert_eq!(tkf_cost.param(1), 2.0);
    assert_eq!(tkf_cost.param(2), 0.33);
    assert_eq!(tkf_cost.param(3), 0.5);
    assert_eq!(tkf_cost.param(4), 0.6);
    assert_eq!(tkf_cost.param(5), 0.77);
    assert_eq!(tkf_cost.param(6), 0.8);
    assert_eq!(tkf_cost.param(7), 0.9);

    assert_eq!(
        tkf_cost.empirical_freqs(),
        setup_test_phylo(Alphabet::dna()).freqs()
    );
}

#[test]
fn tkf91_indel_cost_fmt() {
    let tkf_indel_cost = TKF91IndelCostBuilder::new(1.0, 2.0, setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();

    let fmt = format!("{}", tkf_indel_cost);

    assert_eq!(fmt, "TKF91 with lambda = 1, mu = 2");
}

#[test]
fn tkf91_cost_fmt() {
    let subst_model = SubstModel::<JC69>::new(&[], &[]);
    let tkf_cost = TKF91CostBuilder::new(1.0, 2.0, subst_model, setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();

    let fmt = format!("{}", tkf_cost);

    assert_eq!(fmt, "TKF91 with lambda = 1, mu = 2 and JC69");
}

#[test]
fn tkf92_indel_cost_fmt() {
    let tkf_indel_cost =
        TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();

    let fmt = format!("{}", tkf_indel_cost);

    assert_eq!(fmt, "TKF92 with lambda = 1, mu = 2, r = 0.3");
}

#[test]
fn tkf92_cost_fmt() {
    let subst_model = SubstModel::<JC69>::new(&[], &[]);
    let tkf_cost = TKF92CostBuilder::new(
        1.0,
        2.0,
        0.3,
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();

    let fmt = format!("{}", tkf_cost);

    assert_eq!(fmt, "TKF92 with lambda = 1, mu = 2, r = 0.3 and JC69");
}

#[test]
fn tkf_get_and_set_freqs() {
    let subst_model = SubstModel::<GTR>::new(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8, 0.9]);
    let mut tkf_cost = TKF92CostBuilder::new(
        1.0,
        2.0,
        0.3,
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    assert_eq!(tkf_cost.freqs().as_slice(), &[0.1, 0.2, 0.3, 0.4]);
    tkf_cost.set_freqs(frequencies!(&[0.4, 0.3, 0.2, 0.1]));
    assert_eq!(tkf_cost.freqs().as_slice(), &[0.4, 0.3, 0.2, 0.1]);
}

#[test]
fn tkf91_param_range() {
    let subst_model = SubstModel::<GTR>::new(&[], &[]);
    let tkf_cost = TKF91CostBuilder::new(1.0, 2.0, subst_model, setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();
    let lambda_range = tkf_cost.param_range(usize::from(TKF92Parameters::Lambda));
    let true_lambda_range = (f64::EPSILON, 2.0 - f64::EPSILON);
    assert_eq!(lambda_range, true_lambda_range);
    let mu_range = tkf_cost.param_range(usize::from(TKF92Parameters::Mu));
    let true_mu_range = (1.0 + f64::EPSILON, f64::MAX);
    assert_eq!(mu_range, true_mu_range);

    for subst_param_idx in 2..tkf_cost.param_count() {
        let subst_range = tkf_cost.param_range(subst_param_idx);
        let true_subst_range = PARAM_RANGE_POSITIVE;
        assert_eq!(subst_range, true_subst_range);
    }
}

#[cfg(test)]
fn tkf92_subst_param_range<Q: QMatrix, T: TKFModel, AA: AncestralAlignment>(
    cost: &TKFCost<Q, T, AA>,
) {
    for subst_param_idx in 3..cost.param_count() {
        let subst_range = cost.param_range(subst_param_idx);
        let true_subst_range = PARAM_RANGE_POSITIVE;
        assert_eq!(subst_range, true_subst_range);
    }
}

#[cfg(test)]
fn tkf92_indel_param_range<T: TKFModel, AA: AncestralAlignment>(cost: &TKFIndelCost<T, AA>) {
    let lambda_range = cost.param_range(usize::from(TKF92Parameters::Lambda));
    let true_lambda_range = (f64::EPSILON, 2.0 - f64::EPSILON);
    assert_eq!(lambda_range, true_lambda_range);
    let mu_range = cost.param_range(usize::from(TKF92Parameters::Mu));
    let true_mu_range = (1.0 + f64::EPSILON, f64::MAX);
    assert_eq!(mu_range, true_mu_range);
    let r_range = cost.param_range(usize::from(TKF92Parameters::R));
    let true_r_range = PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE;
    assert_eq!(r_range, true_r_range);
}
#[test]
fn tkf92_param_range() {
    let subst_model = SubstModel::<GTR>::new(&[], &[]);
    let tkf_cost = TKF92CostBuilder::new(
        1.0,
        2.0,
        0.3,
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    tkf92_subst_param_range(&tkf_cost);
    tkf92_indel_param_range(&tkf_cost.indel_cost);
}

#[test]
fn tkf92_fixed_param_range() {
    let tkf_cost =
        TKF92FixedIndelCostBuilder::new(1.0, 2.0, 0.3, vec![], setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
    tkf92_indel_param_range(&tkf_cost);
}

#[test]
fn tkf92_add_param_range() {
    let tkf_cost = TKF92IndelAddBlocksCostBuilder::new(
        1.0,
        2.0,
        0.3,
        vec![],
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    tkf92_indel_param_range(&tkf_cost);
}

#[test]
fn tkf91_indel_logl() {
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
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("A1").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("I3").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("C4").blen));
    // first block ([0:2], insertion at C4)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("C4").blen);
    manual_calculation += x.ln() * 2.0;
    // second block ([2:3], all homologous except B2 deleted)
    let mut x = lambda / mu;
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("C4").blen),
        tree.by_id("C4").blen,
    );
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("A1").blen),
        tree.by_id("A1").blen,
    );
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("I3").blen),
        tree.by_id("I3").blen,
    );
    x *= n0(mu, naive_beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += x.ln();
    // third block ([3:7], insertion at A1)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("C4").blen);
    manual_calculation += x.ln() * 4.0;
    // fourth block ([7:10], insertion at B2)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("B2").blen);
    manual_calculation += x.ln() * 3.0;
    manual_calculation += naive_log_n1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("B2").blen),
        tree.by_id("B2").blen,
    );
    manual_calculation -= n0(mu, naive_beta(lambda, mu, tree.by_id("B2").blen)).ln();
    manual_calculation -= (lambda * naive_beta(lambda, mu, tree.by_id("B2").blen)).ln();

    // assert
    assert_relative_eq!(logl, manual_calculation, epsilon = 1e-12);
    assert_relative_eq!(logl, half_manual, epsilon = 1e-11);
}

#[test]
fn tkf92_indel_logl() {
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
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("A1").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("I3").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("C4").blen));
    // first block ([0:2], insertion at C4)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("C4").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 1.0 * (1.0 + x).ln();
    // second block ([2:3], all homologous except B2 deleted)
    let mut x = lambda / mu * (1.0 - r) / r;
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("C4").blen),
        tree.by_id("C4").blen,
    );
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("A1").blen),
        tree.by_id("A1").blen,
    );
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("I3").blen),
        tree.by_id("I3").blen,
    );
    x *= n0(mu, naive_beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += x.ln();
    // third block ([3:7], insertion at A1)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("C4").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 3.0 * (1.0 + x).ln();
    // fourth block ([7:10], insertion at B2)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("B2").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 2.0 * (1.0 + x).ln();
    manual_calculation += naive_log_n1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("B2").blen),
        tree.by_id("B2").blen,
    );
    manual_calculation -= n0(mu, naive_beta(lambda, mu, tree.by_id("B2").blen)).ln();
    manual_calculation -= (lambda * naive_beta(lambda, mu, tree.by_id("B2").blen)).ln();

    // assert
    assert_relative_eq!(logl, manual_calculation, epsilon = 1e-12);
    assert_relative_eq!(logl, half_manual, epsilon = 1e-11);
}

#[test]
fn tkf91_cost_builder_fails() {
    let phylo = setup_test_phylo(Alphabet::protein());
    let subst_model = SubstModel::<GTR>::new(&[], &[]);

    let tkf91_err = TKF91CostBuilder::new(0.1, 0.2, subst_model, phylo).build();

    assert_matches!(
        tkf91_err, Err(Error::Alphabet(msg)) if msg.contains(
        "alphabet mismatch between model and alignment")
    );
}

#[test]
fn tkf92_cost_builder_fails() {
    let phylo = setup_test_phylo(Alphabet::protein());
    let subst_model = SubstModel::<GTR>::new(&[], &[]);

    let tkf92_err = TKF92CostBuilder::new(0.1, 0.2, 0.3, subst_model, phylo).build();

    assert_matches!(
        tkf92_err, Err(Error::Alphabet(msg)) if msg.contains(
        "alphabet mismatch between model and alignment")
    );
}

#[test]
fn tkf91_logl_with_substitution() {
    // arrange
    let phylo = setup_test_phylo(Alphabet::dna());
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
    let logl = ModelSearchCost::cost(&tkf_cost);
    let subst_logl = ModelSearchCost::cost(&subst_cost);
    let tkf_logl = tkf91_indel_logl_without_aggregation(
        &tkf_cost.indel_cost.model,
        &tkf_cost.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(logl, subst_logl + tkf_logl);
}

#[test]
fn tkf92_logl_with_substitution() {
    // arrange
    let phylo = setup_test_phylo(Alphabet::dna());
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
    let logl = ModelSearchCost::cost(&tkf_cost);
    let subst_logl = ModelSearchCost::cost(&subst_cost);
    let tkf_logl = tkf92_indel_logl_without_aggregation(
        &tkf_cost.indel_cost.model,
        &tkf_cost.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(logl, subst_logl + tkf_logl, epsilon = 1e-12);
}

#[test]
fn tkf_indel_history_doesnt_change_felsenstein() {
    // arrange
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
    let phylo2 = PhyloInfo { msa: msa2, tree };
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
    let tkf_log_1 = ModelSearchCost::cost(&tkf_cost1.clone());
    let tkf_log_2 = ModelSearchCost::cost(&tkf_cost2.clone());
    let tkf_indel_cost_1 = ModelSearchCost::cost(&tkf_cost1.indel_cost);
    let tkf_indel_cost_without_agg_1 = tkf92_indel_logl_without_aggregation(
        &tkf_cost1.indel_cost.model,
        &tkf_cost1.indel_cost.phylo,
    );
    let tkf_indel_cost_2 = ModelSearchCost::cost(&tkf_cost2.indel_cost);
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
fn modify_tkf92_subst_params_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    let phylo = setup_test_phylo(Q::alphabet());
    let subst_original_param = 1.0;
    let subst_changed_param = 0.5;
    let subst_model = SubstModel::<Q>::new(&[], &[subst_original_param]);
    let mut tkf_cost = TKF92CostBuilder::new(0.1, 0.2, 0.3, subst_model, phylo.clone())
        .build()
        .unwrap();

    // sanity check
    let logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl, ModelSearchCost::cost(&tkf_cost));

    // The likelihood should change if we change model parameters
    tkf_cost.set_param(3, subst_changed_param);
    let logl2 = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl2, ModelSearchCost::cost(&tkf_cost));
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let subst_model = SubstModel::<Q>::new(&[], &[subst_changed_param]);
    let tkf_cost = TKF92CostBuilder::new(0.1, 0.2, 0.3, subst_model, phylo)
        .build()
        .unwrap();
    let new_logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(new_logl, ModelSearchCost::cost(&tkf_cost));
    assert_eq!(logl2, new_logl);
}

#[cfg(test)]
fn modify_tkf92_indel_params_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    let phylo = setup_test_phylo(Q::alphabet());
    let subst_model = SubstModel::<Q>::new(&[], &[]);
    let tkf_original_mu = 0.2;
    let tkf_changed_mu = 0.25;
    let mut tkf_cost = TKF92CostBuilder::new(0.1, tkf_original_mu, 0.3, subst_model, phylo.clone())
        .build()
        .unwrap();

    // sanity check
    let logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl, ModelSearchCost::cost(&tkf_cost));

    // The likelihood should change if we change model parameters
    tkf_cost.set_param(1, tkf_changed_mu);
    let logl2 = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl2, ModelSearchCost::cost(&tkf_cost));
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let subst_model = SubstModel::<Q>::new(&[], &[]);
    let tkf_cost = TKF92CostBuilder::new(0.1, tkf_changed_mu, 0.3, subst_model, phylo)
        .build()
        .unwrap();
    let new_logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(new_logl, ModelSearchCost::cost(&tkf_cost));
    assert_eq!(logl2, new_logl);
}

#[test]
fn tkf92_modify_subst_model_params_costs_match() {
    modify_tkf92_subst_params_costs_match_template::<K80>();
    modify_tkf92_subst_params_costs_match_template::<HKY>();
    modify_tkf92_subst_params_costs_match_template::<TN93>();
    modify_tkf92_subst_params_costs_match_template::<GTR>();
}

#[test]
fn tkf_modify_indel_model_params_costs_match() {
    modify_tkf92_indel_params_costs_match_template::<JC69>();
    modify_tkf92_indel_params_costs_match_template::<K80>();
    modify_tkf92_indel_params_costs_match_template::<HKY>();
    modify_tkf92_indel_params_costs_match_template::<TN93>();
    modify_tkf92_indel_params_costs_match_template::<GTR>();
    modify_tkf92_indel_params_costs_match_template::<WAG>();
    modify_tkf92_indel_params_costs_match_template::<BLOSUM>();
    modify_tkf92_indel_params_costs_match_template::<HIVB>();
}

#[test]
fn tkf_update_tree() {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::new(vec![
            record!("A1", b"--GTGGATGC"),
            record!("B2", b"--G----CGA"),
            record!("I3", b"--N----NNN"),
            record!("C4", b"AGC-------"),
            record!("R5", b"--N-------"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };
    let subst_model = SubstModel::<GTR>::new(&[], &[]);
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let mut tkf_cost = TKF92CostBuilder::new(lambda, mu, r, subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let original_logl = TreeSearchCost::cost(&tkf_cost);
    assert_ne!(original_logl, f64::NEG_INFINITY);

    let node_idx = &phylo.tree.by_id("I3").idx;
    let child_idx = &phylo.tree.by_id("A1").idx;
    let new_tree = rooted_nni(&phylo.tree, node_idx, child_idx).unwrap();
    let new_tree_newick = new_tree.to_newick();
    tkf_cost.update_tree(new_tree);

    assert_eq!(new_tree_newick, tkf_cost.tree().to_newick());
    let new_logl = TreeSearchCost::cost(&tkf_cost);

    let new_phylo = PhyloInfo {
        msa: tkf_cost.masa().clone(),
        tree: tkf_cost.tree().clone(),
    };
    let clean_cost = TKF92CostBuilder::new(lambda, mu, r, subst_model.clone(), new_phylo)
        .build()
        .unwrap();
    let clean_logl = TreeSearchCost::cost(&clean_cost);
    assert_ne!(original_logl, new_logl);
    assert_eq!(new_logl, clean_logl);
}

#[test]
fn tkf92_underflow_short_branches() {
    // fails if blens smaller or equal to 1e-17
    let tree = tree!("(((A1:1e-16,B2:2.0)I3:1e-16,C4:2.0)R5:0.0);");

    let msa = MASA::from_aligned_with_ancestral(
        // Testing all events on short branches
        Sequences::new(vec![
            record!("A1", b"-AA-AA"),
            record!("B2", b"A-A--A"),
            record!("I3", b"AAAA-A"),
            record!("C4", b"-----A"),
            record!("R5", b"-----A"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };
    let lambda = 1.0;
    let mu = 4.1;
    let r = 0.8;
    let tkf92_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
        .build()
        .unwrap();
    let logl = tkf92_cost.logl();
    assert!(!logl.is_nan());
    assert!(logl.is_finite());
}

#[test]
fn tkf92_underflow_short_branches_and_large_mu() {
    let tree = tree!("(((A1:1e-16,B2:2.0)I3:1e-16,C4:2.0)R5:0.0);");

    let msa = MASA::from_aligned_with_ancestral(
        // Testing all events on short branches
        Sequences::new(vec![
            record!("A1", b"-AA-AA"),
            record!("B2", b"A-A--A"),
            record!("I3", b"AAAA-A"),
            record!("C4", b"-----A"),
            record!("R5", b"-----A"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };
    let lambda = 1.0;
    let mu = 10000.0;
    let r = 0.8;
    let tkf92_cost = TKF92IndelCostBuilder::new(lambda, mu, r, phylo)
        .build()
        .unwrap();
    let logl = tkf92_cost.logl();
    assert!(!logl.is_nan());
    println!("Log-likelihood: {}", logl);
    assert!(logl.is_finite());
}
