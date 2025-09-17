use std::cell::RefCell;
use std::path::Path;

use crate::optimisers::{NniOptimiser, TopologyOptimiser};
use crate::phylo_info::PhyloInfoBuilder;
use crate::random::DefaultGenerator;
use crate::substitution_models::{QMatrixMaker, JC69};
use crate::{alignment::AncestralAlignment, substitution_models::QMatrix};

use super::TKF92Cost;
use super::{get_blocks, TKF92Model, TKF92ModelInfo};

#[cfg(test)]
fn logl_without_node_values<Q: QMatrix, AA: AncestralAlignment>(cost: &TKF92Cost<Q, AA>) -> f64 {
    use crate::tkf_model::{h1, log_i1, log_n1, n0};

    use super::b;

    let blocks = get_blocks(&cost.phylo.msa);
    let tree = &cost.phylo.tree;
    let model = &cost.model;
    let node_map = cost.phylo.msa.ancestral_maps();
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
        if node_map[&cost.model_info.borrow().virtual_root][fragment - 1].is_some() {
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
            let mut parent_is_gap = node_map[parent_id][fragment - 1].is_none();
            let mut current_is_gap = node_map[node_idx][fragment - 1].is_none();

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
fn test_tkf92() {
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
    topo_opti.run().unwrap();
}
