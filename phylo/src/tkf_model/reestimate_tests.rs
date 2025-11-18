use approx::assert_relative_eq;

use crate::alignment::{AncestralAlignment, Sequences, MASA};
use crate::likelihood::TreeSearchCost;
use crate::phylo_info::PhyloInfo;
use crate::record_wo_desc as record;
use crate::tkf_model::{Reestimator, TKF91IndelCostBuilder, TKF92IndelCostBuilder};
use crate::tree;

#[test]
fn tkf91_reestimate() {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::new(vec![
            record!("A1", b"--GTGGA---"),
            record!("B2", b"-------NNA"),
            record!("I3", b"AAT-------"),
            record!("C4", b"AGG-------"),
            record!("R5", b"--A-------"),
        ]),
        &tree,
    )
    .unwrap();
    let mut phylo = PhyloInfo { msa, tree };

    let tkf_cost = TKF91IndelCostBuilder::new(1.0, 2.0, phylo.clone())
        .build()
        .unwrap();
    let mut reestimator = Reestimator::new(&tkf_cost);
    // to calc model_info tmp nodes values
    let old_logl = tkf_cost.cost();
    let (new_mappings, best_logl) = reestimator
        .reestimate(&tkf_cost.phylo.tree.by_id("I3").idx)
        .unwrap();
    for (node_idx, map) in new_mappings {
        phylo.msa.update_ancestral_map(&node_idx, map);
    }
    let tkf_cost = TKF91IndelCostBuilder::new(1.0, 2.0, phylo).build().unwrap();
    let new_logl = tkf_cost.cost();
    assert_relative_eq!(best_logl, new_logl, epsilon = 1e-12);
    assert!(new_logl > old_logl);
}

#[test]
fn tkf92_reestimate() {
    // TODO use setup default phylo here and above
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::new(vec![
            record!("A1", b"--GTGGA---"),
            record!("B2", b"-------NNA"),
            record!("I3", b"AAT-------"),
            record!("C4", b"AGG-------"),
            record!("R5", b"--A-------"),
        ]),
        &tree,
    )
    .unwrap();
    let mut phylo = PhyloInfo { msa, tree };

    let tkf_cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, phylo.clone())
        .build()
        .unwrap();
    let mut reestimator = Reestimator::new(&tkf_cost);
    // to calc model_info tmp nodes values
    let old_logl = tkf_cost.cost();
    let (new_mappings, best_logl) = reestimator
        .reestimate(&tkf_cost.phylo.tree.by_id("I3").idx)
        .unwrap();
    for (node_idx, map) in new_mappings {
        phylo.msa.update_ancestral_map(&node_idx, map);
    }
    let tkf_cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, phylo)
        .build()
        .unwrap();
    let new_logl = tkf_cost.cost();
    assert_relative_eq!(best_logl, new_logl, epsilon = 1e-12);
    assert!(new_logl > old_logl);
}
