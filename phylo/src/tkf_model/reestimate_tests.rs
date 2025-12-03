use approx::assert_relative_eq;

use crate::alignment::{AncestralAlignment, Sequences, MASA};
use crate::likelihood::TreeSearchCost;
use crate::phylo_info::PhyloInfo;
use crate::record_wo_desc as record;
use crate::tkf_model::get_map_from_any_node;
use crate::tkf_model::{
    possibilities_for_choices, EdgeSeqsReestimator, TKF91IndelCostBuilder, TKF92IndelCostBuilder,
};
use crate::tree;

#[test]
fn tkf_reestimate_possibilities_for_choices_no_choice() {
    let no_choices = [true, false, false, true, false];
    let choices = vec![];
    let possibilities = possibilities_for_choices(&choices, &no_choices);
    let expected = vec![[true, false, false, true, false]];
    assert_eq!(possibilities, expected);
}

#[test]
fn tkf_reestimate_possibilities_for_choices_one() {
    let no_choices = [true, false, false, true, false];
    let choices = vec![0];
    let mut possibilities = possibilities_for_choices(&choices, &no_choices);
    let mut expected = vec![
        [true, false, false, true, false],
        [false, false, false, true, false],
    ];
    // sort
    possibilities.sort();
    expected.sort();
    assert_eq!(possibilities, expected);
}

#[test]
fn tkf_reestimate_possibilities_for_choices() {
    let no_choices = [true, false, false, true, false];
    let choices = vec![1, 3];
    let mut possibilities = possibilities_for_choices(&choices, &no_choices);
    let mut expected = vec![
        [true, false, false, false, false],
        [true, false, false, true, false],
        [true, true, false, false, false],
        [true, true, false, true, false],
    ];
    // sort
    possibilities.sort();
    expected.sort();
    assert_eq!(possibilities, expected);
}

#[test]
fn tkf_reestimate_possibilities_for_choices_all() {
    let no_choices = [true, false, false, true, false];
    let choices = vec![0, 1, 2, 3, 4];
    let possibilities = possibilities_for_choices(&choices, &no_choices);
    let expected = (0..32)
        .map(|i| {
            [
                (i & 0b00001) != 0,
                (i & 0b00010) != 0,
                (i & 0b00100) != 0,
                (i & 0b01000) != 0,
                (i & 0b10000) != 0,
            ]
        })
        .collect::<Vec<[bool; 5]>>();

    assert_eq!(possibilities, expected);
}

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
    let phylo = PhyloInfo { msa, tree };

    let mut tkf_cost = TKF91IndelCostBuilder::new(1.0, 2.0, phylo.clone())
        .build()
        .unwrap();
    let old_logl = tkf_cost.clone().cost();
    let mut reestimator = EdgeSeqsReestimator::new(&mut tkf_cost);
    // to calc model_info tmp nodes values
    let best_logl = reestimator.reestimate(&phylo.tree.by_id("I3").idx);
    let tkf_cost = TKF91IndelCostBuilder::new(1.0, 2.0, reestimator.get_phylo().clone())
        .build()
        .unwrap();
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
            // record!("A1", b"--GTGGA"),
            // record!("B2", b"-------"),
            // record!("I3", b"AAT----"),
            // record!("C4", b"AGG----"),
            // record!("R5", b"--A----"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };

    let mut tkf_cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, phylo.clone())
        .build()
        .unwrap();
    let old_logl = tkf_cost.clone().cost();
    let mut reestimator = EdgeSeqsReestimator::new(&mut tkf_cost);
    // i want to clone here such that intermediate vals are not calculated but must be calculated
    // for in reestimation
    let best_logl = reestimator.reestimate(&phylo.tree.by_id("I3").idx);
    let tkf_cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, reestimator.get_phylo().clone())
        .build()
        .unwrap();
    let new_logl = tkf_cost.cost();
    assert_relative_eq!(best_logl, new_logl, epsilon = 1e-12);
    assert!(new_logl > old_logl);
}

#[cfg(test)]
pub(super) fn assert_masa_is_dollo<AA: AncestralAlignment>(phylo: &PhyloInfo<AA>) {
    for col_idx in 0..phylo.msa.len() {
        let mut num_insertions = 0;
        for node in phylo.tree.postorder() {
            if *node == phylo.tree.root {
                if phylo.msa.ancestral_map(node)[col_idx].is_some() {
                    num_insertions += 1;
                }
            } else {
                let parent = phylo.tree.parent(node).unwrap();
                let parent_site = phylo.msa.ancestral_map(&parent)[col_idx];
                let node_site = get_map_from_any_node(&phylo.msa, node)[col_idx];
                if parent_site.is_none() && node_site.is_some() {
                    num_insertions += 1;
                }
            }}
            assert_eq!(
                num_insertions, 1,
                "Column {col_idx} has {num_insertions} insertions, not Dollo",
            );
    }
}
// again for a larger tree
#[test]
fn tkf92_reestimate_large_tree() {
    let tree = tree!("((((A1:2.0,B2:2.0)I3:0.3,(C4:2.0,D5:2.0)I6:0.4)I7:0.5,E8:2.0)R9:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::new(vec![
            record!("A1", b"----------"),
            record!("B2", b"AAAA------"),
            record!("I3", b"AAAA------"),
            record!("C4", b"--AA-AAA-A"),
            record!("D5", b"--AAA-----"),
            record!("I6", b"--AAA----A"),
            record!("I7", b"--AAA----A"),
            record!("E8", b"--AA----A-"),
            record!("R9", b"--AAA----A"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };

    assert_masa_is_dollo(&phylo);
    let mut tkf_cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, phylo.clone())
        .build()
        .unwrap();
    let mut prev_logl = tkf_cost.clone().cost();
    let mut reestimator = EdgeSeqsReestimator::new(&mut tkf_cost);
    let mut prev_phylo = reestimator.get_phylo().clone();
    for node in ["I3", "I6", "I7", "I3", "I6", "I7"] {
        let backtrack_logl = reestimator.reestimate(&phylo.tree.by_id(node).idx);
        for node in phylo.tree.postorder() {
            if get_map_from_any_node(&prev_phylo.msa, node)
                != get_map_from_any_node(&reestimator.get_phylo().msa, node)
            {
                prev_phylo = reestimator.get_phylo().clone();
                break;
            }
        }
        assert_masa_is_dollo(&reestimator.get_phylo().clone());
        let tkf_cost = TKF92IndelCostBuilder::new(1.0, 2.0, 0.3, reestimator.get_phylo().clone())
            .build()
            .unwrap();
        let new_logl = tkf_cost.cost();
        assert_masa_is_dollo(&tkf_cost.phylo);
        assert_relative_eq!(backtrack_logl, new_logl, epsilon = 1e-12);
        assert!(new_logl >= prev_logl);
        prev_logl = new_logl;
    }
}
