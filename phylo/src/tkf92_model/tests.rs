use std::cell::RefCell;

use crate::{
    alignment::{AncestralAlignmentBuilder, Sequences},
    phylo_info::PhyloInfoAncestors,
    substitution_models::{dna_models::JC69, QMatrix},
    tkf92_model::{TKF92Cost, TKF92Model, TKF92ModelInfo},
    tree::tree_parser::from_newick,
};
use approx::assert_relative_eq;
use bio::io::fasta::Record;

#[test]
fn mytest_logl_all_n() {
    let newick_string = "(((A0:1.0,B1:1.0)I1:1.0,C2:1.0)I2:1.0);";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let seqs = Sequences::new(vec![
        Record::with_attrs("A0", Some("A0 sequence"), b"NN-"),
        Record::with_attrs("B1", Some("B1 sequence"), b"N--"),
        Record::with_attrs("I1", Some("I1 sequence"), b"NNN"),
        Record::with_attrs("C2", Some("C2 sequence"), b"-NN"),
        Record::with_attrs("I2", Some("I2 sequence"), b"NNN"),
    ]);

    let builder = AncestralAlignmentBuilder::new(&tree, seqs);
    let msa: crate::alignment::AncestralAlignment = builder.build().unwrap();

    let phylo = PhyloInfoAncestors { msa, tree };
    let jc = JC69::new(&[0.25, 0.25, 0.25, 0.25], &[0.3]);
    let tkf_model = TKF92Model {
        q: jc,
        params: vec![0.3, 0.4, 0.5],
    };
    let model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));
    let tkf_cost = TKF92Cost {
        model: tkf_model,
        phylo,
        model_info,
    };

    tkf_cost.logl();
    // let felsenstein = &tkf_cost.model_info.borrow().felsenstein;
    // for node in felsenstein {
    //     println!("felsenstein {}", node);
    // }

    // also test if i add a only gap col that that shouldnt change the likelihood

    assert_relative_eq!(tkf_cost.logl_old(), tkf_cost.logl());


}

#[test]
fn mytest_get_blocks() {
    // arrange
    let newick_string = "((A0:1.0,B1:1.0)I1:1.0);";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let seqs = Sequences::new(vec![
        Record::with_attrs("A0", Some("A0 sequence"), b"AAB-D"),
        Record::with_attrs("B1", Some("B1 sequence"), b"-ARAW"),
        Record::with_attrs("I1", Some("I1 sequence"), b"AAA-A"),
    ]);

    let builder = AncestralAlignmentBuilder::new(&tree, seqs);
    let msa: crate::alignment::AncestralAlignment = builder.build().unwrap();

    // act & assert
    assert_eq!(TKF92ModelInfo::<JC69>::get_blocks(&msa), vec![1, 3, 4, 5]);
}

#[test]
fn mytest_get_leaf_seq_info() {
    // arrange
    let newick_string = "((A0:1.0,B1:1.0)I1:1.0);";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let seqs = Sequences::new(vec![
        Record::with_attrs("A0", Some("A0 sequence"), b"ABCD"),
        Record::with_attrs("B1", Some("B1 sequence"), b"-R-W"),
        Record::with_attrs("I1", Some("I1 sequence"), b"AAAA"),
    ]);

    let builder = AncestralAlignmentBuilder::new(&tree, seqs);
    let msa: crate::alignment::AncestralAlignment = builder.build().unwrap();
    let phylo = PhyloInfoAncestors { msa, tree };
    let jc = JC69::new(&[0.25, 0.25, 0.25, 0.25], &[0.3]);
    // let third = 1.0 / 3.0;

    // act
    let leaf_seq_info = TKF92ModelInfo::get_leaf_seq_info(&phylo, &jc);

    // assert
    // assumes ordering of nucleotides T, C, A, G
    assert_eq!(leaf_seq_info.len(), 2);
    // assert_eq!(
    //     leaf_seq_info["A0"],
    //     dmatrix![0.0, third, 0.0, third;
    //             0.0, third, 1.0, 0.0; 
    //             1.0, 0.0, 0.0, third; 
    //             0.0, third, 0.0, third;]
    // );
    // assert_eq!(
    //     leaf_seq_info["B1"],
    //     dmatrix![0.0, 0.0, 0.0, 0.5;
    //             0.0, 0.0, 0.0, 0.0; 
    //             0.0, 0.5, 0.0, 0.5; 
    //             0.0, 0.5, 0.0, 0.0;]
    // );
}

#[test]
fn mytest_b() {
    assert_relative_eq!(TKF92Cost::<JC69>::b(0.3, 0.5, 0.7), 0.5461782813185221) // (1-e^((0.3-0.5)*0.7))/(.5-.3*e^((.3-.5)*0.7))
}

#[test]
fn mytest_log_i1() {
    // arrange
    let l = 2.0;
    let m = 3.0;
    let time = 1.0;
    let b = TKF92Cost::<JC69>::b(l, m, time);
    // act & assert
    assert_relative_eq!(TKF92Cost::<JC69>::log_i1(l, b), -0.8172396554020775) // log((1-2(1-e^(-1))/(3-2*e^(-1)))
}

#[test]
fn mytest_h1() {
    // arrange
    let l = 2.0;
    let m = 3.0;
    let time = 1.5;
    let b = TKF92Cost::<JC69>::b(l, m, time);
    // act & assert
    assert_relative_eq!(TKF92Cost::<JC69>::h1(l, m, b, time), 0.004350089645603061) // e^(-4.5) * (1-2(1-e^(-1.5))/(3-2*e^(-1.5)))
}

#[test]
fn mytest_n0() {
    // arrange
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = TKF92Cost::<JC69>::b(l, m, time);
    // act & assert
    assert_relative_eq!(TKF92Cost::<JC69>::n0(m, b), 0.6605755607027574) // (3(1-e^(-.5))/(3-2*e^(-.5)))
}

#[test]
fn mytest_log_n1() {
    // arrange
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = TKF92Cost::<JC69>::b(l, m, time);
    // act & assert
    assert_relative_eq!(TKF92Cost::<JC69>::log_n1(l, m, b, time), -2.732135332549935) // log((1-e^(-1.5) - 3(1-e^(-.5))/(3-2*e^(-.5)) )* (1-2(1-e^(-.5))/(3-2*e^(-.5)))   (2(1-e^(-1))/(3-2*e^(-1)))^0)
}

// testing felsenstein: if i only use N as nuc then the substitution process should
// vanish and only the indel process should remain