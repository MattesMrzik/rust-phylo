use std::cell::RefCell;

use crate::{
    alignment::{AncestralAlignmentBuilder, Sequences},
    phylo_info::PhyloInfoAncestors,
    substitution_models::{dna_models::JC69, QMatrix},
    tkf92_model::{TKF92Cost, TKF92Model, TKF92ModelInfo},
    tree::tree_parser::from_newick,
};
use bio::io::fasta::Record;
use nalgebra::dmatrix;

#[test]
fn mytest_logl() {
    let newick_string = "(((A0:1.0,B1:1.0)I1:1.0,C2:1.0)I2:1.0);";
    let newick_string = "((A0:1.0,B1:1.0)I1:1.0);";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let seqs = Sequences::new(vec![
        Record::with_attrs("A0", Some("A0 sequence"), b"BAA"),
        Record::with_attrs("B1", Some("B1 sequence"), b"-AA"),
        Record::with_attrs("I1", Some("I1 sequence"), b"TAA"),
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
    let felsenstein = &tkf_cost.model_info.borrow().felsenstein;
    for node in felsenstein {
        println!("felsenstein {}", node);
    }

    // also test if i add a only gap col that that shouldnt change the likelihood
  
    assert_eq!(tkf_cost.logl_old(), tkf_cost.logl());
    
   

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
    let third = 1.0 / 3.0;

    // act
    let leaf_seq_info = TKF92ModelInfo::get_leaf_seq_info(&phylo, &jc);

    // assert
    // assumes ordering of nucleotides T, C, A, G
    assert_eq!(leaf_seq_info.len(), 2);
    assert_eq!(
        leaf_seq_info["A0"],
        dmatrix![0.0, third, 0.0, third;
                0.0, third, 1.0, 0.0; 
                1.0, 0.0, 0.0, third; 
                0.0, third, 0.0, third;]
    );
    assert_eq!(
        leaf_seq_info["B1"],
        dmatrix![0.0, 0.0, 0.0, 0.5;
                0.0, 0.0, 0.0, 0.0; 
                0.0, 0.5, 0.0, 0.5; 
                0.0, 0.5, 0.0, 0.0;]
    );
}


// testing felsenstein: if i only use N as nuc then the substitution process should
// vanish and only the indel process should remain