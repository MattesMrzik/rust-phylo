use std::{cell::RefCell, collections::HashMap};

use crate::{
    alignment::{AlignmentBuilder, AncestralAlignmentBuilder, Sequences},
    evolutionary_models::EvoModel,
    likelihood::ModelSearchCost,
    phylo_info::{PhyloInfo, PhyloInfoAncestors},
    substitution_models::{
        dna_models::JC69, QMatrix, SubstModel, SubstModelInfo, SubstitutionCostBuilder,
    },
    tkf92_model::{TKF92Cost, TKF92Model, TKF92ModelInfo},
    tree::{tree_parser::from_newick, Tree},
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
    println!("leaf seq info {:?}", model_info.borrow().leaf_sequence_info);
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
fn mytest_subst() {
    let sub_model = SubstModel::<JC69>::new(&[0.25, 0.25, 0.25, 0.25], &[0.3]).unwrap();
    let tree = from_newick("(A0:1.0);").unwrap().pop().unwrap();
    let msa = AlignmentBuilder::new(
        &tree,
        Sequences::new(vec![Record::with_attrs("A0", Some("A0 sequence"), b"A")]),
    )
    .build()
    .unwrap();
    let phylo = PhyloInfo { msa, tree };
    let cost = SubstitutionCostBuilder::<JC69>::new(sub_model, phylo)
        .build()
        .unwrap();
    println!("the cost is {}", cost.cost());
}

#[test]
fn mytest_logl_felsenstein() {
    let newick_string = "(((A0:1.0,B1:1.0)I1:1.0,C2:1.0)I2:1.0);";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let subtree_i1 = from_newick("((A0:1.0,B1:1.0)I1:1.0);")
        .unwrap()
        .pop()
        .unwrap();
    let subtree_a0 = from_newick("(A0:1.0);").unwrap().pop().unwrap();
    let mut trees: HashMap<&str, Tree> = HashMap::new();
    trees.insert("I2", tree.clone());
    trees.insert("I1", subtree_i1);
    trees.insert("A0", subtree_a0);
    let records = vec![
        // Record::with_attrs("A0", Some("A0 sequence"), b"AA"),
        // Record::with_attrs("B1", Some("B1 sequence"), b"AG"),
        // Record::with_attrs("I1", Some("I1 sequence"), b"TA"),
        // Record::with_attrs("C2", Some("C2 sequence"), b"-A"),
        // Record::with_attrs("I2", Some("I2 sequence"), b"-A"),
        Record::with_attrs("A0", Some("A0 sequence"), b"AC---"),
        Record::with_attrs("B1", Some("B1 sequence"), b"A----"),
        Record::with_attrs("I1", Some("I1 sequence"), b"TCCAG"),
        Record::with_attrs("C2", Some("C2 sequence"), b"-CCAT"),
        Record::with_attrs("I2", Some("I2 sequence"), b"-CCTA"),
    ];
    let seqs = Sequences::new(records.clone());

    let builder = AncestralAlignmentBuilder::new(&tree, seqs);
    let msa: crate::alignment::AncestralAlignment = builder.build().unwrap();

    let phylo = PhyloInfoAncestors {
        msa,
        tree: tree.clone(),
    };
    let jc = JC69::new(&[0.25, 0.25, 0.25, 0.25], &[0.3]);
    let tkf_model = TKF92Model {
        q: jc,
        params: vec![0.3, 0.4, 0.5],
    };
    let model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));
    println!("leaf seq info {:?}", model_info.borrow().leaf_sequence_info);
    let tkf_cost = TKF92Cost {
        model: tkf_model,
        phylo,
        model_info,
    };

    let tkf_cost_with_out_felsenstein = tkf_cost.logl_old();
    let mut felsenstein_cost = 0.0;
    let n_blocks = tkf_cost.model_info.borrow().blocks.len();
    for block_id in 0..n_blocks {
        let block = tkf_cost.model_info.borrow().blocks[block_id];
        let block_len = tkf_cost.model_info.borrow().block_lens[block_id];
        let mut insertion_node = &tkf_cost.phylo.tree.root;
        for node in tkf_cost.phylo.tree.postorder() {
            let is_insertion = if node == &tkf_cost.phylo.tree.root {
                tkf_cost.is_insertion_at_root(block_id)
            } else {
                tkf_cost.is_insertion_at_non_root(node, block_id)
            };
            if is_insertion {
                insertion_node = node;
                break;
            }
        }
        // extract subtree and submsa for this insertion

        let sub_tree = trees
            .get(&tkf_cost.phylo.tree.node(insertion_node).id[..])
            .unwrap();
        let sub_msa = AlignmentBuilder::new(
            &sub_tree,
            Sequences::new(
                records
                    .iter()
                    .filter(|x| !x.id().starts_with("I"))
                    .map(|x| {
                        Record::with_attrs(x.id(), x.desc(), &x.seq()[(block - block_len)..block])
                    })
                    .collect(),
            ),
        )
        .build()
        .unwrap();
        let subst_phylo = PhyloInfo {
            msa: sub_msa,
            tree: sub_tree.clone(),
        };
        let sub_model = SubstModel::<JC69>::new(&[0.25, 0.25, 0.25, 0.25], &[0.3]).unwrap();
        let cost = SubstitutionCostBuilder::<JC69>::new(sub_model, subst_phylo)
            .build()
            .unwrap();
        let curret_cost = cost.cost();
        felsenstein_cost += curret_cost;
    }
    assert_relative_eq!(
        tkf_cost_with_out_felsenstein + felsenstein_cost,
        tkf_cost.logl()
    );
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
    assert_relative_eq!(TKF92Cost::<JC69>::b(0.3, 0.5, 0.7), 0.5461782813185221)
    // (1-e^((0.3-0.5)*0.7))/(.5-.3*e^((.3-.5)*0.7))
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
    assert_relative_eq!(TKF92Cost::<JC69>::h1(l, m, b, time), 0.004350089645603061)
    // e^(-4.5) * (1-2(1-e^(-1.5))/(3-2*e^(-1.5)))
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
    assert_relative_eq!(TKF92Cost::<JC69>::log_n1(l, m, b, time), -2.732135332549935)
    // log((1-e^(-1.5) - 3(1-e^(-.5))/(3-2*e^(-.5)) )* (1-2(1-e^(-.5))/(3-2*e^(-.5)))   (2(1-e^(-1))/(3-2*e^(-1)))^0)
}

// testing felsenstein: if i only use N as nuc then the substitution process should
// vanish and only the indel process should remain
