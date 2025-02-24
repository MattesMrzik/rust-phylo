use std::{cell::RefCell, process::exit};

use crate::{
    alignment::{AlignmentBuilder, AncestralAlignmentBuilder, Sequences},
    evolutionary_models::EvoModel,
    likelihood::ModelSearchCost,
    phylo_info::{PhyloInfo, PhyloInfoAncestors},
    substitution_models::{
        dna_models::{GTR, JC69},
        QMatrix, SubstModel, SubstitutionCostBuilder,
    },
    tkf92_model::{reassignment::ReassignEdge, TKF92Cost, TKF92Model, TKF92ModelInfo},
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
    // arrange
    let newick_string = "(((A0:1.0,B1:1.0)I1:1.0,C2:1.0)I2:1.0);";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let records = vec![
        Record::with_attrs("A0", Some("A0 sequence"), b"CC--"),
        Record::with_attrs("B1", Some("B1 sequence"), b"A---"),
        Record::with_attrs("I1", Some("I1 sequence"), b"TCCA"),
        Record::with_attrs("C2", Some("C2 sequence"), b"-CCA"),
        Record::with_attrs("I2", Some("I2 sequence"), b"-CCT"),
    ];
    let seqs = Sequences::new(records);

    let msa = AncestralAlignmentBuilder::new(&tree, seqs.clone())
        .build()
        .unwrap();
    let msa_without_ancestors = AlignmentBuilder::new(&tree, seqs).build().unwrap();
    let phylo = PhyloInfoAncestors {
        msa,
        tree: tree.clone(),
    };
    let phylo_without_ancestors = PhyloInfo {
        msa: msa_without_ancestors,
        tree,
    };
    let freqs = &[0.1, 0.2, 0.3, 0.4];
    let params = &[0.1, 0.2, 0.3, 0.4, 0.5];
    let gtr = GTR::new(freqs, params);
    let tkf_model = TKF92Model {
        q: gtr.clone(),
        params: vec![0.3, 0.4, 0.5],
    };
    let model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));
    let tkf_cost = TKF92Cost {
        model: tkf_model,
        phylo: phylo.clone(),
        model_info,
    };

    // act
    let subst = SubstModel::<GTR>::new(freqs, params).unwrap();
    let cost = SubstitutionCostBuilder::<GTR>::new(subst, phylo_without_ancestors)
        .build()
        .unwrap();
    let felsenstein_cost = cost.cost();
    let tkf_cost_with_out_felsenstein = tkf_cost.logl_old();

    // assert
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

#[test]
fn mytest_index_to_bools() {
    assert_eq!(
        ReassignEdge::<GTR>::index_to_bools(0),
        ([false, false], vec![false, false, false, false, false])
    );
    assert_eq!(
        ReassignEdge::<GTR>::index_to_bools(1),
        ([false, false], vec![false, false, false, false, true])
    );
    assert_eq!(
        ReassignEdge::<GTR>::index_to_bools(9),
        ([false, false], vec![false, true, false, false, true])
    );
    assert_eq!(
        ReassignEdge::<GTR>::index_to_bools(65),
        ([true, false], vec![false, false, false, false, true])
    );
}

#[test]
fn mytest_reassignment() {
    // arrange

    // TODO: test for cases where factor n is used in the opti path
    // or cases where the factor n makes a difference in the opti path
    // maybe turn of the factor n and see if dp fails to get the best assignment

    // let newick_string = "(((A0:1.1,B1:1.2)I1:1.3,C2:1.4)I2:1.5);";
    // let tree = from_newick(newick_string).unwrap().pop().unwrap();
    // let records = vec![
    //     // Record::with_attrs("A0", Some("A0 sequence"), b"CC--"),
    //     // Record::with_attrs("B1", Some("B1 sequence"), b"A---"),
    //     // Record::with_attrs("I1", Some("I1 sequence"), b"TCCA"),
    //     // Record::with_attrs("C2", Some("C2 sequence"), b"-CCA"),
    //     // Record::with_attrs("I2", Some("I2 sequence"), b"-CCT"),  Record::with_attrs("A0", Some("A0 sequence"), b"CC--"),
    //     Record::with_attrs("A0", Some("A0 sequence"), b"CC--"),
    //     Record::with_attrs("B1", Some("B1 sequence"), b"A---"),
    //     Record::with_attrs("I1", Some("I1 sequence"), b"AC--"),
    //     Record::with_attrs("C2", Some("C2 sequence"), b"-CCA"),
    //     Record::with_attrs("I2", Some("I2 sequence"), b"AC--"),
    // ];

    // http://www.trex.uqam.ca/index.php?action=randomtreegenerator&project=trex
    let newick_string = "(((3:0.7139,10:1.0807)N15:0.6095,((11:1.3571,(6:0.7200,(1:0.8712,12:0.7348)N16:0.7334)N17:1.8990)N18:0.5412,(15:0.5554,(7:0.6160,8:1.0826)N19:2.3577)N20:1.3745)N21:0.6109)N22:1.3500,(4:0.5974,(13:0.6112,(9:1.8808,(2:1.2448,14:0.9331)N23:1.5854)N24:0.6282)N25:1.0651)N26:0.7182)ROOT;";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let records = vec![
        Record::with_attrs("3", Some("description"), b"-TGAAAAGTCt-C"),
        Record::with_attrs("10", Some("description"), b"-ATTCTCCAAt-T"),
        Record::with_attrs("11", Some("description"), b"-AATGCTCAC--C"),
        Record::with_attrs("6", Some("description"), b"-CACACCCAA-aT"),
        Record::with_attrs("1", Some("description"), b"-CAAT-CCCC--T"),
        Record::with_attrs("12", Some("description"), b"-CTTGTAACA--G"),
        Record::with_attrs("15", Some("description"), b"-TGGTTCGTC--A"),
        Record::with_attrs("7", Some("description"), b"-AGCGGTTA---G"),
        Record::with_attrs("8", Some("description"), b"-TCGGCTAA---G"),
        Record::with_attrs("4", Some("description"), b"-GCATTCTTC--A"),
        Record::with_attrs("13", Some("description"), b"-TGGCGTCCT--G"),
        Record::with_attrs("9", Some("description"), b"-TCGTCTGCA--C"),
        Record::with_attrs("2", Some("description"), b"-CCATATCAA--G"),
        Record::with_attrs("14", Some("description"), b"tCCGTTCCTC--A"),
        Record::with_attrs("N15", Some("description"), b"-AAAGGCCCCt-C"),
        Record::with_attrs("N16", Some("description"), b"-CTACTACCA--G"),
        Record::with_attrs("N17", Some("description"), b"-ATCTTGCCA--C"),
        Record::with_attrs("N18", Some("description"), b"-ATAACTGTC--G"),
        Record::with_attrs("N19", Some("description"), b"-TCCGAGCA---G"),
        Record::with_attrs("N20", Some("description"), b"-TGGATAGTA--A"),
        Record::with_attrs("N21", Some("description"), b"-TTAAGCGTG--C"),
        Record::with_attrs("N22", Some("description"), b"-TTAAGCATG--C"),
        Record::with_attrs("N23", Some("description"), b"-CCATACTGA--G"),
        Record::with_attrs("N24", Some("description"), b"-ACCAATGCT--A"),
        Record::with_attrs("N25", Some("description"), b"-TGCTAATCC--A"),
        Record::with_attrs("N26", Some("description"), b"-CCTAGCGAC--A"),
        Record::with_attrs("ROOT", Some("description"), b"-TATAGCTTC--G"),
    ];

    let seqs = Sequences::new(records.clone());

    let msa = AncestralAlignmentBuilder::new(&tree, seqs.clone())
        .build()
        .unwrap();
    let phylo = PhyloInfoAncestors {
        msa,
        tree: tree.clone(),
    };
    let freqs = &[0.1, 0.2, 0.3, 0.4];
    let params = &[0.1, 0.2, 0.3, 0.4, 0.5];
    let gtr = GTR::new(freqs, params);
    let tkf_model = TKF92Model {
        q: gtr.clone(),
        params: vec![0.3, 0.4, 0.5],
    };
    let model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));
    let tkf_cost = TKF92Cost {
        model: tkf_model,
        phylo: phylo.clone(),
        model_info,
    };
    let v2_idx = tkf_cost
        .phylo
        .tree
        .postorder()
        .iter()
        .find(|x| tkf_cost.phylo.tree.node(x).id == "N19")
        .cloned()
        .unwrap();

    // act

    let original = get_tkf_prob_for_records(records, &tree);
    println!("original {}", original);
    // println!(
    //     "'best'? {}",
    //     get_tkf_prob_for_records(
    //         vec![
    //             Record::with_attrs("A0", Some("A0 sequence"), b"CC--"),
    //             Record::with_attrs("B1", Some("B1 sequence"), b"A---"),
    //             Record::with_attrs("I1", Some("I1 sequence"), b"AC--"),
    //             Record::with_attrs("C2", Some("C2 sequence"), b"-CCA"),
    //             Record::with_attrs("I2", Some("I2 sequence"), b"AC--"),
    //         ],
    //         &tree
    //     )
    // );

    let mut reassign = ReassignEdge::<GTR>::new(tkf_cost);
    reassign.fill_dp(&v2_idx);

    let found_assignment = reassign.backtracking();

    println!("found assignment");
    for block in found_assignment {
        println!("{:?}", block);
    }

    // assert
    let l = reassign.cost.model.lambda();
    let m = reassign.cost.model.mu();
    let r = reassign.cost.model.r();
    let mut logl: f64 = (1.0 - l / m).ln();
    logl += reassign.cost.phylo.msa.len() as f64 * r.ln();
    for node in reassign.cost.phylo.tree.postorder() {
        if node == &reassign.cost.phylo.tree.root {
            continue;
        }
        logl += TKF92Cost::<GTR>::log_i1(
            l,
            reassign.cost.model_info.borrow_mut().beta[usize::from(node)],
        );
    }
    println!("missing prob {}", logl);
    println!("last col should be {}", original - logl);
}

#[cfg(test)]
fn get_tkf_prob_for_records(records: Vec<Record>, tree: &crate::tree::Tree) -> f64 {
    let seqs = Sequences::new(records);

    let msa = AncestralAlignmentBuilder::new(&tree, seqs.clone())
        .build()
        .unwrap();
    let phylo = PhyloInfoAncestors {
        msa,
        tree: tree.clone(),
    };
    let freqs = &[0.1, 0.2, 0.3, 0.4];
    let params = &[0.1, 0.2, 0.3, 0.4, 0.5];
    let gtr = GTR::new(freqs, params);
    let tkf_model = TKF92Model {
        q: gtr.clone(),
        params: vec![0.3, 0.4, 0.5],
    };
    let model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));
    let tkf_cost = TKF92Cost {
        model: tkf_model,
        phylo: phylo.clone(),
        model_info,
    };
    tkf_cost.logl()
}

// #[cfg(test)]
// fn build_random_tree(n_leaves: usize) {
//     let mut rng = rand::thread_rng();
//     let matrix: Vec<Vec<u8>> = (0..n_leaves)
//         .map(|_| (0..n_leaves).map(|_| rng.gen_range(0..=2)).collect())
//         .collect();

//     let nj_distances = NJMat {
//         idx: (0..4).map(NodeIdx::Leaf).collect(),
//         distances: dmatrix![
//                 0.0, 4.0, 5.0, 10.0;
//                 4.0, 0.0, 7.0, 12.0;
//                 5.0, 7.0, 0.0, 9.0;
//                 10.0, 12.0, 9.0, 0.0],
//     };
//     let sequences = Sequences::new(vec![
//         record!("A0", b""),
//         record!("B1", b""),
//         record!("C2", b""),
//         record!("D3", b""),
//     ]);
//     let nj_tree = build_nj_tree_from_matrix(nj_distances, &sequences, |_| 0).unwrap();
// }
