use crate::{likelihood::TreeSearchCost, tree::NodeIdx};
use rand::{seq::IteratorRandom, thread_rng};
use regex::Regex;
use std::cell::RefCell;
use std::fs;
use std::process::Command;

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
fn mytest_virtual_rerooting_with_old() {
    let newick_string = "(((A0:1.0,B1:1.0)I1:1.0,C2:1.0)I2:1.0);";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let seqs = Sequences::new(vec![
        Record::with_attrs("A0", Some("A0 sequence"), b"-N-N-"),
        Record::with_attrs("B1", Some("B1 sequence"), b"N-N-N"),
        Record::with_attrs("I1", Some("I1 sequence"), b"N--N-"),
        Record::with_attrs("C2", Some("C2 sequence"), b"N----"),
        Record::with_attrs("I2", Some("I2 sequence"), b"N--N-"),
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

    let first_old_logl = tkf_cost.logl_old();
    let first_logl = tkf_cost.logl();
    tkf_cost.virtual_reroot_with_id(tkf_cost.tree(), "I1");
    let second_old_logl = tkf_cost.logl_old();
    let second_logl = tkf_cost.logl();
    // before calculating i need to reset the nodes that are effected by the move

    assert_eq!(first_old_logl, second_old_logl);
    assert_relative_eq!(first_old_logl, first_logl);
    assert_relative_eq!(first_logl, second_logl);
}

#[test]
fn mytest_virtual_rerooting_with_substitution() {}

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
    println!("the cost is {}", ModelSearchCost::cost(&cost));
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
    let felsenstein_cost = ModelSearchCost::cost(&cost);
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

    let newick_string = "(((3:0.7139,10:1.0807)N15:0.6095,((11:1.3571,(6:0.7200,(1:0.8712,12:0.7348)N16:0.7334)N17:1.8990)N18:0.5412,(15:0.5554,(7:0.6160,8:1.0826)N19:2.3577)N20:1.3745)N21:0.6109)N22:1.3500,(4:0.5974,(13:0.6112,(9:1.8808,(2:1.2448,14:0.9331)N23:1.5854)N24:0.6282)N25:1.0651)N26:0.7182)ROOT;";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    let records = vec![
        Record::with_attrs("3", Some("description"), b"--C-cC----------T-GT--G"),
        Record::with_attrs("10", Some("description"), b"--T-cC----------C-TT--T"),
        Record::with_attrs("11", Some("description"), b"--G--TacccgaTccTT-TATaC"),
        Record::with_attrs("6", Some("description"), b"--G-aC-----t------CGCaT"),
        Record::with_attrs("1", Some("description"), b"--GttT-----a------CGCcC"),
        Record::with_attrs("12", Some("description"), b"t-GgaC-----c------CGCaC"),
        Record::with_attrs("15", Some("description"), b"--G-aG-----gG--CG-TAA-C"),
        Record::with_attrs("7", Some("description"), b"--T-cC-----aT--GTa-----"),
        Record::with_attrs("8", Some("description"), b"--A-cA-----gC--CGt-AA-T"),
        Record::with_attrs("4", Some("description"), b"-CT--C------A--CT-AGG-T"),
        Record::with_attrs("13", Some("description"), b"-TA--T------A--CT-AG---"),
        Record::with_attrs("9", Some("description"), b"-TA--G------G--AA--C---"),
        Record::with_attrs("2", Some("description"), b"-G---A------T--TT-AT---"),
        Record::with_attrs("14", Some("description"), b"-AC--T------G--TT-TG---"),
        Record::with_attrs("N15", Some("description"), b"--A-gC----------C-TT--A"),
        Record::with_attrs("N16", Some("description"), b"--GgtC-----a------TGCaC"),
        Record::with_attrs("N17", Some("description"), b"--G-tG-----t------CACaT"),
        Record::with_attrs("N18", Some("description"), b"--C-cT-----aA--TG-TTAtT"),
        Record::with_attrs("N19", Some("description"), b"--T-cC-----aT--TGc-GT-G"),
        Record::with_attrs("N20", Some("description"), b"--G-gG-----gG--TG-CGA-C"),
        Record::with_attrs("N21", Some("description"), b"--C-gT-----aA--TG-TTA-T"),
        Record::with_attrs("N22", Some("description"), b"--A-gT------T--CG-TTA-C"),
        Record::with_attrs("N23", Some("description"), b"-TC--A------G--TT-AT---"),
        Record::with_attrs("N24", Some("description"), b"-TC--G------A--GA-CC---"),
        Record::with_attrs("N25", Some("description"), b"-TC--T------A--CA-AG---"),
        Record::with_attrs("N26", Some("description"), b"-TT--A------A--CA-TTG-T"),
        Record::with_attrs("ROOT", Some(""), b"-GT--A------A--CC-CTA-T"),
    ];

    let newick_string = "(((3:0.7139,10:1.0807)N15:0.6095,((11:1.3571,(6:0.7200,(1:0.8712,12:0.7348)N16:0.7334)N17:1.8990)N18:0.5412,(15:0.5554,(7:0.6160,8:1.0826)N19:2.3577)N20:1.3745)N21:0.6109)N22:1.3500,(4:0.5974,(13:0.6112,(9:1.8808,(2:1.2448,14:0.9331)N23:1.5854)N24:0.6282)N25:1.0651)N26:0.7182)ROOT;";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    // [randomseed]	1
    let mut records = vec![
        Record::with_attrs("3",Some(""),b"--------GCC-G-----GTAG-AGa---GTGT-GA--cAG-T-CG--A---TA----------CGA---T----TCG---GTGAG-a--AT--GG-A--GCAAcAG-C---C-CAA-T-CGG-GtG---AC------AACG-GC---C-ATCTCA-GG-TCGCTGTT---------T----TG-AcTCAAAT------ACTT-CGAT----G-GGC---CA-GTGAG--G----A--TaGaGTCa-T--Gg---Gg--GGGgG-GCCGGTG---Agac--G---C-ATA-TGCcT-G---GT-C-AGTC---GT-----G----TCAA-A-G-----CAGGCA-------TAG---T----ACGG-TGCA----CG-TA-T--CaaTGCGT---G-----------------CATCCTTC--T----A-T--G----TA--Gg-T--CCAGTTTcTCAG----C-Cc-cCCCC-T-CG-TCA---ATGCA--TGTCTT--T-G--GCACT--CcGTGT-AAG-TG-ACAC------TAC----A-----tGTaagtACC--TTCT----AG-----T-T-------CGAT---T-CT--ACCATTGActTT--G------ACC-GTAGT---CCT--A--CCA---CTTGG-------"),//G-------CT--GGG---TTTA----CTACc--At----G-CT-CTC---aT--C--GA-AAGGAG-T-CGC-----ACGATGAGG-C----CAT-CGC-TAATTA---TT-CG-AGGA-ACTA-C---TA------A-TAT-C---CTGAAC-T----C-C---TA-C--CCGC-TAGAAA------TTG-TGAGG-ATaaa--TCA-CG------A--G-TT-C-C---CTAC--AG--GG-A-A--CGaGAGCG-gCTA-A----AA--C----T-T--C-A--TA--GACCCAtTGAgA-T---------------G----Ta-AGGAac-A-C----Gc---G-CCGACGGT--C-C-CC--CACAT------T-GC--G--AC--G-A-G--G---GGT-ATA--TAC--C--G-T-T-C-AGT-TggGACGTA-TAG--c--C--------GG-T---T-CTT-T--AGCGCA--AG---GGA-G-GgcgaA---------TAACCGGATCT--CC------CCC----C---A----TTTC-AGGGCt-----GGGCGTTTTT----TA-A--AG-----ACA-------CCAACACCGTTT-T---A-GCC-T-ACg---T--G-----G--GTTTCGAATT-T----CA---CCA--TA---C-A---------A--GGCC-----TCGCT-TCT-G-AA-AG-C-----AT--CC-GgTC--G-----T---------C--G---CGAG-GaCAT-----GCTTAT-----G----t----TGT--C----ACC--C--ATA-CC--C---GATGCT-T-CGTC-ACACATT-CCCGCT-G-A-CGGTCC-C-----GCCG--G--GCCCAtgATG-TAAT--CAC--GGT--------TC-G-G----G-CC-cGGAcgTG-G--------a----C-A-AGTA--tGTAGT-------aT--CGC-ATGCaGCTGA---TA---TGtG--T-----G-C-a-ag--C---T------T------C-G---T-CCA-TT-CCGA---CG-T--tAG--GT--CG----GG--C----G---C--C--A-CA---------------TCG--AG-GG--A--CG-CG-CC-----A----CCA--T-A----C-GAGT--C----CTA-CTA-T-CAAC--CG-CCGA-T-G--TCGTA-GTCG-GA-TGC"),
        Record::with_attrs("10",Some(""),b"--------ATG-A------GGG-TG----TGCT-GGtttGA-C-CT--C---TG------------C---T----TGG---CGGAG-g--CG--AT-G--AAGAtTA-T---A-ACG-T-TAT-GtG---TC------CTTG-AT---C-AAGACT-GT-TGGCTACG---------T----AG-CcTTTAAT------CTAA-TAGC----A-AT----GT-CTTCT--G----T--TcCtCAAt-AG-Cac--Ac--TACgG-ACGTATC---C-----G---C-CTA-TGG-A-A---AA-C-TGTG---GAaccaaG----CCTT-C-G-----CCCCCC-------CTC---C----CATT-TCCA----CT-GA-GCACtaGCCAA---G-----------------CAGG-GGCctA----C-A--T----CC--Ca-A--TTGCAGTcGCAG----A-GgccCTGC-A-GA-TTT---GTGAA--AGGTTA--C-T--TAATA--AgCCAC-GAG-TCGGTCG------AGC----C-----gAG----GGG--TTCGg---TG-----G-CT--AG-GATGC---G-AT--AATACTGTgcGT--G------GGT-GTGAT---CAC--A--CAC---AGCCAggatt--"),//G-------GA--AAGattATCA----GCAC---At----T-CG-GGCattgT--G--CTtGTTAAG-C-AAGcg---GGGCTAAAA-Gg---GCG-TAG-TGACTA---AC-AT-TACC-ATAA-T---GT------G-TAC-C---CTCGAC-A----CAC---T-----G-TC-CTCTCA----------AGTAT-AT-----CGA-AT------G--G-TT-T-A---TGGTg-AG--CA-A-T--CGgGGACGcgACT-A----GC--C-TTGA-C--G-C--AG--TATGCTaTTG-CaG---------------G----G-TAAGGtc-C-A----Tt---G-GTCCATGTA-T-C-TT--CTAATaatttgC-GA--G--A---A-T-A--G----CG-ATG--TCC--C--C-C-C-A-ACA-C-gTAGGCA-GCG--g--Caactca--CC-C---T-CAC-T-TGCGTCA--TGC-TGCC-T-GgtccA---------TACGAGCCTCT--CG------TGC----T---T----GGAA-ATGATg-----CCGACGGATT----TA-C--GT-----GCC-------ATAAGACCCGTG-C---G-TCT-C-GCatt-T--A-----Aa-GTCTCTGGCT-A----AA---TGC--AGgatC-G---------T--GACC-----TAAAT-CCT-A-GG-CT-T-----CTg-CTaA-TC--A-----T--GA-----C--C---TGCA-A-GAG-----TCTCTA-----T----g----AGA--T----GCA--G--AAC-GT--A---AGGCGG-G-AAGT-GCCGACG-GGCTCG-C-C-CATTGT-A-----GTAG--A--TGTCCt---AaGTTA--TAG--GGT--------GG-A-C----A-AT-aAGTcaGT-AG-GCAAAGt----T-G-CATC--aGAACC-------aT--CCC-GGCGaGTGCT---AC---ACgC--T-----C-A-g-gg--C---G------A------A-C---T-CTC--T-CTTA---AG-AC-tATaaTA--CT----TT--C----GG--G--CtaT-CT---------------GAG--CC-AG--G--GC-TA-CC-----G----TCA--CtT----A-ATCC--A----AGT-AGG-A-TGCC--CT-CGGG-T-A--AGGGG-ATAG----GAG"),
        Record::with_attrs("11",Some(""),b"--------AAC-GGT-GtTTCT-T-------CA-CG--tCC-C-GT--TccgCGaatgcc----TAA---G----GAAat-GGAAG-t--CG--GC-CccCATAgAGtT---Ag-CC-C-GAAcAtT---AG------GCGA-ATg--A-TC--------GTACACGTacaccgag-A----GA-A-----CG------GC-A-CTAT----C-GAC---GA-ACTA---G----C--GtTgGTTg-AAgCa---Ac--ATGaG-CTTGGGT---A-----Cg--GgTAC-TAA-C-C---CT-A-AGGA---GT-----T----GCCA-T-ATC-TaGAGGCA-------CTG---T----GGGT-AAGA----CT-C--CCGTacT-TGG---A-T---Gcc---Gt---GCCATTTCC--C----C-G--C----GCaaCa-G---AGTGTGaCTGA----C-Ta-tGTCG-T-CC-GTT-A-TGT-----G--GA--G-C-TTCAAT----CGGC-ACT-AGCTAG--------AG----T-----gGC----TTT----AA----TG-----A-ATatTG-AAGGT---C-TT--AAGGGAGG--AC--C------TCC-GTGAC---CGT--A--GTGaaaACTCT-------"),//T-------GT--CGC---CTG-----GCAA---G-----C-C--CAT---aA--A--AT-CCCTCT---TGA------TACCGTTA-T----ATA-GGT-C-TGCT---AT-CA-CACA-C--G-G---TCC-----C-GAT-Ct--CCCGC--Aaa--AGG---TT-T--AATCTGG--CG------CCT-GATAA-GC-----TAT-TA------G--T-CA-A-A---GCAG------GT-T-G--CT-TTTGC--GTG-T----GT--C-AGGC-A--T-C--CG--CTACAA-ATA-A-GCT------GGgact-T----A-A-TGG---C-C----A-C--C-TGTAATGC--C-T-CC--TAGCA------T--A--G--GGct------------GA--GC--AAC--G--CGT-TaG-TGG-G-cATTGTGcGGC--t--T--------TC-Tct-T-GGG-C-CCGAAAC--CGG-TGCAgC-C----A---------AATAG--TATA--GA-----CATC----Tc--Gtg--AGAA-CGTC-------GCCCTGCCCA----CA-A---------T-C-------GATAAATACCTC-A---T-GTA-T-TTa---T--C-----T--AGTCGACCGT-C----CA----TT--TC---G-Cg-----c-cA--AATC-----ATTCA-AGG-C-AGcTT-A-----AA--CA-C-GA--A-----CA--Ga----TcgG---CGGA-T-TTC-----GCCCCTcg---G-g--g----TGT--Tt---GGA--T--GCG-CG--G---CAGCGT-T-CGGG-ATATTGAaCTCGCT-----ATGAAT-T-----GTTA--A--CCGGA-gTGA-TTGT--TGA--GA---------CA-----GT-TaGA-tCTG--GA-AG-ATCAGGg----A-CaTGTT--cAGTTC-------aTt-CC----TAcACAAG---GG---GT-T--G-----CaGaccgaAaG---A------A------T-G-gg---GC-GA-AATGc--AC-AT-aCT--CG--CAaCA-ACaTC----TCc-A--A--AcAC---------------ATG--CG-CGggC-AGA-TA-GC-----T----TAAg-G-A----A-TT--TAA----CATtTTT-A-AGTG--CT-CCT------gGCTTGgTCGA-CA-ATC"),
        Record::with_attrs("6",Some(""),b"--------GAG-ATG-A-GAGT-TA----GAA-------TAcA-GT--G---CA----------TCC---G----GCC---TAT--ga--AG--TTaT---GACtTCtA---G--CG-T-ACAaCaC---CG------ATAC-TAtc-T-GAC-CC-ATaCCTGGCAG------gg-T----CT-C-----CTtat---CC-G-TACT----A-CAC---TA-TGAGT--T----A--GcAcGTC---A-At---Cc-------------AC---G-----T---A-CGC-AGG---A---GT-G-CGCC---CG-----T-------A-G-CCC-AgCGTTTGta---g-CAA---C----GACG-ACCA----AA-G--AA-AcgCAGAG---T-A---G--gg-Gt---CATCTTGAT--T----T-T---tc--TG--Ca-A--TTGACCGgGCGAta--TcTc-aGC-A-TtCCa-GC-A-GTA-----A--A---T-C-CGGGCG--A-GGCA-TC---GCCTCCT-----CGG----G-----aCG----TAAtcACGA----GC-----G-CCgaCT-GTGTG---TgA---GGGTCA-C--AG--C------GGC-ACTGG---ATT--T--ATG---CGTAA-------"),//T-------AAg-ACC---AAT-----CCCA---G-a------T-CAT---tA--C--AA-CC-TCT-A-AGT------CTGGGTGA-T----CCG-GAA-ACTCCA---CC--C-TCAG-ATGC-T---CTT-----A-ACC-G---CAGGAC-G----GT----CA-C--TGGCCAC--TA------AAA-AT-TG-T-----aAAC-A-------G--A-TT-T-T---TT-C------GC-C-G--AA--TACG--ATG-C----C---G-TCAG-A--T-T--CC--AACTGG-GCA-T-TCGa-----AG----gG----G-GAAAT---G-GaacaC-C--C-GCGTAA--A-A-AcGG--TG-TC------A-TT--A--GA--A-G-T-------TC-A-T--TA---C--CTC-A-A-TAGaG-cATCCTG-CAT-tc--C--------GT-Tga-T-GGA-G-TTAATAA--CAC-GTTTtAtG----T---------ATATG--TATC--AT-----ACGG----Ca--Gg--gGCTA-GTTTA------ACGTTGCCGA----TC-----------T--tta----CCCACACAG--AaA---T-CCTcAgAGa---T--Cg--tgT--TTGTCGTAAA-Cgac-TG----CG--GT---C-Ct-----g-aG--CCAGa----GTCAA-GTG-T-TC-CC-G-----TC--AA-G-TA-cC-----AA--AtatcgT--C---TCTA-A-TAG-----TCTATT-----Gct--c----CGT--Cg---GCT--TttGGC-CT--C---CCCGTGgGcATAGaCGGAA---CTAACA-----CCTAGA-------ACAC--T--AAATC-aATT-AT-C--CCT--AGG--------TG-----AA-CcTC-gC-T--GG-AA-TCTCTCg----T-C-GCT-----CACCtcc----tC--AT----GAtTTA-C---CA---CG-GacT-----T-T-aaggT-C---T------A------A-C-tgC-CAT-CA-TCAT---GC-AGctGT--AC--TTaTC-AA-GG----CT--A--C--T-TC---------------AATt-TG-AG--C-CGA-AG-GCgtt--C----GTG--A-G----A-CCTCGCC----GAA-TAT-A-TA----GT-GTGT-GtCccCATTA-TGTC-AG-ACC"),
        Record::with_attrs("1",Some(""),b"--------CTCaTGG-A-CTGA-G-----G-----T--cGA-T-TT--T---CA----------ATC---G----TGT--cTAA--ct--CT--GAtC---TTGgG--T---G--CC-A-GCAgCtG---GA------CATAtCAt--T-TCT-CT-GAaTCAAGGGC------gc-Acat-GT-C-----TT------CT-T-GGGC----A-TGA---CA-TGGGC--A----------aGAC---G-At---Gg-------------AG---C-----G-tgG-GAC-AAC---A---AG-C-AAGA---GC-----A----TTGA-G-CTT-AtATCGCTac---t-ACT---C----AAGA-GGTC----TA-T--GC-AtaGC-AG---A-T---A----aAc---ATCCTCTCT--T----A-A---ga--TT--Tg-G--ATAAGAGaTAGGaac-C-Ac-aCT-A-A-TTa-GA-C-CAG--------TG--G-C-CTCGCC-cG-TCTG-GT---TAATACA-----TTG----A-----gAG----GCCtgGATT----CTcggctG-CGtaTC-GGTAG---Cg----GGTGGG-G--TG--A------TCC-AAAGA---TGC--C--CGG---TTTAT-------"),//T-------GAt-AA------------CACG---T-c------C-CGC---gA--Ta-AA-GACTAT-A-CCT------TAGAAGCT-C-t--ATA-CGG-TTCCACtt-TA--G-ATCG-ATGT-T----CAgtgacT-GCA-T---CATTTC-A----TG----G--CgcCCACGCA--TG------GAA-TT-GG-TG---ttAAA-TT------A--A--C-G-A---CG-C------TT-G-T--AC--CGAA--C-T-A----C---C-GCACaC--CcG--CG--ATTACC-CCA---AAAa-----TA----aC----A-CGACC---A-C-----------GAGACGC-C-G-GaCG--GC-TA------A-GActT---A--G-A-T-------GG-AG---ATA--C--CCC---CgAA-tC----GCAC-GTG-ct--C--------GT-Tgt-A-TCC-G-ACGTAGG--TCC-GGAGcGaG----G---------AGGTG--CCTA--GA-----CGTA----Ct--Tt-tgTCCT-TCTGG------ATCAGCCAAC----CG-T---------T--ctc----TGCA-CGTA--TaT---C-TCT-AaCCa---T--AtctgtC--TTGCGAGAGT-Cca--AC----GAt-GG---T-Gt-----g-aC--CTTC-ttccGACTTtAGT-C-A---C-G-----AG--GG-C-TT-tA-----AG--Ac-------C---GCAGcC-GTG-----GGCGCT-----Ctt--t----TGT--Ag---GAC--TccGAC--T--T---CGGCTAcG-GCCAgAATCA---CGGCCTa----ACCTGT-C-----T-----------TC-gCGG-GA-C--ACG--GTC--------TT-----CG-CaTA-gTCG--GG-GA-TGGCGAg----T-G-T--------TACggc----cG--TT----T-cTCC-C---CC---TA-TagC-----G-A-agctT-G---A------Atca---C-TcgcT-CCC-GA-TGAG---CA-GC-cCT--CCtaTGgTG-CA-TGaagaCA-cG--A-----A---------------GAGc-CG-TC------C-CA-CTccc--T----TAC-tC-G-tt-G-GGAAATT----ATG-ATC-G-GC----GC-TCCT-GcTttAACCG-GGTT-GTg-GA"),
        Record::with_attrs("12",Some(""),b"--------GAC-TCT-A-AAGCgAT-t--A-----C--tTT-G-AC--A---CT----------ATC---C----GCT--gCAC--ac--CG--AAcC---TATcACtT---G--GA-A-T-AgAgC---CT------GAAT-CAa--A-GAG-GC-TAcCAAGCGTA------gg-T---tAC-T-----CT------GT-G-CGGC----A-TAA---TA-GGAGC--A----C--AcAaGTC---A-Ac---Ga-------------TT---C-----A-caT-GAT-CAA-------CG-C-GCAC---AT-----T----CTAAaC-CTG-GcCTCGCTgactgttGAG---C----CGCG-GTTT-------G--GA-TgtCGCAT---A-G---A----gGa---CATAATACC--G----T-Att-gt--CA--Cg-A--TTGCTCGcCATTat--C-Gc-aCA-A-G-ATt-TG-T-ACC--------AA--T-TaCCGAAG--G-GGTC-GA---CTCGTCC-----TTT----T-----gAG----GATcgGCTG----AGgatggT-GTgtTA-GGTTT---Ag----GGTTAG-G--CC--C------CTC-TTGAT---ACG--G--ACT---GCTAA-------"),//C-------GAt-GGA---ATC-----TCCT---A-a------T-CGT---tA--A--GA-CCCTGT-T-CGA------ACTACGCT-T----ATC-GGA-GGTCTA---TC--T-CGAT-AGTCgG---TGTacacc----A-T---AGGCAC-A----CT----T--G--CTGTCCG--GG------AAT-AC--A-TA---ttACAgTG------G--C--T-T-A---AA-G------AC-A-A--CA--AGAG--TTA-A----A---T-AACCcA--TaA--GG--ATATTG-TTA-A-GCTa-----CA----cT----A-TTTAC---C-C----T-T--C-GAGGGTA-A-G-AtGA--GC-AG------T-CG--T---C--G-A-G-------TT-AG---GTA--A--GGA---AcTAAcA----CAAG-T-G-gt--A--------GC-Tgg-A--CG-G-CGGAAGT--GGA-GGCTtAgC----G---------AGAGC--GATA--CA-----CGAA----Ta--Ac--gACCA-CATAG------TGATCCCTTC----CG-T---------T--agc----TAGT-T--A--CaA---G-CGT-AgGGa---T--Tg--taC--GATATCATAT-Cga--CC----GC--GG---T-Gt-----a-cG--CCTT-----GAGAA-GTC-T-TA-TT-G-----TG--GT-A-AAcgA-----CC--Tt----T--C---TCAG-A-CTAcga--GGACGA-----Gct--g----AGC--Ta---AGC--TtgGGT-TG--T---CTTCAGtC-TTTGcAAAAG---AC-TGG-----ACCGAC-G-----TGG---------CC-tGTA-TC-T--TGG--TCT--------TA-----CT-AaC--aTCT--TG-TC-CCCACAg----T-C-T--------ACTccc----cT--TC----GGtAGC-C---CC---TA-GtcG-----T-C-aacgG-A---A------GtcagcaA-T-gcT-CAA-AC-AACG---GC-GG-tGT--TAccCGtTG-TC-GG----CG-aA--C--A-TG---------------AGGt-AG-AC------G-TT-CTcag--T----TCT--A-Aaga-G-TAACGGT----GTC-GCT-T-TC----AC-T-TA-CcTgtATGCG-TCAC-AAg-TA"),
        Record::with_attrs("15",Some(""),b"--------TTT-TTT-GcGACG-CC----TGGTt-T--tCA-AtAA--C---AA----------TCA--------TCG---CATGA-a--CG--CC-GctGT--cGT-A---T--AG-C-CCAtAgAaatAA------GTGA-TT---CtCGCTGT-AA-AATTACTG---------G----AC-T-C--GCG------A--G-TACTtttaGaCAG---CTgG--CT--C----A--C--tG-------------t--ACTaC-G-ATCCA---C-----A---G-G-A-CCT-C-CcttGTgC-CAGT---AG-----T----TTTC-A-AGG-GgAATAAT-------AAGtccTtct-CTGA-TCCG----TC-GC-GCTTcaCCCAA---C-C---C-----Tt---TTTGTCTAC--G----A-G--C----GG--Ct-G--T-ATGGCcGAGT----C-C----AAT-C-AT-GTC-C-TTT-----C--CA--A-T-G---AC--T-GAA--GCG-TAATACTAggc--TGTgtcaA-----gTC----CTT--ATAT----TG-----C-TActCGaAACCC---A-TT--GCGGCCAC--CG--C------CTA-AGCATaggCTA--A--GTC---T-CCC-------"),//T-------AG-cGGT---CGGA----CTCT-gcG-----G-GC-TGG---cC--T--TA-CAGTTA-A-TTA------CCGTGCAT-T----ATA-TTGgTCATTC---TG-AGcGTTC-ATAA-Cg--TTG-----C-CCG-G--cCTTTCG-C----AACgcc-AaG--GAGTTAT--TC------TCT-C-CC----------GG-GG------T--T-GGgC-T---ATTG------AA-T-C--GT-AAGTT--GCT-A----CTgtTgAATT-G--C-T--AA--GCGTGG-TAC-A-GAT------CG-----Ga---A-ATTGC---C-G----A-A--T-GCATCGACG-C-T-GCt-CCTCT------C-TG--T--CG--A-A-CagGctaT-----C---CC--C--CGGcC-T-TCC-G-tCCATCA-CTA--g--T--------CA-C-c-------------AATt-ATG-CCCC--------T---------AGCAA--CCGC--AA-----AAAC----C---A----CGAG-TAA----------GTTACTTT----GC-----G-----ATG-------GGGACCCTCTAC-C---G-CCA-G-AAa---C--G-----A--GCTTCGTACA-C----GA---CCC--GT---G-Cg-----c-cT--TCGA-----GA--T-AGG-G-TC-AT-T-----TC--CC-T-AC--Gt-g-tCA--Ta----T--C---AGG----GTT-----TTGCGG-----C-c-------ATA--G-a--CCT--C--GTCtGG--G---AACAAT-C-GGGT-GTCGCAG-CCTCTT-----CCTAATcA-----TCCA--C--ATCTTtaCGC-CTAGggGCA--AT---------CTaC-TTCT-G-CC-tCAC--GT-GC-ATGAAAc----A-C-TCAT--gC--CG-------cG--CG----AAtAGGCA---GT--aGG-C--TaggggT-T-ttaaG-C---C------A------T-C---A-ACC-TCg-GCA---GA-TG-aAG--AT--TG-GC-AG-GC----GTg-A--G--A-TC---------------CCA-aAC-CG-gG-GAC-AC-GC-----A----CTG--A-T----T-CTCGTAC----AT---T----TATC--GA-CGGTtT-GagGGTTC-TCCG--C---A"),
        Record::with_attrs("7",Some(""),b"ccaaaa--CCA-GGC-TtGTCA-TG----TGCA-TC--aGC-TtGCat-----A----------G-AccgG----TTC---CTTAG-a-------A-TacG-----G-G------AT-G-TTAcGaAg-cCAt-----AGGG-GT---TaGATACAc-T-GGAGGC-G--------a-----GAc-----GGG------G--C-CGC-gaccTcGTC---GGgT--GA--T----T--GtTaC-------------g--T-------TTAAG---C-----C---A-TAT-GGA-C-CtagTAgC-TATG---AC-----C----GCTG-G-CAT-T--TCGAC-------CAAtaaGact-ATA---TCTgt--CG-GA--ACTtaG-AGA---AcA-gcT-----Ccc--CCCATTA-G--C------G--C--------CccG--A-TGACAtGAGC----C-A----AAG-G-GA-CAA-G-GCA-----C--TC--A-C-A---AT--T-ACG--GTTtATGTAG-Agat--G-T----T-----gGG----CGA--TAGC-c--GG-------TActATcAAAGC---C-GT--GCCCCCGG---C--A------AATgAATGTacgAGGcaA--ATA---CATC-------t"),//Agggttt-AC--CTT---AAGA-----CTG---A-----A-AG-CTA---aA-----CA-AGTTAG-A-CCG------TAGACCGA-T----G---GATcGCTA-T--aGAtCTcTAAC-ACTG-G---TGG-----GaCGGgC-ggTCAGCG-G--t-TTGgct-AgT--TT-GGCG--CA------ACGcG-CTC-C-------TT-CG------TatG-CC-C-A---CGTC------AC-C-C-----TCTTT--C-T-Aaga-AGacTcAAGA-G--C-C--AAtaCCGACC-TTA-C-C-G------TT-----Ac---G-GGGCC--aG-G----C-G--C-------GAT-C-C-AGt-AGACC------A-GA--T--AA--T-AaCa--aaaG-----Ctg-AA-tAgcTAAaC---CAA-A-cCAG---------a--A--------CTaC-c-C-CTC---CGGTCAG--G---CGCC--------G---------GCTGC--ACCCat--ct---GCCAgt-cG---G----TGTG-T-C----------CCCGAACT----CTa----G-----GCG---t-atAGACGAAGCCTC-A---A-ATA-G-GGa--tA--C-----C--TCCCGCAATAcT----CT---TCG-aGC---Cc-attgggtcgAg-AATT-----GC--G-GGG-C-GA-ATcCcaccaTA--TA-G-CC--CaatcgAA--Ac----G--A---TCG----TCT-----AGCGTA-----G-a-------GGGacT-c--ACT--G--AACtAGggG------TAG-T-GCGT-GAAGTGG-CTTTCG-----ACGGTT-T-----GATCtcC--GCT-GtcGTG-AAGCatGCA--AC---------TG-C-TCCC---AGttCCG--AC-AA-TGAAATa----A-G-CTAAtgaC--AT---gggtcG--AA----GTaTA-C-catGGtttGA-C--T-----T-G-agtcC-C---C-------------A-T---A-TAG-AC--AT----AG-TA-tCC--AT--TT-GT-TA-TG----TAa-AtgA--T-GCt-----cttatt---GAG--GA-CC-tC-TAT-GA-GC-----C----AAT--T-G---aAaCATTTGTat--GA---TG-AtTCTA--GCcAG-G-G-CtgCT-TA-TAGC--T-CTA"),
        Record::with_attrs("8",Some(""),b"aga---ctATG-TTC-AgCCTT-GT----A--G-AG--cAG-AgCGta-----A----------G-Ac--C----AGC---GGGGA-c-------A-CtaT-----C-T------AA-G-GGAcTgCa-cGA-t--caAGTC--C---AgAAAT-Tt-T-GCATCCCT--------tG----CCa-----GGG------T--C-CTG-gcgtAgAAT---ACgC--AC--A----T--CcGgA-------------t--T-------TGGGT---C-----C---T-TTT-GTT-T-CtagTTgC-CAAG---CT-----C----AGTC-T-GAA-T--TAGCC-------ACTta--cga-TTA---GACcc--TG-TA--TAAacG-CC----TaG-aaA-----Cc---ACGAGTG-A--T------G--G--------TgtG--A-C--GCgAATC---cT-C----ATG-A-AA-GTC-G-TCT-----A--AT----A-G---CT--T-CCC--GATgAAGCAA-TtaattA-C----T-----gGT----CT---GTGC-a--CT-------TCacACtCGCTA---G-TG--GCAGT-TA---C--A------GATgTAATTatgAGTtaT--ATG---AACC--------"),//AaagtatgAG--TGG---GGAG---t-CTC---A-----C-ATgTTC---aC-----GG-TAGTAGaC-GAA------ATTCGGGT------A---GCGcTCTC-T---TCaCAgAGAT-TCCT-G---TGT-----CcGTAaG-ctAATCAA-G----GTTtgt-AgC--TT-GGTC--TC------CGTcC-TCC-A-------GC-AA------AgcG-CG-C-A---CTTC------ACgG-T-----AGTT---A-G-Aatt-AAatAgTTAG-T--A-C--ACgaGAGTGC-TC--T-C-C------TT-----G-------GTGG--gA-G----A-G--C-------GAA-T-T-GGt-GTTCA------T-GT--G--TC--T-CcGt--gtcG-----Ggt-AG-tTggCTGcC---CAC-C-cCCA---------c--G--------CC-A-t-A-CAA---AGCTCGT--A---CGGC--------A---------GACGC--ACTActTA-----TGAAgggtG---T----GACG-C-G----------CTAGACCA----GG-----G-----TCC---tgctCGCC-AGGTGGG-C---T--CC-T-GGc--aG--G-----T--ATGGAATGTG-T---gTC---GGC-------AaCacg--gg-gGa-GTTA-----AA--A-AGC-T-AT-AGgG---gaTC-aGC-G-TT--Ctttct----At----G--T---CCG----GCC-----TGA--G-----C-a-------TTAtgC-t--GAG--G--CGCcTAtaG---TGAGGT-C-TGGA-CACTA-G-TGTTGC-----GAGGT--G-----AAAC-tC--TCAAAgcCGG-CAGTctTGA--AA---------GT-T-TGTG-G-GGgtGTA--AC--G-AGAGAT-----T-G-TTCCgtaA--GG---gacatG--AA----ACtCG-T-acaACacaGC-G--G-----G-T-acttC-T---Ag-----A------A-T---GcGAC-TT--GT----AG-CA-aTC--CA--GG-CAaTG-GA----TTa--taT--A-TA-actagttgttc---CGA--CG-GC-gCcGTA-GG-CC-----T----TGG--G-A---tG-ATGGCGG----CC---CG-GtGTGA--ATcAA-C-G-AgaAA-TC-CCCC--T-TTT"),
        Record::with_attrs("4",Some(""),b"--------G----CGTG-TTG--------CAAT-TG---CA-A-CA--G---TA----------TGG---A----GCC---CTGCC--C-TT--CG-G--AGCC-CT-T---G-CGCGC-AGG-T-G---TT------TTGT-TT---T-CGGTAT-GA-TAGCTCCT---------A----GT-C-CAGCCA------GGCC-AG-A----T-CGAgct-A-GACTCtaC----T--T-G-GTC--GA-C----G-gtACT-CcCAGCTGT---A-----A---G-CGA-ATA-T-C---TC-T-CACT---AT-----T----CCAC-GgGTA-T-TCACTG-------GTT---A----ATGT-TATC--ACCG-TA-TATT--AGGGA---C-A---G-----A----CGAGTCGTC--C----G-G--C----GG--G--G--GGTGGGA-GGCC----C-C---TGAT-G--A-ACC-G-CTAGT--CGAAGT--T-A-CATTAA--A-CATT-GGA-CCACACTT-----TAC----CAATtA-AA----GA----GTG----TG-----A-AG--AT-GAG-Ga----GG--GCGATAA---TA--A------TAG-GAATA---TAT--CTCAGC---GCTAA-------"),//A-------AT--AGT---TCGG----GAAT---T-----T-CC-CCC----T--C--GG-CTCTTG-T-GAT-----AGAGTACTT-A----GTG---T-CCCAGT---TC-TC-TCTCcAACC-C---GCC-----G-ACA-C---CTGCAT-C---cATG---TT-A--GTTCTCAGCAC------AAC-GGGGA-AA-----GGT-GA---GAAG--G-GG-C-C---CATT---G--AC-CgCctC-----CA--CGC-A----TT--C-CTAT-ATCG-T------TT-ATA-TAA-C-CGA----CCCA-----A-C--T-AGTAG---G-T----T-T--C-GATGCTTGA-G-T-AG--TAC-A------C-GT--C--CC--A-G-G--C---TCC-GGA--TCG--C--GAC-A-G-CCG-A-----CCT-AGTG--acC------G-GA-C---GCCCT-C-TGGGATG---TG-TTCC-G-C----G---------ATATCGTT-AC--AG--ttaATAA----C---T----TCGG-TAAGT------TATTCAAGTG---aGT-A--AT-----ACT-------G----CA--AAC-C---C-ACA-G-CA----A--C-----C-tTCGGGCTCCA-C----CG---TAA--AC---A-A---------G--GCTA-----TCCGA-GGCGC-CG-AA-G-----GT--TA-T-TC--T-----CGAAT-----G--GCTGTATA-A-CCA-----TAAGTC--gagG---------TTC--C-----TT-cG--GTA-TG--T---CTGGAG-G-GGGA-CACAGTA-CCTTGC-T-C-ATAA---------TACA--C--TCCTG--GAG-ACTC--AAC--CCC------tt---C-AGTT-T-TG--ACA--ACTGCc-T-TCC-----C-T-TGCT---CTGTG--------G--AAGTTAGT-TACC----AT---TA-C--C-----T-A-----G-G---C-gg-t-G------C-T---A-CCA-AC-TCGG---GG-AG--CC--AG--CC-CT-GG-GG----TC--C--C--C-CT------------gaaGCG--AG-AC--T--TTCCCtGA-----CA--AACC--G-T----G-TATACAT----AAC-ATATA-CAGA--CA-ATTA-G-G--C-GA----TT-TG-CTC"),
        Record::with_attrs("13",Some(""),b"--------GAT-GCCGA-TTGC-TG----GAGT-AG---CC-A-CA--C---GG----------ACC---G----TTC---TTTTC--CATG--CT-G--TGAG-GG-C---A--TACGtCTC-C-A----A------GTTA-GC---T-GTTGCA-GC-CGGTTAAT---------A----GA-C-CTGCTC------A--C-TGGC----C-AGAgaaTA-GAGCTtgTtattC--T-T-G----GA-T--c-A-atAGA-GgACAAGGAtgaA-----T---G-C-T-GTA-T-A---AT-T-AGCT---TT-----------G---AaTCG-A-CTATCT--------AG---G----TTCCaTCTT--AGCA-ACaTT-T--TTCGT---T-T---C-----C--atCTAAATAAT--T----C-A--C----AC--C--A--CCACCTA-CTAA----A-C---GGCT-A--T-CCG-T-TGGGA--CGG-TG--A-G-GGTCGA--G-AATT-GCT-TAGCTTGG-----GAA----TTT----------GCG--TGGA----AA-----AaCG--CT-TTT-G-ac--GCg-CCGGAGAG--GAggC-a---gTAG-GGCCG---GTC--AAAAAT---TCTAC-------"),//A-------T---GTC---ACAA----TAAA---G-----A-AT-TGT----T--G--C--TGGTTT-A-CCA--ttcCTACCA-AG-C--cgATA-CCT-GCCCAG---TC-TC-CCTCcGATA-A---GTC-----T-ACG-G---CTACCA-C----TAA---GG-A--ATGCCAGACCC------ACC-GATGG-AC-----CGA-CG---GGGC--AaTT-A-T---AGGT-aGGc-AG-A-G--T-----CT--ACT-A----AA--G-TAAT-TCCT-Agt----CTTATT-GAG-G-TGG----TCTA-----T-A--A-AATAG---TgT----C----------TTCGTaGtG--G--CCC-T------T-TA--C--CG--T-T-G--A---CCG-TAA--TAT--C--CTG-T-T-GCC-G-----AAA-CG-T----G------A-TT-A---CTGGT-G-GGGGT-----CA-CCGA-C-A----A--------gGCGCGCTC-CT--GG--cagTTT-----C---A----TTCG-CGAAT-acataGAAGCAAGTG---gGT-AgtGCccg--GGC-------ACCCCAG--CGC-C---G-TCC-G-AA----A--C-------aCAAAG-CAAC-A----AC---TAT---------G---------G--GATA-----GGAGT-CGTACcTG-CA-------GT--AC-G-GA--T-----TTGAA-----T---ACAAGTT-A-GAT---a-ATAGAA-----C-----gcc-GCT--A-----CA-cA--TCC-CC--G---CCTGAA-T-GACG-AGAGAGA-AATAAA-C-G-C------T-----TCGA--A--TACGT--AGA-GCTA--GGG--CGTgcacggga---C-A-AG-T-AA--TAT--ACGACt-A-CCA-----GcT-TGCA---ATGT--------------ACTCGT-ATATT---TT---TC-G--T-----G---------T---G-g--t-T------C-C---A-CGAtTC-TCGC---CT-TG--AG--TC--CA-AG-AT-AG----CT--T--T--C-GT---------------CGT--TG-GC--T--AAGTTcCA-----AC--TACA--A-G----G-CAATATT----CCA-GGAGG-AAGG--AC-GGGC-A-C--G-TAC-TCGCcCA-GTC"),
        Record::with_attrs("9",Some(""),b"--------GAT-TGATG------GT----CGGG-CG---GC-C-AG--G---TC----------AGG---T----TTC---CTGCA--AAAG--AT-C--C--G-AT-C---A-TTC-T-GTC-G-C----A------CATG-A------GCCAGT-AG-TGACGC-----------A----TG----ACGCC------CGG-aATGA----A-GTCgc-TC-CAGATatAcagaT--A-A-GCA-tTA-T--c-T-gcAAC-GtAGGCATTg--A-----C-----A-G-GTG--aG---GT-A-CTAAga-AT-----A-----T-----TC--A-ATATTA--------AC---G----AATA-GCCA--TATGgACcAT-T--TCCCC-----C---G----------ATATGTTAG--AagaaAtG--G--g-CC--T--A--TGGGGAT-TTGC----G-G---TGGAcA--A-GATgCgATTGAt-AAG-GG--GgT-GGCTGG--T-TTAA-ACC-GGATTACA-----ATC----CCCCtT-AT----TTA--TACC--t-TG-----AcAT-----CTC-C-----CAa-AAGGATGG--TAtaC-gccagAA-----TC---CCT--TTGG------TCCT-------"),//C-------G---G--------G----CAGG---G-----TcCT-AAT----A--T-aG--TTCGCC-TcGAC-----CGGTTA-AC-G--g-ATG-GGA-TC--GC---TC-CC-CAATtG----C-ttTCC-----C-TG-------CT-TC-T----TAT---TT-T--TGT---TAGCC------AAC-G-TGC-GG-----CGG-CAtggA-GT--A-GG-T-Tgc-CTTT--GCc--A-A-A--C-----TT--TGCgG---aCC--G----C-TC-T-Tgc-------TGC-CAG-A-TAT----GATC-----T-G--T-A-TCA---TtT----T-C--Gt----CCTGG-GaC-GG-aGTT-T------GcTT--TtcGC--GtC-G--G---CAG-GCT--CAT--A---TT-C-C-AAA-T-----AGG-TT-T----A------C-AA-C---G-TGC-A-TGCCG-----TGgTACG---A----AaactggcggGCACAGGC-G----C--tttTTTG----C-cgT----CAGT-CGCTT-cgggaGGACTGACG--------TttTA-----AGA-------GCTACCA--GTA-C---TcAAG-T-AT----T--T-------aGTC-T-TCCG-T----GC---ATA-------------------A-cGCGT-----T-CC--TGACCgCT-AG-------GT--CA-T-GC--A-----GG--------------GCCCA-A-CTG----gT-ACGG-----T---------CGA--G-----G-gcG--A-T-CA--G---ATTGTG-A-GTAG-TGGAATG-CTATTT-TtGcGCG-TG-Acagg-ACGT--A--AGGA---GTT-GGC-----GaaTAA------cc---A-G-TG-A-GA--GAC--ACAGCa-A-TGT-ttctAcA-TCG----TGCTA--------T-cATGAAGCG-CAC-A---TT---AA-A--T-----A---------TgtaC-c-ca-A------AgC---T-TCCaCT-AGCT-ggCAgTT--GA--GA--TA-CA-GT-G-----GG--A--G--A-TC---------------GGG--TA-AG--C---CG--gAG---tgCT--A--A--T-C----G-ATAACTA----GTC-GGGAA-CACG--GA-CTAC-T-A--G-TA--GACGaAT-CTA"),
        Record::with_attrs("2",Some(""),b"--------ACC-TCATG------AT---tTCTA-T----CC-G-AT--A---TG------ttcaTAT---TcttcATA------GG--GCTG--AC-A--ACCG-TT-G---C-AAG-A-GCT-A-G----C------CGCG-GC--cG-CACTC--------ACAGG---------G----AA----TTATT---ataCAGAaATTA------GTAcgtAT-CC-AAtg-tt-gAcgA-T-GAT-aTC-T--caG-gcGGA-T----TGGGg--T-----G---C-A--gGAC--tT---GG-AgGCTA---CG-----Ttggc-C-------TgT-GGAAAC--------GG---C---gAATA-AGTG--GAGAcGTcAC-A--TGTTGagtC-T---G-----C----CCATGGATA--CccgaT----A--ctAT--A--AtgACCCCGT-TGTT----C-C---CTCA-C--G-CTA--cGGT-----AG-GGatA-G-GCGC----G-T-AA-T----GAGCAAG-----TTG----GGCCgT-CA----GGG--TA-C---aTT-----CtGC-----GTA-T-----CGt-ATGTA-TT--GTatTct---cTT-----CG---TAG--C--A-----------------"),//A-------G---GGA---GATA----ACAT---G--tgaCtA---------C--A--G--GTTTCG-GaTTT-----GGACCG-CGgA--g-CTCtTCA-ACTTAC---AT-AT-GCTTaC----------------C-ATT-T---T-CTCTt-----TCT---CT-G--ACATACAGAGTattacgTGG-AGG-TtTG-----AGG-TT---CAAT--G-CG-T-A---ATCG--ACg-TG-T-T--G-----CG--CCC-G---gTT--A-CGCG-TGTC-Tcc-------ACA-TTG-C-CTT-a--A-GG-----G-TtcT-G-TAC---TgA----T-C--A-----ATAAA-GaC-TC--AAG-G------C-AA--C--A---T-T-C--T---GCGcTTG--TTA--T--TAT-A-G-GCG-C-----TTC-TT------A------GgCT-A--tG-CTCc--AACGT-----AG-GTTT---C----TttccgccgaTTAGCGCA-CC--GC-----TAGT----C---C-------GtCTGCG-atataGGCTAGGTACctagAA-AaaGT---gcTTG-------CTCCAAC--GTT-GtgcA-TTG-T-CG----G--G-------tCT-TC-ATAT-T----GGt---CT---------G--------------AG-----T--TG-CCGC--TT-AG-------GG--CC-A-GA--C-----GTTAC-----C-----ATGAA---CAC--------CAT-----G--------tAAA--A--ga-AGc----G-A-GT--TgtaTGGTGC-A-GTAA-GGAGAAA-TCT--C-CgT-CGGGTA-G----aAATG--CttCGCC---TGC-GGAA--CTG--G-C------gg---A-C-GCcA-TT--GGC--ACTGCa---TCG---atCgA-TGCA---CTTTA--------A--CAGTGGTC-TCCCT---AT---T-----C-----A---------C---A-----a-------CaT---C--ACgAC-ATTT---AC-AA--C-------GT-GA-TG-GC----GC--A--C----TC---------------GAG--ACcGG--A--T-GC-aGC-----CTacACTT----T----C-AA-TAGT--tcTAC-CATAG-TGTCggTG-CTGG-T-T--C-AAG-AGACtGA-TGA"),
        Record::with_attrs("14",Some(""),b"--------TTA-GGGTG------TT---gACTA-C----CC-A-AA--G---T-----------------G----CTT---AATTT--AGTTgcGA-C--CTGT-GA-AtagT-AGG-T-GTG-C-C----A------AATC-AA---T-CTAAGA-C---TCCGATC---------A----AA----CCTTA---a--GTAAtGGAA------GTAcctTA-AG-GGat-attaTgtC-C-TAG-gTA-T--c-T-agAAA-A----G-AAt--A---ccA---T-C-CgCCG--gG---GG-TgCCAA---CC-----G-----G-----GAC-G-CCAGCG--------AA---C---tAAAC-GA-T--AACGcTGtTA-T--GAACCgccT-G---T-----G----TCCCTTCCG--TcattT-C--C--tcTA--A--AcaTTATTA---TTT----A-C---GTGC-T--A-GAA--aTCT-----GG-GCacA-G-TCGCTG--T-G-ATg-----GC-TCCA-----TTC----CTTAcG-AC----TCC--TA-A----CA-----AtGT--TA-GAA-A-----CTt-CTGGTGAT--TAtaT-a---cTA-----GC---GCA--GAAA-----------------"),//A-------T---TTG---TCTT----AGTA---G-----CtG---------CtaA--T--TAAACA-AgGCA-----AGCTCG-TT----c-GCAaTAC-TGTGGC---CA-AA-TAGTtT----A---GGC-----G-GTT-T---GCACTC-G----TAG---GT-T--TGCTAGGGTCA------TCG-ATCGGaTA-----TAA-AA---CGGT--A-CT-AtT--cCCGC--TCga-C-C-A--------T---TTC-A---tAC--T-AGCA-TTTG-Act--------AA-GGT-A-ACG-cggA-GG-----T-TctG-T-TGT---GgG----G-CgaC-----AATCC-AgG-AG--ACT-T------C-GG--A--AG--A-G-A--A---AGC-CTG--TCT--G--TGC-A-G-CAC-T-----GTA-GC------G------A-AC-C--aG-CTG-GaGTGGT-----CA-ACCC---C----TgagtgtttcACGGAGAG-AT--TG--ctgATCG--------T-------G-GAGCA-aacgaCCCCGCCCAGcggcCT-CtcTC---cgACC-------TTACCGT--TCC-Gc--A-GAG-G-AG----AacC-------cCATCT-GCGG-C----CTa-c-CT---------G--------------AC-----G--GC-GATC--CC-TG-------CG--CT-T-GT--A-----GTTGC-----C---ACACCCT-A-TTG--------GAA-----G---------CAG--G--c--AGc----G-G-GA--A---ATACAG-G-GACC-TGTCTTT-CTC--T-AgG-ACCCTA-T-----GAGG--A-tCCCT---GGT-TACT--GCG--C-A------gt---CcG-GA-A-TT--ACC--AATAAc---CT------AgC-TGCG---A-AAC--------C--ACGCGTAC-AACTG---AA---TG-A--A-----G---------A---C-----c-------CtT---A--GAgCA-ATGC---TC-AG--C-------TC-TT-AA-TA----CC--G--A----TC---------------AG---TGtAG--A--G-AC-aGC-----CG--ATGT----G----A-AG-GACC--taAGG-ATCAC-ATCGcaTT-CCAA-A-A--A-TGT-GACAcCC-TTA"),
        Record::with_attrs("N15",Some(""),b"--------AAC-G-----ATAG-TG----GTCC-GA--cGC-C-AG--C---TA----------CGA---G----GTG---GCAAG-g--AC--GT-A--TCGAgGA-A---C-AAA-T-TCG-GcG---GT------AGCA-AA---C-CTCTCA-AA-TCTCTGCT---------T----AG-CcTCGAAT------ACTC-CGTT----C-CGT---GA-CGTCG--G----A--TcGtGTAa-AG-Ag---Gc--GCGgT-ACGGGGG---C-----G---G-AGT-TGT-A-T---GT-G-AGTA---GT-----G----TAGT-A-G-----CGGGCT-------TGG---C----ACGA-TCCA----TG-TA-AAAGcaACCAG---G-----------------CTGGCTTC--A----T-T--C----CT--Gt-T--TGAGTGTaCCAC----C-Gg-cCCCC-T-GC-TCT---ATGCC--CGAACT--T-T--GAATA--CgCTGA-ACG-TCTCCAA------GAC----A-----gAT----ATT--TTCT----TG-----A-TT--AT-GATAG---T-CT--TACATTGTctGT--G------ACC-GAAAG---TAT--A--CCC---ATCGG-------"),//C-------CG--GAG---ACGA----GATC---At----G-CC-GAC---gG--C--TT-CATAAG-T-AGA-----ACGCCAGGA-G----CGT-CGT-TAAATA---TG-CA-TGCA-ACCA-T---TC------C-AAA-A---ATGGGC-C----CAC---AC-T--CGGC-CGTGTA------TTG-CGAAG-AT-----CGG-AT------A--C-GT-C-A---TCGC--AG--AC-A-C--TGcGAGCG-gCTA-A----AA--C-GCTA-C--C-A--TA--CCCCCAgTGA-A-G---------------G----T-TAGGAat-A-C----Cc---A-ACGTGTGTA-C-C-CT--AATAT------G-GC--C--AC--G-A-A--G---AGG-ATC--TAC--C--G-C-T-T-CGT-G-gGGGAGA-TTG--c--C--------CA-C---T-AGA-G-AGCCACA--AGT-TGGC-G-GtggaA---------ACAGACTGCTT--GC------GCC----A---T----GTTC-GGCCAt-----GGGACTTGTT----TA-A--TC-----ACG-------CCGATTGCGCTA-T---T-TGC-T-TGa---T--G-----A--GTTTCGAACG-T----AA---AAA--AT---C-G---------T--GACC-----GTGAG-CCC-G-TG-CT-A-----CT--CC-G-TG--G-----G--CC-----C--C---CGAA-G-CAC-----TTTGAT-----T----t----TGT--C----CTC--C--ATC-CT--G---CAGAAT-T-AGTT-ACCCCTC-GGCTCA-G-A-CGGACC-G-----GCCC--A--GAGCTagTTG-TAAC--GAT--GCT--------TC-G-C----G-AT-gTGAcgGC-TG-GAGTACa----T-G-CGTA--tGGAGT-------aG--CCC-AGGCtGCTGA---CC---TGcG--T-----C-T-g-aa--C---C------A------C-T---G-CCG-TT-CTCG---CC-TA-tTT--AA--CT----AT--C----GT--T--C--A-AC---------------TCT--GG-GG--T--CC-CG-TC-----A----CCA--T-A----C-GGTG--G----ATT-CAA-T-CGAC--AG-CCGT-T-G--ACGGA-ACAT-CA-GGA"),
        Record::with_attrs("N16",Some(""),b"--------CAG-TGG-A-AAGG-AA----A-----T--gTA-T-TA--A---CT----------CTC---C----GCT--gGAA--tc--GC--CAaC---TGTcTCtT---T--TG-A-GTAgCtT---CA------GAAT-TAt--T-GAT-CG-TAcTCAGCGTG------tg-T----GT-C-----CT------GT-G-GAGG----A-TAT---TA-TGAGG--A----A--TgAaGGC---A-Ac---Ga-------------TT---C-----T-cgT-CAT-AAC---A---AG-G-ACCC---CT-----A----TTAA-C-GTG-GcATCGCTga---t-GCG---C----AACG-GCCC----AA-A--AG-AatCCCAT---A-G---A----cCt---CCTACCTCC--T----A-C---gt--CT--Gg-G--TTGCACGaTACTat--C-Ac-aCT-A-T-TTa-AC-T-GCG--------AT--T-T-CTGGCC--G-TGTG-GT---TGCGCGA-----CTA----T-----gAG----GACtgGCGT----GTaaggaT-CGgtTA-GCTTT---Cg----GATTCG-G--CG--G------CTC-AAAAT---TAG--T--CGA---GATAC-------"),//T-------GCt-AGA---AAT-----TCCG---A-t------T-CAG---tA--A--GA-GCCTGT-T-CTG------ACGACCCC-T----TTA-GGG-TGCCAC---CA--T-TTAG-CGTA-T---TTAactacT-TCA-T---CGCTAC-A----CG----T--C--CTACCAG--AG------GAT-AT-GC-TA---cgAAA-TA------A--C--C-C-A---CG-G------TC-C-T--AA--ACCG--GTT-A----G---C-TCACcA--TcA--CG--ATTAGC-GGA-A-CGCa-----TA----cT----A-GTAGA---C-C----T-T--C-GTGAGCC-C-G-AtTC--GC-TA------A-TA--T---A--A-A-G-------TA-AA---GGA--A--GTC---TcAAAaA----CGTG-TTG-aa--A--------GT-Tga-A-TGC-G-AGGAAGG--GTC-GGCTaAcG----G---------AGATG--CATA--CA-----CGTA----Tt--Ac--gGCCA-TATAA------ATCACCCATC----CG-T---------T--ata----TGGA-TCGA--TaA---G-TGT-TgCGa---T--Ag--atC--GAGCAAACTT-Cta--CC----CA--GG---T-Tt-----c-aG--AGTG-----GTGTT-TGC-C-TA-TG-G-----GG--GT-G-AT-tA-----AC--Tt----G--C---TCAG-C-CTA-----GGCTGT-----Gta--a----CGT--Ga---GGG--TcgGAT-TG--T---CTGCGGgG-TCCGtAAACA---TAGACG-----ACCGGA-C-----TTC---------AC-cGGG-AA-C--TCC--ACA--------AG-----AA-TaAC-cTCG--TG-TC-TCGCAAg----T-C-C--------GGCgcc----aA--TT----GGtTGT-C---CC---TA-TagA-----T-G-atgtG-A---A------Gtca---T-T-gcG-CCC-CA-TATG---AA-AC-tCT--TAcaCGtGC-AA-AG----CT-aG--G--C-AC---------------GATc-CC-AC------C-TG-GTcag--T----GCA--A-Gata-G-TAACATT----CTG-GAT-G-AC----GC-TCAA-GtCttATGCG-TATT-TTg-TA"),
        Record::with_attrs("N17",Some(""),b"--------CAG-TGG-A-GACG-AA----AAG---T--gTA-A-GG--A---CC----------CCC---A----GCT---GAA--tc--AG--CCaA---TGTtACtT---T--CA-T-GCAaAtC---GG------AGAT-AAt--T-TAC-CG-AAgACTACCAG------gg-T----CT-C-----CT------GC-G-CACT----T-GAT---TA-GGAGG--T----A--AgAgGGC---A-Ag---Ca-------------GT---G-----T---A-TGC-AAG---C---AT-G-AATC---CG-----A----CTTT-G-GGT-AtAAGGCTgt---g-GAG---C----GACT-ACCC----TA-A--AG-AaaTAGAC---C-C---A-----Cg---CGTCTCCGC--T----G-A---tt--CT--Ac-A--TGGAACGcTCTAtt--A-Ta-aGT-A-T-TCa-AC-A-GCG-----T--AC--T-G-CGGGCC--A-GGAC-CC---TGCGTGT-----CAA----C-----cAA----TAAtgGCGT----GT-----G-CCgcGT-GCGTC---CtC---GAATCG-T--AG--C------CGC-ACTTT---GTG--T--TTG---CGTAC-------"),//T-------TGg-ACC---AGT-----CCCA---G-c------T-CAT---tC--T--GA-CTATCT-T-AGG------ATGGGGTA-G----TCG-GGG-AAACAC---CA--C-TTAG-CGCA-T---CTG-----C-TCC-C---CATGAG-G----CG----TA-A--ATGCCGT--TA------AAA-TT-GC-TA---aaAAA-AT------A--C-GC-C-A---TG-G------TA-C-C--AA--TACG--GTT-C----G---G-ACAT-A--T-A--GC--ATTAGG-GCG-A-TCCa-----TA----cA----A-GCAGT---G-C----C-A--G-ATTTGCC-A-A-AtGC--TT-AG------A-TT--T--GA--A-T-T-------TC-AAC--GGA--A--GAC-T-A-CGCaG-cATTGCG-GCC-tc--C--------GG-Tga-T-GGA-C-ATGTTAG--TAC-GGTGtGtG----G---------TCACG--CTTC--CG-----GGGG----Ta--Ac--gGCTA-GATAA------ATGTCGCGTT----TC-T---------T--aca----TCGAAACAG--AaA---A-CTT-TgCGa---T--Ag--atC--AAGCAGTAAA-Cta--CG----AG--GG---C-Ct-----g-cC--CGTC-----GGCTA-TGG-C-TG-CC-G-----AA--AA-G-TA-tA-----AG--Tg----G--C---TTAC-G-CTA-----GCCTGT-----Tct--a----CAG--Cc---GAT--TtaGAT-TG--C---CCAGTAgG-TTAAtAGACA---TTAACA-----ACTGGC-G-----GGAC--A--AAGAC-cGGG-AA-C--TCT--ACT--------TG-----AA-TaAA-aGCT--GG-TA-TCGCGAg----G-C-TTA-----GACCgcc----aA--AT----GGgTTA-A---CA---CG-GagG-----T-A-atggG-C---C------G------G-T-ggG-CCC-CG-TATT---AC-AA-cGT--TC--TTtGG-AA-AC----CA--A--C--A-TC---------------GATt-TC-AG--T-CCC-TG-GGatg--C----GCA--A-G----T-TAACTCC----AAG-TAT-A-TG----GC-TTAA-AtCttAAGGG-TGCT-CA-GAG"),
        Record::with_attrs("N18",Some(""),b"--------CAT-CTG-GtGTGT-TT----ATTA-GG--tAC-C-AC--A---TG----------CTT---C----GAG---CGGGT-t--AT--GC-CggCGGCtGTtA---T--AT-A-TTAcGgC---GG------ATAA-CCc--A-AAT-CT-TT-GCTCAGGG------tt-T----TT-A-----AG------GG-C-CGGG----G-GGA---GC-CACTC--C----T--AtTgGCTg-AG-Ta---Cc--TGAaA-ATTGACT---T-----T---T-GTA-AAT-C-G---AT-T-AGGA---AT-----C----GCCA-A-GAC-AgCCTACG-------TGT---T----CCTT-GATC----CT-T--TGGGccCAAGA---T-T---G-----Gc---TCTGATAAC--A----T-G--C----CT--Ca-T--TTTGGCAcCTGT----G-Gt-tGTCC-T-CC-ATT-T-TGA-----C--TC--A-C-AGAGAC--C-AAGC-TCT-AGGACGGC-----TCC----G-----tGT----CGT--CGTA----TT-----A-TAtgTT-GGTGT---G-TT--GAATTATT--CC--A------CGT-ATGAG---GCG--A--TGC---TCGTG-------"),//C-------TC--AGG---TAG-----GTGA---A-----C-TT-CTA---tC--G--TC-TCTTTG-C-AGA------CAGCTGAT-T----AGT-TTT-TATATA---TT-CA-CGAG-CCCT-C---TAC-----C-CCC-G---AACGAC-C----AAG---GG-A--CTTCATT--TT------GCC-CTTAC-GT-----GCG-GG------T--G-AT-T-A---CCCG------AC-G-C--GG-ATTGG--CTA-T----AT--C-CCGA-G--C-T--CA--CCATGA-ATG-A-ACT------CC-----T----T-AATGT---G-G----T-C--G-TATCATGCA-A-G-AG--CAACT------G-TC--C--GG--G-A-A-------AC-GTC--CTC--G--GCG-T-A-CGA-A-aGTGTTA-ATC--c--C--------GT-Tga-T-ACG-T-GTGATAG--CCC-CACAtA-G----T---------CGCTC--ATAC--CA-----CGTG----Ta--Cga--ATAA-CTCAC------ATCGTGACCT----TT-T---------A-A-------TACACCGGCTGA-T---C-GCC-T-ATa---A--A-----G--GTGGGAACCT-C----CC----TT--AC---T-At-----t-gA--ATCC-----TTTTA-ACG-A-GG-CG-T-----AA--CA-C-GA--G-----CA--Tg----A--G---GTAC-G-CTA-----GCGCCA-----G-t--a----AGA--Ag---GGC--T--GAT-GC--C---TCACCT-C-CGCT-ATTGTAG-ATCGCT-----GAATAG-T-----CTGC--A--CTATC-gCGT-ATAT--CGA--TTA--------CC-----AC-CaAA-aTTT--GC-CG-GTCATGg----G-C-AAAT--gAGACC-------aT--GT----GCaACCTT---GG---CG-T--T-----T-G-agggT-G---A------G------A-G-ggT-CTG-CA-TGTA---AG-TT-tCT--AC--ACaGC-CT-GC----CAc-C--C--A-TC---------------TTG--CG-GG-gC-GTA-AA-CG-----A----GCA--T-C----A-TGGCGAT----GGC-AAA-A-TGAG--GG-TCGC-T-GaaCTGGG-ATAC-CA-TTA"),
        Record::with_attrs("N19",Some(""),b"tca-----CCC-GGC-TtGACA-CG----TCCA-AG--aGC-TaGAat-----A----------G-GctgA----GGC---CTTAG-c-------A-TgcT-----G-A------AT-G-GAGcCaAa-cGA------AGGG-GA---TcAATATAa-T-GCGGTCCG--------gA----AAc-----GGG------G--T-CGC-gcaaTcGAC---TCgT--CT--C----T--GtTaC-------------t--T-------TCGTC---C-----C---C-TAC-GAT-A-CtagTGgC-AACG---AC-----C----TCCA-A-CAT-T--TCGAC-------ACTaaaGact-ATG---GATgg--CG-GA--GCGtaG-GGT---GcA-gtT-----Cc---CGGAAGG-A--A------G--C--------CccA--A-CGACCtAAGC----C-G----AGG-C-GA-GTA-T-GCT-----G--CC--A-A-G---AT--A-ATA--GAAaCAGTAG-Agaa--G-T----T-----gGG----CGA--TTGC-t--GG-------TAccATcTAAGA---C-GG--GCCCTGGG---C--T------TATgAAGGTacgAGGcaA--AGC---CATG-------a"),//Cgtgttt-AC--CGT---AAAA-----CTG---A-----A-AG-GTA---tG-----CT-CGTTAG-A-CAG------TAGACGGG-A----T---GCTcTACA-T---TAtGGcGCAT-TCGA-G---TGG-----GaAGGgC-gcTAATCG-A----TTGgtt-AtT--GT-CAAC--CC------ATTcA-CAC-C-------TT-CG------GatG-CT-C-A---CGTG------GC-C-C-----TATTA--G-A-Aaga-ACatTaCAGG-T--A-C--ACtaGAGAGC-TGA-T-C-C------TT-----Tt---T-GGGGC--gC-G----A-G--C-------TAT-C-A-AGg-GCACT------C-AA--A--TA--A-AtAg--tctG-----Cag-AA-gAgaTTAaT---CAC-C-gGCG---------g--T--------CC-T-c-C-CCA---CCGTCGT--G---CGTC--------A---------GCCCC--ACCAatTT-----GCTAgg-cG---A----AGAG-T-T----------CTCGAACT----TT-----C-----TGG---t-ctAGCCGAAGTACG-A---A-ATC-G-GGa--tA--G-----C--ACGAGCTCTG-G----CT---TCG--GA---AaCact--gc-gAg-AAGT-----GC--A-GTG-A-AA-ATcA---caTC--TA-G-CC--CcattgAA--Ag----G--C---TGG----GCT-----AGAGTA-----G-a-------ATAcgT-c--CCT--G--AGCtAAggG---TTAGCG-A-TAGA-AAAGAGG-ACATCG-----GAGGAG-T-----ATCCccC--TCTACtaCCG-AACCgtGCA--AC---------TG-T-TTGC-G-GGctGCA--TC-AG-AGAAATa----T-A-CTGGtgaT--AT---ggatcG--AA----GTaCC-C-agcTGcttGA-G--T-----G-G-agttC-T---Cc-----T------A-C---G-AAG-TC--AA----GG-AT-tCC--AT--TG-GT-TG-GC----TAa-CggA--T-TCt-----ctcatt---GTC--GA-CC-tC-TAA-GA-CC-----C----ATT--T-C---tC-CGTGTGAga--AA---TA-TtTCTT--ATaAT-G-G-CtgCT-TC-TATC--T-TCC"),
        Record::with_attrs("N20",Some(""),b"--------TTG-CGC-TcGAAG-CC----TGGT-AG--tTT-AtGC--C---AG----------GCT---A----TAA---GATGC-a--CG--GG-AgtAT--aCG-A---C--AC-C-CGCtAgAaaaAA------GTAA-TT---GaCGTTGC-GC-TATAACTG---------G----AT-C-T--TGC------C--A-CATActaaTtTGC---CTcA--CT--C----A--GtTtG-------------t--ATTaC-A-GTAGG---C-----A---G-GCA-AAT-C-CgttAAgC-TGGC---AT-----T----TTTT-T-ATG-GgGAGACT-------TAActtTaca-GTTT-TTCA----CC-GG-GCTAacCCCGA---C-C---A-----Gt---TTTGTTTAC--C----A-G--C----AC--Ca-G--T-ATAGTtGTGA----T-A----AAT-C-AT-GTG-C-CGA-----C--TA--G-G-G---GT--C-AAC--GCG-TAGTTGTAagc--TTT----A-----gTA----CTG--ATAG----TG-----C-TTttCGaACACT---T-TT--GAAGGCAT--GG--T------CTA-ATGATtggCTA--T--CCT---TATGC-------"),//G-------CG--TGC---CCGA----CTCC---A-----G-GG-ATG---tC--T--GA-ATGTTA-G-ATC------CTTCGCAT-T----AAT-TTAgTGATGC---CG-GAcGTTC-CAGA-C---GGG-----C-CCG-G--tACTACG-C----TGAgcc-AaG--ATGTTAT--GC------TTT-C-CCC-AC-----TGG-GG------T--C-AT-C-T---GTGT------TA-C-G--GA-AAGTA--TAT-A----CTccTgACTT-A--C-G--AA--TCTTCG-TAC-T-GTT------CT-----Aa---C-ATTGC---C-G----A-C--T-GCTTCCGAG-C-G-GGt-CCGCT------C-CT--T--AT--T-T-CcaTccaT-----C---AC--C--CTGcC-T-TCG-G-aACCTCC-ATA--g--T--------CA-C-c-T-CCC---ACGAATT--CTT-CACT--------T---------GTTGC--AGGC--AA-----ACAA----C---T----GAAG-CAT----------GTTATTTT----CC-----G-----CGT-------CGGAACAAATAT-T---G-CCC-T-ATa---C--G-----A--TATTCGTACT-C----AG---TAC--GT---A-Cg-----c-aA--ACGG-----GA--G-AGG-G-GC-AC-T-----TC--TG-C-AG--Gt-g-tCA--Tg----T--C---ATG----ATT-----ATGCTG-----C-t-------GCA--G-a--TCT--T--CTTtGG--G---AAAACT-C-CGGT-GTCGCAA-CCAGCT-----GATAAT-A-----CTCA--A--GTATTtaCAA-CTAGaaGCT--GT---------AT-T-CTAT-G-CG-gCAC--GT-GG-ATAAAAc----A-C-GCAT--aC--CA-------cG--CG----AAtAAACA---GGgcaAG-C--T-----T-T-ttaaG-C---C------A------A-C---A-TCC-TG-TGCC---CC-TG-tGC--CA--TG-GC-AT-GA----GCg-T--G--A-TC---------------CAA--AC-CA-gC-GGA-AC-CC-----A----CTA--C-T----T-TTCGTAC----AT---CA-G-TCTC--GT-CGGT-T-GtaCGATG-TCCG--C-TGG"),
        Record::with_attrs("N21",Some(""),b"--------CGA-TTC-GcGAAG-TT----CTAT-GC--gAA-C-CC--C---GA----------TTA---A----GTG---CTGTT-t--AG--GC-CgcCGGCtGA-A---T--AT-T-TTTcGcC---GT------AGAA-AA---A-CGTGCC-TC-TCTCAGGG---------A----TT-T-T--AAG------GG-C-CGTC----G-CGA---GC-CGCTT--C----T--AtTaGCAg-GC-Ag---Tc--TGTaA-ATGGGGA---C-----T---G-GTA-AAT-C-G---AA-T-ATAA---GT-----T----TCTG-A-GAC-TaAGTACG-------TCT---T----GCTT-TATC----CT-TG-AGCGgcAGCAA---T-T---G-----Ac---TCTGGTATG--A----T-G--C----CT--Ca-T--AGTTGTAgCTGG----A-Gg-tGTCT-G-CC-ATT-T-TGA-----A--TT--A-C-AGATAC--C-GAGC-GCT-TGGACACT-----CCC----G-----tTT----CTT--GGTA----TT-----A-AAcgTT-GGTGT---T-GT--GAATTACT--GG--G------CTT-ACAAG---GCG--A--GCC---TCCAG-------"),//C-------AT--CGC---GCGA----GTCC---A-----G-CA-TTT---tC--T--CT-ATCTTA-A-AGA------CATCTGAT-T----TGT-CTT-TAAATA---TC-CA-AGCA-ACCT-T---TGA-----C-CCC-A---AATGTC-C----AAG---GA-A--GTTAGTT--TG------GTC-CTTAC-GT-----CCC-GC------T--G-AT-T-A---CCGT------AC-G-C--GG-AACAG--CTT-T----AT--T-GCTA-G--C-A--TA--GCAGCG-TTG-T-ACC------CG-----T----T-AATCC---G-G----T-G--T-TATCCGGGA-G-G-AG--CATCT------G-CA--C--GA--G-A-A--T---CAT-GTC--CAC--G--GCG-T-C-CGA-G-aGTGTCT-ATT--c--C--------CT-Cat-G-CCA-T-AAGGTAC--CCT-CGCT-G-G----T---------CGCGC--GGAA--AC-----ACTG----A---C----ATAG-CTCCC------TGCTAGACCT----CG-T--GA-----AGA-------CGAACCAGCAGA-T---C-CCA-T-AAa---A--A-----A--GTGTGTACCT-C----CC---CGG--CA---A-At-----t-aT--ATCC-----TTGTG-AGA-A-GA-GG-G-----TC--CA-C-GT--G-----CA--Ta----A--C---GTAA-G-CAT-----CCGGCA-----G-t--g----AGA--A----CTA--G--GCT-GC--G---TAACCT-T-CGAC-ATATCTG-GCCGCT-----TCATCA-T-----CTGC--A--TTGTCcgCAC-ATAG--CGT--TTA--------TG-A-CTAT-G-TC-gTGA--GG-AG-GTCCAGg----A-C-AGAA--aAGAGC-------aA--GC----GAaGCGTG---AC---CG-T--C-----C-A-agatA-C---C------A------A-C---A-CTG-TT-TTCA---AT-AA-tCT--AA--AC-GA-GA-GC----GAg-C--C--A-TC---------------TTA--CG-CG-gT-GTA-AA-CC-----A----CCA--T-C----G-GGGTGAT----GTC-AAC-A-TGAG--GG-TGGA-T-GaaCCGGG-ATAT-CC-TTA"),
        Record::with_attrs("N22",Some(""),b"--------AGA-TTC-G-GAAG-TA----CTCC-GA--gAA-C-CC--A---TA----------CAA---A----TTG---ATGAG-c--AC--GT-A--CATAgGA-A---T-AAC-T-TCT-GcC---GT------GGCA-AA---A-CTTTCC-CA-TCTCAGAG---------T----AG-T-AGGAAC------AGTC-CGGT----T-CGG---GG-CGACG--G----T--AtGtGTAg-GA-Ag---Tc--CCTgT-AGGTGGA---C-----G---G-AGT-GAT-T-G---GG-C-AGGC---GT-----T----TCTG-T-GAC-C-CGGGGT-------TCT---A----TCAA-TCCA----TC-TG-ATAGgaATCAG---T-T---G-----C----TCTGGCATC--T----T-T--C----CT--Aa-T--AGTGGGTaCTGG----A-Cg-gCTCC-G-GC-ACT-C-TGACC--CTAATT--T-T-GGAAAC--C-TAGA-GCC-TCGGCAGA-----CGA----G-----gAT----CTT--CCCT----TC-----A-AT--AC-GGTGA---G-AT--TATTTCGA--GT--G------CCC-GCAAG---GAT--A--GAC---TTCTC-------"),//C-------AC--GTC---TCGA----GGTC---A-----A-CC-TTC---gC--G--CT-CCGAAT-A-AAA-----ACACGAGGT-G----TGT-TGT-TAAATA---TG-CC-TGCA-ACCT-C---TCA-----C-AAA-A---AATCTC-C----AAG---AA-A--GACAGCTTGTC------TTG-CCATG-AT-----CCG-GC------A--G-GT-T-A---CCGC--GG--AC-G-A--GG-GAGGT--CTG-T----AT--T-GCTC-G--C-A--GA--CCCCCC-TGA-A-GTT------CT-----T----A-TAGGA---A-T----T-G--G-CTTTGGGAA-C-G-CG--GACCT------G-CA--C--AA--G-A-A--A---CAT-GTC--CAC--G--GCG-A-T-CGG-G-gGGGTCT-CTG--c--C--------CT-C---C-TCG-G-AATCAAA--AGT-CACG-G-G----T---------GGCGACTGCTG--AC-----AGAG----A---G----ATAG-CACCC------TGTACTTGTT----TA-G--AC-----AGG-------CCGCTTAAGTGA-T---C-CCC-T-TGa---T--G-----A--TTCTCGACCC-T----AC---CCG--AA---A-A---------T--CCCC-----TTGTG-AGG-T-GG-CG-C-----TC--CG-C-GT--G-----CATCG-----C--C---CGAA-G-CAG-----TTTGTA-----G----t----AGT--G----CTC--T--GTT-GC--G---CGAAGT-T-CGTC-ATGTAGC-GCCCGG-G-C-CGAGGC-T-----GTTC--A--TTACCagTTC-ATTC--CGT--TTA--------TT-G-CTAA-G-TT-gTGA--GG-AG-GCGTCGg----T-G-CGAA--aTGAGT-------aA--GCG-ATGCaGCCTG---CC---TG-T--C-----C-T-aggaG-G---C------A------C-C---G-CTG-TT-TTCA---CC-AG-tTA--AA--GT-GT-AA-AC----GT--C--C--A-AC---------------TTG--GG-CG--G-ATC-AA-CC-----A----CCA--T-A----G-GGTGGAG----ATG-AAA-T-GGTC--GG-CGGA-T-G--TAGGA-ACAT-CG-TTA"),
        Record::with_attrs("N23",Some(""),b"--------CCG-GAAAT------CT---gTCTA-G----AC-A-AG--T---CA----------TCT---T----CAT---ACAGT--CGTT--CT-C--ATTG-GA-C---T-AAT-A-GTG-A-C----C------CGTG-TC---C-ATAAGA-G---TCAGATG---------G----AA----ACTTC---c--GGTAaGATA------TTGcctAT-AA-CCtt-attaAgcC-C-GGA-gTC-C--c-T-agGAT-A----TAACt--G-----T---T-A-CcAAG--tG---GC-GgCTGA---CG-----G-----C-----ACC-T-CCTCAG--------AG---C---aGTAC-TAAA--AACCcTGcTC-A--CATCCatcG-T---T-----C----GACTCAATT--CcatcT-T--C--acTC--C--AtgTGTGCGT-TTCT----A-T---ATGA-C--A-ATA--aTAT-----AT-GCaaA-A-AGGATG--T-G-TT-C----GAGTCCG-----TTG----CTTAcG-AT----GCA--TA-A----GG-----AtCT--TA-GAA-A-----TAt-TTCCGAAC--TAgaT-c---cTA-----CC---ACG--CGAT-----------------"),//A-------A---GCG---TCTT----TTTC---G-----AtA---------C--C--T--GAGCCA-GaTAA-----GACAAG-AG-A--c-CCCaTAC-AATCGC---TC-CG-CTCAcT----A---GGT-----G-GCT-T---GACCTC-G----GCA---GG-T--TCGTTAAGTCA------TCG-ACCGAcGA-----CGC-CC---ACGT--A-CA-A-A---CTCT--ACg-TG-C-A--G-----TG--CAC-A---tAG--C-CTCA-TGTG-Gac-------TGC-GAG-A-CTG-c--C-GA-----G-GtcT-G-TGT---TcG----T-A--T-----CCCCG-GgG-AC--AAG-T------A-CG--A--AC--G-G-A--A---CGT-CTG--TTA--C--TAT-C-G-GGC-C-----GTA-AC------A------A-CT-C--aG-CTC-C-CTTGC-----CT-ACCG---C----TtgacgtgcaGCCGCGAC-TT--GT--taaCTTG----C---A-------G-TTGTA-agggcACCAAGGCAGagagAT-CgcTG---ccGGG-------TTCCAGA--CGG-Tg--G-GCC-T-AG----A-tC-------cGAGGC-TAGG-C----GTa---CA---------G--------------TA-----G--CG-AGAC--CC-AG-------CG--TC-A-TT--C-----ATGGG-----G---CTTCTCT-T-TTC--------CAT-----T---------GTA--C--ag-AAc----G-C-GA--T---ATCCGA-G-GATC-TGTTACT-TTG--T-GgC-CAGGTA-G-----AACT--A-tCGCT---CGA-CCAA--TGG--A-G------ag---C-G-GCcA-TA--ATC--ATTGCc---CGG---gaCgC-TTAC---ATACA--------C--CAGTCTTC-GTCTA---GT---TC-G--G-----A---------A---C-----c-------GaT---T--CCgTC-ATTC---CC-AG--T-------TT-GT-TG-TT----GA--C--T----TA---------------AGA--GGtAA--C--T-AC-aGT-----CT--AGTT----G----T-CT-AACA--gaGAG-CACGT-TGGGagTC-CCAG-C-C--A-AGC-TAAAcGT-CTA"),
        Record::with_attrs("N24",Some(""),b"--------CTT-CAGAA------GG----ACGA-CC---AC-T-CG--G---GC----------AAG---G----TTC---TCGTG--TAAG--CT-C--TGAG-CA-C---C-TTTGA-CTC-A-G----G------CCTA-GA---G-GTGACG-GG-TCATTAAT---------T----TG----TACTA------TGAGcAGGC----C-ACTcctAA-CATCTttTtcttT--C-A-TGA-tTA-C--c-G-ctTGT-AgAAGCAAGt--G-----C---C-A-G-GCA--aC---TG-G-CGCC---GT-----G-----A-----ACG-A-GCATAT--------TG---G----AGGC-ACAT--AAGGgCTcTC-T--CACCT---T-T---A-----A----CTATAGGAA--CggagC-A--T--t-AA--A--C--CGACCGA-GTGC----A-A---TGGC-G--T-AAC-AcGAGGGgaGGA-TG--T-A-ACTCTG--C-ATGT-AGG-AGACCTCT-----TAA----TAAAcG-TA----TGA--TCGC----TT-----GtCC--CT-ACC-A-----GCt-TCGCAGCC--GAcaG-c---cGA-----TT---CAG--CAATGA---ACTTC-------"),//A-------T---GCC---ACGA----TTGG---G-----AcGT-CAC----C--T--T--TCGGTG-GaGAA-----GTATCA-AG-A--c-ATA-GCT-GCACCT---TT-CC-CGGGaT----C---CGA-----T-CGC-C---TATATG-G----TCA---TA-T--GTTTTGCAGCC------AAC-GATGA-CC-----GTT-GC---ACGT--C-CA-T-C---AGGA--GCc-TG-T-G--A-----CT--GCA-A---aCT--C-AGGA-GGGA-Taa-------CTC-CAG-G-AGG----CATA-----C-G--T-G-TAA---GaT----T-C--T-----ACGTG-TgT-CA--TAT-C------G-TA--T--GC--G-G-T--G---CGC-TCA--CAA--A--ATT-G-T-AAC-C-----GAA-TTAA----A------T-TT-A---A-AGA-T-TGCTC-----CA-CGGA---A----TcaacggcagTCTCAGTC-CA--GC--cttTCCG----C---A----TTTG-CAAAA-atttaTTCACGACTC---gCT-TgtGC-----AGG-------TTCCCTA--AAC-T---C-GTC-T-AG----A--G-------cACCAG-CGGG-A----TG---ACT---------G---------G--GATC-----GCCCG-GGCCAtTC-GG-------TG--TC-A-AG--A-----GTAAT-----T---TTCCAAT-T-GAT-----TTCCAA-----G---------ACG--C-----ACctT--T-A-GC--C---TACTCG-T-GCAT-AGGAGGG-AAAGAG-CtA-TTATGG-C-----AATC--G--AGAT---GTA-GCAA--CAG--CAC------cc---G-A-GA-G-AT--TCG--AAAACt-C-TCG---ggAgC-TTGG---TACCA--------G--TAACAAGC-CACTT---TT---CA-G--A-----A---------A---T-c--c-G------CgA---T-ACTgTG-TCTT---CG-AG--TA--TG--GA-CT-CT-AG----AG--T--T--A-TT---------------TAC--TG-AG--C--GCGT-gAA-----TT--GGAG--G-G----A-CATGACA----GCT-TTGGC-CAAG--AC-CGTG-A-A--G-TTA-CAATcCC-GTG"),
        Record::with_attrs("N25",Some(""),b"--------GAT-CTGGA-TTGT-TG----CAGT-CC---CC-T-CG--G---GC----------ACC---G----GTA---TCGTG--CATG--CT-T--AAAG-CA-G---G-ATTTA-CTT-A-G----A------TCAA-GT---T-GTGACG-GC-TAGTTGGT---------T----GA-C-TTGCTG------AGAG-AGGG----C-AGAgcgTA-CAGCTtcTtattT--T-A-TCA--GA-T--c-G-ctTGA-AgAAGCGAGa--A-----A---G-C-A-GGA-T-C---TG-T-AGCT---TG-----T-----G---AaACG-A-GCATAT--------TG---G----AGTC-ATTC--AGGA-GCtTA-T--CACAT---A-T---C-----C----GTAACGTAC--T----C-A--A----AA--A--C--CGACCGA-CTAC----A-A---GGGT-G--T-AAG-A-AAGGG--GGA-TG--T-A-TATCGG--T-ATGT-GGT-TGCCCTCG-----CAA----CATTcT-TG----GAA--ACGT----AT-----GtCA--TT-ATT-G-----GCa-ACGGAGAG--CAgaC-a---cTAG-GGACG---GTG--AGATAC---TCTAC-------"),//A-------T---GTC---ACGA----TACA---T-----A-CT-CGT----C--T--C--TGGGTG-G-GCA-----ATATCA-AG-A--c-ATC-CCT-GAACAT---TC-TC-CGACcGTTA-C---GGC-----T-CCA-C---GATGTT-C----TAA---GG-T--GTGTCACACCC------ACC-GCTGT-GG-----CGA-GA---GGGT--G-CA-A-T---CGGT--GGc-TG-G-G--A-----CT--CCT-T----CA--T-AGGG-TGGT-Tgt----CTTCTA-CAT-G-AGG----CCTA-----T-G--G-GATAG---GaT----C-G--G-----CCCGG-CtT-GG--TAC-T------T-CT--T--GC--T-C-T--A---CCT-TAA--CAC--A--CTA-G-T-GAC-C-----GGT-CGTT----G------A-TT-A---GACGT-T-TGGGC-----CA-CGGA-C-A----A--------gGCGCGCTC-CT--GG--ctgAGTG----C---A----TTTG-CGAAT-acatcTTAGCGAGTG---gGT-TgtGC-----AGC-------AACCCTG--CGC-C---C-GGC-T-AA----A--C-------aCATAG-CGGC-A----AC---TAT---------A---------G--GATC-----GCAGT-GGCACtTG-CG-------GT--AC-T-GG--A-----TTAAA-----T---ACCCAAT-A-GTG-----TTACGA-----G---------GCA--G-----CC-tA--TAA-GC--G---GCTAAG-T-GCAT-AGAGGGG-AATGAG-C-T-TTATGA-C-----TCGA--A--AGCTG--GGA-GCTA--ATG--CGC------cc---C-A-AG-C-AT--GAG--ACGACt-A-CAA-----TcT-AGCA---CTGTG--------T--GCACTTGT-AAATT---TT---CC-G--T-----A---------G---G-c--c-T------C-A---A-CGAtTG-TCGT---CG-AG--GA--TG--GA-CG-GT-AG----AT--T--T--A-GT---------------CGT--TG-TA--C--ATGTTcCA-----TA--TACG--G-G----G-CGCTAGA----CCG-CGGGC-CAAC--GC-CGTC-A-A--G-TAT-TCCTcCA-GTA"),
        Record::with_attrs("N26",Some(""),b"--------TTC-CTGTA-TAGT-TA----CATT-TG---CC-A-CA--G---GA----------TCG---A----GCA---CCGCG--CATC--CC-G--AGAC-CT-T---G-CTTCG-AGG-G-A---TT------TCAA-GT---T-CGGGAA-GT-TAGCTGCT---------A----GT-C-GTACAA------AGTG-ACGA----T-TTGgggTA-CACTCtaT----T--A-G-GAA--GA-T----C-ctGCT-GcCAGCTGC---G-----A---G-CCA-ATA-T-G---TA-T-AACT---GT-----T----TCAC-AgGGA-T-GCTTTT-------GTC---G----ATAT-TATA--ATTG-CT-TATT--AGGAA---T-A---G-----G----AAAGTTTCC--C----C-T--A----AT--A--T--CTTCGGA-CATC----A-T---CTCT-G--T-ACC-G-ACGGT--GCCAGT--A-A-AGTCAA--A-TGGG-GGT-TCACAGTT-----GAC----CATTaT-AA----GTA--ACGG----TT-----C-AT--AT-GAG-G-----GG--CCGCAATT--CA--C------TAG-TAATC---GGG--ATCGGC---ACTAC-------"),//G-------GC--GGT---GCGA----AAAA---A-----A-GT-CCG----G--G--GG-CCCGTG-G-GCA-----ATATTACAA-A----GAC-CGT-GCAAGT---TC-CC-TGTCaTACA-C---GCG-----G-CCA-G---GTTCAT-C----CAG---GT-A--GTGACGAACCC------ACC-AAGGT-AG-----GGT-TA---GAAT--G-GG-C-A---CATT--GG--CA-G-G--T-----CG--CCC-A----GG--C-CTTG-TAAG-G------ATAATA-TGC-C-AGG----CCTA-----T-G--T-AATAG---G-T----A-T--G-GATTCACTA-G-T-GG--TAC-T------G-CA--G--AC--T-T-T--G---GCC-GGA--TCC--G--CCC-G-T-CCC-T-----CCT-AGTG----A------A-GG-T---GACGT-T-TGCGATG---TA-TGCC-G-G----G---------ATAAGCCT-CT--AA--taaAGAT----A---T----TTCA-TAAAT------TAAACAAGGG---aGT-G--AT-----AGT-------AAGCCCA--CCC-C---C-ACC-T-CA----A--C-----G-tCGCAGCGCGA-T----AC---TAG--GG---A-C---------G--CATA-----ACCCT-AACGC-CA-AG-G-----AC--CT-T-TC--G-----AGGTA-----A--GCCGCAAT-A-CCG-----TAAGTG-----G---------GCA--G-----TT-gT--TAG-TT--T---GCTGAG-G-GCGA-ACGATAA-CGTTGC-T-C-ACGGGC-G-----TATA--A--TCCTC--GAT-ACTA--CTT--CCC------tt---C-AGTT-T-GG--ACG--ACAACc-G-AGC-----G-T-CGCA---CTGGG--------A--TCTGTTGT-GAACT---TT---GC-C--T-----A-A-----G-C---C-tc-t-C------C-G---A-CTA-AC-TCGG---GG-AG--CA--AA--GT-CT-AG-GG----TT--T--C--C-CC---------------GGC--CG-AC--G--ATCTAtGC-----CA--AACC--G-T----G-AATACAA----ATC-CTAGA-CGTC--AA-ACTA-A-C--A-TAG-ACCT-CC-GTC"),
        Record::with_attrs("ROOT",Some(""),b"--------CTC-CTTGA-TCGG-TT----CATT-GA---CC-C-TC--T---TA----------TCG---A----TTA---AGGTG--CATG--CG-G--AATA-TT-T---T-CGCCG-ATG-G-C---GT------GTAA-AT---C-CTGCGC-TA-ACCCATCT---------T----GC-C-GGTTCC------AATG-CCGT----T-CTA---AA-GGCTC--T----G--A-G-GCT--GA-A----A---GCA-G-TAGTTCC---G-----A---G-AAC-AGA-T-C---CA-G-AACT---GT-----T----TATC-A-GCC-A-GCCTTG-------GCC---G----ATAT-TTTA--TCTC-CA-TTTG--AAGAT---T-T---A-----T----AGAAAATCC--G----C-C--A----GT--A--T--CTTCGGT-CTGC----A-C---CTCA-C-TT-ACG-G-TGAGA--GACAGC--T-A-CGTCGG--C-TGGA-TCC-TCTTTGGC-----AAC----CATT-T-AT----GAA--CACG----TG-----G-AT--GG-GTGAA---T-GG--TTAAAACT--CA--C------TAG-GAGTC---GGT--GGCAAC---AGTAA-------"),//G-------GC--CTT---GCGT----AGAA---A-----T-GG-CCG----T--G--AA-ACCATT-A-GCG-----ATATTTCCG-G----TAT-CTT-TCAAGC---TC-CC-TGAC-ACCA-C---TCC-----G-CCA-A---GACCGC-C----TAG---TT-C--TGTAGGGAGCC------GCT-AAGGG-AT-----TTG-AC---GAAA--G-GG-A-A---GTTA--GA--TT-C-C--GG-TTGTG--CCA-C----GG--A-CATG-GAGC-C--TG--ACAATC-CAA-T-GGG----CTTA-----T-T--T-AATAT---G-T----A-T--G-GATGCCTAA-T-T-CC--TGCGT------G-CA--G--TC--T-G-C--A---ACC-GCA--CCC--G--ACT-C-A-CTA-T--CCGACT-AGCT----G------C-GG-T---GATGA-C-AGCCCAG-ACTT-TCCC-G-C----C---------AGGTGCGTAAT--AC-----AACT----A---C----TACA-CTCAT------TATACAAATT----AT-G--AT-----GGT-------GCGCCCATACGC-G---G-AGA-C-CA----T--T-----A--AACTGGAGAA-T----TC---TGG--GA---A-T---------T--CAAA-----ACGCA-ATGCG-CA-CG-C-----TC--GT-T-TT--G-----CGTTA-----A--TGCATAAT-A-TCT-----TTAGCT-----G---------GCA--A----CTT--A--TCT-TT--C---GCTCAG-C-CAGA-AGGACTT-CGATGG-T-A-TCGGGT-G-----AATG--A--TTCAC--GTG-ACGA--CGT--CCA--------TT-C-CATC-A-TC--CCA--GAAGC-TCCTAC-----G-T-GCCA---CGCTT--------A--TCTGGTGT-GCTGG---TT---GC-C--A-----T-A-----T-C---C------C------C-G---A-CTA-TC-TCTC---AG-CG--CA--AC--GT-GG-AG-GA----TA--T--A--C-CC---------------GGC--CG-AC--G-CCTGTC-CC-----CC--AACT--C-A----G-TGTTGTT----TCC-ATAGT-CGTG--AG-CCCA-A-C--AATGG-CCAG-CG-GTC"),
    ];
    // Record::with_attrs("3",      Some(""), b"********GCC*G----*GTAG*AGa***GTGT*GA**cAG*T*CG**A***TA**********CGA***T****TCG***GTGAG*a--AT**GG*A**GCAAcAG*C***C*CAA-T*CGG*GtG***AC******AACG*GC***C*ATCTCA*GG*TCGCTGTT*********T****TG*AcTCAAAT******ACTT*CGAT****G*GGC***CA*GTGAG**G****A**TaGaGTCa*T-*Gg***Gg**GGGgG*GCCGGTG***Agac**G***C*ATA*TGCcT*G***GT*C*AGTC***GT*****G****TCAA*A*G--*-*CAGGCA*******TAG***T****ACGG*TGCA**--CG*TA*T--CaaTGCGT***G*-***-*****-****-CATCCTTC**T****A*T**G****TA**Gg*T**CCAGTTTcTCAG****C*Cc*cCCCC*T*CG*TCA*-*ATGCA**TGTCTT**T*G*-GCACT**CcGTGT*AAG*TG-ACAC-*****TAC****A---*-tGTaagtACC**TTCT****AG*****T*T-**--*-CGAT***T*CT**ACCATTGActTT**G******ACC*GTAGT***CCT**A--CCA***CTTGG*******G*******CT**GGG***TTTA****CTACc**At****G*CT*CTC***aT**C**GA*AAGGAG*T*CGC*****ACGATGAGG*C****CAT*CGC*TAATTA***TT*CG*AGGA*ACTA*C***TA-*****A*TAT*C***CTGAAC*T****C-C***TA*C**CCGC-TAGAAA******TTG*TGAGG*ATaaa**TCA*CG***---A**G*TT*C*C***CTAC**AG**GG*A*A**CGaGAGCG*gCTA*A****AA**C*---T*T--C*A**TA**GACCCAtTGAgA*T--****----*****G*-**Ta-AGGAac*A*C****Gc-**G*CCGACGGT-*C*C*CC**CACAT******T*GC**G**AC**G*A*G**G***GGT*ATA**TAC**C**G-T*T*C*AGT*TggGACGTA*TAG-*c**C******-*GG*T***T-CTT*T*-AGCGCA*-AG-*-GGA*G*GgcgaA*********TAACCGGATCT**CC*****-CCC****C***A****TTTC*AGGGCt*****GGGCGTTTTT****TA*A**AG*****ACA*******CCAACACCGTTT*T***A*GCC*T*ACg***T**G*****G**GTTTCGAATT*T****CA***CCA**TA***C*A*********A**GGCC*****TCGCT*TCT-G*AA*AG*C*****AT**CC*GgTC**G*****T----*****C**G---CGAG*GaCAT*****GCTTAT*****G****t****TGT**C****ACC**C**ATA*CC**C***GATGCT*T*CGTC*ACACATT*CCCGCT*G*A*CGGTCC*C*****GCCG**G**GCCCAtgATG*TAAT**CAC**GGT********TC*G*G---*G*CC*cGGAcgTG-G-*------a****C*A*AGTA**tGTAGT*******aT**CGC-ATGCaGCTGA***TA***TGtG**T*****G*C*a*ag-*C***T******T******C*G***T*CCA*TT*CCGA***CG*T-*tAG**GT**CG*--*GG*-C****G-**C**C**A*CA***************TCG**AG*GG**A*-CG-CG*CC*****A-**-CCA**T*A****C*GAGT--C****CTA*CTA-T*CAAC**CG*CCGA*T*G**TCGTA*GTCG*GA*TGC"),
    // Record::with_attrs("10",     Some(""), b"********ATG*A----*-GGG*TG****TGCT*GGtttGA*C*CT**C***TG**********--C***T****TGG***CGGAG*g--CG**AT*G**AAGAtTA*T***A*ACG-T*TAT*GtG***TC******CTTG*AT***C*AAGACT*GT*TGGCTACG*********T****AG*CcTTTAAT******CTAA*TAGC****A*AT-***GT*CTTCT**G****T**TcCtCAAt*AG*Cac**Ac**TACgG*ACGTATC***C*****G***C*CTA*TGG*A*A***AA*C*TGTG***GAaccaaG****CCTT*C*G--*-*CCCCCC*******CTC***C****CATT*TCCA**--CT*GA*GCACtaGCCAA***G*-***-*****-****-CAGG-GGCctA****C*A**T****CC**Ca*A**TTGCAGTcGCAG****A*GgccCTGC*A*GA*TTT*-*GTGAA**AGGTTA**C*T*-TAATA**AgCCAC*GAG*TCGGTCG-*****AGC****C---*-gAG****GGG**TTCGg***TG*****G*CT**AG*GATGC***G*AT**AATACTGTgcGT**G******GGT*GTGAT***CAC**A--CAC***AGCCAggatt**G*******GA**AAGattATCA****GCAC***At****T*CG*GGCattgT**G**CTtGTTAAG*C*AAGcg***GGGCTAAAA*Gg***GCG*TAG*TGACTA***AC*AT*TACC*ATAA*T***GT-*****G*TAC*C***CTCGAC*A****CAC***T-*-**G-TC-CTCTCA******---*AGTAT*AT*****CGA*AT***---G**G*TT*T*A***TGGTg*AG**CA*A*T**CGgGGACGcgACT*A****GC**C*TTGA*C--G*C**AG**TATGCTaTTG*CaG--****----*****G*-**G*TAAGGtc*C*A****Tt-**G*GTCCATGTA*T*C*TT**CTAATaatttgC*GA**G**A-**A*T*A**G***-CG*ATG**TCC**C**C-C*C*A*ACA*C*gTAGGCA*GCG-*g**Caactca-*CC*C***T-CAC*T*TGCGTCA*-TGC*TGCC*T*GgtccA*********TACGAGCCTCT**CG*****-TGC****T***T****GGAA*ATGATg*****CCGACGGATT****TA*C**GT*****GCC*******ATAAGACCCGTG*C***G*TCT*C*GCatt*T**A*****Aa*GTCTCTGGCT*A****AA***TGC**AGgatC*G*********T**GACC*****TAAAT*CCT-A*GG*CT*T*****CTg*CTaA*TC**A*****T--GA*****C**C---TGCA*A*GAG*****TCTCTA*****T****g****AGA**T****GCA**G**AAC*GT**A***AGGCGG*G*AAGT*GCCGACG*GGCTCG*C*C*CATTGT*A*****GTAG**A**TGTCCt*--AaGTTA**TAG**GGT********GG*A*C---*A*AT*aAGTcaGT-AG*GCAAAGt****T*G*CATC**aGAACC*******aT**CCC-GGCGaGTGCT***AC***ACgC**T*****C*A*g*gg-*C***G******A******A*C***T*CTC*-T*CTTA***AG*AC*tATaaTA**CT*--*TT*-C****GG**G**CtaT*CT***************GAG**CC*AG**G*-GC-TA*CC*****G-**-TCA**CtT****A*ATCC--A****AGT*AGG-A*TGCC**CT*CGGG*T*A**AGGGG*ATAG*--*GAG"),
    // Record::with_attrs("11",     Some(""), b"********AAC*GGT-GtTTCT*T-****--CA*CG**tCC*C*GT**TccgCGaatgcc****TAA***G****GAAat*GGAAG*t--CG**GC*CccCATAgAGtT***Ag-CC-C*GAAcAtT***AG******GCGA*ATg**A*TC----*--*GTACACGTacaccgag*A****GA*A*----CG******GC-A*CTAT****C*GAC***GA*ACTA-**G****C**GtTgGTTg*AAgCa***Ac**ATGaG*CTTGGGT***A*****Cg**GgTAC*TAA*C*C***CT*A*AGGA***GT*****T****GCCA*T*ATC*TaGAGGCA*******CTG***T****GGGT*AAGA**--CT*C-*CCGTacT-TGG***A*T***Gcc***Gt***GCCATTTCC**C****C*G**C****GCaaCa*G**-AGTGTGaCTGA****C*Ta*tGTCG*T*CC*GTT*A*TGT--**-G--GA**G*C*TTCAAT**-*CGGC*ACT*AGCTAG--*****-AG****T---*-gGC****TTT**--AA****TG*****A*ATatTG*AAGGT***C*TT**AAGGGAGG**AC**C******TCC*GTGAC***CGT**A--GTGaaaACTCT*******T*******GT**CGC***CTG-****GCAA***G*****C*C-*CAT***aA**A**AT*CCCTCT*-*TGA*****-TACCGTTA*T****ATA*GGT*C-TGCT***AT*CA*CACA*C--G*G***TCC*****C*GAT*Ct**CCCGC-*Aaa**AGG***TT*T**AATCTGG--CG******CCT*GATAA*GC*****TAT*TA***---G**T*CA*A*A***GCAG**--**GT*T*G**CT*TTTGC**GTG*T****GT**C*AGGC*A--T*C**CG**CTACAA*ATA*A*GCT****--GGgact*T*-**A*A-TGG***C*C****A*C**C*TGTAATGC-*C*T*CC**TAGCA******T*-A**G**GGct-*-*-**-***-GA*-GC**AAC**G**CGT*TaG*TGG*G*cATTGTGcGGC-*t**T******-*TC*Tct*T-GGG*C*CCGAAAC*-CGG*TGCAgC*C****A*********AATAG--TATA**GA*****CATC****Tc**Gtg**AGAA*CGTC-******GCCCTGCCCA****CA*A**--*****T-C*******GATAAATACCTC*A***T*GTA*T*TTa***T**C*****T**AGTCGACCGT*C****CA***-TT**TC***G*Cg*****c*cA**AATC*****ATTCA*AGG-C*AGcTT*A*****AA**CA*C*GA**A*****CA--Ga****TcgG---CGGA*T*TTC*****GCCCCTcg***G*g**g****TGT**Tt***GGA**T**GCG*CG**G***CAGCGT*T*CGGG*ATATTGAaCTCGCT*-*-*ATGAAT*T*****GTTA**A**CCGGA*gTGA*TTGT**TGA**GA-********CA*-*--GT*TaGA*tCTG**GA-AG*ATCAGGg****A*CaTGTT**cAGTTC*******aTt*CC----TAcACAAG***GG***GT*T**G*****CaGaccgaAaG***A******A******T*G*gg-*-GC*GA*AATGc**AC*AT*aCT**CG**CAaCA*ACaTC****TCc*A**A**AcAC***************ATG**CG*CGggC*AGA-TA*GC*****T-**-TAAg*G*A****A*TT--TAA****CATtTTT-A*AGTG**CT*CCT-*-*-*gGCTTGgTCGA*CA*ATC"),
    // Record::with_attrs("6",      Some(""), b"********GAG*ATG-A*GAGT*TA****GAA-*--***TAcA*GT**G***CA**********TCC***G****GCC***TAT--ga--AG**TTaT**-GACtTCtA***G*-CG-T*ACAaCaC***CG******ATAC*TAtc*T*GAC-CC*ATaCCTGGCAG******gg*T****CT*C*----CTtat***CC-G*TACT****A*CAC***TA*TGAGT**T****A**GcAcGTC**-A*At***Cc**---*-*-----AC***G*****T***A*CGC*AGG*-*A***GT*G*CGCC***CG*****T****---A*G*CCC*AgCGTTTGta***g*CAA***C****GACG*ACCA**--AA*G-*AA-AcgCAGAG***T*A***G**gg*Gt***CATCTTGAT**T****T*T**-tc**TG**Ca*A**TTGACCGgGCGAta**TcTc*aGC-A*TtCCa-GC*A*GTA--**-A--A-**T*C*CGGGCG**A*GGCA*TC-*-GCCTCCT*****CGG****G---*-aCG****TAAtcACGA****GC*****G*CCgaCT*GTGTG***TgA-**GGGTCA-C**AG**C******GGC*ACTGG***ATT**T--ATG***CGTAA*******T*******AAg*ACC***AAT-****CCCA***G*a***-*-T*CAT***tA**C**AA*CC-TCT*A*AGT*****-CTGGGTGA*T****CCG*GAA*ACTCCA***CC*-C*TCAG*ATGC*T***CTT*****A*ACC*G***CAGGAC*G****GT-***CA*C**TGGCCAC--TA******AAA*AT-TG*T-****aAAC*A-***---G**A*TT*T*T***TT-C**--**GC*C*G**AA*-TACG**ATG*C****C-**G*TCAG*A--T*T**CC**AACTGG*GCA*T*TCGa***--AG****gG*-**G*GAAAT***G*GaacaC*C**C*GCGTAA--A*A*AcGG**TG-TC******A*TT**A**GA**A*G*T**-***-TC*A-T**TA-**C**CTC*A*A*TAGaG*cATCCTG*CAT-tc**C******-*GT*Tga*T-GGA*G*TTAATAA*-CAC*GTTTtAtG****T*********ATATG--TATC**AT*****ACGG****Ca**Gg**gGCTA*GTTTA******ACGTTGCCGA****TC*-**--*****T--tta****CCCACACAG--AaA***T*CCTcAgAGa***T**Cg**tgT**TTGTCGTAAA*Cgac*TG***-CG**GT***C*Ct*****g*aG**CCAGa****GTCAA*GTG-T*TC*CC*G*****TC**AA*G*TA*cC*****AA--AtatcgT**C---TCTA*A*TAG*****TCTATT*****Gct**c****CGT**Cg***GCT**TttGGC*CT**C***CCCGTGgGcATAGaCGGAA--*CTAACA*-*-*CCTAGA*-*****ACAC**T**AAATC*aATT*AT-C**CCT**AGG********TG*-*--AA*CcTC*gC-T**GG-AA*TCTCTCg****T*C*GCT-***-CACCtcc****tC**AT----GAtTTA-C***CA***CG*GacT*****T*T*aaggT*C***T******A******A*C*tgC*CAT*CA*TCAT***GC*AGctGT**AC**TTaTC*AA*GG****CT**A**C**T*TC***************AATt*TG*AG**C*CGA-AG*GCgtt**C-**-GTG**A*G****A*CCTCGCC****GAA*TAT-A*TA--**GT*GTGT*GtCccCATTA*TGTC*AG*ACC"),
    // Record::with_attrs("1",      Some(""), b"********CTCaTGG-A*CTGA*G-****G---*-T**cGA*T*TT**T***CA**********ATC***G****TGT**cTAA--ct--CT**GAtC**-TTGgG-*T***G*-CC-A*GCAgCtG***GA******CATAtCAt**T*TCT-CT*GAaTCAAGGGC******gc*Acat*GT*C*----TT******CT-T*GGGC****A*TGA***CA*TGGGC**A****-**-*-aGAC**-G*At***Gg**---*-*-----AG***C*****G*tgG*GAC*AAC*-*A***AG*C*AAGA***GC*****A****TTGA*G*CTT*AtATCGCTac***t*ACT***C****AAGA*GGTC**--TA*T-*GC-AtaGC-AG***A*T***A****aAc***ATCCTCTCT**T****A*A**-ga**TT**Tg*G**ATAAGAGaTAGGaac*C*Ac*aCT-A*A*TTa-GA*C*CAG--**----TG**G*C*CTCGCC*cG*TCTG*GT-*-TAATACA*****TTG****A---*-gAG****GCCtgGATT****CTcggctG*CGtaTC*GGTAG***Cg--**GGTGGG-G**TG**A******TCC*AAAGA***TGC**C--CGG***TTTAT*******T*******GAt*AA-***----****CACG***T*c***-*-C*CGC***gA**Ta*AA*GACTAT*A*CCT*****-TAGAAGCT*C*t**ATA*CGG*TTCCACtt*TA*-G*ATCG*ATGT*T***-CAgtgacT*GCA*T***CATTTC*A****TG-***G-*CgcCCACGCA--TG******GAA*TT-GG*TG***ttAAA*TT***---A**A*-C*G*A***CG-C**--**TT*G*T**AC*-CGAA**C-T*A****C-**C*GCACaC--CcG**CG**ATTACC*CCA*-*AAAa***--TA****aC*-**A*CGACC***A*C****-*-**-*GAGACGC-C*G*GaCG**GC-TA******A*GActT**-A**G*A*T**-***-GG*AG-**ATA**C**CCC*-*CgAA-tC**--GCAC*GTG-ct**C******-*GT*Tgt*A-TCC*G*ACGTAGG*-TCC*GGAGcGaG****G*********AGGTG--CCTA**GA*****CGTA****Ct**Tt*tgTCCT*TCTGG******ATCAGCCAAC****CG*T**--*****T--ctc****TGCA-CGTA--TaT***C*TCT*AaCCa***T**AtctgtC**TTGCGAGAGT*Cca**AC***-GAt*GG***T*Gt*****g*aC**CTTC*ttccGACTTtAGT-C*A-*-C*G*****AG**GG*C*TT*tA*****AG--Ac****-**C---GCAGcC*GTG*****GGCGCT*****Ctt**t****TGT**Ag***GAC**TccGAC*-T**T***CGGCTAcG*GCCAgAATCA--*CGGCCTa-*-*ACCTGT*C*****T---**-**---TC*gCGG*GA-C**ACG**GTC********TT*-*--CG*CaTA*gTCG**GG-GA*TGGCGAg****T*G*T---***--TACggc****cG**TT----T-cTCC-C***CC***TA*TagC*****G*A*agctT*G***A******Atca***C*TcgcT*CCC*GA*TGAG***CA*GC*cCT**CCtaTGgTG*CA*TGaagaCA*cG**A**-*-A***************GAGc*CG*TC**-*--C-CA*CTccc**T-**-TAC*tC*G*tt*G*GGAAATT****ATG*ATC-G*GC--**GC*TCCT*GcTttAACCG*GGTT*GTg-GA"),
    // Record::with_attrs("12",     Some(""), b"********GAC*TCT-A*AAGCgAT*t**A---*-C**tTT*G*AC**A***CT**********ATC***C****GCT**gCAC--ac--CG**AAcC**-TATcACtT***G*-GA-A*T-AgAgC***CT******GAAT*CAa**A*GAG-GC*TAcCAAGCGTA******gg*T***tAC*T*----CT******GT-G*CGGC****A*TAA***TA*GGAGC**A****C**AcAaGTC**-A*Ac***Ga**---*-*-----TT***C*****A*caT*GAT*CAA*-*-***CG*C*GCAC***AT*****T****CTAAaC*CTG*GcCTCGCTgactgttGAG***C****CGCG*GTTT**----*G-*GA-TgtCGCAT***A*G***A****gGa***CATAATACC**G****T*Att-gt**CA**Cg*A**TTGCTCGcCATTat**C*Gc*aCA-A*G*ATt-TG*T*ACC--**----AA**T*TaCCGAAG**G*GGTC*GA-*-CTCGTCC*****TTT****T---*-gAG****GATcgGCTG****AGgatggT*GTgtTA*GGTTT***Ag--**GGTTAG-G**CC**C******CTC*TTGAT***ACG**G--ACT***GCTAA*******C*******GAt*GGA***ATC-****TCCT***A*a***-*-T*CGT***tA**A**GA*CCCTGT*T*CGA*****-ACTACGCT*T****ATC*GGA*GGTCTA***TC*-T*CGAT*AGTCgG***TGTacacc-*--A*T***AGGCAC*A****CT-***T-*G**CTGTCCG--GG******AAT*AC--A*TA***ttACAgTG***---G**C*-T*T*A***AA-G**--**AC*A*A**CA*-AGAG**TTA*A****A-**T*AACCcA--TaA**GG**ATATTG*TTA*A*GCTa***--CA****cT*-**A*TTTAC***C*C****T*T**C*GAGGGTA-A*G*AtGA**GC-AG******T*CG**T**-C**G*A*G**-***-TT*AG-**GTA**A**GGA*-*AcTAAcA**--CAAG*T-G-gt**A******-*GC*Tgg*A--CG*G*CGGAAGT*-GGA*GGCTtAgC****G*********AGAGC--GATA**CA*****CGAA****Ta**Ac**gACCA*CATAG******TGATCCCTTC****CG*T**--*****T--agc****TAGT-T--A--CaA***G*CGT*AgGGa***T**Tg**taC**GATATCATAT*Cga**CC***-GC**GG***T*Gt*****a*cG**CCTT*****GAGAA*GTC-T*TA*TT*G*****TG**GT*A*AAcgA*****CC--Tt****T**C---TCAG*A*CTAcga**GGACGA*****Gct**g****AGC**Ta***AGC**TtgGGT*TG**T***CTTCAGtC*TTTGcAAAAG--*AC-TGG*-*-*ACCGAC*G*****TGG-**-**---CC*tGTA*TC-T**TGG**TCT********TA*-*--CT*AaC-*aTCT**TG-TC*CCCACAg****T*C*T---***--ACTccc****cT**TC----GGtAGC-C***CC***TA*GtcG*****T*C*aacgG*A***A******GtcagcaA*T*gcT*CAA*AC*AACG***GC*GG*tGT**TAccCGtTG*TC*GG****CG*aA**C**A*TG***************AGGt*AG*AC**-*--G-TT*CTcag**T-**-TCT**A*Aaga*G*TAACGGT****GTC*GCT-T*TC--**AC*T-TA*CcTgtATGCG*TCAC*AAg-TA"),
    // Record::with_attrs("15",     Some(""), b"********TTT*TTT-GcGACG*CC****TGGTt-T**tCA*AtAA**C***AA**********TCA***-****TCG***CATGA*a--CG**CC*GctGT--cGT*A***T*-AG-C*CCAtAgAaatAA******GTGA*TT***CtCGCTGT*AA*AATTACTG*********G****AC*T*C--GCG******A--G*TACTtttaGaCAG***CTgG--CT**C****A**C*-tG--**--*-****-t**ACTaC*G-ATCCA***C*****A***G*G-A*CCT*C*CcttGTgC*CAGT***AG*****T****TTTC*A*AGG*GgAATAAT*******AAGtccTtct*CTGA*TCCG**--TC*GC*GCTTcaCCCAA***C*C***C*****Tt***TTTGTCTAC**G****A*G**C****GG**Ct*G**T-ATGGCcGAGT****C*C***-AAT*C*AT*GTC*C*TTT--**-C--CA**A*T*G---AC**T*GAA-*GCG*TAATACTAggc**TGTgtcaA---*-gTC****CTT**ATAT****TG*****C*TActCGaAACCC***A*TT**GCGGCCAC**CG**C******CTA*AGCATaggCTA**A--GTC***T-CCC*******T*******AG*cGGT***CGGA****CTCT*gcG*****G*GC*TGG***cC**T**TA*CAGTTA*A*TTA*****-CCGTGCAT*T****ATA*TTGgTCATTC***TG*AGcGTTC*ATAA*Cg**TTG*****C*CCG*G**cCTTTCG*C****AACgcc-AaG**GAGTTAT--TC******TCT*C-CC-*--*****-GG*GG***---T**T*GGgC*T***ATTG**--**AA*T*C**GT*AAGTT**GCT*A****CTgtTgAATT*G--C*T**AA**GCGTGG*TAC*A*GAT****--CG*****Ga-**A*ATTGC***C*G****A*A**T*GCATCGACG*C*T*GCt*CCTCT******C*TG**T**CG**A*A*CagGctaT--*--C**-CC**C**CGGcC*T*TCC*G*tCCATCA*CTA-*g**T******-*CA*C*c*-----*-*----AATt-ATG*CCCC*-*-****T*********AGCAA--CCGC**AA*****AAAC****C***A****CGAG*TAA--******--GTTACTTT****GC*-**-G*****ATG*******GGGACCCTCTAC*C***G*CCA*G*AAa***C**G*****A**GCTTCGTACA*C****GA***CCC**GT***G*Cg*****c*cT**TCGA*****GA--T*AGG-G*TC*AT*T*****TC**CC*T*AC**Gt*g*tCA--Ta****T**C---AGG-*-*GTT*****TTGCGG*****C*c*******ATA**G*a**CCT**C**GTCtGG**G***AACAAT*C*GGGT*GTCGCAG*CCTCTT*-*-*CCTAATcA*****TCCA**C**ATCTTtaCGC*CTAGggGCA**AT-********CTaC*TTCT*G*CC*tCAC**GT-GC*ATGAAAc****A*C*TCAT**gC--CG*******cG**CG----AAtAGGCA***GT**aGG*C**TaggggT*T*ttaaG*C***C******A******T*C***A*ACC*TCg-GCA***GA*TG*aAG**AT**TG*GC*AG*GC****GTg*A**G**A*TC***************CCA*aAC*CG*gG*GAC-AC*GC*****A-**-CTG**A*T****T*CTCGTAC****AT-*-T---*TATC**GA*CGGTtT*GagGGTTC*TCCG*-C*--A"),
    // Record::with_attrs("7",      Some(""), b"ccaaaa**CCA*GGC-TtGTCA*TG****TGCA*TC**aGC*TtGCat-***-A**********G-AccgG****TTC***CTTAG*a----**-A*TacG---*-G*G***-*-AT-G*TTAcGaAg*cCAt*****AGGG*GT***TaGATACAc-T*GGAGGC-G********a-****GAc-*---GGG******G--C*CGC-gaccTcGTC***GGgT--GA**T****T**GtTaC--**--*-****-g**T--*-*--TTAAG***C*****C***A*TAT*GGA*C*CtagTAgC*TATG***AC*****C****GCTG*G*CAT*T*-TCGAC*******CAAtaaGact*ATA-*-TCTgt--CG*GA*-ACTtaG-AGA***AcA*gcT*****Ccc**CCCATTA-G**C****-*G**C****--**CccG**A-TGACAtGAGC****C*A***-AAG*G*GA*CAA*G*GCA--**-C--TC**A*C*A---AT**T*ACG-*GTTtATGTAG-Agat**G-T****T---*-gGG****CGA**TAGC*c**GG*****-*TActATcAAAGC***C*GT**GCCCCCGG**-C**A******AATgAATGTacgAGGcaA--ATA***CATC-******tAgggttt*AC**CTT***AAGA****-CTG***A*****A*AG*CTA***aA**-**CA*AGTTAG*A*CCG*****-TAGACCGA*T****G--*GATcGCTA-T**aGAtCTcTAAC*ACTG*G***TGG*****GaCGGgC*ggTCAGCG*G**t*TTGgct-AgT**TT-GGCG--CA******ACGcG-CTC*C-*****-TT*CG***---TatG*CC*C*A***CGTC**--**AC*C*C**--*TCTTT**C-T*Aaga*AGacTcAAGA*G--C*C**AAtaCCGACC*TTA*C*C-G****--TT*****Ac-**G*GGGCC**aG*G****C*G**C*------GAT*C*C*AGt*AGACC******A*GA**T**AA**T*AaCa*-aaaG--*--Ctg-AA*tAgcTAAaC*-*CAA*A*cCAG---*----*a**A******-*CTaC*c*C-CTC*-*CGGTCAG*-G--*CGCC*-*-****G*********GCTGC--ACCCat--ct***GCCAgt*cG***G****TGTG*T-C--******--CCCGAACT****CTa-**-G*****GCG***t*atAGACGAAGCCTC*A***A*ATA*G*GGa**tA**C*****C**TCCCGCAATAcT****CT***TCG*aGC***Cc-attgggtcgAg*AATT*****GC--G*GGG-C*GA*ATcCcaccaTA**TA*G*CC**CaatcgAA--Ac****G**A---TCG-*-*TCT*****AGCGTA*****G*a*******GGGacT*c**ACT**G**AACtAGggG***---TAG*T*GCGT*GAAGTGG*CTTTCG*-*-*ACGGTT*T*****GATCtcC**GCT-GtcGTG*AAGCatGCA**AC-********TG*C*TCCC*-*AGttCCG**AC-AA*TGAAATa****A*G*CTAAtgaC--AT***gggtcG**AA----GTaTA-C-catGGtttGA*C**T*****T*G*agtcC*C***C******-******A*T***A*TAG*AC*-AT-***AG*TA*tCC**AT**TT*GT*TA*TG****TAa*AtgA**T*GCt*****cttatt***GAG**GA*CC*tC*TAT-GA*GC*****C-**-AAT**T*G***aAaCATTTGTat**GA-*-TG-AtTCTA**GCcAG-G*G*CtgCT-TA*TAGC*-T*CTA"),
    // Record::with_attrs("8",      Some(""), b"aga***ctATG*TTC-AgCCTT*GT****A--G*AG**cAG*AgCGta-***-A**********G-Ac**C****AGC***GGGGA*c----**-A*CtaT---*-C*T***-*-AA-G*GGAcTgCa*cGA*t**caAGTC*-C***AgAAAT-Tt-T*GCATCCCT********tG****CCa-*---GGG******T--C*CTG-gcgtAgAAT***ACgC--AC**A****T**CcGgA--**--*-****-t**T--*-*--TGGGT***C*****C***T*TTT*GTT*T*CtagTTgC*CAAG***CT*****C****AGTC*T*GAA*T*-TAGCC*******ACTta*-cga*TTA-*-GACcc--TG*TA*-TAAacG-CC-***TaG*aaA*****Cc***ACGAGTG-A**T****-*G**G****--**TgtG**A-C--GCgAATC***cT*C***-ATG*A*AA*GTC*G*TCT--**-A--AT**-*A*G---CT**T*CCC-*GATgAAGCAA-TtaattA-C****T---*-gGT****CT-**GTGC*a**CT*****-*TCacACtCGCTA***G*TG**GCAGT-TA**-C**A******GATgTAATTatgAGTtaT--ATG***AACC-*******AaagtatgAG**TGG***GGAG***t-CTC***A*****C*ATgTTC***aC**-**GG*TAGTAGaC*GAA*****-ATTCGGGT*-****A--*GCGcTCTC-T***TCaCAgAGAT*TCCT*G***TGT*****CcGTAaG*ctAATCAA*G****GTTtgt-AgC**TT-GGTC--TC******CGTcC-TCC*A-*****-GC*AA***---AgcG*CG*C*A***CTTC**--**ACgG*T**--*AGTT-**A-G*Aatt*AAatAgTTAG*T--A*C**ACgaGAGTGC*TC-*T*C-C****--TT*****G*-**-*-GTGG**gA*G****A*G**C*------GAA*T*T*GGt*GTTCA******T*GT**G**TC**T*CcGt*-gtcG--*--Ggt-AG*tTggCTGcC*-*CAC*C*cCCA---*----*c**G******-*CC*A*t*A-CAA*-*AGCTCGT*-A--*CGGC*-*-****A*********GACGC--ACTActTA*****TGAAgggtG***T****GACG*C-G--******--CTAGACCA****GG*-**-G*****TCC***tgctCGCC-AGGTGGG*C***T*-CC*T*GGc**aG**G*****T**ATGGAATGTG*T***gTC***GGC**--***AaCacg**gg*gGa*GTTA*****AA--A*AGC-T*AT*AGgG***gaTC*aGC*G*TT**Ctttct----At****G**T---CCG-*-*GCC*****TGA--G*****C*a*******TTAtgC*t**GAG**G**CGCcTAtaG***TGAGGT*C*TGGA*CACTA-G*TGTTGC*-*-*GAGGT-*G*****AAAC*tC**TCAAAgcCGG*CAGTctTGA**AA-********GT*T*TGTG*G*GGgtGTA**AC--G*AGAGAT*****T*G*TTCCgtaA--GG***gacatG**AA----ACtCG-T-acaACacaGC*G**G*****G*T*acttC*T***Ag*****A******A*T***GcGAC*TT*-GT-***AG*CA*aTC**CA**GG*CAaTG*GA****TTa*-taT**A*TA*actagttgttc***CGA**CG*GC*gCcGTA-GG*CC*****T-**-TGG**G*A***tG*ATGGCGG****CC-*-CG-GtGTGA**ATcAA-C*G*AgaAA-TC*CCCC*-T*TTT"),
    // Record::with_attrs("4",      Some(""), b"********G--*-CGTG*TTG-*--****CAAT*TG***CA*A*CA**G***TA**********TGG***A****GCC***CTGCC**C-TT**CG*G**AGCC*CT*T***G*CGCGC*AGG*T*G***TT******TTGT*TT***T*CGGTAT*GA*TAGCTCCT*********A****GT*C*CAGCCA******GGCC*AG-A****T*CGAgct-A*GACTCtaC****T**T*G*GTC**GA*C****G*gtACT*CcCAGCTGT***A*****A***G*CGA*ATA*T*C***TC*T*CACT***AT*****T****CCAC*GgGTA*T*TCACTG*******GTT***A****ATGT*TATC**ACCG*TA*TATT**AGGGA***C*A***G*****A****CGAGTCGTC**C****G*G**C****GG**G**G**GGTGGGA*GGCC****C*C***TGAT*G*-A*ACC*G*CTAGT**CGAAGT**T*A*CATTAA**A*CATT*GGA*CCACACTT*****TAC****CAATtA*AA****GA-**-GTG****TG*****A*AG**AT*GAG-Ga**-*GG**GCGATAA-**TA**A******TAG*GAATA***TAT**CTCAGC***GCTAA*******A*******AT**AGT***TCGG****GAAT***T*****T*CC*CCC****T**C**GG*CTCTTG*T*GAT*****AGAGTACTT*A****GTG*--T*CCCAGT***TC*TC*TCTCcAACC*C***GCC*****G*ACA*C***CTGCAT*C***cATG***TT*A**GTTCTCAGCAC******AAC*GGGGA*AA*****GGT*GA***GAAG**G*GG*C*C***CATT**-G**AC*CgCctC-*---CA**CGC*A****TT**C*CTAT*ATCG*T**--**TT-ATA*TAA*C*CGA****CCCA*****A*C**T*AGTAG***G*T****T*T**C*GATGCTTGA*G*T*AG**TAC-A******C*GT**C**CC**A*G*G**C***TCC*GGA**TCG**C**GAC*A*G*CCG*A**---CCT*AGTG**acC******G*GA*C***GCCCT*C*TGGGATG*--TG*TTCC*G*C****G*********ATATCGTT-AC**AG**ttaATAA****C***T****TCGG*TAAGT******TATTCAAGTG***aGT*A**AT*****ACT*******G----CA--AAC*C***C*ACA*G*CA****A**C*****C*tTCGGGCTCCA*C****CG***TAA**AC***A*A*********G**GCTA*****TCCGA*GGCGC*CG*AA*G*****GT**TA*T*TC**T*****CGAAT*****G**GCTGTATA*A*CCA*****TAAGTC**gagG*********TTC**C****-TT*cG**GTA*TG**T***CTGGAG*G*GGGA*CACAGTA*CCTTGC*T*C*ATAA--*-*****TACA**C**TCCTG**GAG*ACTC**AAC**CCC******tt--*C*AGTT*T*TG**ACA**ACTGCc-T-TCC*****C*T*TGCT***CTGTG********G**AAGTTAGT*TACC-***AT***TA*C**C*****T*A*****G*G***C*gg*t*G******C*T***A*CCA*AC*TCGG***GG*AG**CC**AG**CC*CT*GG*GG****TC**C**C**C*CT************gaaGCG**AG*AC**T*-TTCCCtGA*****CA**AACC**G*T****G*TATACAT****AAC*ATATA*CAGA**CA*ATTA*G*G**C-GA-*--TT*TG*CTC"),
    // Record::with_attrs("13",     Some(""), b"********GAT*GCCGA*TTGC*TG****GAGT*AG***CC*A*CA**C***GG**********ACC***G****TTC***TTTTC**CATG**CT*G**TGAG*GG*C***A*-TACGtCTC*C*A***-A******GTTA*GC***T*GTTGCA*GC*CGGTTAAT*********A****GA*C*CTGCTC******A--C*TGGC****C*AGAgaaTA*GAGCTtgTtattC**T*T*G--**GA*T**c*A*atAGA*GgACAAGGAtgaA*****T***G*C-T*GTA*T*A***AT*T*AGCT***TT*****-****-G--*AaTCG*A*CTATCT*******-AG***G****TTCCaTCTT**AGCA*ACaTT-T**TTCGT***T*T***C*****C**atCTAAATAAT**T****C*A**C****AC**C**A**CCACCTA*CTAA****A*C***GGCT*A*-T*CCG*T*TGGGA**CGG-TG**A*G*GGTCGA**G*AATT*GCT*TAGCTTGG*****GAA****TTT-*-*--****GCG**TGGA****AA*****AaCG**CT*TTT-G*ac-*GCg*CCGGAGAG**GAggC*a***gTAG*GGCCG***GTC**AAAAAT***TCTAC*******A*******T-**GTC***ACAA****TAAA***G*****A*AT*TGT****T**G**C-*TGGTTT*A*CCA**ttcCTACCA-AG*C**cgATA*CCT*GCCCAG***TC*TC*CCTCcGATA*A***GTC*****T*ACG*G***CTACCA*C****TAA***GG*A**ATGCCAGACCC******ACC*GATGG*AC*****CGA*CG***GGGC**AaTT*A*T***AGGT*aGGc*AG*A*G**T-*---CT**ACT*A****AA**G*TAAT*TCCT*Agt--**CTTATT*GAG*G*TGG****TCTA*****T*A**A*AATAG***TgT****C*-**-*----TTCGTaGtG*-G**CCC-T******T*TA**C**CG**T*T*G**A***CCG*TAA**TAT**C**CTG*T*T*GCC*G**---AAA*CG-T****G******A*TT*A***CTGGT*G*GGGGT--*--CA*CCGA*C*A****A********gGCGCGCTC-CT**GG**cagTTT-****C***A****TTCG*CGAAT*acataGAAGCAAGTG***gGT*AgtGCccg**GGC*******ACCCCAG--CGC*C***G*TCC*G*AA****A**C*****-*aCAAAG-CAAC*A****AC***TAT**--***-*G*********G**GATA*****GGAGT*CGTACcTG*CA*-*****GT**AC*G*GA**T*****TTGAA*****T**-ACAAGTT*A*GAT***a*ATAGAA*****C*****gcc*GCT**A****-CA*cA**TCC*CC**G***CCTGAA*T*GACG*AGAGAGA*AATAAA*C*G*C-----*T*****TCGA**A**TACGT**AGA*GCTA**GGG**CGTgcacggga--*C*A-AG*T*AA**TAT**ACGACt-A-CCA*****GcT*TGCA***ATGT-********-**--ACTCGT*ATATT***TT***TC*G**T*****G*-*****-*T***G*g**t*T******C*C***A*CGAtTC*TCGC***CT*TG**AG**TC**CA*AG*AT*AG****CT**T**T**C*GT***************CGT**TG*GC**T*-AAGTTcCA*****AC**TACA**A*G****G*CAATATT****CCA*GGAGG*AAGG**AC*GGGC*A*C**G-TAC*TCGCcCA*GTC"),
    // Record::with_attrs("9",      Some(""), b"********GAT*TGATG*----*GT****CGGG*CG***GC*C*AG**G***TC**********AGG***T****TTC***CTGCA**AAAG**AT*C**C--G*AT*C***A*TTC-T*GTC*G*C***-A******CATG*A-***-*GCCAGT*AG*TGACGC--*********A****TG*-*-ACGCC******CGG-aATGA****A*GTCgc*TC*CAGATatAcagaT**A*A*GCA*tTA*T**c*T*gcAAC*GtAGGCATTg**A*****C***-*A-G*GTG*-aG***GT*A*CTAAga*AT*****A****-T--*-*TC-*A*ATATTA*******-AC***G****AATA*GCCA**TATGgACcAT-T**TCCCC***-*C***G*****-****ATATGTTAG**AagaaAtG**G**g*CC**T**A**TGGGGAT*TTGC****G*G***TGGAcA*-A*GATgCgATTGAt*AAG-GG**GgT*GGCTGG**T*TTAA*ACC*GGATTACA*****ATC****CCCCtT*AT****TTA**TACC**t*TG*****AcAT**--*CTC-C***-*CAa*AAGGATGG**TAtaC*gccagAA-*---TC***CCT**TTGG--***-TCCT*******C*******G-**G--***---G****CAGG***G*****TcCT*AAT****A**T*aG-*TTCGCC*TcGAC*****CGGTTA-AC*G**g*ATG*GGA*TC--GC***TC*CC*CAATtG---*C*ttTCC*****C*TG-*-***-CT-TC*T****TAT***TT*T**TGT---TAGCC******AAC*G-TGC*GG*****CGG*CAtggA-GT**A*GG*T*Tgc*CTTT**GCc*-A*A*A**C-*---TT**TGCgG***aCC**G*---C*TC-T*Tgc--**---TGC*CAG*A*TAT****GATC*****T*G**T*A-TCA***TtT****T*C**Gt----CCTGG*GaC*GG*aGTT-T******GcTT**TtcGC**GtC*G**G***CAG*GCT**CAT**A**-TT*C*C*AAA*T**---AGG*TT-T****A******C*AA*C***G-TGC*A*TGCCG--*--TGgTACG*-*A****AaactggcggGCACAGGC-G-**-C**tttTTTG****C*cgT****CAGT*CGCTT*cgggaGGACTGACG-****--*TttTA*****AGA*******GCTACCA--GTA*C***TcAAG*T*AT****T**T*****-*aGTC-T-TCCG*T****GC***ATA**--***-*-*********A*cGCGT*****T-CC-*TGACCgCT*AG*-*****GT**CA*T*GC**A*****GG---*****-**---GCCCA*A*CTG****gT-ACGG*****T*********CGA**G****-G-gcG**A-T*CA**G***ATTGTG*A*GTAG*TGGAATG*CTATTT*TtGcGCG-TG*Acagg*ACGT**A**AGGA-**GTT*GGC-**--GaaTAA******cc--*A*G-TG*A*GA**GAC**ACAGCa-A-TGT*ttctAcA*TCG-***TGCTA********T*cATGAAGCG*CAC-A***TT***AA*A**T*****A*-*****-*TgtaC*c*ca*A******AgC***T*TCCaCT*AGCT*ggCAgTT**GA**GA**TA*CA*GT*G-****GG**A**G**A*TC***************GGG**TA*AG**C*--CG--gAG***tgCT**A--A**T*C****G*ATAACTA****GTC*GGGAA*CACG**GA*CTAC*T*A**G-TA-*GACGaAT*CTA"),
    // Record::with_attrs("2",      Some(""), b"********ACC*TCATG*----*AT***tTCTA*T-***CC*G*AT**A***TG******ttcaTAT***TcttcATA***---GG**GCTG**AC*A**ACCG*TT*G***C*AAG-A*GCT*A*G***-C******CGCG*GC**cG*CACTC-*--*---ACAGG*********G****AA*-*-TTATT***ataCAGAaATTA****-*GTAcgtAT*CC-AAtg-tt*gAcgA*T*GAT*aTC*T**caG*gcGGA*T*---TGGGg**T*****G***C*A--gGAC*-tT***GG*AgGCTA***CG*****Ttggc-C--*-*--TgT*GGAAAC*******-GG***C***gAATA*AGTG**GAGAcGTcAC-A**TGTTGagtC*T***G*****C****CCATGGATA**CccgaT*-**A**ctAT**A**AtgACCCCGT*TGTT****C*C***CTCA*C*-G*CTA*-cGGT--**-AG-GGatA*G*GCGC--**G*T-AA*T--*-GAGCAAG*****TTG****GGCCgT*CA****GGG**TA-C***aTT*****CtGC**--*GTA-T***-*CGt*ATGTA-TT**GTatTct***cTT-*---CG***TAG**C--A--***-----*******A*******G-**GGA***GATA****ACAT***G**tgaCtA-*---****C**A**G-*GTTTCG*GaTTT*****GGACCG-CGgA**g*CTCtTCA*ACTTAC***AT*AT*GCTTaC---*-***---*****C*ATT*T***T-CTCTt-****TCT***CT*G**ACATACAGAGTattacgTGG*AGG-TtTG*****AGG*TT***CAAT**G*CG*T*A***ATCG**ACg*TG*T*T**G-*---CG**CCC*G***gTT**A*CGCG*TGTC*Tcc--**---ACA*TTG*C*CTT*a**A-GG*****G*TtcT*G-TAC***TgA****T*C**A*----ATAAA*GaC*TC**AAG-G******C*AA**C**A-**T*T*C**T***GCGcTTG**TTA**T**TAT*A*G*GCG*C**---TTC*TT--****A******GgCT*A**tG-CTCc-*AACGT--*--AG*GTTT*-*C****TttccgccgaTTAGCGCA-CC**GC*****TAGT****C***C****---GtCTGCG*atataGGCTAGGTACctagAA*AaaGT***gcTTG*******CTCCAAC--GTT*GtgcA*TTG*T*CG****G**G*****-*tCT-TC-ATAT*T****GGt**-CT**--***-*G*********-**--AG*****T--TG*CCGC-*TT*AG*-*****GG**CC*A*GA**C*****GTTAC*****C**---ATGAA*-*CAC*****---CAT*****G********tAAA**A**ga-AGc*-**G-A*GT**TgtaTGGTGC*A*GTAA*GGAGAAA*TCT--C*CgT*CGGGTA*G****aAATG**CttCGCC-**TGC*GGAA**CTG**G-C******gg--*A*C-GCcA*TT**GGC**ACTGCa---TCG***atCgA*TGCA***CTTTA********A**CAGTGGTC*TCCCT***AT***T-*-**C*****A*-*****-*C***A*****a-******CaT***C*-ACgAC*ATTT***AC*AA**C-**--**GT*GA*TG*GC****GC**A**C**-*TC***************GAG**ACcGG**A*-T-GC-aGC*****CTacACTT**-*T****C*AA-TAGT**tcTAC*CATAG*TGTCggTG*CTGG*T*T**C-AAG*AGACtGA*TGA"),
    // Record::with_attrs("14",     Some(""), b"********TTA*GGGTG*----*TT***gACTA*C-***CC*A*AA**G***T-**********---***G****CTT***AATTT**AGTTgcGA*C**CTGT*GA*AtagT*AGG-T*GTG*C*C***-A******AATC*AA***T*CTAAGA*C-*-TCCGATC*********A****AA*-*-CCTTA***a**GTAAtGGAA****-*GTAcctTA*AG-GGat-attaTgtC*C*TAG*gTA*T**c*T*agAAA*A*---G-AAt**A***ccA***T*C-CgCCG*-gG***GG*TgCCAA***CC*****G****-G--*-*GAC*G*CCAGCG*******-AA***C***tAAAC*GA-T**AACGcTGtTA-T**GAACCgccT*G***T*****G****TCCCTTCCG**TcattT*C**C**tcTA**A**AcaTTATTA-*-TTT****A*C***GTGC*T*-A*GAA*-aTCT--**-GG-GCacA*G*TCGCTG**T*G-ATg---*-GC-TCCA*****TTC****CTTAcG*AC****TCC**TA-A****CA*****AtGT**TA*GAA-A***-*CTt*CTGGTGAT**TAtaT*a***cTA-*---GC***GCA**GAAA--***-----*******A*******T-**TTG***TCTT****AGTA***G*****CtG-*---****CtaA**T-*TAAACA*AgGCA*****AGCTCG-TT*-**c*GCAaTAC*TGTGGC***CA*AA*TAGTtT---*A***GGC*****G*GTT*T***GCACTC*G****TAG***GT*T**TGCTAGGGTCA******TCG*ATCGGaTA*****TAA*AA***CGGT**A*CT*AtT**cCCGC**TCga-C*C*A**--*---T-**TTC*A***tAC**T*AGCA*TTTG*Act--**----AA*GGT*A*ACG*cggA-GG*****T*TctG*T-TGT***GgG****G*CgaC*----AATCC*AgG*AG**ACT-T******C*GG**A**AG**A*G*A**A***AGC*CTG**TCT**G**TGC*A*G*CAC*T**---GTA*GC--****G******A*AC*C**aG-CTG*GaGTGGT--*--CA*ACCC*-*C****TgagtgtttcACGGAGAG-AT**TG**ctgATCG****-***T****---G*GAGCA*aacgaCCCCGCCCAGcggcCT*CtcTC***cgACC*******TTACCGT--TCC*Gc**A*GAG*G*AG****AacC*****-*cCATCT-GCGG*C****CTa*c-CT**--***-*G*********-**--AC*****G--GC*GATC-*CC*TG*-*****CG**CT*T*GT**A*****GTTGC*****C**-ACACCCT*A*TTG*****---GAA*****G*********CAG**G**c*-AGc*-**G-G*GA**A***ATACAG*G*GACC*TGTCTTT*CTC--T*AgG*ACCCTA*T*****GAGG**A*tCCCT-**GGT*TACT**GCG**C-A******gt--*CcG-GA*A*TT**ACC**AATAAc---CT-*****AgC*TGCG***A-AAC********C**ACGCGTAC*AACTG***AA***TG*A**A*****G*-*****-*A***C*****c-******CtT***A*-GAgCA*ATGC***TC*AG**C-**--**TC*TT*AA*TA****CC**G**A**-*TC***************AG-**TGtAG**A*-G-AC-aGC*****CG**ATGT**-*G****A*AG-GACC**taAGG*ATCAC*ATCGcaTT*CCAA*A*A**A-TGT*GACAcCC*TTA"),
    // Record::with_attrs("N15",    Some(""), b"********AAC*G----*ATAG*TG****GTCC*GA**cGC*C*AG**C***TA**********CGA***G****GTG***GCAAG*g--AC**GT*A**TCGAgGA*A***C*AAA-T*TCG*GcG***GT******AGCA*AA***C*CTCTCA*AA*TCTCTGCT*********T****AG*CcTCGAAT******ACTC*CGTT****C*CGT***GA*CGTCG**G****A**TcGtGTAa*AG*Ag***Gc**GCGgT*ACGGGGG***C*****G***G*AGT*TGT*A*T***GT*G*AGTA***GT*****G****TAGT*A*G--*-*CGGGCT*******TGG***C****ACGA*TCCA**--TG*TA*AAAGcaACCAG***G*-***-*****-****-CTGGCTTC**A****T*T**C****CT**Gt*T**TGAGTGTaCCAC****C*Gg*cCCCC*T*GC*TCT*-*ATGCC**CGAACT**T*T*-GAATA**CgCTGA*ACG*TCTCCAA-*****GAC****A---*-gAT****ATT**TTCT****TG*****A*TT**AT*GATAG***T*CT**TACATTGTctGT**G******ACC*GAAAG***TAT**A--CCC***ATCGG*******C*******CG**GAG***ACGA****GATC***At****G*CC*GAC***gG**C**TT*CATAAG*T*AGA*****ACGCCAGGA*G****CGT*CGT*TAAATA***TG*CA*TGCA*ACCA*T***TC-*****C*AAA*A***ATGGGC*C****CAC***AC*T**CGGC-CGTGTA******TTG*CGAAG*AT*****CGG*AT***---A**C*GT*C*A***TCGC**AG**AC*A*C**TGcGAGCG*gCTA*A****AA**C*GCTA*C--C*A**TA**CCCCCAgTGA*A*G--****----*****G*-**T*TAGGAat*A*C****Cc-**A*ACGTGTGTA*C*C*CT**AATAT******G*GC**C**AC**G*A*A**G***AGG*ATC**TAC**C**G-C*T*T*CGT*G*gGGGAGA*TTG-*c**C******-*CA*C***T-AGA*G*AGCCACA*-AGT*TGGC*G*GtggaA*********ACAGACTGCTT**GC*****-GCC****A***T****GTTC*GGCCAt*****GGGACTTGTT****TA*A**TC*****ACG*******CCGATTGCGCTA*T***T*TGC*T*TGa***T**G*****A**GTTTCGAACG*T****AA***AAA**AT***C*G*********T**GACC*****GTGAG*CCC-G*TG*CT*A*****CT**CC*G*TG**G*****G--CC*****C**C---CGAA*G*CAC*****TTTGAT*****T****t****TGT**C****CTC**C**ATC*CT**G***CAGAAT*T*AGTT*ACCCCTC*GGCTCA*G*A*CGGACC*G*****GCCC**A**GAGCTagTTG*TAAC**GAT**GCT********TC*G*C---*G*AT*gTGAcgGC-TG*GAGTACa****T*G*CGTA**tGGAGT*******aG**CCC-AGGCtGCTGA***CC***TGcG**T*****C*T*g*aa-*C***C******A******C*T***G*CCG*TT*CTCG***CC*TA*tTT**AA**CT*--*AT*-C****GT**T**C**A*AC***************TCT**GG*GG**T*-CC-CG*TC*****A-**-CCA**T*A****C*GGTG--G****ATT*CAA-T*CGAC**AG*CCGT*T*G**ACGGA*ACAT*CA*GGA"),
    // Record::with_attrs("N16",    Some(""), b"********CAG*TGG-A*AAGG*AA****A---*-T**gTA*T*TA**A***CT**********CTC***C****GCT**gGAA--tc--GC**CAaC**-TGTcTCtT***T*-TG-A*GTAgCtT***CA******GAAT*TAt**T*GAT-CG*TAcTCAGCGTG******tg*T****GT*C*----CT******GT-G*GAGG****A*TAT***TA*TGAGG**A****A**TgAaGGC**-A*Ac***Ga**---*-*-----TT***C*****T*cgT*CAT*AAC*-*A***AG*G*ACCC***CT*****A****TTAA*C*GTG*GcATCGCTga***t*GCG***C****AACG*GCCC**--AA*A-*AG-AatCCCAT***A*G***A****cCt***CCTACCTCC**T****A*C**-gt**CT**Gg*G**TTGCACGaTACTat**C*Ac*aCT-A*T*TTa-AC*T*GCG--**----AT**T*T*CTGGCC**G*TGTG*GT-*-TGCGCGA*****CTA****T---*-gAG****GACtgGCGT****GTaaggaT*CGgtTA*GCTTT***Cg--**GATTCG-G**CG**G******CTC*AAAAT***TAG**T--CGA***GATAC*******T*******GCt*AGA***AAT-****TCCG***A*t***-*-T*CAG***tA**A**GA*GCCTGT*T*CTG*****-ACGACCCC*T****TTA*GGG*TGCCAC***CA*-T*TTAG*CGTA*T***TTAactacT*TCA*T***CGCTAC*A****CG-***T-*C**CTACCAG--AG******GAT*AT-GC*TA***cgAAA*TA***---A**C*-C*C*A***CG-G**--**TC*C*T**AA*-ACCG**GTT*A****G-**C*TCACcA--TcA**CG**ATTAGC*GGA*A*CGCa***--TA****cT*-**A*GTAGA***C*C****T*T**C*GTGAGCC-C*G*AtTC**GC-TA******A*TA**T**-A**A*A*G**-***-TA*AA-**GGA**A**GTC*-*TcAAAaA**--CGTG*TTG-aa**A******-*GT*Tga*A-TGC*G*AGGAAGG*-GTC*GGCTaAcG****G*********AGATG--CATA**CA*****CGTA****Tt**Ac**gGCCA*TATAA******ATCACCCATC****CG*T**--*****T--ata****TGGA-TCGA--TaA***G*TGT*TgCGa***T**Ag**atC**GAGCAAACTT*Cta**CC***-CA**GG***T*Tt*****c*aG**AGTG*****GTGTT*TGC-C*TA*TG*G*****GG**GT*G*AT*tA*****AC--Tt****G**C---TCAG*C*CTA*****GGCTGT*****Gta**a****CGT**Ga***GGG**TcgGAT*TG**T***CTGCGGgG*TCCGtAAACA--*TAGACG*-*-*ACCGGA*C*****TTC-**-**---AC*cGGG*AA-C**TCC**ACA********AG*-*--AA*TaAC*cTCG**TG-TC*TCGCAAg****T*C*C---***--GGCgcc****aA**TT----GGtTGT-C***CC***TA*TagA*****T*G*atgtG*A***A******Gtca***T*T*gcG*CCC*CA*TATG***AA*AC*tCT**TAcaCGtGC*AA*AG****CT*aG**G**C*AC***************GATc*CC*AC**-*--C-TG*GTcag**T-**-GCA**A*Gata*G*TAACATT****CTG*GAT-G*AC--**GC*TCAA*GtCttATGCG*TATT*TTg-TA"),
    // Record::with_attrs("N17",    Some(""), b"********CAG*TGG-A*GACG*AA****AAG-*-T**gTA*A*GG**A***CC**********CCC***A****GCT***GAA--tc--AG**CCaA**-TGTtACtT***T*-CA-T*GCAaAtC***GG******AGAT*AAt**T*TAC-CG*AAgACTACCAG******gg*T****CT*C*----CT******GC-G*CACT****T*GAT***TA*GGAGG**T****A**AgAgGGC**-A*Ag***Ca**---*-*-----GT***G*****T***A*TGC*AAG*-*C***AT*G*AATC***CG*****A****CTTT*G*GGT*AtAAGGCTgt***g*GAG***C****GACT*ACCC**--TA*A-*AG-AaaTAGAC***C*C***A*****Cg***CGTCTCCGC**T****G*A**-tt**CT**Ac*A**TGGAACGcTCTAtt**A*Ta*aGT-A*T*TCa-AC*A*GCG--**-T--AC**T*G*CGGGCC**A*GGAC*CC-*-TGCGTGT*****CAA****C---*-cAA****TAAtgGCGT****GT*****G*CCgcGT*GCGTC***CtC-**GAATCG-T**AG**C******CGC*ACTTT***GTG**T--TTG***CGTAC*******T*******TGg*ACC***AGT-****CCCA***G*c***-*-T*CAT***tC**T**GA*CTATCT*T*AGG*****-ATGGGGTA*G****TCG*GGG*AAACAC***CA*-C*TTAG*CGCA*T***CTG*****C*TCC*C***CATGAG*G****CG-***TA*A**ATGCCGT--TA******AAA*TT-GC*TA***aaAAA*AT***---A**C*GC*C*A***TG-G**--**TA*C*C**AA*-TACG**GTT*C****G-**G*ACAT*A--T*A**GC**ATTAGG*GCG*A*TCCa***--TA****cA*-**A*GCAGT***G*C****C*A**G*ATTTGCC-A*A*AtGC**TT-AG******A*TT**T**GA**A*T*T**-***-TC*AAC**GGA**A**GAC*T*A*CGCaG*cATTGCG*GCC-tc**C******-*GG*Tga*T-GGA*C*ATGTTAG*-TAC*GGTGtGtG****G*********TCACG--CTTC**CG*****GGGG****Ta**Ac**gGCTA*GATAA******ATGTCGCGTT****TC*T**--*****T--aca****TCGAAACAG--AaA***A*CTT*TgCGa***T**Ag**atC**AAGCAGTAAA*Cta**CG***-AG**GG***C*Ct*****g*cC**CGTC*****GGCTA*TGG-C*TG*CC*G*****AA**AA*G*TA*tA*****AG--Tg****G**C---TTAC*G*CTA*****GCCTGT*****Tct**a****CAG**Cc***GAT**TtaGAT*TG**C***CCAGTAgG*TTAAtAGACA--*TTAACA*-*-*ACTGGC*G*****GGAC**A**AAGAC*cGGG*AA-C**TCT**ACT********TG*-*--AA*TaAA*aGCT**GG-TA*TCGCGAg****G*C*TTA-***-GACCgcc****aA**AT----GGgTTA-A***CA***CG*GagG*****T*A*atggG*C***C******G******G*T*ggG*CCC*CG*TATT***AC*AA*cGT**TC**TTtGG*AA*AC****CA**A**C**A*TC***************GATt*TC*AG**T*CCC-TG*GGatg**C-**-GCA**A*G****T*TAACTCC****AAG*TAT-A*TG--**GC*TTAA*AtCttAAGGG*TGCT*CA*GAG"),
    // Record::with_attrs("N18",    Some(""), b"********CAT*CTG-GtGTGT*TT****ATTA*GG**tAC*C*AC**A***TG**********CTT***C****GAG***CGGGT*t--AT**GC*CggCGGCtGTtA***T*-AT-A*TTAcGgC***GG******ATAA*CCc**A*AAT-CT*TT*GCTCAGGG******tt*T****TT*A*----AG******GG-C*CGGG****G*GGA***GC*CACTC**C****T**AtTgGCTg*AG*Ta***Cc**TGAaA*ATTGACT***T*****T***T*GTA*AAT*C*G***AT*T*AGGA***AT*****C****GCCA*A*GAC*AgCCTACG*******TGT***T****CCTT*GATC**--CT*T-*TGGGccCAAGA***T*T***G*****Gc***TCTGATAAC**A****T*G**C****CT**Ca*T**TTTGGCAcCTGT****G*Gt*tGTCC*T*CC*ATT*T*TGA--**-C--TC**A*C*AGAGAC**C*AAGC*TCT*AGGACGGC*****TCC****G---*-tGT****CGT**CGTA****TT*****A*TAtgTT*GGTGT***G*TT**GAATTATT**CC**A******CGT*ATGAG***GCG**A--TGC***TCGTG*******C*******TC**AGG***TAG-****GTGA***A*****C*TT*CTA***tC**G**TC*TCTTTG*C*AGA*****-CAGCTGAT*T****AGT*TTT*TATATA***TT*CA*CGAG*CCCT*C***TAC*****C*CCC*G***AACGAC*C****AAG***GG*A**CTTCATT--TT******GCC*CTTAC*GT*****GCG*GG***---T**G*AT*T*A***CCCG**--**AC*G*C**GG*ATTGG**CTA*T****AT**C*CCGA*G--C*T**CA**CCATGA*ATG*A*ACT****--CC*****T*-**T*AATGT***G*G****T*C**G*TATCATGCA*A*G*AG**CAACT******G*TC**C**GG**G*A*A**-***-AC*GTC**CTC**G**GCG*T*A*CGA*A*aGTGTTA*ATC-*c**C******-*GT*Tga*T-ACG*T*GTGATAG*-CCC*CACAtA*G****T*********CGCTC--ATAC**CA*****CGTG****Ta**Cga**ATAA*CTCAC******ATCGTGACCT****TT*T**--*****A-A*******TACACCGGCTGA*T***C*GCC*T*ATa***A**A*****G**GTGGGAACCT*C****CC***-TT**AC***T*At*****t*gA**ATCC*****TTTTA*ACG-A*GG*CG*T*****AA**CA*C*GA**G*****CA--Tg****A**G---GTAC*G*CTA*****GCGCCA*****G*t**a****AGA**Ag***GGC**T**GAT*GC**C***TCACCT*C*CGCT*ATTGTAG*ATCGCT*-*-*GAATAG*T*****CTGC**A**CTATC*gCGT*ATAT**CGA**TTA********CC*-*--AC*CaAA*aTTT**GC-CG*GTCATGg****G*C*AAAT**gAGACC*******aT**GT----GCaACCTT***GG***CG*T**T*****T*G*agggT*G***A******G******A*G*ggT*CTG*CA*TGTA***AG*TT*tCT**AC**ACaGC*CT*GC****CAc*C**C**A*TC***************TTG**CG*GG*gC*GTA-AA*CG*****A-**-GCA**T*C****A*TGGCGAT****GGC*AAA-A*TGAG**GG*TCGC*T*GaaCTGGG*ATAC*CA*TTA"),
    // Record::with_attrs("N19",    Some(""), b"tca*****CCC*GGC-TtGACA*CG****TCCA*AG**aGC*TaGAat-***-A**********G-GctgA****GGC***CTTAG*c----**-A*TgcT---*-G*A***-*-AT-G*GAGcCaAa*cGA******AGGG*GA***TcAATATAa-T*GCGGTCCG********gA****AAc-*---GGG******G--T*CGC-gcaaTcGAC***TCgT--CT**C****T**GtTaC--**--*-****-t**T--*-*--TCGTC***C*****C***C*TAC*GAT*A*CtagTGgC*AACG***AC*****C****TCCA*A*CAT*T*-TCGAC*******ACTaaaGact*ATG-*-GATgg--CG*GA*-GCGtaG-GGT***GcA*gtT*****Cc***CGGAAGG-A**A****-*G**C****--**CccA**A-CGACCtAAGC****C*G***-AGG*C*GA*GTA*T*GCT--**-G--CC**A*A*G---AT**A*ATA-*GAAaCAGTAG-Agaa**G-T****T---*-gGG****CGA**TTGC*t**GG*****-*TAccATcTAAGA***C*GG**GCCCTGGG**-C**T******TATgAAGGTacgAGGcaA--AGC***CATG-******aCgtgttt*AC**CGT***AAAA****-CTG***A*****A*AG*GTA***tG**-**CT*CGTTAG*A*CAG*****-TAGACGGG*A****T--*GCTcTACA-T***TAtGGcGCAT*TCGA*G***TGG*****GaAGGgC*gcTAATCG*A****TTGgtt-AtT**GT-CAAC--CC******ATTcA-CAC*C-*****-TT*CG***---GatG*CT*C*A***CGTG**--**GC*C*C**--*TATTA**G-A*Aaga*ACatTaCAGG*T--A*C**ACtaGAGAGC*TGA*T*C-C****--TT*****Tt-**T*GGGGC**gC*G****A*G**C*------TAT*C*A*AGg*GCACT******C*AA**A**TA**A*AtAg*-tctG--*--Cag-AA*gAgaTTAaT*-*CAC*C*gGCG---*----*g**T******-*CC*T*c*C-CCA*-*CCGTCGT*-G--*CGTC*-*-****A*********GCCCC--ACCAatTT*****GCTAgg*cG***A****AGAG*T-T--******--CTCGAACT****TT*-**-C*****TGG***t*ctAGCCGAAGTACG*A***A*ATC*G*GGa**tA**G*****C**ACGAGCTCTG*G****CT***TCG**GA***AaCact**gc*gAg*AAGT*****GC--A*GTG-A*AA*ATcA***caTC**TA*G*CC**CcattgAA--Ag****G**C---TGG-*-*GCT*****AGAGTA*****G*a*******ATAcgT*c**CCT**G**AGCtAAggG***TTAGCG*A*TAGA*AAAGAGG*ACATCG*-*-*GAGGAG*T*****ATCCccC**TCTACtaCCG*AACCgtGCA**AC-********TG*T*TTGC*G*GGctGCA**TC-AG*AGAAATa****T*A*CTGGtgaT--AT***ggatcG**AA----GTaCC-C-agcTGcttGA*G**T*****G*G*agttC*T***Cc*****T******A*C***G*AAG*TC*-AA-***GG*AT*tCC**AT**TG*GT*TG*GC****TAa*CggA**T*TCt*****ctcatt***GTC**GA*CC*tC*TAA-GA*CC*****C-**-ATT**T*C***tC*CGTGTGAga**AA-*-TA-TtTCTT**ATaAT-G*G*CtgCT-TC*TATC*-T*TCC"),
    // Record::with_attrs("N20",    Some(""), b"********TTG*CGC-TcGAAG*CC****TGGT*AG**tTT*AtGC**C***AG**********GCT***A****TAA***GATGC*a--CG**GG*AgtAT--aCG*A***C*-AC-C*CGCtAgAaaaAA******GTAA*TT***GaCGTTGC*GC*TATAACTG*********G****AT*C*T--TGC******C--A*CATActaaTtTGC***CTcA--CT**C****A**GtTtG--**--*-****-t**ATTaC*A-GTAGG***C*****A***G*GCA*AAT*C*CgttAAgC*TGGC***AT*****T****TTTT*T*ATG*GgGAGACT*******TAActtTaca*GTTT*TTCA**--CC*GG*GCTAacCCCGA***C*C***A*****Gt***TTTGTTTAC**C****A*G**C****AC**Ca*G**T-ATAGTtGTGA****T*A***-AAT*C*AT*GTG*C*CGA--**-C--TA**G*G*G---GT**C*AAC-*GCG*TAGTTGTAagc**TTT****A---*-gTA****CTG**ATAG****TG*****C*TTttCGaACACT***T*TT**GAAGGCAT**GG**T******CTA*ATGATtggCTA**T--CCT***TATGC*******G*******CG**TGC***CCGA****CTCC***A*****G*GG*ATG***tC**T**GA*ATGTTA*G*ATC*****-CTTCGCAT*T****AAT*TTAgTGATGC***CG*GAcGTTC*CAGA*C***GGG*****C*CCG*G**tACTACG*C****TGAgcc-AaG**ATGTTAT--GC******TTT*C-CCC*AC*****TGG*GG***---T**C*AT*C*T***GTGT**--**TA*C*G**GA*AAGTA**TAT*A****CTccTgACTT*A--C*G**AA**TCTTCG*TAC*T*GTT****--CT*****Aa-**C*ATTGC***C*G****A*C**T*GCTTCCGAG*C*G*GGt*CCGCT******C*CT**T**AT**T*T*CcaTccaT--*--C**-AC**C**CTGcC*T*TCG*G*aACCTCC*ATA-*g**T******-*CA*C*c*T-CCC*-*ACGAATT*-CTT*CACT*-*-****T*********GTTGC--AGGC**AA*****ACAA****C***T****GAAG*CAT--******--GTTATTTT****CC*-**-G*****CGT*******CGGAACAAATAT*T***G*CCC*T*ATa***C**G*****A**TATTCGTACT*C****AG***TAC**GT***A*Cg*****c*aA**ACGG*****GA--G*AGG-G*GC*AC*T*****TC**TG*C*AG**Gt*g*tCA--Tg****T**C---ATG-*-*ATT*****ATGCTG*****C*t*******GCA**G*a**TCT**T**CTTtGG**G***AAAACT*C*CGGT*GTCGCAA*CCAGCT*-*-*GATAAT*A*****CTCA**A**GTATTtaCAA*CTAGaaGCT**GT-********AT*T*CTAT*G*CG*gCAC**GT-GG*ATAAAAc****A*C*GCAT**aC--CA*******cG**CG----AAtAAACA***GGgcaAG*C**T*****T*T*ttaaG*C***C******A******A*C***A*TCC*TG*TGCC***CC*TG*tGC**CA**TG*GC*AT*GA****GCg*T**G**A*TC***************CAA**AC*CA*gC*GGA-AC*CC*****A-**-CTA**C*T****T*TTCGTAC****AT-*-CA-G*TCTC**GT*CGGT*T*GtaCGATG*TCCG*-C*TGG"),
    // Record::with_attrs("N21",    Some(""), b"********CGA*TTC-GcGAAG*TT****CTAT*GC**gAA*C*CC**C***GA**********TTA***A****GTG***CTGTT*t--AG**GC*CgcCGGCtGA*A***T*-AT-T*TTTcGcC***GT******AGAA*AA***A*CGTGCC*TC*TCTCAGGG*********A****TT*T*T--AAG******GG-C*CGTC****G*CGA***GC*CGCTT**C****T**AtTaGCAg*GC*Ag***Tc**TGTaA*ATGGGGA***C*****T***G*GTA*AAT*C*G***AA*T*ATAA***GT*****T****TCTG*A*GAC*TaAGTACG*******TCT***T****GCTT*TATC**--CT*TG*AGCGgcAGCAA***T*T***G*****Ac***TCTGGTATG**A****T*G**C****CT**Ca*T**AGTTGTAgCTGG****A*Gg*tGTCT*G*CC*ATT*T*TGA--**-A--TT**A*C*AGATAC**C*GAGC*GCT*TGGACACT*****CCC****G---*-tTT****CTT**GGTA****TT*****A*AAcgTT*GGTGT***T*GT**GAATTACT**GG**G******CTT*ACAAG***GCG**A--GCC***TCCAG*******C*******AT**CGC***GCGA****GTCC***A*****G*CA*TTT***tC**T**CT*ATCTTA*A*AGA*****-CATCTGAT*T****TGT*CTT*TAAATA***TC*CA*AGCA*ACCT*T***TGA*****C*CCC*A***AATGTC*C****AAG***GA*A**GTTAGTT--TG******GTC*CTTAC*GT*****CCC*GC***---T**G*AT*T*A***CCGT**--**AC*G*C**GG*AACAG**CTT*T****AT**T*GCTA*G--C*A**TA**GCAGCG*TTG*T*ACC****--CG*****T*-**T*AATCC***G*G****T*G**T*TATCCGGGA*G*G*AG**CATCT******G*CA**C**GA**G*A*A**T***CAT*GTC**CAC**G**GCG*T*C*CGA*G*aGTGTCT*ATT-*c**C******-*CT*Cat*G-CCA*T*AAGGTAC*-CCT*CGCT*G*G****T*********CGCGC--GGAA**AC*****ACTG****A***C****ATAG*CTCCC******TGCTAGACCT****CG*T**GA*****AGA*******CGAACCAGCAGA*T***C*CCA*T*AAa***A**A*****A**GTGTGTACCT*C****CC***CGG**CA***A*At*****t*aT**ATCC*****TTGTG*AGA-A*GA*GG*G*****TC**CA*C*GT**G*****CA--Ta****A**C---GTAA*G*CAT*****CCGGCA*****G*t**g****AGA**A****CTA**G**GCT*GC**G***TAACCT*T*CGAC*ATATCTG*GCCGCT*-*-*TCATCA*T*****CTGC**A**TTGTCcgCAC*ATAG**CGT**TTA********TG*A*CTAT*G*TC*gTGA**GG-AG*GTCCAGg****A*C*AGAA**aAGAGC*******aA**GC----GAaGCGTG***AC***CG*T**C*****C*A*agatA*C***C******A******A*C***A*CTG*TT*TTCA***AT*AA*tCT**AA**AC*GA*GA*GC****GAg*C**C**A*TC***************TTA**CG*CG*gT*GTA-AA*CC*****A-**-CCA**T*C****G*GGGTGAT****GTC*AAC-A*TGAG**GG*TGGA*T*GaaCCGGG*ATAT*CC*TTA"),
    // Record::with_attrs("N22",    Some(""), b"********AGA*TTC-G*GAAG*TA****CTCC*GA**gAA*C*CC**A***TA**********CAA***A****TTG***ATGAG*c--AC**GT*A**CATAgGA*A***T*AAC-T*TCT*GcC***GT******GGCA*AA***A*CTTTCC*CA*TCTCAGAG*********T****AG*T*AGGAAC******AGTC*CGGT****T*CGG***GG*CGACG**G****T**AtGtGTAg*GA*Ag***Tc**CCTgT*AGGTGGA***C*****G***G*AGT*GAT*T*G***GG*C*AGGC***GT*****T****TCTG*T*GAC*C*CGGGGT*******TCT***A****TCAA*TCCA**--TC*TG*ATAGgaATCAG***T*T***G*****C****TCTGGCATC**T****T*T**C****CT**Aa*T**AGTGGGTaCTGG****A*Cg*gCTCC*G*GC*ACT*C*TGACC**CTAATT**T*T*GGAAAC**C*TAGA*GCC*TCGGCAGA*****CGA****G---*-gAT****CTT**CCCT****TC*****A*AT**AC*GGTGA***G*AT**TATTTCGA**GT**G******CCC*GCAAG***GAT**A--GAC***TTCTC*******C*******AC**GTC***TCGA****GGTC***A*****A*CC*TTC***gC**G**CT*CCGAAT*A*AAA*****ACACGAGGT*G****TGT*TGT*TAAATA***TG*CC*TGCA*ACCT*C***TCA*****C*AAA*A***AATCTC*C****AAG***AA*A**GACAGCTTGTC******TTG*CCATG*AT*****CCG*GC***---A**G*GT*T*A***CCGC**GG**AC*G*A**GG*GAGGT**CTG*T****AT**T*GCTC*G--C*A**GA**CCCCCC*TGA*A*GTT****--CT*****T*-**A*TAGGA***A*T****T*G**G*CTTTGGGAA*C*G*CG**GACCT******G*CA**C**AA**G*A*A**A***CAT*GTC**CAC**G**GCG*A*T*CGG*G*gGGGTCT*CTG-*c**C******-*CT*C***C-TCG*G*AATCAAA*-AGT*CACG*G*G****T*********GGCGACTGCTG**AC*****AGAG****A***G****ATAG*CACCC******TGTACTTGTT****TA*G**AC*****AGG*******CCGCTTAAGTGA*T***C*CCC*T*TGa***T**G*****A**TTCTCGACCC*T****AC***CCG**AA***A*A*********T**CCCC*****TTGTG*AGG-T*GG*CG*C*****TC**CG*C*GT**G*****CATCG*****C**C---CGAA*G*CAG*****TTTGTA*****G****t****AGT**G****CTC**T**GTT*GC**G***CGAAGT*T*CGTC*ATGTAGC*GCCCGG*G*C*CGAGGC*T*****GTTC**A**TTACCagTTC*ATTC**CGT**TTA********TT*G*CTAA*G*TT*gTGA**GG-AG*GCGTCGg****T*G*CGAA**aTGAGT*******aA**GCG-ATGCaGCCTG***CC***TG*T**C*****C*T*aggaG*G***C******A******C*C***G*CTG*TT*TTCA***CC*AG*tTA**AA**GT*GT*AA*AC****GT**C**C**A*AC***************TTG**GG*CG**G*ATC-AA*CC*****A-**-CCA**T*A****G*GGTGGAG****ATG*AAA-T*GGTC**GG*CGGA*T*G**TAGGA*ACAT*CG*TTA"),
    // Record::with_attrs("N23",    Some(""), b"********CCG*GAAAT*----*CT***gTCTA*G-***AC*A*AG**T***CA**********TCT***T****CAT***ACAGT**CGTT**CT*C**ATTG*GA*C***T*AAT-A*GTG*A*C***-C******CGTG*TC***C*ATAAGA*G-*-TCAGATG*********G****AA*-*-ACTTC***c**GGTAaGATA****-*TTGcctAT*AA-CCtt-attaAgcC*C*GGA*gTC*C**c*T*agGAT*A*---TAACt**G*****T***T*A-CcAAG*-tG***GC*GgCTGA***CG*****G****-C--*-*ACC*T*CCTCAG*******-AG***C***aGTAC*TAAA**AACCcTGcTC-A**CATCCatcG*T***T*****C****GACTCAATT**CcatcT*T**C**acTC**C**AtgTGTGCGT*TTCT****A*T***ATGA*C*-A*ATA*-aTAT--**-AT-GCaaA*A*AGGATG**T*G-TT*C--*-GAGTCCG*****TTG****CTTAcG*AT****GCA**TA-A****GG*****AtCT**TA*GAA-A***-*TAt*TTCCGAAC**TAgaT*c***cTA-*---CC***ACG**CGAT--***-----*******A*******A-**GCG***TCTT****TTTC***G*****AtA-*---****C**C**T-*GAGCCA*GaTAA*****GACAAG-AG*A**c*CCCaTAC*AATCGC***TC*CG*CTCAcT---*A***GGT*****G*GCT*T***GACCTC*G****GCA***GG*T**TCGTTAAGTCA******TCG*ACCGAcGA*****CGC*CC***ACGT**A*CA*A*A***CTCT**ACg*TG*C*A**G-*---TG**CAC*A***tAG**C*CTCA*TGTG*Gac--**---TGC*GAG*A*CTG*c**C-GA*****G*GtcT*G-TGT***TcG****T*A**T*----CCCCG*GgG*AC**AAG-T******A*CG**A**AC**G*G*A**A***CGT*CTG**TTA**C**TAT*C*G*GGC*C**---GTA*AC--****A******A*CT*C**aG-CTC*C*CTTGC--*--CT*ACCG*-*C****TtgacgtgcaGCCGCGAC-TT**GT**taaCTTG****C***A****---G*TTGTA*agggcACCAAGGCAGagagAT*CgcTG***ccGGG*******TTCCAGA--CGG*Tg**G*GCC*T*AG****A*tC*****-*cGAGGC-TAGG*C****GTa**-CA**--***-*G*********-**--TA*****G--CG*AGAC-*CC*AG*-*****CG**TC*A*TT**C*****ATGGG*****G**-CTTCTCT*T*TTC*****---CAT*****T*********GTA**C**ag-AAc*-**G-C*GA**T***ATCCGA*G*GATC*TGTTACT*TTG--T*GgC*CAGGTA*G*****AACT**A*tCGCT-**CGA*CCAA**TGG**A-G******ag--*C*G-GCcA*TA**ATC**ATTGCc---CGG***gaCgC*TTAC***ATACA********C**CAGTCTTC*GTCTA***GT***TC*G**G*****A*-*****-*A***C*****c-******GaT***T*-CCgTC*ATTC***CC*AG**T-**--**TT*GT*TG*TT****GA**C**T**-*TA***************AGA**GGtAA**C*-T-AC-aGT*****CT**AGTT**-*G****T*CT-AACA**gaGAG*CACGT*TGGGagTC*CCAG*C*C**A-AGC*TAAAcGT*CTA"),
    // Record::with_attrs("N24",    Some(""), b"********CTT*CAGAA*----*GG****ACGA*CC***AC*T*CG**G***GC**********AAG***G****TTC***TCGTG**TAAG**CT*C**TGAG*CA*C***C*TTTGA*CTC*A*G***-G******CCTA*GA***G*GTGACG*GG*TCATTAAT*********T****TG*-*-TACTA******TGAGcAGGC****C*ACTcctAA*CATCTttTtcttT**C*A*TGA*tTA*C**c*G*ctTGT*AgAAGCAAGt**G*****C***C*A-G*GCA*-aC***TG*G*CGCC***GT*****G****-A--*-*ACG*A*GCATAT*******-TG***G****AGGC*ACAT**AAGGgCTcTC-T**CACCT***T*T***A*****A****CTATAGGAA**CggagC*A**T**t*AA**A**C**CGACCGA*GTGC****A*A***TGGC*G*-T*AAC*AcGAGGGgaGGA-TG**T*A*ACTCTG**C*ATGT*AGG*AGACCTCT*****TAA****TAAAcG*TA****TGA**TCGC****TT*****GtCC**CT*ACC-A***-*GCt*TCGCAGCC**GAcaG*c***cGA-*---TT***CAG**CAATGA***ACTTC*******A*******T-**GCC***ACGA****TTGG***G*****AcGT*CAC****C**T**T-*TCGGTG*GaGAA*****GTATCA-AG*A**c*ATA*GCT*GCACCT***TT*CC*CGGGaT---*C***CGA*****T*CGC*C***TATATG*G****TCA***TA*T**GTTTTGCAGCC******AAC*GATGA*CC*****GTT*GC***ACGT**C*CA*T*C***AGGA**GCc*TG*T*G**A-*---CT**GCA*A***aCT**C*AGGA*GGGA*Taa--**---CTC*CAG*G*AGG****CATA*****C*G**T*G-TAA***GaT****T*C**T*----ACGTG*TgT*CA**TAT-C******G*TA**T**GC**G*G*T**G***CGC*TCA**CAA**A**ATT*G*T*AAC*C**---GAA*TTAA****A******T*TT*A***A-AGA*T*TGCTC--*--CA*CGGA*-*A****TcaacggcagTCTCAGTC-CA**GC**cttTCCG****C***A****TTTG*CAAAA*atttaTTCACGACTC***gCT*TgtGC*****AGG*******TTCCCTA--AAC*T***C*GTC*T*AG****A**G*****-*cACCAG-CGGG*A****TG***ACT**--***-*G*********G**GATC*****GCCCG*GGCCAtTC*GG*-*****TG**TC*A*AG**A*****GTAAT*****T**-TTCCAAT*T*GAT*****TTCCAA*****G*********ACG**C****-ACctT**T-A*GC**C***TACTCG*T*GCAT*AGGAGGG*AAAGAG*CtA*TTATGG*C*****AATC**G**AGAT-**GTA*GCAA**CAG**CAC******cc--*G*A-GA*G*AT**TCG**AAAACt-C-TCG***ggAgC*TTGG***TACCA********G**TAACAAGC*CACTT***TT***CA*G**A*****A*-*****-*A***T*c**c*G******CgA***T*ACTgTG*TCTT***CG*AG**TA**TG**GA*CT*CT*AG****AG**T**T**A*TT***************TAC**TG*AG**C*-GCGT-gAA*****TT**GGAG**G*G****A*CATGACA****GCT*TTGGC*CAAG**AC*CGTG*A*A**G-TTA*CAATcCC*GTG"),
    // Record::with_attrs("N25",    Some(""), b"********GAT*CTGGA*TTGT*TG****CAGT*CC***CC*T*CG**G***GC**********ACC***G****GTA***TCGTG**CATG**CT*T**AAAG*CA*G***G*ATTTA*CTT*A*G***-A******TCAA*GT***T*GTGACG*GC*TAGTTGGT*********T****GA*C*TTGCTG******AGAG*AGGG****C*AGAgcgTA*CAGCTtcTtattT**T*A*TCA**GA*T**c*G*ctTGA*AgAAGCGAGa**A*****A***G*C-A*GGA*T*C***TG*T*AGCT***TG*****T****-G--*AaACG*A*GCATAT*******-TG***G****AGTC*ATTC**AGGA*GCtTA-T**CACAT***A*T***C*****C****GTAACGTAC**T****C*A**A****AA**A**C**CGACCGA*CTAC****A*A***GGGT*G*-T*AAG*A*AAGGG**GGA-TG**T*A*TATCGG**T*ATGT*GGT*TGCCCTCG*****CAA****CATTcT*TG****GAA**ACGT****AT*****GtCA**TT*ATT-G***-*GCa*ACGGAGAG**CAgaC*a***cTAG*GGACG***GTG**AGATAC***TCTAC*******A*******T-**GTC***ACGA****TACA***T*****A*CT*CGT****C**T**C-*TGGGTG*G*GCA*****ATATCA-AG*A**c*ATC*CCT*GAACAT***TC*TC*CGACcGTTA*C***GGC*****T*CCA*C***GATGTT*C****TAA***GG*T**GTGTCACACCC******ACC*GCTGT*GG*****CGA*GA***GGGT**G*CA*A*T***CGGT**GGc*TG*G*G**A-*---CT**CCT*T****CA**T*AGGG*TGGT*Tgt--**CTTCTA*CAT*G*AGG****CCTA*****T*G**G*GATAG***GaT****C*G**G*----CCCGG*CtT*GG**TAC-T******T*CT**T**GC**T*C*T**A***CCT*TAA**CAC**A**CTA*G*T*GAC*C**---GGT*CGTT****G******A*TT*A***GACGT*T*TGGGC--*--CA*CGGA*C*A****A********gGCGCGCTC-CT**GG**ctgAGTG****C***A****TTTG*CGAAT*acatcTTAGCGAGTG***gGT*TgtGC*****AGC*******AACCCTG--CGC*C***C*GGC*T*AA****A**C*****-*aCATAG-CGGC*A****AC***TAT**--***-*A*********G**GATC*****GCAGT*GGCACtTG*CG*-*****GT**AC*T*GG**A*****TTAAA*****T**-ACCCAAT*A*GTG*****TTACGA*****G*********GCA**G****-CC*tA**TAA*GC**G***GCTAAG*T*GCAT*AGAGGGG*AATGAG*C*T*TTATGA*C*****TCGA**A**AGCTG**GGA*GCTA**ATG**CGC******cc--*C*A-AG*C*AT**GAG**ACGACt-A-CAA*****TcT*AGCA***CTGTG********T**GCACTTGT*AAATT***TT***CC*G**T*****A*-*****-*G***G*c**c*T******C*A***A*CGAtTG*TCGT***CG*AG**GA**TG**GA*CG*GT*AG****AT**T**T**A*GT***************CGT**TG*TA**C*-ATGTTcCA*****TA**TACG**G*G****G*CGCTAGA****CCG*CGGGC*CAAC**GC*CGTC*A*A**G-TAT*TCCTcCA*GTA"),
    // Record::with_attrs("N26",    Some(""), b"********TTC*CTGTA*TAGT*TA****CATT*TG***CC*A*CA**G***GA**********TCG***A****GCA***CCGCG**CATC**CC*G**AGAC*CT*T***G*CTTCG*AGG*G*A***TT******TCAA*GT***T*CGGGAA*GT*TAGCTGCT*********A****GT*C*GTACAA******AGTG*ACGA****T*TTGgggTA*CACTCtaT****T**A*G*GAA**GA*T****C*ctGCT*GcCAGCTGC***G*****A***G*CCA*ATA*T*G***TA*T*AACT***GT*****T****TCAC*AgGGA*T*GCTTTT*******GTC***G****ATAT*TATA**ATTG*CT*TATT**AGGAA***T*A***G*****G****AAAGTTTCC**C****C*T**A****AT**A**T**CTTCGGA*CATC****A*T***CTCT*G*-T*ACC*G*ACGGT**GCCAGT**A*A*AGTCAA**A*TGGG*GGT*TCACAGTT*****GAC****CATTaT*AA****GTA**ACGG****TT*****C*AT**AT*GAG-G***-*GG**CCGCAATT**CA**C******TAG*TAATC***GGG**ATCGGC***ACTAC*******G*******GC**GGT***GCGA****AAAA***A*****A*GT*CCG****G**G**GG*CCCGTG*G*GCA*****ATATTACAA*A****GAC*CGT*GCAAGT***TC*CC*TGTCaTACA*C***GCG*****G*CCA*G***GTTCAT*C****CAG***GT*A**GTGACGAACCC******ACC*AAGGT*AG*****GGT*TA***GAAT**G*GG*C*A***CATT**GG**CA*G*G**T-*---CG**CCC*A****GG**C*CTTG*TAAG*G**--**ATAATA*TGC*C*AGG****CCTA*****T*G**T*AATAG***G*T****A*T**G*GATTCACTA*G*T*GG**TAC-T******G*CA**G**AC**T*T*T**G***GCC*GGA**TCC**G**CCC*G*T*CCC*T**---CCT*AGTG****A******A*GG*T***GACGT*T*TGCGATG*--TA*TGCC*G*G****G*********ATAAGCCT-CT**AA**taaAGAT****A***T****TTCA*TAAAT******TAAACAAGGG***aGT*G**AT*****AGT*******AAGCCCA--CCC*C***C*ACC*T*CA****A**C*****G*tCGCAGCGCGA*T****AC***TAG**GG***A*C*********G**CATA*****ACCCT*AACGC*CA*AG*G*****AC**CT*T*TC**G*****AGGTA*****A**GCCGCAAT*A*CCG*****TAAGTG*****G*********GCA**G****-TT*gT**TAG*TT**T***GCTGAG*G*GCGA*ACGATAA*CGTTGC*T*C*ACGGGC*G*****TATA**A**TCCTC**GAT*ACTA**CTT**CCC******tt--*C*AGTT*T*GG**ACG**ACAACc-G-AGC*****G*T*CGCA***CTGGG********A**TCTGTTGT*GAACT***TT***GC*C**T*****A*A*****G*C***C*tc*t*C******C*G***A*CTA*AC*TCGG***GG*AG**CA**AA**GT*CT*AG*GG****TT**T**C**C*CC***************GGC**CG*AC**G*-ATCTAtGC*****CA**AACC**G*T****G*AATACAA****ATC*CTAGA*CGTC**AA*ACTA*A*C**A-TAG*ACCT*CC*GTC"),
    // Record::with_attrs("ROOT",   Some(""), b"********CTC*CTTGA*TCGG*TT****CATT*GA***CC*C*TC**T***TA**********TCG***A****TTA***AGGTG**CATG**CG*G**AATA*TT*T***T*CGCCG*ATG*G*C***GT******GTAA*AT***C*CTGCGC*TA*ACCCATCT*********T****GC*C*GGTTCC******AATG*CCGT****T*CTA***AA*GGCTC**T****G**A*G*GCT**GA*A****A***GCA*G*TAGTTCC***G*****A***G*AAC*AGA*T*C***CA*G*AACT***GT*****T****TATC*A*GCC*A*GCCTTG*******GCC***G****ATAT*TTTA**TCTC*CA*TTTG**AAGAT***T*T***A*****T****AGAAAATCC**G****C*C**A****GT**A**T**CTTCGGT*CTGC****A*C***CTCA*C*TT*ACG*G*TGAGA**GACAGC**T*A*CGTCGG**C*TGGA*TCC*TCTTTGGC*****AAC****CATT*T*AT****GAA**CACG****TG*****G*AT**GG*GTGAA***T*GG**TTAAAACT**CA**C******TAG*GAGTC***GGT**GGCAAC***AGTAA*******G*******GC**CTT***GCGT****AGAA***A*****T*GG*CCG****T**G**AA*ACCATT*A*GCG*****ATATTTCCG*G****TAT*CTT*TCAAGC***TC*CC*TGAC*ACCA*C***TCC*****G*CCA*A***GACCGC*C****TAG***TT*C**TGTAGGGAGCC******GCT*AAGGG*AT*****TTG*AC***GAAA**G*GG*A*A***GTTA**GA**TT*C*C**GG*TTGTG**CCA*C****GG**A*CATG*GAGC*C**TG**ACAATC*CAA*T*GGG****CTTA*****T*T**T*AATAT***G*T****A*T**G*GATGCCTAA*T*T*CC**TGCGT******G*CA**G**TC**T*G*C**A***ACC*GCA**CCC**G**ACT*C*A*CTA*T**CCGACT*AGCT****G******C*GG*T***GATGA*C*AGCCCAG*ACTT*TCCC*G*C****C*********AGGTGCGTAAT**AC*****AACT****A***C****TACA*CTCAT******TATACAAATT****AT*G**AT*****GGT*******GCGCCCATACGC*G***G*AGA*C*CA****T**T*****A**AACTGGAGAA*T****TC***TGG**GA***A*T*********T**CAAA*****ACGCA*ATGCG*CA*CG*C*****TC**GT*T*TT**G*****CGTTA*****A**TGCATAAT*A*TCT*****TTAGCT*****G*********GCA**A****CTT**A**TCT*TT**C***GCTCAG*C*CAGA*AGGACTT*CGATGG*T*A*TCGGGT*G*****AATG**A**TTCAC**GTG*ACGA**CGT**CCA********TT*C*CATC*A*TC**CCA**GAAGC*TCCTAC*****G*T*GCCA***CGCTT********A**TCTGGTGT*GCTGG***TT***GC*C**A*****T*A*****T*C***C******C******C*G***A*CTA*TC*TCTC***AG*CG**CA**AC**GT*GG*AG*GA****TA**T**A**C*CC***************GGC**CG*AC**G*CCTGTC*CC*****CC**AACT**C*A****G*TGTTGTT****TCC*ATAGT*CGTG**AG*CCCA*A*C**AATGG*CCAG*CG*GTC"),

    let len = records.len();
    let strip_to = 19;
    for i in 0..len {
        records[i] = Record::with_attrs(
            records[i].id(),
            records[i].desc(),
            &records[i].seq()[..strip_to],
        );
    }
    let seqs = Sequences::new(records.clone());

    let mut msa = AncestralAlignmentBuilder::new(&tree, seqs.clone())
        .build()
        .unwrap();
    let mut phylo = PhyloInfoAncestors {
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
    let mut model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));
    let mut tkf_cost = TKF92Cost {
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

    let original = get_tkf_prob_for_records(records.clone(), &tree);
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

    reassign.print_backtracking_assignment(&v2_idx);

    let v2_mapping_before_update = reassign.cost.phylo.msa.get_node_map()[&v2_idx].clone();
    let v1_idx = reassign.cost.phylo.tree.node(&v2_idx).parent.unwrap();
    let v1_mapping_before_update = reassign.cost.phylo.msa.get_node_map()[&v1_idx].clone();

    let new_mapping = reassign.get_mapping_from_backtracking(&v2_idx);
    reassign.cost.phylo.msa.update_nodes(new_mapping);
    reassign.cost.model_info.borrow_mut().valid = false;
    println!("cost of backtracking = {}", reassign.cost.logl());
    println!("original cost = {}", original);

    let v2_mapping_after_update = reassign.cost.phylo.msa.get_node_map()[&v2_idx].clone();
    let v1_mapping_after_update = reassign.cost.phylo.msa.get_node_map()[&v1_idx].clone();

    // assert_eq!(v2_mapping_before_update, v2_mapping_after_update);
    // assert_eq!(v1_mapping_before_update, v1_mapping_after_update);

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
    println!(
        "last col should be (only if backtracking and original are the same) {}",
        original - logl
    );
    find_brute_force_max(records, &tree, &v2_idx);
}

#[test]
fn mytest_manual_msa_with_many_factor_ns() {
    // let newick_string = "(((3:0.7139,10:1.0807)N15:0.6095,((11:1.3571,(6:0.7200,(1:0.8712,12:0.7348)N16:0.7334)N17:1.8990)N18:0.5412,(15:0.5554,(7:0.6160,8:1.0826)N19:2.3577)N20:1.3745)N21:0.6109)N22:1.3500,(4:0.5974,(13:0.6112,(9:1.8808,(2:1.2448,14:0.9331)N23:1.5854)N24:0.6282)N25:1.0651)N26:0.7182)ROOT;";
    let newick_string = "(((3:0.7139,10:1.0807)N15:0.6095,((11:1.3571,(6:0.7200,(1:0.8712,12:0.7348)N16:0.7334)N17:1.8990)N18:0.5412,(15:0.5554,(7:0.6160,8:1.0826)N19:2.3577)N20:0.3745)N21:0.6109)N22:1.3500,(4:0.5974,(13:0.6112,(9:1.8808,(2:1.2448,14:0.9331)N23:1.5854)N24:0.6282)N25:1.0651)N26:0.7182)ROOT;";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    // once its better to take the factor n and once its better to avoid it
    let records = vec![
        Record::with_attrs("3", Some("descn  "), b"Aa-A-A---"),
        Record::with_attrs("10", Some("descn "), b"A--A-A---"),
        Record::with_attrs("11", Some("descn "), b"A--A-A---"),
        Record::with_attrs("6", Some("descn  "), b"A--A-A---"),
        Record::with_attrs("1", Some("descn  "), b"A--A-A---"),
        Record::with_attrs("12", Some("descn "), b"A--A-AA--"),
        Record::with_attrs("4", Some("descn  "), b"A--A-A---"),
        Record::with_attrs("13", Some("descn "), b"A--A-A-A-"),
        Record::with_attrs("9", Some("descn  "), b"A--A-A-A-"),
        Record::with_attrs("2", Some("descn  "), b"A--A-A---"),
        Record::with_attrs("14", Some("descn "), b"A--A-A---"),
        Record::with_attrs("N15", Some("descn"), b"A--A-A---"),
        Record::with_attrs("N16", Some("descn"), b"A--A-AA--"),
        Record::with_attrs("N17", Some("descn"), b"A--A-AA--"),
        Record::with_attrs("N18", Some("descn"), b"A--A----A"),
        Record::with_attrs("7", Some("descn  "), b"--A-A---A"),
        Record::with_attrs("8", Some("descn  "), b"--A-A---A"),
        Record::with_attrs("15", Some("descn "), b"--A-A---A"),
        Record::with_attrs("N19", Some("descn"), b"--A-A---A"),
        Record::with_attrs("N20", Some("descn"), b"A-A-A---A"),
        Record::with_attrs("N21", Some("descn"), b"A--A-A--A"),
        Record::with_attrs("N22", Some("descn"), b"A--A-A---"),
        Record::with_attrs("N23", Some("descn"), b"A--A-A---"),
        Record::with_attrs("N24", Some("descn"), b"A--A-A-A-"),
        Record::with_attrs("N25", Some("descn"), b"A--A-A-A-"),
        Record::with_attrs("N26", Some("descn"), b"A--A-A---"),
        Record::with_attrs("ROOT", Some("desc"), b"A--A-A---"),
    ];

    let newick_string = "(L01:0.0402583,((((L02:0.592808,(L03:0.303166,L04:0.415641)N33:0.269621)N34:0.538741,((((L05:0.676477,L06:0.676477)N35:0.499026,L07:1.1755)N36:0.0636639,L08:0.150534)N37:0.353044,(L09:0.773536,(((L10:0.227436,L11:0.238621)N38:0.727303,L12:0.00126538)N39:0.0321444,((L13:0.0185434,L14:0.0185434)N40:0.196494,L15:0.215038)N41:0.78303)N42:0.163427)N43:0.430716)N44:0.698717)N45:0.205707,(L16:1.51253,L17:0.359572)N46:0.640356)N47:0.0811874,((((L18:0.304886,(L19:0.443065,((L20:0.0645092,L21:0.0645092)N48:0.388477,(L22:0.343509,L23:0.343509)N49:0.109477)N50:0.0871658)N51:0.348784)N52:1.11909,(((L24:0.160409,L25:0.160409)N53:0.406448,(L26:0.529192,L27:0.529192)N54:0.0376659)N55:0.726152,(L28:1.2921,(L29:0.0564165,L30:0.0564165)N56:1.23568)N57:0.000912024)N58:0.715021)N59:0.460082,L31:1.33426)N60:0.0182408,L32:0.0982867)N61:0.0914687)N62:0.341837)ROOT;";
    let tree = from_newick(newick_string).unwrap().pop().unwrap();
    // [randomseed]	1
    let mut records = vec![
	Record::with_attrs("L01", Some(""), b"-TTGTGCCCATC-------AA--T-GAGA-TT-ACGACC----ATG-ATATA---GGT-ACG-ATC-T-GG-C-C-ATA----G---CC--AGAA-TG----A--AA----GCC-G-CCGG--CG--GTA---C-A--TTGGTCC--AGC-CT-G--GGAC-TTA-T-C"),
	Record::with_attrs("L02", Some(""), b"-CTGTTCAAGTC-------AC--A-AGGAtAT-ATGC-C---cGCC-CCCTCg--AGG-CTG-TCC-G-AA-T-T-TCA----T---AC--CGAA-TG----C--CG----AAC-T-TATG--CG--TCC---C-G--TA-ACTA--ATG-CC-G--AAAG-GCG-G-A"),
	Record::with_attrs("L03", Some(""), b"-TATTTCAAACC-------TA--G-GTGAaGT-GTGC-C---aCC--GAAACc--TGG-TCG-CCT-G----T-T-AAAtgctGgggCA--TGCT-TC----C--CA----AAA-G-GGCG--CA--CGC---A-G--CATACAT--TTG-CT-G--CAGG-GCA-G-C"),
	Record::with_attrs("L04", Some(""), b"-CCTTGGCTACC-------TT--C-AGGCaCT-TTGC-A---cCCC-G---Tc--AGG-ACT-GCC-G-CC-T-G-CAA----T---TA--AGCAcTA----C--CA----AAA-G-AAGG--CA--GTC---C-G--CACACAC--ATG-TT-G--CGAG-GTG-C-C"),
	Record::with_attrs("L05", Some(""), b"-CAC-CACACTG-------CTa-A-ACC-----CCCTCC----CGA-G-CACa--TGTaACT-GAT-G-AG-C-C-ACC----A---CG--TAAA-AC----T--AC----TTA-T-CCTG--GC--G-A---A----TTGTGGG--GAT-TG-G--GCCT-GTA-G-T"),
	Record::with_attrs("L06", Some(""), b"-CGC-ATTCCTG-------CCt-T-AAC-----ATCTCT----AGA-T-AAGc--TGTaTTG-TAT-C-TA-G-C-CTA----G---CA--AAAA-GG----C--AC----CAG-T-TGGT--AT--A-A---G-Ga-TAGCTAG--TCC-CA-G---TGC-AAA-AtA"),
	Record::with_attrs("L07", Some(""), b"--AG-CTTATAT-------AA--T-GTG-----ATAATT----ATC-ACGTCt--TTCgGAC-TAT-A-TT-T-C-AAA----G---CA--CCGC-GG----Ag-GG----CTA-T-ACTT--GGtcC-ActcA-At-CGCCTGT--CCC-TT-G--ATCC-GGT-G-T"),
	Record::with_attrs("L08", Some(""), b"-TCC-ACACAAA-------CA--A-GTG-----AGACGT----ATG-CTGAGc--TTGaAAA-CAT-T-GT-C-T-CTA----C---CG--CGAA-CT----T--AT----TAC-A-TCCA--CT--G-A---C-Gc-TCCCTTG--GGA-TT-G--CTAC-GTC-G-C"),
	Record::with_attrs("L09", Some(""), b"-TG--AGGGCCC------aGG--T-ATAG-CG-AGGCTA---tAGA-CGTTGa--TTCtTGA-G-T-C-CG-A-C-TCC----G---TG--G-AA-TC----C--GC----TGC-G-CGCG--TG--AGT---T-G--GATTAGG--ACA-TA-C--TTAA-ACA-C-A"),
	Record::with_attrs("L10", Some(""), b"-AG--ACGCCACaccactcGG--A-AATC-GA-AAGCTG---cCGA-CCTTGc--GCTaT-A-AAT-A-CT-G-T-GCG----G---TC--CCTC-CT----A---G----TGG-A-TAGG--CA--ACT---C-G--TCTTGTA--CCC-GT-A--CTGA-GCA-C-C"),
	Record::with_attrs("L11", Some(""), b"-CG--CCTCGAC-cctcttGG--A-AGTT-GG-AAGTAG---cCGT-TCTTGt--TCAaC-C-AAT-A-CA-T-T-CAG----G---TC--CGTA-AA----A---G----TGC-A-TAAG--CA--ACT---C-G--ACTTAAA--CGC-AC-A--AGGG-G-T-A-T"),
	Record::with_attrs("L12", Some(""), b"-TG--ACGGGGT------aGG--C-ATGT-AG-GAGCGT---tTTA-GCAGGa--CTGgGAA-ACT-A-CA-A-T-CCT----G---TG--CGAA-CA----T---G----CCC-G-TAGG--TA--ACT---C-G--AACTATG--ACC-CC-G--GTGC-GTA-G-A"),
	Record::with_attrs("L13", Some(""), b"-TC--GCGAGAT------gG---T-ATCC-GA-AGCTCA---tTCG-AGTGGcctT-TaTAT---C-G-CA-C-AcACG----A---GC--ATCA-TC----G---G----CTC-AtTAGG--GA--AGT---G-C--ATGTTTG--TCC-CC-C--CTGC-GTG-G-C"),
	Record::with_attrs("L14", Some(""), b"-TG--GCGAGAT------gG---T-AGCC-GA-AGCTCA---tTCG-AGTGGcctT-TaTAT---C-G-CA-C-AcACG----A---GC--ATCA-TC----G---G----CTC-AtTAGG--GA--AGT---G-C--ATGTTTG--TCC-CC-C--CTGC-GTG-G-C"),
	Record::with_attrs("L15", Some(""), b"-GG--GTGTGAT------gGA--G-AGGC-TA-AGCTCC---tTAG-GGTGGcctT-TtTAT---C-G-GG-C-GaACG----T---TG--ATCA-TG----C---G----GCG-AgGAGG--GA--AGT---G-G--AGGACTG--TGC-CC-C--CTGG-TTC-G-C"),
	Record::with_attrs("L16", Some(""), b"-GAGTGCAACTC-------GA--A-ACTA-CT--CCATG-cccATC-GACAC---ACA-CGTgGCA-T-AA------CA----G---TAtgTTAA-CG----C--TC-----GC-G-ACTT--TT--TGC---T-C--CTACTCG--TTCaCC-AagGCTA-TCT-G-G"),
	Record::with_attrs("L17", Some(""), b"-CTTAGCCGCCA-------AC--C-C--G-CT--TTGAC--ttGCG-GATTC---GCA-CGTgAGA-G-GG------GC----G---TG--AGGG-AG----C--CA----AGC-T-GCGT--AT--GAA---A-A--CCTAAGT--AGC-AT-G--GAAG-CTC-G-A"),
	Record::with_attrs("L18", Some(""), b"-CACATCCATCG-------GT--T-CGAA-AT-GGGGTTt--tCTG-TAGCG---CCG-AAT-AGG-G-GC-C-G-AG---------TT--AAAG-CA----G--TA---tAAT-C-GAAC--CA--TAG---G-G--GTTCTGC--CAC-AAaG--TCTCcAAT-C-A"),
	Record::with_attrs("L19", Some(""), b"-TCTGTGAGAGA-------AA-aC-CGAA-GG-AGGAATa--gGTG-AAGTG---ACG-ACT-AGT-G-TG-C-T-AA---------TC--GCAG-CG----T--GA---tAGT-C-GAAGt-CG--ATT---A-G--GTGCCTT--CGC-AAaT--TCAC-ATT-C-A"),
	Record::with_attrs("L20", Some(""), b"-CATGTGTTACC-------AT-cC-CGAG-TC-ACTACGa--aGTC-CAGTG---GCG-CCT-TTGtT-GA-G-A-CC---------AT--TTCG-CG----C--CT---tAAA-T-GTAA--AG--AGC---A-G--TCGCTTC--CGC-GGaT--TCTC-CAG-C-A"),
	Record::with_attrs("L21", Some(""), b"-CATGTGAAAGC-------AT-cC-CGAG-TG-ACTACGa--aGTC-CAGTG---GCG-ACT-TTGtT-GC---G-GC---------AT--CTCG-CG----C--AT---tAAA-T-GTAT--AG--AGC---A-G--TCTCTTC--CGC-GGaT--TCTC-CAG-C-A"),
	Record::with_attrs("L22", Some(""), b"tCGCGCGAAAGA-------AA-aA-CGTT-TC-AGAG-Ga--aGTA-AGTTG---TAG-ACT-TGG-G-AC-A-T-CC---------AC--AGGG-TC----A--CT---tTAC-C-GAAG--CG--TAC---A-C--CCCCAGC--CGG-TGgT--ACAT-ATG-C-A"),
	Record::with_attrs("L23", Some(""), b"-ATTACGGACGA-------AC-aG-CGCA-AG-AGGGCCc--aGAG-TATTG---CAG-ACA-TGG-G-GC-C-G-AC---------AT--GAGG-CC----T-aCAacggTAA-C-TAGA--AT--GAC---C-G--GCTAT-C--CTG-TAgT--TCGT-ATT-C-A"),
	Record::with_attrs("L24", Some(""), b"-TACGAATGGAA-------AA--TtTCTA-GC-GCCTCT---aTAG-GACTC---GTC-CCG-ACG-A-TT-A-C--G---------GT--CGTT-AG----G--TC-------aA-GAGT-aGA--CTT-----G--TG-CGTTctCGC-A--T--TGTC-GAA-G-A"),
	Record::with_attrs("L25", Some(""), b"-AACGAATGGGA-------AA--TtTGTC-GC-CGCTCG---cTAG-GACTC---GTC---G-ACC-A-GT-A-T--T---------AT--CGCT-CG----C--AC-------aA-AAGT-aGC--CTG-----G--TG-CGTCttCGC-A--G--GTTC-TCA-G-A"),
	Record::with_attrs("L26", Some(""), b"-CAGTGGG---A-------AG--GcTCGA-GA-CC--AT---cCAGcGTGT-----GG-ATG-TAG-CtGT-A-T--A---------AG--AGTA-CG----G--CC-------gT------------GA-----C--AA-GACA-tGCT-A--C--GATA-TTA-C-C"),
	Record::with_attrs("L27", Some(""), b"-CGCAGAAGACA-------CC--TtTCTT-GC-TGTATA---cTAC-CCGTC---GTG-CTG-TCG-T-GTcA-A--G---------AT--ACAA-CG----G--CT-------gA-GTAT--GC--GGC-----G--GT-CG-CctGGC-------GGTC-CTT-C-G"),
	Record::with_attrs("L28", Some(""), b"-CACGTGCCATC-------CG--AgTCTG-C--GCGTCT---cAGC-TCAAT---ATT-TGG-CTG-A-A--T-G--T---------CG--AGTG-CAcgg-A--TA----A-----CTCC--TA--GAC---C-T-tATC-TCCttTAT-GGcC--TTAC-ACG-G-G"),
	Record::with_attrs("L29", Some(""), b"-CTTGTCGAAGC-------TT--TgCACG-TT-AGAG-----cG-G-TGGCA---AAC-G-A-TGT-A-GA-CcC--A---------CG--AGTC-CC----G--CA----CATaT-CTGG--TG--ATG---CaA--ACA-GCTaaAG---AtG--TGCT-ATTgC-C"),
	Record::with_attrs("L30", Some(""), b"-CTTGTCGTAGG-------TG--TgCCCG-TT-ATAA-----cG-G-TGGAA---AAC-G-A-TGT-A-AA-CcC--C---------CG--CGTC-CC----G--CA----CATaT-CTAG--CG--ATG---CaT--ACA-TCTaaAA---GtA--AGCT-ATTgC-C"),
	Record::with_attrs("L31", Some(""), b"-CCGC-CCGTTA-------AG--T-TATA-TTgATCCAT-----TA-GACTG---ATG-CGC-AAG-A-CG-G-T-CAC----G---T---CCAG-TC----T--------CCT-G-CTCC--CA--A-----G-T--TTGGGCC--GGT-CTtG--TCTA-TCA-G-G"),
	Record::with_attrs("L32", Some(""), b"-TAATGCCGTAA-------AA--T-ATGA-AT-ATCTTC---aATG-ATATG---CGT-AGG-ACG-T-AA-C-T-ATA----G---TT--CGCT-TA----C--TC----AGC-G-CAGG--CG--TCA---G-A--CACGTCC--AGC-CGcG--GCAC-TAA-G-G"),
	Record::with_attrs("N33", Some(""), b"-CTTTTTATACC-------TG--T-GGGAaAT-GTGC-C---aCCA-GGAAGc--TGG-TCT-TCT-G-TC-T-T-GAA----G---CA--AGCT-TA----C--CA----AAA-G-AAGG--CA--GCC---C-G--CACACAC--ATG-CT-G--CGGC-GCG-G-C"),
	Record::with_attrs("N34", Some(""), b"-CTTCTCATACC-------AC--T-TGGAaAT-TTGC-C---tCCA-CGATCg--TGG-AGT-TCG-G-TA-T-T-TAA----G---TA--CGCT-TA----C--CA----ATA-C-TAGG--CA--TCC---C-G--CAAACCC--ATG-CT-G--CAGC-GCG-G-C"),
	Record::with_attrs("N35", Some(""), b"-CCC-TATCCAG-------CTt-A-ACC-----ACCCAT----CGG-G-CAGc--TGTaACG-CAT-A-AA-C-C-ATA----A---CG--TAAA-TC----T--AC----AAC-T-CATT--AT--G-A---C-Ga-TTGCTTG--GAA-TG-G--GTTT-ATA-G-C"),
	Record::with_attrs("N36", Some(""), b"-TAC-ACTCAAA-------CA--A-GCG-----ATACGT----ATG-CGGAGc--TTGaAAA-CAT-A-GT-C-C-ATA----C---CG--CGAA-CT----T--GC----TAC-A-CCTT--CT--G-A---C-Ga-TTCCTTG--GGA-TT-G--GTAC-GTA-G-C"),
	Record::with_attrs("N37", Some(""), b"-TAC-ACTCAAA-------CA--A-GTG-----ATACGT----ATA-CGGAGc--TTGaAAA-CAT-A-GT-C-C-CTA----C---CG--CGAA-CT----T--GC----TAC-A-TCCA--CT--G-A---C-Ga-TTCCTTG--GGA-TT-G--GTAC-GTA-G-C"),
	Record::with_attrs("N38", Some(""), b"-AG--ACGCCACtcctcttGG--A-ACTT-GG-AAGACG---cCGA-CCTTGc--CCTaT-C-AAT-A-CA-G-T-GCG----G---TC--CCTC-CA----A---G----TGC-A-TAAG--CA--ACT---C-G--ACTTATA--CCC-GC-A--CTGA-GCT-A-A"),
	Record::with_attrs("N39", Some(""), b"-TG--ACGGGGT------aGG--C-ATGT-AG-GAGCGT---tTTA-GCAGGa--CTGgGAA-ACT-A-CA-A-T-CCT----G---TG--CGAA-CA----T---G----CCC-G-TAGG--TA--ACT---C-G--AACTATG--ACC-CC-G--GTGC-GTA-G-A"),
	Record::with_attrs("N40", Some(""), b"-TG--GCGAGAT------gG---T-AGCC-GA-AGCTCA---tTCG-AGTGGcctT-TaTAT---C-G-CA-C-AcACG----A---GC--ATCA-TC----G---G----CTC-AtTAGG--GA--AGT---G-C--ATGTTTG--TCC-CC-C--CTGC-GTG-G-C"),
	Record::with_attrs("N41", Some(""), b"-GG--GCGAGCT------gGC--C-AGGC-GA-AGCTCC---tTCG-AGTGGcctT-TaTAT---T-G-CG-C-GcACG----T---TC--ATCA-TG----T---G----GCG-AtTAGG--GA--AGT---G-G--ATGTTTG--TCC-CC-C--CTGT-TTC-G-C"),
	Record::with_attrs("N42", Some(""), b"-TG--ACGGGGT------aGG--C-ATGT-AG-AAGCGT---tTTA-GCAGGa--CTTgGAA-ACT-A-CA-A-T-CCT----G---TG--CGAA-CA----T---G----CCC-G-TAGG--TA--ACT---C-G--AACCATG--ACC-CC-G--GTGC-GTA-G-A"),
	Record::with_attrs("N43", Some(""), b"-TG--ACGGGGT------aGG--C-ATGT-AG-AAGCGT---tTTA-CCAGGa--CTTgGAA-ACT-A-CA-A-T-CCA----G---TG--CGAA-CA----T--GG----TCC-G-TAGG--TT--AAT---C-G--ATCCATG--ACA-CT-G--GTGC-GTA-G-A"),
	Record::with_attrs("N44", Some(""), b"-TGCCACTCCAT-------AG--C-AAGT-AT-ATACGT---tATA-TTAGGc--CATaGAG-TCT-A-GA-T-T-CTA----G---TG--CGAG-CA----C--GC----TAC-G-TAGG--CT--AAA---C-G--TTCCTTG--GAA-CT-G--GTCC-GTA-G-C"),
	Record::with_attrs("N45", Some(""), b"-TATCTCCGAAT-------AC--T-AAGA-AT-TTGCAC---tATA-GTATGc--CGT-AGG-TCG-A-GA-T-T-TTA----G---TC--CGCT-CA----T--TA----ATC-C-CAGG--CG--TCC---C-G--TACACCC--AGC-CT-G--GGAC-GCA-G-C"),
	Record::with_attrs("N46", Some(""), b"-TTTTTCCGCTC-------AC--T-TACC-CT--TTCAC--ctATG-AAATC---GGA-TGGaAAA-G-GG------GA----G---TC--CGGG-AG----C--CA----AGC-T-GCGT--CT--GAA---T-A--CTGAAGT--AGC-CG-G--CAAG-GTA-G-G"),
	Record::with_attrs("N47", Some(""), b"-TATTGCCGTAA-------AC--T-AAGA-AT-TTGCAC---tATG-ATATG---CGT-AGG-ACG-T-GA-T-T-ATA----G---TC--CGCT-AA----C--TA----AGC-G-CAGG--CG--TCA---C-A--CACATTC--AGC-CT-G--GGAC-GTA-G-C"),
	Record::with_attrs("N48", Some(""), b"-CATGTGAAAGC-------AT-cC-CGAG-TG-ACTACGa--aGTC-CAGTG---GCG-CCT-TTGtT-GC-G-G-CC---------AT--CTCG-CG----C--CT---tAAA-T-GTAT--AG--AGC---A-G--TCGCTTC--CGC-GGaT--TCTC-CAG-C-A"),
	Record::with_attrs("N49", Some(""), b"-CGTGCGAAAGA-------AT-aA-CGCG-AG-ATGGCGa--aGTG-TATTG---CAG-ACT-TGG-G-GC-C-G-CC---------AT--CTGG-CC----T--CA---tTAA-C-GAAG--AG--TAC---A-G--GCGCTGC--CTC-TGgT--TCAC-ATT-C-A"),
	Record::with_attrs("N50", Some(""), b"-CATGTGAAAGA-------AT-aA-CGAG-AG-ATGGCGa--aGTG-TATTG---CAG-ACT-TGG-G-GC-C-G-CC---------AT--CTAG-CG----T--CA---tAAA-C-GAAG--AG--TAC---A-G--GCGCTTC--CAC-TGaT--TCAC-ATT-C-A"),
	Record::with_attrs("N51", Some(""), b"-CATATGAAAGA-------AT-aT-CGAA-AG-AGGGAGa--aGTG-TATTG---CAG-ACT-AGG-G-GC-C-T-CC---------AT--CTAG-CG----T--CA---tAAT-C-GAAG--CG--CAC---A-G--GCGCTTC--CAC-TGaT--TCAC-ATT-C-A"),
	Record::with_attrs("N52", Some(""), b"-CAGATCCAACG-------GT--T-CGAA-AG-AGGGATa--aCTG-TAGCG---CAG-ATT-AGC-G-GC-C-G-AG---------TT--CAAG-CG----T--CA---tAAT-C-GAAG--CG--TAC---A-G--GTGCTGC--CAC-AGaT--TCCC-AAT-C-A"),
	Record::with_attrs("N53", Some(""), b"-AACGAATGGCA-------AA--TtTCTA-GC-CCCTCA---cTAG-GACTC---GTC-CCG-ACG-A-GT-A-T--G---------AT--CGTT-CG----G--CC-------aA-GAGT-aGG--CTG-----G--TG-CGTCctCGC-A--G--GTTC-TCA-G-A"),
	Record::with_attrs("N54", Some(""), b"-CACTGAGGTCA-------CA--TtTCTA-GC-CCCTAA---cTAG-GAGTC---GTG-CAG-TCG-A-GT-A-T--G---------AG--AGTA-CG----G--CC-------gT-TAGA--GG--TAA-----G--AT-CGTActGGT-T--T--GGTC-TTC-C-A"),
	Record::with_attrs("N55", Some(""), b"-CACTGACGTCA-------CA--TtTCTA-GC-CCCTAA---cTAG-GAGTC---GTG-CTG-TCG-A-GT-A-T--G---------AG--AGTA-CG----G--CC-------gT-TAGG--GG--TAA-----G--AT-CGTActGGT-T--T--GGTC-TTC-C-A"),
	Record::with_attrs("N56", Some(""), b"-CTTGTCGTAGC-------TG--TgCCCG-TT-ATAG-----cG-G-TGGCA---AAC-G-A-TGT-A-GA-CcC--A---------CG--CGTC-CC----G--CA----CATaT-CTGG--CG--ATG---CaT--ACA-TCTaaAG---AtG--AGCT-ATTgC-C"),
	Record::with_attrs("N57", Some(""), b"-CACGGCCGTCC-------CA--TgTCTA-AG-CTAAAG---cATA-TTGTG---GTT-GGA-TGC-A-GT-A-T--T---------TT--AGTC-CG----G--TC----GTTtT-CAGG--CG--AAA---A-G--ATTCTCAgcAGA-CGaT--TGCG-GTT-C-C"),
	Record::with_attrs("N58", Some(""), b"-CACAGCCGTCC-------CA--TgTCTA-AG-CTAAAG---cATA-TTGTG---GTT-GGA-TGC-A-GT-A-T--T---------TT--AGTC-CG----G--TC----GTTtT-CAGG--CG--AAA---A-G--ATTCTCAgcAGA-CGaT--TGCG-GTT-C-C"),
	Record::with_attrs("N59", Some(""), b"-TTGTGCCCCAG-------CT--T-CCAA-AG-ATAAAC---aATA-ATGCG---CGT-AGG-CCC-G-TA-C-T-AT---------TT--CGTG-CG----C--TC---tACC-G-CAGG--CG--TTA---A-T--CTCGTCC--AGC-CGgT--TCAC-GAA-C-C"),
	Record::with_attrs("N60", Some(""), b"-TACTGCCGTAA-------AA--T-ATGA-GT-ATCTTC---tATG-ATATG---CGT-AGG-ACG-T-GA-C-T-ATA----G---TT--CGCT-CA----C--TC----AGC-G-CAGG--CG--TCA---C-A--CACGTCC--AGC-CGcT--GCAC-GAA-G-C"),
	Record::with_attrs("N61", Some(""), b"-TACTGCCGTAA-------AA--T-ATGA-GT-ATCTTC---tATG-ATATG---CGT-AGG-ACG-T-GA-C-T-ATA----G---TT--CGCT-TA----C--TC----AGC-G-CAGG--CG--TCA---C-A--CACGTCC--AGC-CGcG--GCAC-GAA-G-C"),
	Record::with_attrs("N62", Some(""), b"-TACTGCCGTAA-------AA--T-AAGA-AT-ATGTTC---tATG-ATATG---CGT-AGG-ACG-T-GA-C-T-ATA----G---TC--CGCT-TA----C--AC----AGC-G-CAGG--CG--TCA---C-A--CACGTTC--AGC-CG-G--GGAC-GTA-G-C"),
	Record::with_attrs("ROOT", Some(""), b"-TAGTGCCCATA-------AA--T-GACA-TT-ACGACC----ATG-ATATA---GGT-AGG-ATC-T-GG-C-C-ATA----G---CC--AGAA-TG----A--AA----GCC-G-CCGG--CG--GTA---C-A--TTGGTCC--AGC-CT-G--GGAC-TGA-T-C"),
];
    // Record::with_attrs("L01",    Some(""), b"*TTGTGCCCATC*******AA**T*GAGA*TT*ACGACC****ATG*ATATA***GGT*ACG*ATC*T*GG*C*C*ATA****G***CC**AGAA*TG****A**AA****GCC*G*CCGG**CG**GTA***C*A**TTGGTCC**AGC*CT*G**GGAC*TTA*T*C"),
    // Record::with_attrs("L02",    Some(""), b"*CTGTTCAAGTC*******AC**A*AGGAtAT*ATGC-C***cGCC*CCCTCg**AGG*CTG*TCC*G*AA*T*T*TCA****T***AC**CGAA*TG****C**CG****AAC*T*TATG**CG**TCC***C*G**TA-ACTA**ATG*CC*G**AAAG*GCG*G*A"),
    // Record::with_attrs("L03",    Some(""), b"*TATTTCAAACC*******TA**G*GTGAaGT*GTGC-C***aCC-*GAAACc**TGG*TCG*CCT*G*--*T*T*AAAtgctGgggCA**TGCT*TC****C**CA****AAA*G*GGCG**CA**CGC***A*G**CATACAT**TTG*CT*G**CAGG*GCA*G*C"),
    // Record::with_attrs("L04",    Some(""), b"*CCTTGGCTACC*******TT**C*AGGCaCT*TTGC-A***cCCC*G---Tc**AGG*ACT*GCC*G*CC*T*G*CAA****T***TA**AGCAcTA****C**CA****AAA*G*AAGG**CA**GTC***C*G**CACACAC**ATG*TT*G**CGAG*GTG*C*C"),
    // Record::with_attrs("L05",    Some(""), b"*CAC-CACACTG*******CTa*A*ACC-*--*CCCTCC****CGA*G-CACa**TGTaACT*GAT*G*AG*C*C*ACC****A***CG**TAAA*AC****T**AC****TTA*T*CCTG**GC**G-A***A*-**TTGTGGG**GAT*TG*G**GCCT*GTA*G*T"),
    // Record::with_attrs("L06",    Some(""), b"*CGC-ATTCCTG*******CCt*T*AAC-*--*ATCTCT****AGA*T-AAGc**TGTaTTG*TAT*C*TA*G*C*CTA****G***CA**AAAA*GG****C**AC****CAG*T*TGGT**AT**A-A***G*Ga*TAGCTAG**TCC*CA*G**-TGC*AAA*AtA"),
    // Record::with_attrs("L07",    Some(""), b"*-AG-CTTATAT*******AA**T*GTG-*--*ATAATT****ATC*ACGTCt**TTCgGAC*TAT*A*TT*T*C*AAA****G***CA**CCGC*GG****Ag*GG****CTA*T*ACTT**GGtcC-ActcA*At*CGCCTGT**CCC*TT*G**ATCC*GGT*G*T"),
    // Record::with_attrs("L08",    Some(""), b"*TCC-ACACAAA*******CA**A*GTG-*--*AGACGT****ATG*CTGAGc**TTGaAAA*CAT*T*GT*C*T*CTA****C***CG**CGAA*CT****T**AT****TAC*A*TCCA**CT**G-A***C*Gc*TCCCTTG**GGA*TT*G**CTAC*GTC*G*C"),
    // Record::with_attrs("L09",    Some(""), b"*TG--AGGGCCC******aGG**T*ATAG*CG*AGGCTA***tAGA*CGTTGa**TTCtTGA*G-T*C*CG*A*C*TCC****G***TG**G-AA*TC****C**GC****TGC*G*CGCG**TG**AGT***T*G**GATTAGG**ACA*TA*C**TTAA*ACA*C*A"),
    // Record::with_attrs("L10",    Some(""), b"*AG--ACGCCACaccactcGG**A*AATC*GA*AAGCTG***cCGA*CCTTGc**GCTaT-A*AAT*A*CT*G*T*GCG****G***TC**CCTC*CT****A**-G****TGG*A*TAGG**CA**ACT***C*G**TCTTGTA**CCC*GT*A**CTGA*GCA*C*C"),
    // Record::with_attrs("L11",    Some(""), b"*CG--CCTCGAC*cctcttGG**A*AGTT*GG*AAGTAG***cCGT*TCTTGt**TCAaC-C*AAT*A*CA*T*T*CAG****G***TC**CGTA*AA****A**-G****TGC*A*TAAG**CA**ACT***C*G**ACTTAAA**CGC*AC*A**AGGG*G-T*A*T"),
    // Record::with_attrs("L12",    Some(""), b"*TG--ACGGGGT******aGG**C*ATGT*AG*GAGCGT***tTTA*GCAGGa**CTGgGAA*ACT*A*CA*A*T*CCT****G***TG**CGAA*CA****T**-G****CCC*G*TAGG**TA**ACT***C*G**AACTATG**ACC*CC*G**GTGC*GTA*G*A"),
    // Record::with_attrs("L13",    Some(""), b"*TC--GCGAGAT******gG-**T*ATCC*GA*AGCTCA***tTCG*AGTGGcctT-TaTAT*--C*G*CA*C*AcACG****A***GC**ATCA*TC****G**-G****CTC*AtTAGG**GA**AGT***G*C**ATGTTTG**TCC*CC*C**CTGC*GTG*G*C"),
    // Record::with_attrs("L14",    Some(""), b"*TG--GCGAGAT******gG-**T*AGCC*GA*AGCTCA***tTCG*AGTGGcctT-TaTAT*--C*G*CA*C*AcACG****A***GC**ATCA*TC****G**-G****CTC*AtTAGG**GA**AGT***G*C**ATGTTTG**TCC*CC*C**CTGC*GTG*G*C"),
    // Record::with_attrs("L15",    Some(""), b"*GG--GTGTGAT******gGA**G*AGGC*TA*AGCTCC***tTAG*GGTGGcctT-TtTAT*--C*G*GG*C*GaACG****T***TG**ATCA*TG****C**-G****GCG*AgGAGG**GA**AGT***G*G**AGGACTG**TGC*CC*C**CTGG*TTC*G*C"),
    // Record::with_attrs("L16",    Some(""), b"*GAGTGCAACTC*******GA**A*ACTA*CT*-CCATG*cccATC*GACAC***ACA*CGTgGCA*T*AA*-*-*-CA****G***TAtgTTAA*CG****C**TC****-GC*G*ACTT**TT**TGC***T*C**CTACTCG**TTCaCC*AagGCTA*TCT*G*G"),
    // Record::with_attrs("L17",    Some(""), b"*CTTAGCCGCCA*******AC**C*C--G*CT*-TTGAC**ttGCG*GATTC***GCA*CGTgAGA*G*GG*-*-*-GC****G***TG**AGGG*AG****C**CA****AGC*T*GCGT**AT**GAA***A*A**CCTAAGT**AGC*AT*G**GAAG*CTC*G*A"),
    // Record::with_attrs("L18",    Some(""), b"*CACATCCATCG*******GT**T*CGAA*AT*GGGGTTt**tCTG*TAGCG***CCG*AAT*AGG*G*GC*C*G*AG-****-***TT**AAAG*CA****G**TA***tAAT*C*GAAC**CA**TAG***G*G**GTTCTGC**CAC*AAaG**TCTCcAAT*C*A"),
    // Record::with_attrs("L19",    Some(""), b"*TCTGTGAGAGA*******AA*aC*CGAA*GG*AGGAATa**gGTG*AAGTG***ACG*ACT*AGT*G*TG*C*T*AA-****-***TC**GCAG*CG****T**GA***tAGT*C*GAAGt*CG**ATT***A*G**GTGCCTT**CGC*AAaT**TCAC*ATT*C*A"),
    // Record::with_attrs("L20",    Some(""), b"*CATGTGTTACC*******AT*cC*CGAG*TC*ACTACGa**aGTC*CAGTG***GCG*CCT*TTGtT*GA*G*A*CC-****-***AT**TTCG*CG****C**CT***tAAA*T*GTAA**AG**AGC***A*G**TCGCTTC**CGC*GGaT**TCTC*CAG*C*A"),
    // Record::with_attrs("L21",    Some(""), b"*CATGTGAAAGC*******AT*cC*CGAG*TG*ACTACGa**aGTC*CAGTG***GCG*ACT*TTGtT*GC*-*G*GC-****-***AT**CTCG*CG****C**AT***tAAA*T*GTAT**AG**AGC***A*G**TCTCTTC**CGC*GGaT**TCTC*CAG*C*A"),
    // Record::with_attrs("L22",    Some(""), b"tCGCGCGAAAGA*******AA*aA*CGTT*TC*AGAG-Ga**aGTA*AGTTG***TAG*ACT*TGG*G*AC*A*T*CC-****-***AC**AGGG*TC****A**CT***tTAC*C*GAAG**CG**TAC***A*C**CCCCAGC**CGG*TGgT**ACAT*ATG*C*A"),
    // Record::with_attrs("L23",    Some(""), b"*ATTACGGACGA*******AC*aG*CGCA*AG*AGGGCCc**aGAG*TATTG***CAG*ACA*TGG*G*GC*C*G*AC-****-***AT**GAGG*CC****T*aCAacggTAA*C*TAGA**AT**GAC***C*G**GCTAT-C**CTG*TAgT**TCGT*ATT*C*A"),
    // Record::with_attrs("L24",    Some(""), b"*TACGAATGGAA*******AA**TtTCTA*GC*GCCTCT***aTAG*GACTC***GTC*CCG*ACG*A*TT*A*C*-G-****-***GT**CGTT*AG****G**TC****---aA*GAGT*aGA**CTT***-*G**TG-CGTTctCGC*A-*T**TGTC*GAA*G*A"),
    // Record::with_attrs("L25",    Some(""), b"*AACGAATGGGA*******AA**TtTGTC*GC*CGCTCG***cTAG*GACTC***GTC*--G*ACC*A*GT*A*T*-T-****-***AT**CGCT*CG****C**AC****---aA*AAGT*aGC**CTG***-*G**TG-CGTCttCGC*A-*G**GTTC*TCA*G*A"),
    // Record::with_attrs("L26",    Some(""), b"*CAGTGGG---A*******AG**GcTCGA*GA*CC--AT***cCAGcGTGT-***-GG*ATG*TAG*CtGT*A*T*-A-****-***AG**AGTA*CG****G**CC****---gT*----**--**-GA***-*C**AA-GACA*tGCT*A-*C**GATA*TTA*C*C"),
    // Record::with_attrs("L27",    Some(""), b"*CGCAGAAGACA*******CC**TtTCTT*GC*TGTATA***cTAC*CCGTC***GTG*CTG*TCG*T*GTcA*A*-G-****-***AT**ACAA*CG****G**CT****---gA*GTAT**GC**GGC***-*G**GT-CG-CctGGC*--*-**GGTC*CTT*C*G"),
    // Record::with_attrs("L28",    Some(""), b"*CACGTGCCATC*******CG**AgTCTG*C-*GCGTCT***cAGC*TCAAT***ATT*TGG*CTG*A*A-*T*G*-T-****-***CG**AGTG*CAcgg*A**TA****A--*-*CTCC**TA**GAC***C*T*tATC-TCCttTAT*GGcC**TTAC*ACG*G*G"),
    // Record::with_attrs("L29",    Some(""), b"*CTTGTCGAAGC*******TT**TgCACG*TT*AGAG--***cG-G*TGGCA***AAC*G-A*TGT*A*GA*CcC*-A-****-***CG**AGTC*CC****G**CA****CATaT*CTGG**TG**ATG***CaA**ACA-GCTaaAG-*-AtG**TGCT*ATTgC*C"),
    // Record::with_attrs("L30",    Some(""), b"*CTTGTCGTAGG*******TG**TgCCCG*TT*ATAA--***cG-G*TGGAA***AAC*G-A*TGT*A*AA*CcC*-C-****-***CG**CGTC*CC****G**CA****CATaT*CTAG**CG**ATG***CaT**ACA-TCTaaAA-*-GtA**AGCT*ATTgC*C"),
    // Record::with_attrs("L31",    Some(""), b"*CCGC-CCGTTA*******AG**T*TATA*TTgATCCAT****-TA*GACTG***ATG*CGC*AAG*A*CG*G*T*CAC****G***T-**CCAG*TC****T**--****CCT*G*CTCC**CA**A--***G*T**TTGGGCC**GGT*CTtG**TCTA*TCA*G*G"),
    // Record::with_attrs("L32",    Some(""), b"*TAATGCCGTAA*******AA**T*ATGA*AT*ATCTTC***aATG*ATATG***CGT*AGG*ACG*T*AA*C*T*ATA****G***TT**CGCT*TA****C**TC****AGC*G*CAGG**CG**TCA***G*A**CACGTCC**AGC*CGcG**GCAC*TAA*G*G"),
    // Record::with_attrs("N33",    Some(""), b"*CTTTTTATACC*******TG**T*GGGAaAT*GTGC-C***aCCA*GGAAGc**TGG*TCT*TCT*G*TC*T*T*GAA****G***CA**AGCT*TA****C**CA****AAA*G*AAGG**CA**GCC***C*G**CACACAC**ATG*CT*G**CGGC*GCG*G*C"),
    // Record::with_attrs("N34",    Some(""), b"*CTTCTCATACC*******AC**T*TGGAaAT*TTGC-C***tCCA*CGATCg**TGG*AGT*TCG*G*TA*T*T*TAA****G***TA**CGCT*TA****C**CA****ATA*C*TAGG**CA**TCC***C*G**CAAACCC**ATG*CT*G**CAGC*GCG*G*C"),
    // Record::with_attrs("N35",    Some(""), b"*CCC-TATCCAG*******CTt*A*ACC-*--*ACCCAT****CGG*G-CAGc**TGTaACG*CAT*A*AA*C*C*ATA****A***CG**TAAA*TC****T**AC****AAC*T*CATT**AT**G-A***C*Ga*TTGCTTG**GAA*TG*G**GTTT*ATA*G*C"),
    // Record::with_attrs("N36",    Some(""), b"*TAC-ACTCAAA*******CA**A*GCG-*--*ATACGT****ATG*CGGAGc**TTGaAAA*CAT*A*GT*C*C*ATA****C***CG**CGAA*CT****T**GC****TAC*A*CCTT**CT**G-A***C*Ga*TTCCTTG**GGA*TT*G**GTAC*GTA*G*C"),
    // Record::with_attrs("N37",    Some(""), b"*TAC-ACTCAAA*******CA**A*GTG-*--*ATACGT****ATA*CGGAGc**TTGaAAA*CAT*A*GT*C*C*CTA****C***CG**CGAA*CT****T**GC****TAC*A*TCCA**CT**G-A***C*Ga*TTCCTTG**GGA*TT*G**GTAC*GTA*G*C"),
    // Record::with_attrs("N38",    Some(""), b"*AG--ACGCCACtcctcttGG**A*ACTT*GG*AAGACG***cCGA*CCTTGc**CCTaT-C*AAT*A*CA*G*T*GCG****G***TC**CCTC*CA****A**-G****TGC*A*TAAG**CA**ACT***C*G**ACTTATA**CCC*GC*A**CTGA*GCT*A*A"),
    // Record::with_attrs("N39",    Some(""), b"*TG--ACGGGGT******aGG**C*ATGT*AG*GAGCGT***tTTA*GCAGGa**CTGgGAA*ACT*A*CA*A*T*CCT****G***TG**CGAA*CA****T**-G****CCC*G*TAGG**TA**ACT***C*G**AACTATG**ACC*CC*G**GTGC*GTA*G*A"),
    // Record::with_attrs("N40",    Some(""), b"*TG--GCGAGAT******gG-**T*AGCC*GA*AGCTCA***tTCG*AGTGGcctT-TaTAT*--C*G*CA*C*AcACG****A***GC**ATCA*TC****G**-G****CTC*AtTAGG**GA**AGT***G*C**ATGTTTG**TCC*CC*C**CTGC*GTG*G*C"),
    // Record::with_attrs("N41",    Some(""), b"*GG--GCGAGCT******gGC**C*AGGC*GA*AGCTCC***tTCG*AGTGGcctT-TaTAT*--T*G*CG*C*GcACG****T***TC**ATCA*TG****T**-G****GCG*AtTAGG**GA**AGT***G*G**ATGTTTG**TCC*CC*C**CTGT*TTC*G*C"),
    // Record::with_attrs("N42",    Some(""), b"*TG--ACGGGGT******aGG**C*ATGT*AG*AAGCGT***tTTA*GCAGGa**CTTgGAA*ACT*A*CA*A*T*CCT****G***TG**CGAA*CA****T**-G****CCC*G*TAGG**TA**ACT***C*G**AACCATG**ACC*CC*G**GTGC*GTA*G*A"),
    // Record::with_attrs("N43",    Some(""), b"*TG--ACGGGGT******aGG**C*ATGT*AG*AAGCGT***tTTA*CCAGGa**CTTgGAA*ACT*A*CA*A*T*CCA****G***TG**CGAA*CA****T**GG****TCC*G*TAGG**TT**AAT***C*G**ATCCATG**ACA*CT*G**GTGC*GTA*G*A"),
    // Record::with_attrs("N44",    Some(""), b"*TGCCACTCCAT*******AG**C*AAGT*AT*ATACGT***tATA*TTAGGc**CATaGAG*TCT*A*GA*T*T*CTA****G***TG**CGAG*CA****C**GC****TAC*G*TAGG**CT**AAA***C*G**TTCCTTG**GAA*CT*G**GTCC*GTA*G*C"),
    // Record::with_attrs("N45",    Some(""), b"*TATCTCCGAAT*******AC**T*AAGA*AT*TTGCAC***tATA*GTATGc**CGT*AGG*TCG*A*GA*T*T*TTA****G***TC**CGCT*CA****T**TA****ATC*C*CAGG**CG**TCC***C*G**TACACCC**AGC*CT*G**GGAC*GCA*G*C"),
    // Record::with_attrs("N46",    Some(""), b"*TTTTTCCGCTC*******AC**T*TACC*CT*-TTCAC**ctATG*AAATC***GGA*TGGaAAA*G*GG*-*-*-GA****G***TC**CGGG*AG****C**CA****AGC*T*GCGT**CT**GAA***T*A**CTGAAGT**AGC*CG*G**CAAG*GTA*G*G"),
    // Record::with_attrs("N47",    Some(""), b"*TATTGCCGTAA*******AC**T*AAGA*AT*TTGCAC***tATG*ATATG***CGT*AGG*ACG*T*GA*T*T*ATA****G***TC**CGCT*AA****C**TA****AGC*G*CAGG**CG**TCA***C*A**CACATTC**AGC*CT*G**GGAC*GTA*G*C"),
    // Record::with_attrs("N48",    Some(""), b"*CATGTGAAAGC*******AT*cC*CGAG*TG*ACTACGa**aGTC*CAGTG***GCG*CCT*TTGtT*GC*G*G*CC-****-***AT**CTCG*CG****C**CT***tAAA*T*GTAT**AG**AGC***A*G**TCGCTTC**CGC*GGaT**TCTC*CAG*C*A"),
    // Record::with_attrs("N49",    Some(""), b"*CGTGCGAAAGA*******AT*aA*CGCG*AG*ATGGCGa**aGTG*TATTG***CAG*ACT*TGG*G*GC*C*G*CC-****-***AT**CTGG*CC****T**CA***tTAA*C*GAAG**AG**TAC***A*G**GCGCTGC**CTC*TGgT**TCAC*ATT*C*A"),
    // Record::with_attrs("N50",    Some(""), b"*CATGTGAAAGA*******AT*aA*CGAG*AG*ATGGCGa**aGTG*TATTG***CAG*ACT*TGG*G*GC*C*G*CC-****-***AT**CTAG*CG****T**CA***tAAA*C*GAAG**AG**TAC***A*G**GCGCTTC**CAC*TGaT**TCAC*ATT*C*A"),
    // Record::with_attrs("N51",    Some(""), b"*CATATGAAAGA*******AT*aT*CGAA*AG*AGGGAGa**aGTG*TATTG***CAG*ACT*AGG*G*GC*C*T*CC-****-***AT**CTAG*CG****T**CA***tAAT*C*GAAG**CG**CAC***A*G**GCGCTTC**CAC*TGaT**TCAC*ATT*C*A"),
    // Record::with_attrs("N52",    Some(""), b"*CAGATCCAACG*******GT**T*CGAA*AG*AGGGATa**aCTG*TAGCG***CAG*ATT*AGC*G*GC*C*G*AG-****-***TT**CAAG*CG****T**CA***tAAT*C*GAAG**CG**TAC***A*G**GTGCTGC**CAC*AGaT**TCCC*AAT*C*A"),
    // Record::with_attrs("N53",    Some(""), b"*AACGAATGGCA*******AA**TtTCTA*GC*CCCTCA***cTAG*GACTC***GTC*CCG*ACG*A*GT*A*T*-G-****-***AT**CGTT*CG****G**CC****---aA*GAGT*aGG**CTG***-*G**TG-CGTCctCGC*A-*G**GTTC*TCA*G*A"),
    // Record::with_attrs("N54",    Some(""), b"*CACTGAGGTCA*******CA**TtTCTA*GC*CCCTAA***cTAG*GAGTC***GTG*CAG*TCG*A*GT*A*T*-G-****-***AG**AGTA*CG****G**CC****---gT*TAGA**GG**TAA***-*G**AT-CGTActGGT*T-*T**GGTC*TTC*C*A"),
    // Record::with_attrs("N55",    Some(""), b"*CACTGACGTCA*******CA**TtTCTA*GC*CCCTAA***cTAG*GAGTC***GTG*CTG*TCG*A*GT*A*T*-G-****-***AG**AGTA*CG****G**CC****---gT*TAGG**GG**TAA***-*G**AT-CGTActGGT*T-*T**GGTC*TTC*C*A"),
    // Record::with_attrs("N56",    Some(""), b"*CTTGTCGTAGC*******TG**TgCCCG*TT*ATAG--***cG-G*TGGCA***AAC*G-A*TGT*A*GA*CcC*-A-****-***CG**CGTC*CC****G**CA****CATaT*CTGG**CG**ATG***CaT**ACA-TCTaaAG-*-AtG**AGCT*ATTgC*C"),
    // Record::with_attrs("N57",    Some(""), b"*CACGGCCGTCC*******CA**TgTCTA*AG*CTAAAG***cATA*TTGTG***GTT*GGA*TGC*A*GT*A*T*-T-****-***TT**AGTC*CG****G**TC****GTTtT*CAGG**CG**AAA***A*G**ATTCTCAgcAGA*CGaT**TGCG*GTT*C*C"),
    // Record::with_attrs("N58",    Some(""), b"*CACAGCCGTCC*******CA**TgTCTA*AG*CTAAAG***cATA*TTGTG***GTT*GGA*TGC*A*GT*A*T*-T-****-***TT**AGTC*CG****G**TC****GTTtT*CAGG**CG**AAA***A*G**ATTCTCAgcAGA*CGaT**TGCG*GTT*C*C"),
    // Record::with_attrs("N59",    Some(""), b"*TTGTGCCCCAG*******CT**T*CCAA*AG*ATAAAC***aATA*ATGCG***CGT*AGG*CCC*G*TA*C*T*AT-****-***TT**CGTG*CG****C**TC***tACC*G*CAGG**CG**TTA***A*T**CTCGTCC**AGC*CGgT**TCAC*GAA*C*C"),
    // Record::with_attrs("N60",    Some(""), b"*TACTGCCGTAA*******AA**T*ATGA*GT*ATCTTC***tATG*ATATG***CGT*AGG*ACG*T*GA*C*T*ATA****G***TT**CGCT*CA****C**TC****AGC*G*CAGG**CG**TCA***C*A**CACGTCC**AGC*CGcT**GCAC*GAA*G*C"),
    // Record::with_attrs("N61",    Some(""), b"*TACTGCCGTAA*******AA**T*ATGA*GT*ATCTTC***tATG*ATATG***CGT*AGG*ACG*T*GA*C*T*ATA****G***TT**CGCT*TA****C**TC****AGC*G*CAGG**CG**TCA***C*A**CACGTCC**AGC*CGcG**GCAC*GAA*G*C"),
    // Record::with_attrs("N62",    Some(""), b"*TACTGCCGTAA*******AA**T*AAGA*AT*ATGTTC***tATG*ATATG***CGT*AGG*ACG*T*GA*C*T*ATA****G***TC**CGCT*TA****C**AC****AGC*G*CAGG**CG**TCA***C*A**CACGTTC**AGC*CG*G**GGAC*GTA*G*C"),
    // Record::with_attrs("ROOT",   Some(""), b"*TAGTGCCCATA*******AA**T*GACA*TT*ACGACC****ATG*ATATA***GGT*AGG*ATC*T*GG*C*C*ATA****G***CC**AGAA*TG****A**AA****GCC*G*CCGG**CG**GTA***C*A**TTGGTCC**AGC*CT*G**GGAC*TGA*T*C"),

    let len = records.len();
    let strip_to = 169;
    let strip_from = 160;
    for i in 0..len {
        records[i] = Record::with_attrs(
            records[i].id(),
            records[i].desc(),
            &records[i].seq()[strip_from..strip_to],
        );
        println!("{}: {}", records[i].id(), records[i].seq()[0] as char,);
    }

    let seqs = Sequences::new(records.clone());

    let mut msa = AncestralAlignmentBuilder::new(&tree, seqs.clone())
        .build()
        .unwrap();
    let mut phylo = PhyloInfoAncestors {
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
    let mut model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &tkf_model));
    let mut tkf_cost = TKF92Cost {
        model: tkf_model,
        phylo: phylo.clone(),
        model_info,
    };

    let v2_idx = tkf_cost
        .phylo
        .tree
        .postorder()
        .iter()
        .find(|x| tkf_cost.phylo.tree.node(x).id == "N56")
        .cloned()
        .unwrap();

    // act

    let original = get_tkf_prob_for_records(records.clone(), &tree);
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

    reassign.print_backtracking_assignment(&v2_idx);

    let v2_mapping_before_update = reassign.cost.phylo.msa.get_node_map()[&v2_idx].clone();
    let v1_idx = reassign.cost.phylo.tree.node(&v2_idx).parent.unwrap();
    let v1_mapping_before_update = reassign.cost.phylo.msa.get_node_map()[&v1_idx].clone();

    let new_mapping = reassign.get_mapping_from_backtracking(&v2_idx);
    reassign.cost.phylo.msa.update_nodes(new_mapping);
    reassign.cost.model_info.borrow_mut().valid = false;
    let backtracking_result = reassign.cost.logl();
    println!("cost of backtracking = {}", backtracking_result);
    println!("original cost = {}", original);

    let v2_mapping_after_update = reassign.cost.phylo.msa.get_node_map()[&v2_idx].clone();
    let v1_mapping_after_update = reassign.cost.phylo.msa.get_node_map()[&v1_idx].clone();

    // assert_eq!(v2_mapping_before_update, v2_mapping_after_update);
    // assert_eq!(v1_mapping_before_update, v1_mapping_after_update);

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
    println!(
        "last col should be (only if backtracking and original are the same) {}",
        original - logl
    );
    let force = find_brute_force_max(records, &tree, &v2_idx);
    assert_eq!(backtracking_result, force);
}

#[test]
fn mytest_prev_max_denied_by_factor_n() {
    // Check for example where one prev max is not taken bc factor n. Where current action is nothing and both pass throughs aren't neg inf
}

#[test]
fn mytest_automatic_indelible() {
    let content = fs::read_to_string("many_trees.txt");
    if content.is_err() {
        return;
    }
    let unwrapped = content.unwrap();
    for line in unwrapped.lines() {
        if line.len() > 1 {
            println!("read line {}", line);
            update_control_file(line);
            let output = Command::new("./indelible")
                .output()
                .expect("Failed to execute command");

            println!("Status: {}", output.status);
            println!("Stdout: {}", String::from_utf8_lossy(&output.stdout));
            println!("Stderr: {}", String::from_utf8_lossy(&output.stderr));
            let newick = get_newick_str_from_file();
            let records = get_records_from_file();
            for record in &records {
                println!("recorfd = {}", record);
            }
            let seqs = Sequences::new(records.clone());
            let tree = from_newick(&newick).unwrap().pop().unwrap();

            let mut msa = AncestralAlignmentBuilder::new(&tree, seqs.clone())
                .build()
                .unwrap();
            let mut phylo = PhyloInfoAncestors {
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

            // TODO: instead i could just run it for every interal node
            // let random_node = loop {
            //     let random = tkf_cost
            //         .phylo
            //         .tree
            //         .postorder()
            //         .iter()
            //         .cloned()
            //         .choose_multiple(&mut thread_rng(), 1)[0];
            //     if tkf_cost.phylo.tree.node(&random).id.starts_with("N") {
            //         break random;
            //     }
            // };

            let model = tkf_model.clone();
            let records = records.clone();

            for v2_idx in tree.postorder() {
                if !tree.node(v2_idx).id.starts_with("N") {
                    continue;
                }
                let model = model.clone();
                let mut model_info = RefCell::new(TKF92ModelInfo::new(&phylo, &model));
                let mut tkf_cost = TKF92Cost {
                    model,
                    phylo: phylo.clone(),
                    model_info,
                };
                let mut reassign = ReassignEdge::<GTR>::new(tkf_cost);
                reassign.fill_dp(&v2_idx);

                let new_mapping = reassign.get_mapping_from_backtracking(&v2_idx);
                reassign.cost.phylo.msa.update_nodes(new_mapping);
                reassign.cost.model_info.borrow_mut().valid = false;
                let my_cost = reassign.cost.logl();

                println!("node is = {}", reassign.cost.phylo.tree.node(&v2_idx).id);

                let force = find_brute_force_max(records.clone(), &tree, &v2_idx);
                if force != 1.0 {
                    assert_eq!(my_cost, force);
                }
            }
        }
    }
}
#[cfg(test)]
fn get_records_from_file() -> Vec<Record> {
    let content = fs::read_to_string("outputname_TRUE.phy").unwrap();
    let lines: Vec<&str> = content.lines().collect();

    let re = Regex::new(r"\s+").unwrap(); // Regex to split whitespace
    let mut records = Vec::new();

    println!("let records = vec![");
    for line in &lines[1..] {
        // Skip the first line
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = re.split(line).collect();
        if parts.len() < 2 {
            continue; // Skip malformed lines
        }

        let name = parts[0];
        let sequence = parts[1].replace("*", "-");

        records.push(Record::with_attrs(name, Some("desc"), sequence.as_bytes()));
    }
    records
}

#[cfg(test)]
fn get_newick_str_from_file() -> String {
    let content = fs::read_to_string("trees.txt").unwrap(); // Read file
    content
        .lines()
        .find(|line| line.contains("ROOT;")) // Find first matching line
        .and_then(|line| line.split_whitespace().last()) // Get last column
        .unwrap_or("") // Default to empty string if not found
        .to_string()
}

#[cfg(test)]
fn update_control_file(tree: &str) -> Result<(), std::io::Error> {
    let mut content = fs::read_to_string("before_tree.txt")?;
    let mut content = String::from(content.strip_suffix("\n").unwrap());
    content.push_str(tree);
    content.push_str("\n\n");
    content.push_str("[EVOLVE] partitionname 1 outputname");
    content.push_str("\n\n");

    fs::write("control.txt", content)?;
    println!("writing succeced");
    Ok(())
}

#[cfg(test)]
fn get_tkf_prob_for_records(records: Vec<Record>, tree: &crate::tree::Tree) -> f64 {
    get_tkf_prob_for_records_strip(records, tree, 1000, false)
}
#[cfg(test)]
fn get_tkf_prob_for_records_strip(
    records: Vec<Record>,
    tree: &crate::tree::Tree,
    only_until: usize,
    exclude_const: bool,
) -> f64 {
    let seqs = Sequences::new(records);

    let msa = AncestralAlignmentBuilder::new(&tree, seqs).build().unwrap();
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
    tkf_cost.logl_strip(only_until, exclude_const)
}

#[cfg(test)]
fn find_brute_force_max(records: Vec<Record>, tree: &crate::tree::Tree, v2_idx: &NodeIdx) -> f64 {
    // should be same value aus dp last
    use itertools::Itertools;

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
    let mut reassign = ReassignEdge::<GTR>::new(tkf_cost);
    let mut possible_edge_assignments: Vec<Vec<[bool; 2]>> =
        vec![vec![]; reassign.cost.model_info.borrow().blocks.len()];
    for block_id in 0..reassign.cost.model_info.borrow().blocks.len() {
        let (t1, t2, t3, t4) = reassign.are_chars_at_leafs(v2_idx, block_id);
        possible_edge_assignments[block_id] =
            ReassignEdge::<JC69>::get_allowed_assignments(t1, t2, t3, t4);

        if possible_edge_assignments[block_id].len() > 1 {
            println!(
                "the len for block_id = {} is {}",
                block_id,
                possible_edge_assignments[block_id].len()
            );
        }
    }

    // let possibilities: Vec<Vec<[bool; 2]>> = possible_edge_assignments
    //     .into_iter()
    //     .multi_cartesian_product()
    //     .collect();
    let mut number_of_possibilities = 1;
    for poss in &possible_edge_assignments {
        number_of_possibilities *= poss.len();
    }
    if number_of_possibilities > 1000000 {
        return 1.0;
    }
    use std::io::{self, Write};

    let mut max: Option<f64> = None;
    let mut arg_max: Option<Vec<[bool; 2]>> = None;
    for (i, possibility) in possible_edge_assignments
        .into_iter()
        .multi_cartesian_product()
        .enumerate()
    {
        // print!("calculating {} of {}\r", i, possibilities.len());
        if i % 100 == 0 {
            print!(
                "calculating {} of {}, which is {:.4}% \r",
                i,
                number_of_possibilities,
                i as f64 / number_of_possibilities as f64
            );
            let _ = io::stdout().flush();
        }

        let new_mapping = reassign.get_mapping_from_vec(v2_idx, &possibility);
        reassign.cost.phylo.msa.update_nodes(new_mapping);
        reassign.cost.model_info.borrow_mut().valid = false;
        let current = reassign.cost.logl();
        if let Some(ref mut m) = max {
            if current > *m {
                *m = current;
                arg_max = Some(possibility);
            }
        } else {
            max = Some(current);
            arg_max = Some(possibility);
        }
    }

    println!("the brute force max = {}", max.unwrap());
    println!("and the argmax is:");
    for ass in &arg_max.unwrap() {
        println!("{:?}", ass);
    }
    max.unwrap()
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

#[test]
fn limit_factor_n() {
    let l = 1.0;
    let m = 2.0;
    let t = 0.0000000001;
    let b = TKF92Cost::<JC69>::b(l, m, t);
    let log_factor_n =
        TKF92Cost::<JC69>::log_n1(l, m, b, t) - TKF92Cost::<JC69>::n0(m, b) - (l * b).ln();
    println!("factor n = {}", log_factor_n.exp());
}
