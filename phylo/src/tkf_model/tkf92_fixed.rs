use std::cell::RefCell;
use std::fmt::Display;

use hashbrown::HashSet;
use log::warn;
use num_enum::{FromPrimitive, IntoPrimitive};

use crate::likelihood::{ParamRange, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE};
use crate::phylo_info::PhyloInfo;
use crate::tkf_model::{validate_lambda_and_mu, validate_r, TKFIndelCost, TKFIndelModelInfo};
use crate::Result;
use crate::{alignment::AncestralAlignment, tkf_model::TKFModel};

#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
pub(crate) enum TKF92Parameters {
    Lambda = 0,
    Mu = 1,
    R = 2,
    #[num_enum(catch_all)]
    Invalid(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TKF92FixedIndelModel {
    pub(super) params: Vec<f64>,
    /// precomputed r.ln()
    pub(super) log_r: f64,
    /// The fixed fragmentation to be used
    pub(super) fragmentation: Vec<usize>,
}

// TODO: this whole thing could be compiled only when testing
impl TKF92FixedIndelModel {
    pub(crate) fn r(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::R)]
    }
}

impl TKFModel for TKF92FixedIndelModel {
    fn lambda(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::Lambda)]
    }

    fn mu(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::Mu)]
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        let param = TKF92Parameters::from_primitive(idx);
        match param {
            TKF92Parameters::R => {
                self.params[usize::from(TKF92Parameters::R)] = value;
                self.log_r = value.ln();
            }
            _ => {
                self.params[idx] = value;
            }
        };
    }

    fn param_range(&self, idx: usize) -> ParamRange {
        let param = TKF92Parameters::from_primitive(idx);
        match param {
            TKF92Parameters::Lambda => (f64::EPSILON, self.mu() - f64::EPSILON),
            TKF92Parameters::Mu => (self.lambda() + f64::EPSILON, f64::MAX),
            TKF92Parameters::R => PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE,
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
        }
    }

    fn insertion_prob_at_root(&self) -> f64 {
        self.lambda() / self.mu()
    }

    // this is not a prob but a factor since it can be > 1
    // TODO i really should rename also the node_event_prob to node_event_factor or something
    fn insertion_prob_at_non_root(&self, beta: f64) -> f64 {
        self.lambda() * beta
    }

    fn block_prob(&self, tree_event_prob: f64, block_len: usize) -> f64 {
        if tree_event_prob == 1.0 {
            0.0
        } else {
            tree_event_prob.ln() + (block_len as f64 - 1.0) * self.log_r + (1.0 - self.r()).ln()
        }
    }

    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        let mut blocks: HashSet<usize> = HashSet::new();
        for map in msa
            .ancestral_maps()
            .values()
            .chain(msa.leaf_maps().values())
        {
            let mut previous_is_char = map[0].is_some();
            for (i, c) in map.iter().skip(1).enumerate() {
                let current_is_char = c.is_some();
                // whenever there is a change from gap to not gap or vice versa, we have a block border
                if previous_is_char ^ current_is_char {
                    blocks.insert(i + 1);
                }
                previous_is_char = current_is_char;
            }
            blocks.insert(map.len());
        }
        let mut blocks: Vec<usize> = blocks.iter().copied().collect();
        blocks.sort();
        merge_fragmentations_with_blocks(&self.fragmentation, &blocks)
    }
}

/// Merges the user defined fragmentation with the observed block borders in the MSA.
/// Assumes both inputs are sorted and within MSA length.
pub(super) fn merge_fragmentations_with_blocks(
    fragmentation: &[usize],
    blocks: &[usize],
) -> Vec<usize> {
    let mut frag_iter = fragmentation.iter().peekable();
    let mut missing = Vec::new();
    for block in blocks.iter() {
        let mut next_block = false;
        while let Some(&frag) = frag_iter.peek() {
            if frag > block {
                println!("Observed right border of block {block} in MSA not in fragmentation, adding it.");
                warn!("Observed right border of block {block} in MSA not in fragmentation, adding it.");
                missing.push(*block);
                next_block = true;
                break;
            } else if frag == block {
                next_block = true;
                break;
            } else {
                frag_iter.next();
            }
        }
        if next_block {
            continue;
        } else {
            println!(
                "Observed right border of block {block} in MSA not in fragmentation, adding it."
            );
            warn!("Observed right border of block {block} in MSA not in fragmentation, adding it.");
            missing.push(*block);
        }
    }
    for frag in fragmentation {
        missing.push(*frag);
    }

    missing.sort();
    missing
}

impl Display for TKF92FixedIndelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with lambda = {}, mu = {}, r = {}, fragmentation = {:?}",
            self.lambda(),
            self.mu(),
            self.r(),
            self.fragmentation,
        )
    }
}

/// Builder for TKF92 indel cost, i.e., without substitution model and a fixed fragmentation
pub struct TKF92FixedIndelCostBuilder<AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    r: f64,
    fragmentation: Vec<usize>,
    phylo: PhyloInfo<AA>,
}

fn validate_fragmentation(fragmentation: &[usize], msa_len: usize) -> Vec<usize> {
    let mut fragmentation = fragmentation.to_vec();
    let original_len = fragmentation.len();
    if original_len == 0 {
        return fragmentation.to_vec();
    }
    fragmentation.sort();
    fragmentation.dedup();
    let deduped_len = fragmentation.len();
    if deduped_len < original_len {
        warn!("Fragmentation had duplicate entries, which were removed.");
    }
    fragmentation.retain(|&x| x > 0 && x <= msa_len);
    let retained_len = fragmentation.len();
    if retained_len < deduped_len {
        warn!("Fragmentation had entries out of bounds (0, seq_len], which were removed.");
    }
    fragmentation.to_vec()
}

impl<AA: AncestralAlignment> TKF92FixedIndelCostBuilder<AA> {
    pub fn new(
        lambda: f64,
        mu: f64,
        r: f64,
        fragmentation: Vec<usize>,
        phylo: PhyloInfo<AA>,
    ) -> Self {
        Self {
            lambda,
            mu,
            r,
            fragmentation,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFIndelCost<TKF92FixedIndelModel, AA>> {
        let (lambda, mu) = validate_lambda_and_mu(self.lambda, self.mu);
        let r = validate_r(self.r);
        let fragmentation = validate_fragmentation(&self.fragmentation, self.phylo.msa.len());
        let model = TKF92FixedIndelModel {
            params: vec![lambda, mu, r],
            log_r: r.ln(),
            fragmentation,
        };
        let info = TKFIndelModelInfo::new(&model, &self.phylo);
        Ok(TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        })
    }
}

#[cfg(test)]
mod private_tests {

    use approx::assert_relative_eq;

    use crate::alignment::{Sequences, MASA};
    use crate::phylo_info::PhyloInfo;
    use crate::tkf_model::{TKF92IndelCostBuilder, DEFAULT_LAMBDA, DEFAULT_MU, DEFAULT_R};
    use crate::{record_wo_desc as record, tree};

    use super::*;

    #[test]
    fn test_validate_fragmentation() {
        let fragmentation = vec![3, 19, 3, 4, 58, 13, 0, 1, 0, 3, 4, 15, 16];
        let msa_len = 15;
        let validated = validate_fragmentation(&fragmentation, msa_len);
        assert_eq!(validated, vec![1, 3, 4, 13, 15]);
    }

    #[test]
    fn test_merge_fragmentations_with_blocks_case1() {
        let fragmentation = vec![3, 5, 7, 10];
        let blocks = vec![5, 10, 12];
        let merged = merge_fragmentations_with_blocks(&fragmentation, &blocks);
        assert_eq!(merged, vec![3, 5, 7, 10, 12]);
    }

    #[test]
    fn test_merge_fragmentations_with_blocks_case2() {
        let fragmentation = vec![3, 7, 10, 12];
        let blocks = vec![5, 10, 12];
        let merged = merge_fragmentations_with_blocks(&fragmentation, &blocks);
        assert_eq!(merged, vec![3, 5, 7, 10, 12]);
    }

    #[test]
    fn test_merge_fragmentations_with_blocks_case3() {
        let fragmentation = vec![];
        let blocks = vec![5, 10, 12];
        let merged = merge_fragmentations_with_blocks(&fragmentation, &blocks);
        assert_eq!(merged, vec![ 5,  10, 12]);
    }

    #[test]
    fn test_integration() {
        let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
        let seqs = Sequences::new(vec![
            record!("A0", b"AAB---D"),
            record!("B1", b"-ARAAAW"),
            record!("I1", b"AAA---A"),
        ]);
        let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
        let phylo_info = PhyloInfo { msa, tree };

        let tkf92_cost =
            TKF92IndelCostBuilder::new(DEFAULT_LAMBDA, DEFAULT_MU, DEFAULT_R, phylo_info.clone())
                .build()
                .unwrap();
        let cost = tkf92_cost.logl();

        let mut sum_over_fragmentations_cost = 0.0;
        let fragmentations = [
            vec![1, 3, 6, 7],
            vec![1, 2, 3, 6, 7],
            vec![1, 3, 4, 6, 7],
            vec![1, 3, 5, 6, 7],
            vec![1, 3, 4, 5, 6, 7],
            vec![1, 2, 3, 4, 6, 7],
            vec![1, 2, 3, 5, 6, 7],
            vec![1, 2, 3, 4, 5, 6, 7],
        ];

        for fragmentation in fragmentations {
            let fragment_cost = TKF92FixedIndelCostBuilder::new(
                DEFAULT_LAMBDA,
                DEFAULT_MU,
                DEFAULT_R,
                fragmentation,
                phylo_info.clone(),
            )
            .build()
            .unwrap();
            sum_over_fragmentations_cost += fragment_cost.logl().exp();
        }
        sum_over_fragmentations_cost = sum_over_fragmentations_cost.ln();
        assert_relative_eq!(cost, sum_over_fragmentations_cost);
    }

    #[test]
    fn test_simulation() {
        let tree = tree!("((A:0.5,B:0.5)I1:0.7,(C:0.6,D:0.6)I2:0.6)R:1.0;");
        let seqs = Sequences::new(vec![
            record!("R", b"--------------AA------AAAA--------------A--------------AAA----------------A-AAAA-----------AA-----------A------A------A----------AAAAAA-A--AA---AAAAA---AAAAAAAA-----------------------AA--------AAAAAA------A--------------AAA-AA-A------A----A--------AAA---AAA----------------------------A-------------A------AA-A--------------AAAAAAA------------------AAAAA---------AA----AAA----"),
            record!("I1", b"--------------TA-----TGCTT---------------------A-------ATC----GAACA--AA---AT--CAGAA-TA-----CA---AA-------------A--ACAA------------GCCTA-A----------AA--C--ATATTG-----------------------AA---------------A----A-----GG------A--C-AA--------C---------------------A------------AA----CAAAA-----A----AA---CC--T---------T------------ACATAAATC-------------AAG----------------AC---------TT"),
            record!("A", b"-----CG--------------C-CTC--G------------------A------G-------TCACC--CAATTAT--GAGACTTA-----CC----A--CCGT-------A--GTAA------------GTTTAAA-----------------GATGAG--------CA--------------------------------TCG------C-CTTACC---CCTGG-----------------------------A--------------GCCG---AC-GAAC-T---AATGACC------------TCA----------CCATAAGTA-ATA---------AC-----------------TG---------TT"),
            record!("B", b"--------------GA------GCCTTT--------------------CACGAT-GAACTGTGAACAACGA----T----GAA-G-AACGA-----AATC--------------CCGT------------CCCCT-A------------TTC--------GCTATTTT---------------AAGACC-----------AG---A-----T-------A--C-----------C----------------------------------------TA--GG----A-----------AGT---------T------------TT--------------------AAGCT--------------TGTCAC-------"),
            record!("I2", b"-------CGGA--CAA-------------A-AG-------A---------------------------------A-GGCA-----------TA-----------C---AG-A------G-AAAAAGG---AGAAC-C---A-----------------------------AAAA--A---A--------A---CCAAAA------C--------------CCC-AA-A---G-AT---AA---GAAAA---A-----A---AAAGCCAC----------------T-A-----------AT-----TTAA--AATACGACG---ATATCGG----AGATCG-A---------AAA--------AA----AAAAA--"),
            record!("C", b"----C--------G-----------------AC--------TGAGAA---------------------------A----------------TA-----------C---AG-C-C-----AACAATGG---AGAAC-----------------------------------GAAT-A-------------AATA-----A---------------------TC-----TGTATA-CTAA--TGTGATCT---ACA-----AAATAGCCAC----------------T--------------T-----T-A---AATCCTCCGA--GTCCCGGG----CTGCG-GA----------A-----ATAAG----AGA----"),
            record!("D", b"GTGT---CGAATA-GCATATG--------ATAGTGCAGTAA---------------------------------A---TA-------------CCG---------AATTTG-T--------------CG-------CGC-ACAG--------------------------CAATT-ACAA-AG---------------AG-----CAATGG---------CT--CA-C------G---TG---CATGG----------C--CCAGCCGC-------------------AA----------TCTAGAT-A---TACACGACC---ATGCCTC----ACGGTATT---------AG-ATAAT----------------"),
        ]);
        let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
        let phylo_info = PhyloInfo { msa, tree };

        let fragmentation = vec![
            4, 5, 6, 7, 11, 12, 13, 14, 16, 21, 22, 23, 26, 28, 29, 30, 31, 33, 40, 41, 47, 48, 54,
            55, 58, 61, 62, 63, 67, 69, 70, 71, 74, 75, 76, 78, 80, 83, 84, 85, 86, 88, 91, 93, 96,
            97, 98, 100, 104, 105, 108, 110, 111, 112, 113, 114, 118, 119, 120, 127, 129, 130, 135,
            136, 137, 138, 139, 140, 141, 143, 144, 147, 149, 151, 152, 154, 160, 163, 164, 166,
            168, 169, 170, 172, 174, 175, 176, 177, 180, 181, 183, 185, 187, 189, 190, 193, 195,
            198, 199, 200, 201, 202, 205, 206, 211, 212, 213, 219, 220, 222, 223, 224, 226, 227,
            228, 231, 232, 233, 234, 235, 238, 239, 240, 243, 248, 249, 251, 252, 254, 256, 257,
            258, 259, 261, 265, 269, 271, 272, 275, 277, 278, 279, 280, 281, 282, 284, 285, 286,
            287, 288, 290, 292, 295, 297, 299, 300, 301, 302, 303, 305, 306, 307, 308, 309, 310,
            312, 321, 322, 324, 331, 332, 334, 335, 336, 341, 342, 343, 344, 346, 347, 349, 352,
            354, 355, 356, 359, 360, 363, 365, 369, 372, 374, 376,
        ];
        let fragment_cost = TKF92FixedIndelCostBuilder::new(
            DEFAULT_LAMBDA,
            DEFAULT_MU,
            DEFAULT_R,
            fragmentation,
            phylo_info,
        )
        .build()
        .unwrap();
        // logl from simulation
        assert_relative_eq!(fragment_cost.logl(), -769.4115065236674, epsilon = 1e-10);
    }
}
