use std::cell::RefCell;
use std::fmt::Display;

use num_enum::FromPrimitive;

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ParamRange, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE};
use crate::phylo_info::PhyloInfo;
use crate::tkf_model::{
    blocks_of_alignment, merge_fragmentation_with_blocks, validate_fragmentation,
    validate_lambda_and_mu, validate_r, TKF92Parameters, TKFIndelCost, TKFIndelModelInfo, TKFModel,
};
use crate::Result;

/// [TKF92IndelModel](`super::TKF92IndelModel`) with additional block borders (and without a substitution model),
/// which means that the provided blocks will be used in addition to the blocks determined from
/// the alignment, see [`super::TKFModel::get_blocks`].
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct TKF92IndelModelAddBlocks {
    params: Vec<f64>,
    /// precomputed r.ln()
    log_r: f64,
    /// precomputed (1 - r)/r
    ln_one_minus_r_over_r: f64,
    /// Blocks to be used in addition to those determined from the alignment
    additional_blocks: Vec<usize>,
}

#[cfg(test)]
impl TKF92IndelModelAddBlocks {
    pub fn r(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::R)]
    }
}

#[cfg(test)]
impl TKFModel for TKF92IndelModelAddBlocks {
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
                self.ln_one_minus_r_over_r = ((1.0 - value) / value).ln();
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

    fn ln_insertion_factor_at_root(&self) -> f64 {
        self.lambda().ln() - self.mu().ln() + self.ln_one_minus_r_over_r
    }

    fn ln_insertion_factor_at_non_root(&self, ln_beta: f64) -> f64 {
        self.lambda().ln() + ln_beta + self.ln_one_minus_r_over_r
    }

    fn block_prob(&self, ln_tree_event_factor: f64, block_len: usize) -> f64 {
        ln_tree_event_factor
            + (block_len as f64 - 1.0) * (ln_tree_event_factor.exp()).ln_1p()
            + (block_len as f64) * self.log_r
    }

    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        let blocks = blocks_of_alignment(msa);
        merge_fragmentation_with_blocks(&blocks, &self.additional_blocks)
    }
}

#[cfg(test)]
impl Display for TKF92IndelModelAddBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with lambda = {}, mu = {}, r = {}, and additional blocks = {:?}",
            self.lambda(),
            self.mu(),
            self.r(),
            self.additional_blocks
        )
    }
}

/// Builder for the cost using the [`TKF92IndelModelAddBlocks`].
pub struct TKF92IndelAddBlocksCostBuilder<AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    r: f64,
    phylo: PhyloInfo<AA>,
    additional_blocks: Vec<usize>,
}

#[cfg(test)]
impl<AA: AncestralAlignment> TKF92IndelAddBlocksCostBuilder<AA> {
    pub fn new(
        lambda: f64,
        mu: f64,
        r: f64,
        additional_blocks: Vec<usize>,
        phylo: PhyloInfo<AA>,
    ) -> Self {
        Self {
            lambda,
            mu,
            r,
            phylo,
            additional_blocks,
        }
    }

    pub fn build(self) -> Result<TKFIndelCost<TKF92IndelModelAddBlocks, AA>> {
        let (lambda, mu) = validate_lambda_and_mu(self.lambda, self.mu);
        let r = validate_r(self.r);
        let additional_blocks =
            validate_fragmentation(&self.additional_blocks, self.phylo.msa.len());
        let model = TKF92IndelModelAddBlocks {
            params: vec![lambda, mu, r],
            log_r: r.ln(),
            ln_one_minus_r_over_r: ((1.0 - r) / r).ln(),
            additional_blocks,
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
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use approx::assert_relative_eq;

    use crate::alignment::{Sequences, MASA};
    use crate::tkf_model::TKF92FixedIndelCostBuilder;
    use crate::{record_wo_desc as record, tree};

    use super::*;

    #[test]
    #[should_panic]
    fn tkf92_param_range_invalid_index() {
        let model = TKF92IndelModelAddBlocks {
            params: vec![0.5, 1.0, 0.3],
            log_r: 0.0, // cache filled with dummy since it is not needed here
            ln_one_minus_r_over_r: 0.0, // cache filled with dummy since it is not needed here
            additional_blocks: vec![],
        };
        // Use an invalid index
        model.param_range(3);
    }

    #[test]
    fn tkf92_add_blocks_model_fmt() {
        let tkf_indel_model = TKF92IndelModelAddBlocks {
            params: vec![1.1, 2.0, 0.3],
            log_r: 0.0,                 // cache filled with dummy since it is not printed
            ln_one_minus_r_over_r: 0.0, // cache filled with dummy since it is not printed
            additional_blocks: vec![1, 2],
        };

        let fmt = format!("{}", tkf_indel_model);

        assert_eq!(
            fmt,
            "TKF92 with lambda = 1.1, mu = 2, r = 0.3, and additional blocks = [1, 2]"
        );
    }

    #[test]
    fn tkf92_add_blocks_indel_set_param() {
        let mut model = TKF92IndelModelAddBlocks {
            params: vec![1.0, 2.0, 0.3],
            log_r: 0.0,                 // dummy
            ln_one_minus_r_over_r: 0.0, // dummy
            additional_blocks: vec![],  // dummy
        };
        model.set_param(usize::from(TKF92Parameters::Lambda), 1.1);
        assert_eq!(model.lambda(), 1.1);
        model.set_param(usize::from(TKF92Parameters::Mu), 2.1);
        assert_eq!(model.mu(), 2.1);
        model.set_param(usize::from(TKF92Parameters::R), 0.4);
        assert_eq!(model.r(), 0.4);
        assert_eq!(model.log_r, 0.4f64.ln());
        assert_eq!(model.ln_one_minus_r_over_r, ((1.0f64 - 0.4) / 0.4).ln());
    }

    #[test]
    fn tkf_add_blocks_manual_integration_over_fragmentations() {
        // By manually summing over unobserved fragmentations (that confirm with the additionally
        // provided block borders) we can verify that this TKF92 model integrates over all possible
        // fragmentations that are consistent with the MSA and the additional block borders.
        let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
        let seqs = Sequences::new(vec![
            record!("A0", b"AAB---DD"),
            record!("B1", b"-ARAAAWD"),
            record!("I1", b"AAA---AD"),
        ]);
        let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
        let phylo_info = PhyloInfo { msa, tree };
        let lambda = 1.0;
        let mu = 1.1;
        let r = 0.5;
        let additional_blocks = vec![2, 4];

        let tkf92_cost = TKF92IndelAddBlocksCostBuilder::new(
            lambda,
            mu,
            r,
            additional_blocks,
            phylo_info.clone(),
        )
        .build()
        .unwrap();
        let cost = tkf92_cost.logl();

        let mut sum_over_fragmentations_cost = 0.0;

        let fragmentations = [vec![2, 4], vec![2, 4, 5], vec![2, 4, 7], vec![2, 4, 5, 7]];
        for fragmentation in fragmentations {
            let fragment_cost =
                TKF92FixedIndelCostBuilder::new(lambda, mu, r, fragmentation, phylo_info.clone())
                    .build()
                    .unwrap();
            sum_over_fragmentations_cost += fragment_cost.logl().exp();
        }
        sum_over_fragmentations_cost = sum_over_fragmentations_cost.ln();
        assert_relative_eq!(cost, sum_over_fragmentations_cost);
    }
}
