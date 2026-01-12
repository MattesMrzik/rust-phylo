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

/// TKF92 indel model with additional block borders (and without a substitution model),
/// which means that the provided blocks will be used in addition to the blocks determined from
/// the alignment, see [`super::TKFModel::get_blocks`].
#[derive(Clone, Debug, PartialEq)]
pub struct TKF92IndelModelAddBlocks {
    params: Vec<f64>,
    /// precomputed r.ln()
    log_r: f64,
    /// precomputed (1 - r)/r
    one_minus_r_over_r: f64,
    /// Blocks to be used in addition to those determined from the alignment
    additional_blocks: Vec<usize>,
}

impl TKF92IndelModelAddBlocks {
    pub fn r(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::R)]
    }
}

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
                self.one_minus_r_over_r = (1.0 - value) / value;
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
        self.lambda() / self.mu() * self.one_minus_r_over_r
    }

    // TODO: this is not a prob but a factor since it can be > 1, rename?
    fn insertion_prob_at_non_root(&self, beta: f64) -> f64 {
        self.lambda() * beta * self.one_minus_r_over_r
    }

    fn block_prob(&self, tree_event_prob: f64, block_len: usize) -> f64 {
        if tree_event_prob == 1.0 {
            0.0
        } else {
            tree_event_prob.ln()
                + (block_len as f64 - 1.0) * (1.0 + tree_event_prob).ln()
                + (block_len as f64) * self.log_r
        }
    }

    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        let blocks = blocks_of_alignment(msa);
        merge_fragmentation_with_blocks(&blocks, &self.additional_blocks)
    }
}

impl Display for TKF92IndelModelAddBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with lambda = {}, mu = {}, r = {}",
            self.lambda(),
            self.mu(),
            self.r(),
        )
    }
}

/// Builder for the cost using [`TKF92IndelModelAddBlocks`].
pub struct TKF92IndelAddBlocksCostBuilder<AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    r: f64,
    phylo: PhyloInfo<AA>,
    additional_blocks: Vec<usize>,
}

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
            one_minus_r_over_r: (1.0 - r) / r,
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
