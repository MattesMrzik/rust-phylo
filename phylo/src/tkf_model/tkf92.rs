use std::cell::RefCell;
use std::fmt::Display;

use anyhow::bail;
use hashbrown::HashSet;
use log::warn;
use num_enum::{FromPrimitive, IntoPrimitive};

use crate::evolutionary_models::EvoModel;
use crate::likelihood::PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE;
use crate::substitution_models::{QMatrix, SubstModel, SubstitutionCostBuilder as SCB};
use crate::tkf_model::{
    validate_lambda_and_mu, TKFCost, TKFIndelCost, TKFIndelModelInfo, DEFAULT_R,
};
use crate::Result;
use crate::{alignment::AncestralAlignment, phylo_info::PhyloInfo, tkf_model::TKFModel};

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
pub struct TKF92IndelModel {
    params: Vec<f64>,
    /// precomputed r.ln()
    log_r: f64,
    /// precomputed (1 - r)/r
    one_minus_r_over_r: f64,
}

impl TKF92IndelModel {
    pub(crate) fn r(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::R)]
    }
}

impl TKFModel for TKF92IndelModel {
    fn lambda(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::Lambda)]
    }

    fn mu(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::Mu)]
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    /// Sets the parameter if it is valid then returns true,
    /// otherwise the parameter is not changed and false is returned.
    /// This assumes that the other parameters are valid
    fn set_param(&mut self, idx: usize, value: f64) -> bool {
        let param = TKF92Parameters::from_primitive(idx);
        match param {
            TKF92Parameters::Lambda => {
                if value > 0.0 && value < self.mu() {
                    self.params[usize::from(TKF92Parameters::Lambda)] = value;
                    return true;
                }
            }
            TKF92Parameters::Mu => {
                if value > self.lambda() {
                    self.params[usize::from(TKF92Parameters::Mu)] = value;
                    return true;
                }
            }
            TKF92Parameters::R => {
                if value > 0.0 && value < 1.0 {
                    self.params[usize::from(TKF92Parameters::R)] = value;
                    self.log_r = value.ln();
                    self.one_minus_r_over_r = (1.0 - value) / value;
                    return true;
                }
            }
            TKF92Parameters::Invalid(_) => return false,
        };
        false
    }

    fn param_range(&self, idx: usize) -> crate::likelihood::ParamRange {
        let param = TKF92Parameters::from_primitive(idx);
        match param {
            TKF92Parameters::Lambda => (f64::EPSILON, self.mu()),
            TKF92Parameters::Mu => (self.lambda(), f64::MAX),
            TKF92Parameters::R => PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE,
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
        }
    }

    fn insertion_prob_at_root(&self) -> f64 {
        self.lambda() / self.mu() * self.one_minus_r_over_r
    }

    fn insertion_prob_at_non_root(&self, beta: f64) -> f64 {
        self.lambda() * beta * self.one_minus_r_over_r
    }

    fn block_prob(&self, x: f64, block_len: usize) -> f64 {
        if x == 1.0 {
            0.0
        } else {
            x.ln() + (block_len as f64 - 1.0) * (1.0 + x).ln() + (block_len as f64) * self.log_r
        }
    }

    // Determines the block borders from the alignment. A block border is defined as a
    // position where any sequence changes from gap to non-gap or vice versa. Returns a sorted
    // vector of the right exclusive block borders.
    fn get_blocks<AA: AncestralAlignment>(msa: &AA) -> Vec<usize> {
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
        blocks
    }
}

impl Display for TKF92IndelModel {
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

fn validate_r(r: f64) -> f64 {
    let mut valid_r = r;
    if r == 0.0 {
        warn!(
            "Tried to set r to invalid value 0. It must be in (0, 1). Setting r to {DEFAULT_R}. Hint: r = 0 yields special case: TKF91 model, consider using that instead."
        );
        valid_r = DEFAULT_R;
    } else if r <= 0.0 || r >= 1.0 {
        warn!(
            "Tried to set r to invalid value {r}. It must be in (0, 1). Setting r to {DEFAULT_R}",
        );
        valid_r = DEFAULT_R;
    }
    valid_r
}

pub struct TKF92IndelCostBuilder<AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    r: f64,
    phylo: PhyloInfo<AA>,
}

impl<AA: AncestralAlignment> TKF92IndelCostBuilder<AA> {
    pub fn new(lambda: f64, mu: f64, r: f64, phylo: PhyloInfo<AA>) -> Self {
        Self {
            lambda,
            mu,
            r,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFIndelCost<AA, TKF92IndelModel>> {
        let (lambda, mu) = validate_lambda_and_mu(self.lambda, self.mu);
        let r = validate_r(self.r);
        let model = TKF92IndelModel {
            params: vec![lambda, mu, r],
            log_r: r.ln(),
            one_minus_r_over_r: (1.0 - r) / r,
        };
        let info = TKFIndelModelInfo::new::<_, TKF92IndelModel>(&self.phylo);
        Ok(TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        })
    }
}

pub struct TKF92CostBuilder<Q: QMatrix, AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    r: f64,
    subst_model: SubstModel<Q>,
    phylo: PhyloInfo<AA>,
}

impl<Q: QMatrix, AA: AncestralAlignment> TKF92CostBuilder<Q, AA> {
    pub fn new(
        lambda: f64,
        mu: f64,
        r: f64,
        subst_model: SubstModel<Q>,
        phylo: PhyloInfo<AA>,
    ) -> Self {
        Self {
            lambda,
            mu,
            r,
            subst_model,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFCost<Q, TKF92IndelModel, AA>> {
        if self.phylo.msa.alphabet() != self.subst_model.alphabet() {
            bail!("Alphabet mismatch between model and alignment");
        }

        let (lambda, mu) = validate_lambda_and_mu(self.lambda, self.mu);
        let r = validate_r(self.r);
        let model = TKF92IndelModel {
            params: vec![lambda, mu, r],
            log_r: r.ln(),
            one_minus_r_over_r: (1.0 - r) / r,
        };
        let info = TKFIndelModelInfo::new::<_, TKF92IndelModel>(&self.phylo);
        let tkf_cost = TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        };
        Ok(TKFCost {
            indel_cost: tkf_cost,
            subst_cost: SCB::new(self.subst_model, self.phylo).build().unwrap(),
        })
    }
}
