use std::cell::RefCell;
use std::fmt::Display;

use anyhow::bail;
use log::warn;
use num_enum::{FromPrimitive, IntoPrimitive};

use crate::evolutionary_models::EvoModel;
use crate::substitution_models::{QMatrix, SubstModel, SubstitutionCostBuilder as SCB};
use crate::tkf_model::{
    TKFCost, TKFIndelCost, TKFIndelModelInfo, DEFAULT_LAMBDA, DEFAULT_LAMBDA_MU_RATIO, DEFAULT_MU,
};
use crate::Result;
use crate::{alignment::AncestralAlignment, phylo_info::PhyloInfo, tkf_model::TKFModel};

#[derive(Debug, Eq, PartialEq, FromPrimitive, IntoPrimitive)]
#[repr(usize)]
enum TKF91Parameters {
    Lambda = 0,
    Mu = 1,
    #[num_enum(catch_all)]
    Invalid(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TKF91IndelModel {
    params: Vec<f64>,
}

impl TKFModel for TKF91IndelModel {
    fn lambda(&self) -> f64 {
        self.params[usize::from(TKF91Parameters::Lambda)]
    }

    fn mu(&self) -> f64 {
        self.params[usize::from(TKF91Parameters::Mu)]
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    /// Sets the parameter if it is valid then returns true, otherwise the parameter is not changed
    /// and false is returned.
    /// This assumes that the other parameter is valid
    fn set_param(&mut self, idx: usize, value: f64) -> bool {
        let param = TKF91Parameters::from_primitive(idx);
        match param {
            TKF91Parameters::Lambda => {
                if value > 0.0 && value < self.mu() {
                    self.params[usize::from(TKF91Parameters::Lambda)] = value;
                    return true;
                }
            }
            TKF91Parameters::Mu => {
                if value > self.lambda() {
                    self.params[usize::from(TKF91Parameters::Mu)] = value;
                    return true;
                }
            }
            TKF91Parameters::Invalid(_) => return false,
        };
        false
    }

    fn param_range(&self, idx: usize) -> crate::likelihood::ParamRange {
        let param = TKF91Parameters::from_primitive(idx);
        match param {
            TKF91Parameters::Lambda => (f64::EPSILON, self.mu()),
            TKF91Parameters::Mu => (self.lambda(), f64::MAX),
            _ => panic!("Invalid parameter index for TKF model: {param:?}"),
        }
    }

    fn insertion_prob_at_root(&self) -> f64 {
        self.lambda() / self.mu()
    }

    fn insertion_prob_at_non_root(&self, beta: f64) -> f64 {
        self.lambda() * beta
    }

    fn block_prob(&self, x: f64, block_len: usize) -> f64 {
        if x == 1.0 {
            0.0
        } else {
            (block_len as f64) * x.ln()
        }
    }

    fn get_blocks<AA: AncestralAlignment>(msa: &AA) -> Vec<usize> {
        (1..msa.len() + 1).collect()
    }
}

impl Display for TKF91IndelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF91 with lambda = {}, mu = {}",
            self.lambda(),
            self.mu(),
        )
    }
}

/// Validates the TKF indel parameters lambda and mu. If they are not valid, they are set to
/// default values and a warning is logged.
/// Returns valid (lambda, mu).
pub(crate) fn validate_lambda_and_mu(lambda: f64, mu: f64) -> (f64, f64) {
    let mut valid_lambda = lambda;
    let mut valid_mu = mu;
    if lambda <= 0.0 && mu <= 0.0 {
        warn!(
            "Both lambda and mu must be positive. Setting lambda to {DEFAULT_LAMBDA} and mu to {DEFAULT_MU}."
        );
        valid_lambda = DEFAULT_LAMBDA;
        valid_mu = DEFAULT_MU;
    } else if lambda <= 0.0 {
        valid_lambda = DEFAULT_LAMBDA_MU_RATIO * mu;
        warn!(
            "Tried to set lambda to invalid value {lambda}. It must be in (0, mu) with mu = {mu}. Setting lambda to {DEFAULT_LAMBDA_MU_RATIO}*mu = {valid_lambda}",
        );
    } else if mu <= lambda {
        valid_mu = lambda / DEFAULT_LAMBDA_MU_RATIO;
        warn!(
            "Tried to set mu to invalid value {mu}. It must be in (lambda, infinity) with lambda = {lambda}. Setting mu to lambda/{DEFAULT_LAMBDA_MU_RATIO} = {valid_mu}"
        );
    }
    (valid_lambda, valid_mu)
}

pub struct TKF91IndelCostBuilder<AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    phylo: PhyloInfo<AA>,
}

impl<AA: AncestralAlignment> TKF91IndelCostBuilder<AA> {
    pub fn new(lambda: f64, mu: f64, phylo: PhyloInfo<AA>) -> Self {
        Self { lambda, mu, phylo }
    }

    pub fn build(self) -> Result<TKFIndelCost<AA, TKF91IndelModel>> {
        let (lambda, mu) = validate_lambda_and_mu(self.lambda, self.mu);
        let model = TKF91IndelModel {
            params: vec![lambda, mu],
        };
        let info = TKFIndelModelInfo::new::<_, TKF91IndelModel>(&self.phylo);
        Ok(TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        })
    }
}

pub struct TKF91CostBuilder<Q: QMatrix, AA: AncestralAlignment> {
    lambda: f64,
    mu: f64,
    subst_model: SubstModel<Q>,
    phylo: PhyloInfo<AA>,
}

impl<Q: QMatrix, AA: AncestralAlignment> TKF91CostBuilder<Q, AA> {
    pub fn new(lambda: f64, mu: f64, subst_model: SubstModel<Q>, phylo: PhyloInfo<AA>) -> Self {
        Self {
            lambda,
            mu,
            subst_model,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFCost<Q, TKF91IndelModel, AA>> {
        if self.phylo.msa.alphabet() != self.subst_model.alphabet() {
            bail!("Alphabet mismatch between model and alignment");
        }

        let (lambda, mu) = validate_lambda_and_mu(self.lambda, self.mu);
        let model = TKF91IndelModel {
            params: vec![lambda, mu],
        };
        let info = TKFIndelModelInfo::new::<_, TKF91IndelModel>(&self.phylo);
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
