use std::cell::RefCell;

use hashbrown::HashMap;
use log::warn;
use rand::{distr::weighted::WeightedIndex, Rng, RngCore, SeedableRng};

use crate::alignment::{AlignmentSimulation, AncestralAlignment, Sequences};
use crate::alphabets::Alphabet;
use crate::random::RandomGenerator;
use crate::substitution_models::{QMatrix, SubstModel};
use crate::tree::{NodeIdx, Tree};
use crate::{bail, record_wo_desc as record};
use crate::{Result, MAX_BLEN};

pub struct SubstitutionSimulatorBuilder<Q, R>
where
    Q: QMatrix,
    R: Rng + SeedableRng + RngCore,
{
    model: SubstModel<Q>,
    tree: Tree,
    rng: RandomGenerator<R>,
    alignment_length: Option<usize>,
}

impl<Q, R> SubstitutionSimulatorBuilder<Q, R>
where
    Q: QMatrix,
    R: Rng + SeedableRng + RngCore,
{
    pub fn new(
        model: crate::substitution_models::SubstModel<Q>,
        tree: Tree,
        rng: RandomGenerator<R>,
    ) -> Self {
        Self {
            model,
            tree,
            rng,
            alignment_length: None,
        }
    }

    pub fn alignment_length(mut self, length: usize) -> Self {
        if length == 0 {
            warn!("Setting alignment_length to 0 will produce an empty alignment");
        }
        self.alignment_length = Some(length);
        self
    }

    pub fn build(self) -> Result<SubstitutionSimulator<R>> {
        let alignment_length = match self.alignment_length {
            Some(x) => x,
            None => {
                bail!(
                    AlignmentSimulation,
                    "alignment_length must be set before building the substitution simulator"
                )
            }
        };

        let root_dist = WeightedIndex::new(self.model.qmatrix.freqs().as_slice()).unwrap();

        let mut p_weighted = HashMap::with_capacity(self.tree.len());
        for idx in self.tree.preorder().iter().skip(1) {
            let blen = self.tree.node(idx).blen;
            let qmat = self.model.qmatrix.q();
            let p = if blen > MAX_BLEN {
                (qmat * MAX_BLEN).exp()
            } else {
                (qmat * blen).exp()
            };

            let mut column_dists = Vec::with_capacity(p.ncols());
            for col in 0..p.ncols() {
                let column = p.column(col);
                column_dists.push(WeightedIndex::new(column.as_slice()).unwrap());
            }
            p_weighted.insert(*idx, column_dists);
        }

        let alphabet = *Q::alphabet();

        Ok(SubstitutionSimulator {
            tree: self.tree,
            alphabet,
            root_dist,
            p_weighted,
            rng: RefCell::new(self.rng),
            alignment_length,
        })
    }
}

#[derive(Debug)]
pub struct SubstitutionSimulator<R>
where
    R: Rng + SeedableRng + RngCore,
{
    tree: Tree,
    alphabet: Alphabet,
    root_dist: WeightedIndex<f64>,
    /// Probability distributions for each branch (NodeIdx) and each parent character (index in the Vec).
    p_weighted: HashMap<NodeIdx, Vec<WeightedIndex<f64>>>,
    rng: RefCell<RandomGenerator<R>>,
    alignment_length: usize,
}

impl<R> AlignmentSimulation for SubstitutionSimulator<R>
where
    R: Rng + SeedableRng + RngCore,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(&self) -> AA {
        let mut sequences: HashMap<NodeIdx, Vec<usize>> = HashMap::with_capacity(self.tree.len());

        let mut rng = self.rng.borrow_mut();
        let root_seq: Vec<usize> = (0..self.alignment_length)
            .map(|_| rng.sample(&self.root_dist))
            .collect();
        drop(rng);
        sequences.insert(self.tree.root, root_seq);

        for node_idx in self.tree.preorder().iter().skip(1) {
            let parent_idx = self.tree.parent(node_idx).unwrap();
            let parent_seq = sequences.get(&parent_idx).unwrap();
            let column_dists = self.p_weighted.get(node_idx).unwrap();

            let mut rng = self.rng.borrow_mut();
            let child_seq: Vec<usize> = parent_seq
                .iter()
                .map(|&parent_state| rng.sample(&column_dists[parent_state]))
                .collect();

            sequences.insert(*node_idx, child_seq);
        }

        let records: Vec<_> = sequences
            .iter()
            .map(|(node_idx, seq)| {
                let id = self.tree.node_id(node_idx);
                let char_seq: Vec<u8> = seq.iter().map(|&s| self.alphabet.symbols()[s]).collect();
                record!(id, &char_seq)
            })
            .collect();

        let seqs = Sequences::new(records);
        AA::from_aligned_with_ancestral(seqs, &self.tree).unwrap()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use assert_matches::assert_matches;

    use crate::alignment::{Alignment, MASA};
    use crate::random::DefaultGenerator;
    use crate::substitution_models::{dna_models::GTR, SubstModel};
    use crate::{tree, Error};

    use super::*;

    #[test]
    fn test_substitution_simulator() {
        // GTR with chosen freqs and rate parameters
        let model = SubstModel::<GTR>::new(&[0.3, 0.2, 0.2, 0.3], &[0.8, 1.2, 0.9, 1.1, 0.7]);
        let tree = tree!("((A:2.0,B:2.0)AB:2.0,(C:2.0,D:2.0)CD:2.0)R;");
        let rng = DefaultGenerator::new(123);

        let simulator = SubstitutionSimulatorBuilder::new(model, tree.clone(), rng)
            .alignment_length(50)
            .build()
            .unwrap();

        let alignment: MASA = simulator.simulate_ancestral_alignment();

        assert_eq!(alignment.len(), 50);
        assert_eq!(
            alignment.seq_count() + alignment.ancestral_seqs().len(),
            tree.len()
        );
        // no gaps in this simulation, so all sequences should have the same length as the alignment
        for seq in alignment.seqs() {
            assert_eq!(seq.seq().len(), alignment.len());
        }
    }

    #[test]
    fn test_reproducibility() {
        // GTR with same parameters to ensure reproducibility across RNGs
        let model = SubstModel::<GTR>::new(&[0.3, 0.2, 0.2, 0.3], &[0.8, 1.2, 0.9, 1.1, 0.7]);
        let tree = tree!("((A:0.5,B:0.5)AB:0.7,(C:0.6,D:0.6)CD:0.6)R;");

        let rng1 = DefaultGenerator::new(42);
        let rng2 = DefaultGenerator::new(42);

        let simulator1 = SubstitutionSimulatorBuilder::new(model.clone(), tree.clone(), rng1)
            .alignment_length(100)
            .build()
            .unwrap();

        let simulator2 = SubstitutionSimulatorBuilder::new(model, tree.clone(), rng2)
            .alignment_length(100)
            .build()
            .unwrap();

        let alignment1: MASA = simulator1.simulate_ancestral_alignment();
        let alignment2: MASA = simulator2.simulate_ancestral_alignment();

        assert_eq!(
            alignment1.to_string(),
            alignment2.to_string(),
            "Same seed should produce identical alignments"
        );
    }

    #[test]
    fn test_builder_alignment_length_zero() {
        let model = SubstModel::<GTR>::new(&[0.3, 0.2, 0.2, 0.3], &[0.8, 1.2, 0.9, 1.1, 0.7]);
        let tree = tree!("((A:0.5,B:0.5)AB:0.7);");
        let rng = DefaultGenerator::new(42);

        let masa = SubstitutionSimulatorBuilder::new(model, tree, rng)
            .alignment_length(0)
            .build()
            .unwrap()
            .simulate_ancestral_alignment::<MASA>();

        assert_eq!(masa.len(), 0);
    }

    #[test]
    fn test_builder_requires_alignment_length() {
        let model = SubstModel::<GTR>::new(&[0.3, 0.2, 0.2, 0.3], &[0.8, 1.2, 0.9, 1.1, 0.7]);
        let tree = tree!("((A:0.5,B:0.5)AB:0.7);");
        let rng = DefaultGenerator::new(42);

        let result = SubstitutionSimulatorBuilder::new(model, tree, rng).build();
        assert_matches!(result, Err(Error::AlignmentSimulation(msg)) if msg.contains("alignment_length must be set"));
    }
}
